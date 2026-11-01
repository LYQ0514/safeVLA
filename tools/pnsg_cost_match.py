"""PNSG 代价匹配诊断（结果后的探索性解释分析，不改写批次结论）。

范围冻结：
- 只分析两个 source_only 状态（ep06/q28、ep07/q18）。
- 取 Source 成功候选的最终部署动作代价 C_act（5 rep 平均）。
- 沿原 Action 修正方向 Δa* 缩放幅度 s，使 clipping 后的 C_act 匹配该值。
  匹配只查询动作代价，不查询认证结果。
- 每个状态只产生一个匹配候选（统一 s），按原五次重复 + 同一 C_2 评估。
- 目标代价不可达时明确记录（不改变方向补救）。

解释边界（预先固定）：
- Action 成功 → 原独占修复差异可由候选幅度覆盖不足解释。
- Action 仍失败 → 该状态上 Source 的修正方向在相同动作代价下更有效。
- 结果翻转或不可达 → 该诊断无法明确归因。
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import time
import warnings

warnings.filterwarnings("ignore")

ROOT = "/data/safeVLA"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "dataset", "SafeLIBERO", "safelibero"))
sys.path.insert(0, "/data/2608.SDL/MojocoProj/src/pi05")

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

from src.pnsg.certifier import Certifier  # noqa: E402

AUDIT = pathlib.Path(ROOT, "data", "audit")
N_REPRO = 5
EXECUTE_STEPS = 2
TRANS, ROT = 0.01039, 0.09197
S_WEIGHT = np.array([TRANS] * 3 + [ROT] * 3)

# source_only 状态：episode -> (source 成功候选 key)
SOURCE_ONLY = {6: "source_b0.25", 7: "source_b1.0"}


def c_act_pose(actions_pose, nominal_pose):
    """部署动作代价：候选 pose 相对 nominal pose 的 L2 范数。"""
    return float(np.linalg.norm(np.asarray(actions_pose) - np.asarray(nominal_pose)))


def load_state(ep):
    meta = json.loads((AUDIT / f"stable_dangerous_state_batch_{ep}" / "meta.json").read_text())
    snap = np.load(AUDIT / f"stable_dangerous_state_batch_{ep}" / "snapshot.npz")["flattened"]
    z0 = np.load(AUDIT / f"stable_dangerous_state_batch_{ep}" / "z0.npz")["z0"]
    fz = np.load(AUDIT / f"candidate_repro_frozen_batch_s{ep}" / "frozen.npz")
    da = fz["da"].reshape(10, 6)  # (60,) -> (10,6)
    return meta, snap, z0, da


def collect_reps(ep, src_key):
    """收集 5 个 rep 的 nominal + source 成功候选动作。"""
    nom_acts = []
    src_acts = []
    for rep in range(N_REPRO):
        d = json.loads((AUDIT / f"candidate_repro_batch_s{ep}_rep{rep}.json").read_text())
        nom_acts.append(np.array(d["candidates"]["nominal"]["actions"]))
        src_acts.append(np.array(d["candidates"][src_key]["actions"]))
    return nom_acts, src_acts


def compute_target_and_reachability(ep, src_key):
    """算 target C_act，并求沿 Δa* 方向的匹配 s 及可达性。"""
    nom_acts, src_acts = collect_reps(ep, src_key)
    _, _, _, da = load_state(ep)

    # target C_act：source 成功候选 5 rep 平均
    c_act_src = [c_act_pose(s[:, :6], n[:, :6]) for s, n in zip(src_acts, nom_acts)]
    target = float(np.mean(c_act_src))

    # 沿 Δa* 缩放：C_act(s) = mean ||clip(nom + s·da) - nom||
    def c_act_match(s):
        vals = []
        for n in nom_acts:
            posed = np.clip(n[:, :6] + s * da, -1.0, 1.0)
            vals.append(c_act_pose(posed, n[:, :6]))
        return float(np.mean(vals))

    # 扫描 s ∈ [0, 30]，找最接近 target 的 s
    ss = np.linspace(0.0, 30.0, 30001)
    best_s, best_resid = None, np.inf
    for s in ss:
        v = c_act_match(s)
        resid = abs(v - target)
        if resid < best_resid:
            best_s, best_resid = float(s), resid
    # 饱和值（s -> inf 时的 C_act）
    sat_val = c_act_match(1e9)

    return {
        "target_c_act": target,
        "c_act_src_per_rep": c_act_src,
        "best_s": best_s,
        "best_residual": best_resid,
        "saturation_c_act": sat_val,
        "reachable": sat_val >= target - 1e-6,
    }


def certify_match(ep, src_key, s):
    """用统一 s 生成匹配候选，5 次 C_2 认证。"""
    meta, snap, z0, da = load_state(ep)
    nom_acts, _ = collect_reps(ep, src_key)
    state_id = meta["state_id"]

    # 匹配候选动作（每个 rep 用该 rep 的 nominal）
    match_acts = []
    for n in nom_acts:
        posed = np.clip(n[:, :6] + s * da, -1.0, 1.0)
        act = n.copy()
        act[:, :6] = posed
        act[:, 6] = n[:, 6]  # gripper 保持 nominal
        match_acts.append(act)

    results = []
    for rep in range(N_REPRO):
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        ts = benchmark.get_benchmark_dict()["safelibero_spatial"](safety_level="I")
        task = ts.get_task(0)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256,
                                 controller="OSC_POSE")
        env.seed(7)
        env.reset()
        env.set_init_state(ts.get_task_init_states(0)[state_id])
        dummy = np.zeros(7, dtype=np.float32); dummy[-1] = -1.0
        for _ in range(10):
            env.step(dummy)
        env.sim.set_state_from_flattened(snap)
        env.sim.forward()
        certifier = Certifier(env)
        t0 = time.time()
        c2 = certifier.certify_prefix(match_acts[rep][:EXECUTE_STEPS])
        dt = time.time() - t0
        progress = float(env.check_success())
        results.append({
            "rep": rep,
            "c2_pass": c2.pass_,
            "c2_reason": c2.reason,
            "native_dist": c2.min_robot_obstacle_dist,
            "progress": progress,
            "time_s": dt,
            "c_act": c_act_pose(match_acts[rep][:, :6], nom_acts[rep][:, :6]),
        })
        env.close()
    return results


def main():
    out_records = []
    for ep, src_key in sorted(SOURCE_ONLY.items()):
        meta, snap, z0, da = load_state(ep)
        print(f"\n{'=' * 70}")
        print(f"=== ep{ep} query{meta['query']} {meta['obstacle']} (source_only) ===")
        print(f"  Source 成功候选: {src_key}")

        info = compute_target_and_reachability(ep, src_key)
        print(f"  target C_act = {info['target_c_act']:.4f}  "
              f"(per-rep: {[round(x, 4) for x in info['c_act_src_per_rep']]})")
        print(f"  Action 方向饱和 C_act = {info['saturation_c_act']:.4f}")
        print(f"  可达: {info['reachable']}  best_s={info['best_s']:.4f}  "
              f"residual={info['best_residual']:.4f}")

        rec = {
            "episode": ep, "query": meta["query"], "obstacle": meta["obstacle"],
            "source_candidate": src_key,
            "target_c_act": info["target_c_act"],
            "c_act_src_per_rep": info["c_act_src_per_rep"],
            "saturation_c_act": info["saturation_c_act"],
            "reachable": info["reachable"],
        }

        if not info["reachable"]:
            print(f"  → 目标代价不可达（Action 方向 clip 后最大 C_act = "
                  f"{info['saturation_c_act']:.4f} < target {info['target_c_act']:.4f}）")
            rec["outcome"] = "unreachable"
            out_records.append(rec)
            continue

        s = info["best_s"]
        results = certify_match(ep, src_key, s)
        n_pass = sum(1 for r in results if r["c2_pass"])
        c_act_match_mean = float(np.mean([r["c_act"] for r in results]))
        residual = abs(c_act_match_mean - info["target_c_act"])

        print(f"  → 匹配候选 s={s:.4f}，C_act 匹配均值={c_act_match_mean:.4f} "
              f"(residual={residual:.4f})")
        print(f"  → C_2 通过 {n_pass}/5")
        for r in results:
            nd = "无" if r["native_dist"] is None else f"{r['native_dist']*1000:.2f}mm"
            print(f"     rep{r['rep']}: pass={r['c2_pass']} native={nd} "
                  f"progress={r['progress']} C_act={r['c_act']:.4f} {r['c2_reason']}")

        if n_pass == 5:
            outcome = "action_success"
        elif n_pass == 0:
            outcome = "action_fail"
        else:
            outcome = "flip"

        rec.update({
            "outcome": outcome,
            "match_s": s,
            "c_act_match_mean": c_act_match_mean,
            "residual": residual,
            "n_pass": n_pass,
            "results": results,
        })
        out_records.append(rec)

    out = AUDIT / "cost_match_result.json"
    out.write_text(json.dumps(out_records, ensure_ascii=False, indent=2, default=str))
    print(f"\n结果: {out}")

    # 解释表
    print(f"\n=== 解释（按预先冻结的边界）===")
    for rec in out_records:
        print(f"  ep{rec['episode']}: {rec['outcome']}")


if __name__ == "__main__":
    main()
