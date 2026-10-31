"""候选跨进程复现评估（协议 v13）。

5 个独立进程 × 9 候选（Nominal + Source×4 + Action×4）= 45 次两步认证。

冻结：C,b,J,Δz*,Δa*（只在主进程求解一次，落盘）。
每个独立进程：加载同一 snapshot/obs/z0（验证身份），重新生成 nominal 和
Source 候选（跨进程推理差异），Action 候选用同一 nominal 基准 + 冻结 Δa*。

每个候选报告：C_2 通过次数/5、native 距离逐次值、动作跨重复误差、耗时。

标签：5/5 全通过；1-4/5 翻转；0/5 全失败。

保存：实际动作数组、hash、clipping、native 距离、接触 pair、执行后进度。
"""

from __future__ import annotations

import hashlib
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
import pi05_libero_runner as R  # noqa: E402

from src.pnsg.crn import action_sha256, z_sha256  # noqa: E402
from src.pnsg.source_solver import minimum_norm_halfspaces  # noqa: E402
from src.pnsg.certifier import Certifier  # noqa: E402
from src.pnsg.geometry_proxy import (  # noqa: E402
    build_control_points, build_obstacles, selected_constraints,
)

# obs 身份哈希覆盖策略实际消费的字段（与 obs_to_policy_obs 一致）
OBS_IDENTITY_KEYS = ("agentview_image", "robot0_eye_in_hand_image",
                     "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")


def obs_sha256(obs):
    h = hashlib.sha256()
    for k in OBS_IDENTITY_KEYS:
        if k in obs:
            h.update(k.encode())
            h.update(np.asarray(obs[k]).tobytes())
    return h.hexdigest()

SUITE = "safelibero_spatial"
LEVEL = "I"
TASK_ID = 0
EXECUTE_STEPS = 2
NUM_STEPS_WAIT = 10
R_BUDGET = 12.0
N_REPRO = 5
BETA = [0.25, 0.5, 0.75, 1.0]
H_FLOOR = 0.005
MAX_CONSTRAINTS = 3
MARGIN = 0.01

STATE_DIR_NAME = os.environ.get("PNSG_STATE_DIR", "stable_dangerous_state")
TAG = os.environ.get("PNSG_TAG", "state0")
STATE_DIR = pathlib.Path(ROOT, "data", "audit", STATE_DIR_NAME)
FROZEN_DIR = pathlib.Path(ROOT, "data", "audit", f"candidate_repro_frozen_{TAG}")


def _sha256(arr):
    return hashlib.sha256(np.asarray(arr).tobytes()).hexdigest()


def load_state():
    meta = json.loads((STATE_DIR / "meta.json").read_text())
    snap = np.load(STATE_DIR / "snapshot.npz")
    z0 = np.load(STATE_DIR / "z0.npz")["z0"]
    nom = np.load(STATE_DIR / "nominal.npz")
    return meta, snap["flattened"], z0, nom["raw"], nom["nominal"]


def load_obs():
    """加载筛选时保存的 obs（策略前向输入，避免从 snapshot 重建引入不一致）。"""
    with np.load(STATE_DIR / "obs.npz") as f:
        return {k: f[k] for k in f.files}


def make_env(ts, task, state_id):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256,
                             controller="OSC_POSE")
    env.seed(7)
    obs = env.reset()
    obs = env.set_init_state(ts.get_task_init_states(TASK_ID)[state_id])
    dummy = np.zeros(7, dtype=np.float32); dummy[-1] = -1.0
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = env.step(dummy)
    return env, obs


def restore(env, flattened):
    env.sim.set_state_from_flattened(flattened)
    env.sim.forward()


def get_obs(env):
    env._update_observables(force=True)
    return env.env._get_observations()


def _role_geoms(env):
    m = env.sim.model._model
    robot = {
        g for g in range(m.ngeom)
        if m.geom_contype[g] > 0
        and (env.sim.model.body_names[m.geom_bodyid[g]].startswith("robot0")
             or env.sim.model.body_names[m.geom_bodyid[g]].startswith("gripper0"))
    }
    obstacle_names = set(
        n.replace("_joint0", "") for n in env.sim.model.joint_names if "obstacle" in n
    )
    obstacle = {
        g for g in range(m.ngeom)
        if m.geom_contype[g] > 0
        and any(env.sim.model.body_names[m.geom_bodyid[g]].startswith(o)
                for o in obstacle_names)
    }
    return robot, obstacle


def robot_obstacle_pairs(env):
    m = env.sim.model._model
    d = env.sim.data._data
    robot, obstacle = _role_geoms(env)
    pairs = []
    for c in range(d.ncon):
        g1, g2 = d.contact[c].geom1, d.contact[c].geom2
        if (g1 in robot and g2 in obstacle) or (g2 in robot and g1 in obstacle):
            b1 = env.sim.model.body_names[m.geom_bodyid[g1]]
            b2 = env.sim.model.body_names[m.geom_bodyid[g2]]
            pairs.append({"body1": b1, "body2": b2, "dist": float(d.contact[c].dist)})
    return pairs


def certify(ts, task, state_id, flattened, actions):
    env, _ = make_env(ts, task, state_id)
    restore(env, flattened)
    certifier = Certifier(env)
    initial_pairs = robot_obstacle_pairs(env)
    c2 = certifier.certify_prefix(actions[:EXECUTE_STEPS])
    final_pairs = robot_obstacle_pairs(env)
    progress = float(env.check_success())
    init_keys = {(p["body1"], p["body2"]) for p in initial_pairs}
    new_pairs = [p for p in final_pairs if (p["body1"], p["body2"]) not in init_keys
                 and (p["body2"], p["body1"]) not in init_keys]
    env.close()
    return {
        "c2_pass": c2.pass_,
        "c2_reason": c2.reason,
        "native_dist": c2.min_robot_obstacle_dist,
        "initial_pairs": initial_pairs,
        "final_pairs": final_pairs,
        "new_pairs": new_pairs,
        "progress": progress,
    }


def clip_report(raw, clipped):
    diff = np.asarray(clipped, dtype=np.float64) - np.asarray(raw, dtype=np.float64)
    triggered = bool(np.any(np.abs(diff) > 1e-9))
    return {"triggered": triggered, "max_abs": float(np.max(np.abs(diff))) if diff.size else 0.0}


def solve_and_freeze():
    """主进程：求解并冻结 C,b,J,Δz*,Δa*，落盘。"""
    from libero.libero import benchmark
    from src.pnsg.policy_adapter import PnsgPolicy

    meta, flattened, z0, raw0, nominal0 = load_state()
    state_id = meta["state_id"]
    prompt = meta["prompt"]

    print("=== 主进程：求解冻结量 ===")
    print(f"snapshot sha256: {_sha256(flattened)}")
    print(f"z0 sha256: {z_sha256(z0)}")

    ts = benchmark.get_benchmark_dict()[SUITE](safety_level=LEVEL)
    task = ts.get_task(TASK_ID)
    policy = PnsgPolicy()

    env, obs = make_env(ts, task, state_id)
    restore(env, flattened)
    obs_snap = load_obs()
    ob_key = f"{meta['obstacle']}_pos"
    assert ob_key in obs_snap, f"obs 缺少 {ob_key}"
    print(f"恢复后 {ob_key} = {np.round(obs_snap[ob_key], 4)}")

    cps = build_control_points(env, obs_snap)
    obstacles = build_obstacles(env, obs_snap)
    rows = selected_constraints(cps, obstacles, nominal0[:, :6],
                                h_floor=H_FLOOR, max_constraints=MAX_CONSTRAINTS, margin=MARGIN)
    env.close()

    print(f"选中 {len(rows)} 条约束")
    C = np.vstack([c for c, _ in rows])
    b = np.array([bb for _, bb in rows])
    _, _, _, ext_J = policy.generate_with_jacobian(
        z0, R.obs_to_policy_obs(obs_snap, prompt, image_size=224))

    dz, _ = minimum_norm_halfspaces(C @ ext_J, b)
    da, _ = minimum_norm_halfspaces(C, b)
    r_star = float(np.linalg.norm(dz))
    alpha_scale = min(1.0, R_BUDGET / r_star)
    print(f"Δz* 范数 r_star={r_star:.1f}, budget_scale={alpha_scale:.3f}")
    print(f"Δa* 范数={np.linalg.norm(da):.3f}")

    policy.close()

    FROZEN_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(FROZEN_DIR / "frozen.npz", dz=dz, da=da)
    frozen_meta = {
        "r_star": r_star, "alpha_scale": alpha_scale,
        "da_norm": float(np.linalg.norm(da)),
        "n_constraints": len(rows),
        "C": C.tolist(), "b": b.tolist(),
        "z0_sha256": z_sha256(z0),
        "snapshot_sha256": _sha256(flattened),
    }
    (FROZEN_DIR / "frozen_meta.json").write_text(
        json.dumps(frozen_meta, ensure_ascii=False, indent=2, default=str))
    print(f"冻结量已落盘: {FROZEN_DIR}")
    return frozen_meta


def worker(rep_id):
    """独立进程：加载冻结量，重新生成 nominal + source，认证 9 候选。"""
    from libero.libero import benchmark
    from src.pnsg.policy_adapter import PnsgPolicy

    meta, flattened, z0, raw0, nominal0 = load_state()
    state_id = meta["state_id"]
    prompt = meta["prompt"]
    frozen_meta = json.loads((FROZEN_DIR / "frozen_meta.json").read_text())
    fz = np.load(FROZEN_DIR / "frozen.npz")
    dz = fz["dz"]; da = fz["da"]
    alpha_scale = frozen_meta["alpha_scale"]

    # 身份验证
    assert z_sha256(z0) == frozen_meta["z0_sha256"], "z0 身份不符"
    assert _sha256(flattened) == frozen_meta["snapshot_sha256"], "snapshot 身份不符"

    ts = benchmark.get_benchmark_dict()[SUITE](safety_level=LEVEL)
    task = ts.get_task(TASK_ID)

    policy = PnsgPolicy()

    env, obs = make_env(ts, task, state_id)
    restore(env, flattened)
    obs_snap = load_obs()
    env.close()

    # obs 身份校验
    if "obs_sha256" in meta:
        assert obs_sha256(obs_snap) == meta["obs_sha256"], "obs 身份不符"

    po = R.obs_to_policy_obs(obs_snap, prompt, image_size=224)

    cand_keys = ["nominal"] + [f"source_b{beta}" for beta in BETA] + [f"action_b{beta}" for beta in BETA]
    rep_result = {}

    # 重新生成 nominal（跨进程推理差异）
    raw_rep = policy.generate(z0, po)
    nominal_rep = np.clip(raw_rep, -1, 1)
    nominal_rep[:, 6] = nominal0[:, 6]

    # 生成 9 个候选（记录 pre-clip raw + clipped）
    cands = {"nominal": (raw_rep, nominal_rep)}
    for beta in BETA:
        alpha = beta * alpha_scale
        z_c = z0 + (alpha * dz).reshape(z0.shape)
        src_raw = policy.generate(z_c, po)
        src_clip = np.clip(src_raw, -1, 1)
        src_clip[:, 6] = nominal0[:, 6]
        cands[f"source_b{beta}"] = (src_raw, src_clip)
    for beta in BETA:
        act_raw = nominal_rep.copy()
        act_raw[:, :6] = nominal_rep[:, :6] + beta * da.reshape(nominal_rep[:, :6].shape)
        act = np.clip(act_raw, -1, 1)
        act[:, 6] = nominal0[:, 6]
        cands[f"action_b{beta}"] = (act_raw, act)

    for key, (raw_arr, actions) in cands.items():
        t0 = time.time()
        ev = certify(ts, task, state_id, flattened, actions)
        dt = time.time() - t0
        rep_result[key] = {
            "c2_pass": ev["c2_pass"],
            "c2_reason": ev["c2_reason"],
            "native_dist": ev["native_dist"],
            "final_pairs": ev["new_pairs"],
            "progress": ev["progress"],
            "action_hash": action_sha256(actions),
            "clip": clip_report(raw_arr, actions),
            "time_s": dt,
            "actions": actions.tolist(),
        }
        print(f"[rep{rep_id}] {key:<14} C2={ev['c2_pass']} "
              f"native={'无' if ev['native_dist'] is None else round(ev['native_dist']*1000, 3)} "
              f"| {ev['c2_reason']}")

    policy.close()

    out = pathlib.Path(ROOT, "data", "audit", f"candidate_repro_{TAG}_rep{rep_id}.json")
    out.write_text(json.dumps({"rep_id": rep_id, "candidates": rep_result},
                              ensure_ascii=False, indent=2, default=str))
    print(f"[rep{rep_id}] 结果: {out}")


def aggregate():
    cand_keys = ["nominal"] + [f"source_b{beta}" for beta in BETA] + [f"action_b{beta}" for beta in BETA]
    agg = {k: {"pass": 0, "native_dists": [], "c2_reasons": [], "times": [],
               "hashes": [], "progress": [], "actions": [], "clip": [],
               "final_pairs": []} for k in cand_keys}
    for rep in range(N_REPRO):
        d = json.loads(pathlib.Path(ROOT, "data", "audit", f"candidate_repro_{TAG}_rep{rep}.json").read_text())
        for k in cand_keys:
            r = d["candidates"][k]
            agg[k]["native_dists"].append(r["native_dist"])
            agg[k]["c2_reasons"].append(r["c2_reason"])
            agg[k]["times"].append(r["time_s"])
            agg[k]["hashes"].append(r["action_hash"])
            agg[k]["progress"].append(r["progress"])
            agg[k]["actions"].append(r["actions"])
            agg[k]["clip"].append(r["clip"])
            agg[k]["final_pairs"].append(r["final_pairs"])
            if r["c2_pass"]:
                agg[k]["pass"] += 1
    return agg, cand_keys


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=None)
    args = ap.parse_args()

    if args.worker is not None:
        worker(args.worker)
        return

    solve_and_freeze()

    print(f"\n=== 启动 {N_REPRO} 个独立进程 ===")
    for rep in range(N_REPRO):
        print(f"\n--- 独立进程 {rep} ---")
        subprocess.run([sys.executable, __file__, "--worker", str(rep)],
                       check=True, cwd=ROOT,
                       env={**os.environ, "PYTHONPATH": ROOT})

    agg, cand_keys = aggregate()

    print("\n" + "=" * 72)
    print("=== 候选跨进程复现结果（5 独立进程）===")
    print(f"{'候选':<14} {'通过':>6} {'native逐次(mm)':>48} {'标签':>10}")
    for key in cand_keys:
        r = agg[key]
        nd = [('无' if d is None else f"{d*1000:.2f}") for d in r["native_dists"]]
        if r["pass"] == N_REPRO:
            label = "5/5"
        elif r["pass"] == 0:
            label = "0/5"
        else:
            label = f"{r['pass']}/5 翻转"
        print(f"{key:<14} {r['pass']:>4}/5  {str(nd):>48}  {label:>10}")

    print("\n=== 动作跨进程唯一 hash（1=5 进程完全一致）===")
    for key in cand_keys:
        unique = len(set(agg[key]["hashes"]))
        print(f"  {key:<14}: {unique}/5 唯一 action hash")

    print("\n=== native 距离范围（mm，None=无接触）===")
    for key in cand_keys:
        nds = [d for d in agg[key]["native_dists"] if d is not None]
        if nds:
            print(f"  {key:<14}: min={min(nds)*1000:.2f} max={max(nds)*1000:.2f} (n_contact={len(nds)})")
        else:
            print(f"  {key:<14}: 全部无接触")

    frozen_meta = json.loads((FROZEN_DIR / "frozen_meta.json").read_text())
    state_meta = json.loads((STATE_DIR / "meta.json").read_text())
    out = pathlib.Path(ROOT, "data", "audit", f"candidate_repro_{TAG}.json")
    payload = {
        "state_id": int(state_meta["state_id"]),
        "query": state_meta["query"],
        "obstacle": state_meta["obstacle"],
        "obs_protocol": state_meta.get("obs_protocol"),
        "obs_sha256": state_meta.get("obs_sha256"),
        "sensing_lag": state_meta.get("sensing_lag"),
        "nominal_sha256": state_meta.get("nominal_sha256"),
        "frozen": frozen_meta,
        "candidates": {
            k: {
                "pass_count": r["pass"],
                "native_dists": r["native_dists"],
                "c2_reasons": r["c2_reasons"],
                "times": r["times"],
                "action_hashes": r["hashes"],
                "progress": r["progress"],
                "clip": r["clip"],
                "final_pairs": r["final_pairs"],
                "actions": r["actions"],
            } for k, r in agg.items()
        },
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    print(f"\n汇总结果: {out}")


if __name__ == "__main__":
    main()
