"""PNSG 12 状态固定批次（deployment 身份）。

冻结规则：
- 从 episode 4 起（state0/1/3 已作为开发案例单列），扫描上限 30 个 episode。
- 每个独立 episode 最多纳入一个状态：取首个通过两层 5/5 nominal 稳定性门的 query。
- deployment obs 协议、完整 snapshot、冻结九候选、五进程评估。
- 达到 12 个合格状态或耗尽 30 个 episode 即停止。
- 不根据 Source/Action 结果决定纳入/跳过/提前结束；所有筛选失败与原因保留。

编排：主进程加载策略一次，逐 episode 筛选并保存；每找到一个合格状态，
subprocess 调 pnsg_candidate_repro.py 跑五进程评估（保持跨进程隔离），
再调 pnsg_measure_progress.py 测连续进度。

主比较单位 = 状态：两组都能修复 / 仅 Source / 仅 Action / 都不能。
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")

ROOT = "/data/safeVLA"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "dataset", "SafeLIBERO", "safelibero"))
sys.path.insert(0, "/data/2608.SDL/MojocoProj/src/pi05")

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

from tools.pnsg_find_stable_state import (  # noqa: E402
    SUITE, LEVEL, TASK_ID, find_stable_dangerous_state,
)

BATCH_START_EPISODE = 4
BATCH_MAX_EPISODES = 30
BATCH_TARGET_STATES = 12
AUDIT = pathlib.Path(ROOT, "data", "audit")
REPRO_SCRIPT = pathlib.Path(ROOT, "tools", "pnsg_candidate_repro.py")
PROGRESS_SCRIPT = pathlib.Path(ROOT, "tools", "pnsg_measure_progress.py")


def save_state(ds, out_dir):
    """保存入选状态（snapshot/z0/nominal/obs 用 npz，meta 用 json）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {k: v for k, v in ds.items() if k not in ("snapshot", "obs", "z0", "raw", "nominal",
                                                     "layer1_results", "layer2_results")}
    meta["layer1_results"] = ds["layer1_results"]
    meta["layer2_results"] = ds["layer2_results"]
    meta["nominal_sha256"] = hashlib.sha256(
        np.asarray(ds["nominal"]).tobytes()).hexdigest()
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str))
    np.savez(out_dir / "snapshot.npz", flattened=ds["snapshot"]["flattened"])
    np.savez(out_dir / "z0.npz", z0=ds["z0"])
    np.savez(out_dir / "nominal.npz", raw=ds["raw"], nominal=ds["nominal"])
    obs_arrays = {k: np.asarray(v) for k, v in ds["obs"].items() if v is not None}
    np.savez(out_dir / "obs.npz", **obs_arrays)


def run_subprocess(script, state_dir_name, tag):
    subprocess.run(
        [sys.executable, str(script)],
        check=True, cwd=ROOT,
        env={**os.environ, "PYTHONPATH": ROOT,
             "PNSG_STATE_DIR": state_dir_name, "PNSG_TAG": tag},
    )


def classify_state(repro):
    """状态级分类：Source 是否能修复、Action 是否能修复。"""
    cands = repro["candidates"]
    src_pass = any(v["pass_count"] == 5 for k, v in cands.items() if k.startswith("source"))
    act_pass = any(v["pass_count"] == 5 for k, v in cands.items() if k.startswith("action"))
    if src_pass and act_pass:
        return "both"
    if src_pass:
        return "source_only"
    if act_pass:
        return "action_only"
    return "neither"


def min_passing(repro, group):
    """按冻结规则选「修正幅度最小」的通过候选，返回 (key, record, delta_norm)。"""
    cands = repro["candidates"]
    passing = []
    for k, v in cands.items():
        if not k.startswith(group):
            continue
        if v["pass_count"] != 5:
            continue
        # 幅度：source 用 delta_norm（α‖Δz*‖），action 用 ‖βΔa*‖
        # 从 frozen 恢复 α_scale / da_norm / r_star
        f = repro["frozen"]
        beta = float(k.split("_b")[1])
        if group == "source":
            delta = beta * f["alpha_scale"] * f["r_star"]
        else:
            delta = beta * f["da_norm"]
        passing.append((delta, k, v))
    if not passing:
        return None
    passing.sort(key=lambda x: x[0])
    delta, k, v = passing[0]
    return {"key": k, "beta": float(k.split("_b")[1]), "delta_norm": delta,
            "c_act": None, "c_act_weighted": None, "native_dists": v["native_dists"]}


def compute_c_act(repro, cand_key):
    """该候选实际部署 pose 相对 nominal 的修正代价（5 进程平均）。"""
    from src.pnsg.crn import action_sha256  # noqa
    cands = repro["candidates"]
    nom_acts = np.array(cands["nominal"]["actions"])
    acts = np.array(cands[cand_key]["actions"])
    TRANS, ROT = 0.01039, 0.09197
    S = np.array([TRANS] * 3 + [ROT] * 3)
    u = w = 0.0
    for i in range(len(acts)):
        dp = acts[i, :, :6] - nom_acts[i, :, :6]
        u += float(np.linalg.norm(dp))
        w += float(np.linalg.norm(dp @ S))
    n = len(acts)
    return {"c_act": u / n, "c_act_weighted": w / n}


def main():
    import argparse
    from libero.libero import benchmark
    from src.pnsg.policy_adapter import PnsgPolicy

    ap = argparse.ArgumentParser()
    ap.add_argument("--start-episode", type=int, default=BATCH_START_EPISODE)
    ap.add_argument("--target-states", type=int, default=BATCH_TARGET_STATES)
    ap.add_argument("--max-episodes", type=int, default=BATCH_MAX_EPISODES)
    ap.add_argument("--limit", type=int, default=None,
                    help="调试用：最多扫描的 episode 数（覆盖 max-episodes）")
    args = ap.parse_args()

    ts = benchmark.get_benchmark_dict()[SUITE](safety_level=LEVEL)
    task = ts.get_task(TASK_ID)
    policy = PnsgPolicy()

    n_episodes = args.limit if args.limit is not None else args.max_episodes
    episodes = list(range(args.start_episode, args.start_episode + n_episodes))

    print(f"=== PNSG 12 状态批次（deployment，episode {episodes[0]}..{episodes[-1]}，"
          f"目标 {args.target_states} 状态）===")

    collected = []
    failures = []

    try:
        for ep in episodes:
            if len(collected) >= args.target_states:
                print(f"\n已达目标状态数 {args.target_states}，停止。")
                break

            print(f"\n{'=' * 60}\nepisode {ep} 筛选…")
            ep_fail_log = []
            ds = find_stable_dangerous_state(ts, task, policy, [ep],
                                             fail_log=ep_fail_log,
                                             obs_protocol="deployment")
            if ds is None:
                failures.append({"episode": ep, "reason": "无合格危险状态",
                                 "query_failures": ep_fail_log})
                print(f"episode {ep}: 无合格状态（query 级失败 {len(ep_fail_log)} 条）")
                continue

            state_dir_name = f"stable_dangerous_state_batch_{ep}"
            tag = f"batch_s{ep}"
            save_state(ds, AUDIT / state_dir_name)
            print(f"episode {ep}: 入选 query{ds['query']} obstacle={ds['obstacle']} "
                  f"native={ds['layer1_results'][0]['native_dist']*1000 if ds['layer1_results'][0]['native_dist'] is not None else None}")

            # 五进程评估
            print(f"  跑五进程评估…")
            run_subprocess(REPRO_SCRIPT, state_dir_name, tag)
            # 连续进度
            print(f"  测连续进度…")
            run_subprocess(PROGRESS_SCRIPT, state_dir_name, tag)

            repro = json.loads((AUDIT / f"candidate_repro_{tag}.json").read_text())
            cat = classify_state(repro)

            src_min = min_passing(repro, "source")
            act_min = min_passing(repro, "action")
            for m in (src_min, act_min):
                if m is not None:
                    m.update(compute_c_act(repro, m["key"]))

            rec = {
                "episode": ep, "query": ds["query"], "obstacle": ds["obstacle"],
                "category": cat,
                "nominal_native_mm": ds["layer1_results"][0]["native_dist"] * 1000
                if ds["layer1_results"][0]["native_dist"] is not None else None,
                "snapshot_sha256": repro["frozen"]["snapshot_sha256"],
                "obs_sha256": ds["obs_sha256"],
                "source_pass": any(v["pass_count"] == 5 for k, v in repro["candidates"].items()
                                   if k.startswith("source")),
                "action_pass": any(v["pass_count"] == 5 for k, v in repro["candidates"].items()
                                   if k.startswith("action")),
                "source_min_passing": src_min,
                "action_min_passing": act_min,
            }
            collected.append(rec)
            print(f"  结果: {cat}  (source={rec['source_pass']}, action={rec['action_pass']})")
    finally:
        policy.close()

    # 汇总
    from collections import Counter
    cat_count = Counter(r["category"] for r in collected)
    summary = {
        "start_episode": args.start_episode,
        "n_episodes_scanned": len(episodes),
        "target_states": args.target_states,
        "n_states_collected": len(collected),
        "category_counts": dict(cat_count),
        "collected": collected,
        "failures": failures,
    }
    out = AUDIT / "batch_12state_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print(f"\n{'=' * 60}")
    print(f"=== 批次汇总 ===")
    print(f"扫描 episode: {len(episodes)}，收集状态: {len(collected)}")
    for cat, n in sorted(cat_count.items()):
        print(f"  {cat}: {n}")
    print(f"汇总: {out}")


if __name__ == "__main__":
    main()
