"""补测连续进度指标（state0 候选，用已保存动作数组重放）。

不重跑策略：加载 candidate_repro_state0.json 中保存的实际动作数组，
从同一 snapshot 恢复，逐步执行两步前缀，记录连续进度：

- eef_target_dist: ||robot0_eef_pos - akita_black_bowl_1_pos||
- target_lift: akita_black_bowl_1_pos 的 z（相对初始 z 的变化 = lift）
- target_goal_dist: ||akita_black_bowl_1_pos - plate_1_pos||

每步记录：初始 / step1 后 / step2 后。并区分：
- 零变化（|Δ| ≤ 1e-4）
- 未达阈值（有变化但远未达抓取/放置成功阈值）
- 未测量（指标缺失）
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")

ROOT = "/data/safeVLA"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "dataset", "SafeLIBERO", "safelibero"))
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

TASK_ID = 0
NUM_STEPS_WAIT = 10
EXECUTE_STEPS = 2

STATE_DIR_NAME = os.environ.get("PNSG_STATE_DIR", "stable_dangerous_state")
TAG = os.environ.get("PNSG_TAG", "state0")
STATE_DIR = pathlib.Path(ROOT, "data", "audit", STATE_DIR_NAME)
REPRO_FILE = pathlib.Path(ROOT, "data", "audit", f"candidate_repro_{TAG}.json")

EEF_KEY = "robot0_eef_pos"
TARGET_KEY = "akita_black_bowl_1_pos"
GOAL_KEY = "plate_1_pos"


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


def get_obs(env):
    env._update_observables(force=True)
    return env.env._get_observations()


def measure_trajectory(env, flattened, actions):
    """从 snapshot 恢复，逐步执行，记录进度指标轨迹。"""
    env.sim.set_state_from_flattened(flattened)
    env.sim.forward()

    def snapshot_metrics():
        o = get_obs(env)
        eef = np.asarray(o[EEF_KEY], dtype=np.float64)
        target = np.asarray(o[TARGET_KEY], dtype=np.float64)
        goal = np.asarray(o[GOAL_KEY], dtype=np.float64)
        return {
            "eef": eef.tolist(),
            "target": target.tolist(),
            "goal": goal.tolist(),
            "eef_target_dist": float(np.linalg.norm(eef - target)),
            "target_z": float(target[2]),
            "target_goal_dist": float(np.linalg.norm(target - goal)),
        }

    traj = [snapshot_metrics()]
    for act in actions[:EXECUTE_STEPS]:
        env.step(np.asarray(act, dtype=np.float64))
        traj.append(snapshot_metrics())
    return traj


def classify(delta, eps=1e-4):
    if delta is None:
        return "未测量"
    if abs(delta) <= eps:
        return "零变化"
    return "未达阈值"


def main():
    from libero.libero import benchmark

    meta = json.loads((STATE_DIR / "meta.json").read_text())
    snap = np.load(STATE_DIR / "snapshot.npz")["flattened"]
    repro = json.loads(REPRO_FILE.read_text())
    state_id = meta["state_id"]

    ts = benchmark.get_benchmark_dict()["safelibero_spatial"](safety_level="I")
    task = ts.get_task(TASK_ID)

    cand_keys = ["nominal"] + [f"source_b{b}" for b in [0.25, 0.5, 0.75, 1.0]] + \
                [f"action_b{b}" for b in [0.25, 0.5, 0.75, 1.0]]

    out = {}
    for key in cand_keys:
        acts_list = repro["candidates"][key]["actions"]  # (5, 10, 7)
        # 用 5 个 rep 的动作各测一次，记录轨迹
        trajs = []
        for rep in range(5):
            env, _ = make_env(ts, task, state_id)
            traj = measure_trajectory(env, snap, np.array(acts_list[rep]))
            env.close()
            trajs.append(traj)
        out[key] = trajs

    # 汇总：每候选 3 指标的 初始/末值/变化（5 rep 平均 + 范围）
    summary = {}
    for key in cand_keys:
        trajs = out[key]
        m = {}
        for metric in ("eef_target_dist", "target_z", "target_goal_dist"):
            init_vals = [t[0][metric] for t in trajs]
            final_vals = [t[-1][metric] for t in trajs]
            deltas = [f - i for i, f in zip(init_vals, final_vals)]
            m[metric] = {
                "init_mean": float(np.mean(init_vals)),
                "final_mean": float(np.mean(final_vals)),
                "delta_mean": float(np.mean(deltas)),
                "delta_min": float(min(deltas)),
                "delta_max": float(max(deltas)),
            }
        summary[key] = m

    # 打印报告
    print("=== 连续进度指标（2 步前缀，5 rep 平均，单位 m）===")
    print(f"{'候选':<14} {'EEF-target Δ':>13} {'lift Δ(z)':>11} {'target-goal Δ':>14} {'分类'}")
    for key in cand_keys:
        m = summary[key]
        det = m["eef_target_dist"]["delta_mean"]
        dtz = m["target_z"]["delta_mean"]
        dtg = m["target_goal_dist"]["delta_mean"]
        # 分类以变化最大的指标为准
        cls = max(classify(det), classify(dtz), classify(dtg))
        print(f"{key:<14} {det*1000:>10.2f}mm {dtz*1000:>8.2f}mm {dtg*1000:>11.2f}mm  {cls}")

    out_file = pathlib.Path(ROOT, "data", "audit", f"progress_{TAG}.json")
    out_file.write_text(json.dumps({
        "candidate_trajectories": {k: v for k, v in out.items()},
        "summary": summary,
    }, ensure_ascii=False, indent=2, default=str))
    print(f"\n已保存: {out_file}")


if __name__ == "__main__":
    main()
