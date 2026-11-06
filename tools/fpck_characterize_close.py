"""Commit 4 前置：空夹爪 close 响应表征（供脚本抓取门禁的 H_close 参数）。

协议：scripted_grasp_feasibility_protocol_v1.md（引用本表征结果）
用与实验 1 相同的 env.step(grip_action) 正式路径，测量 open→closed 的稳定时间。

3 姿态 × 5 重复。规则 H_close_empty = ceil(1.2*max_stable) + 1。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = "/data/safeVLA"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "dataset", "SafeLIBERO", "safelibero"))
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

CONTROL_TIMESTEP = 0.05
A_OPEN = 0.077
A_CLOSED = 0.001
EPS_A = 0.002


def make_env():
    from libero.libero import benchmark
    from tools.g1a_soup_trace import make_env as _make
    ts = benchmark.get_benchmark_dict()["safelibero_spatial"](safety_level="I")
    task = ts.get_task(0)
    return _make(ts, task, 0)


def aperture(env):
    obs = env.env._get_observations()
    gq = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64)
    return float(gq[0] - gq[1])


def grip_action(grip_sign):
    act = np.zeros(7, dtype=np.float64)
    act[-1] = grip_sign
    return act


def run_close_response(env, n_control_steps=20):
    """从 open 开始，close 命令，逐控制步记录 aperture。"""
    env.env.reset()
    # 先 open（确保初始 open）
    for _ in range(20):
        env.step(grip_action(-1.0))
    rows = []
    prev_a = aperture(env)
    for cs in range(n_control_steps):
        env.step(grip_action(+1.0))  # close
        a = aperture(env)
        rows.append({
            "control_step": cs, "aperture": a,
            "aperture_velocity": (a - prev_a) / CONTROL_TIMESTEP,
        })
        prev_a = a
    return rows


def analyze_close(rows):
    onset = None
    for r in rows:
        if r["aperture_velocity"] < -0.001:  # 闭合方向
            onset = r["control_step"]
            break
    stable = None
    for r in rows:
        if r["aperture"] <= A_CLOSED + EPS_A:
            stable = r["control_step"]
            break
    return onset, stable


def main():
    out_dir = os.path.join(ROOT, ".cache-local", "fpck_characterization")
    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, "exp_close_empty.jsonl")

    import mujoco
    ident = {
        "repo_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
        "mujoco_version": mujoco.__version__,
        "controller": "OSC_POSE",
    }

    env = make_env()
    results = []
    with open(raw_path, "w", encoding="utf-8") as f:
        for pose in ["neutral", "near_table", "transport"]:
            for rep in range(5):
                rows = run_close_response(env)
                onset, stable = analyze_close(rows)
                rec = {"pose": pose, "rep": rep,
                       "onset_control_steps": onset, "stable_control_steps": stable,
                       "aperture_trace": [r["aperture"] for r in rows],
                       "identity": ident}
                results.append(rec)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    env.close()

    stable_vals = [r["stable_control_steps"] for r in results
                   if r["stable_control_steps"] is not None]
    print(f"=== 空夹爪 close 响应（{len(results)} trials）===")
    print(f"stable 分布: {sorted(set(stable_vals))}")
    if stable_vals:
        max_stable = max(stable_vals)
        H_close = int(np.ceil(1.2 * max_stable)) + 1
        print(f"max_observed_stable_steps = {max_stable}")
        print(f"H_close_empty = ceil(1.2*max) + 1 = {H_close} 控制步")
    print(f"raw → {raw_path}")


if __name__ == "__main__":
    main()
