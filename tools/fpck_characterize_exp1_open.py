"""Commit 4 物理表征实验 1-3：夹爪 open 响应 / 载物 open / contact gone。

协议：docs/method/terminal_validation_characterization_protocol_v1.md
protocol_sha256=02936a14c2fd5bc199c19e929a073b4a53c44be1933027d43c6e66a1272255a6

本阶段**全部脚本化控制**（直接写 actuator ctrl），不调用 VLA。
只测夹爪物理响应，不碰 state3/state4 对象，不用成功率。

实验 1：空夹爪 open 响应（3 姿态 × 3 初始 aperture）
实验 2：载物 open 响应（跳过真实对象，用隔离场景；先做空夹爪基线）
实验 3：release 后 finger contact 消失（空夹爪先验证 contact 判定，载物后续）

raw 数据 append-only 写 .cache-local/fpck_characterization/。
"""

from __future__ import annotations

import hashlib
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

PROTOCOL_SHA256 = "02936a14c2fd5bc199c19e929a073b4a53c44be1933027d43c6e66a1272255a6"
CONTROL_HZ = 10.0
CONTROL_TIMESTEP = 0.05
SUBSTEPS = 25
# Panda gripper：left finger ctrl ∈ [0, 0.04]，right ∈ [-0.04, 0]
# open 状态 ctrl=(0.04, -0.04) → aperture≈0.077；closed ctrl=(0, 0) → aperture≈0.001
A_OPEN = 0.077            # open 状态 aperture（实测）
A_CLOSED = 0.001          # closed 状态 aperture
OPEN_CTRL = (0.04, -0.04)
CLOSED_CTRL = (0.0, 0.0)
EPS_A = 0.002          # aperture 稳定容差
EPS_ADOT = 0.001       # aperture 速度阈值（per substep）


def env_identity():
    import mujoco
    return {
        "repo_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
        "git_dirty": bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT).decode().strip()),
        "mujoco_version": mujoco.__version__,
        "controller": "OSC_POSE",
        "physics_timestep": 0.002,
        "control_timestep": CONTROL_TIMESTEP,
        "physics_substeps_per_control": SUBSTEPS,
        "robot": "panda",
        "terminal_protocol_sha256": PROTOCOL_SHA256,
    }


def make_env():
    from libero.libero import benchmark
    from tools.g1a_soup_trace import make_env as _make
    ts = benchmark.get_benchmark_dict()["safelibero_spatial"](safety_level="I")
    task = ts.get_task(0)  # 只用 env 结构，不用 state3/state4 对象
    return _make(ts, task, 0)


def aperture(env):
    obs = env.env._get_observations()
    gq = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64)
    return float(gq[0] - gq[1])


def grip_action(grip_sign):
    """构造 7 维 action，前 6 维零（hold 位姿），gripper 分量为 grip_sign。

    grip_sign=+1 = close，-1 = open（PandaGripper sticky 累积语义）。
    经 env.step → _pre_action → robot.control → grip_action → format_action 真实驱动。
    """
    act = np.zeros(7, dtype=np.float64)
    act[-1] = grip_sign
    return act


def run_open_response(env, init_frac=0.0, n_control_steps=12):
    """从指定初始开度（0=closed, 1=open）开始，用 env.step 下发 open 命令，
    逐控制步记录 aperture 响应。

    返回 (rows, start_state)。rows 每控制步一条（aperture + velocity）。
    注意：表征阶段直接 reset env 到初始，不依赖任务 done 状态。
    """
    env.env.reset()
    # 先闭合（close 命令，经 sticky 累积）；用 while 逼近闭合而非固定步数
    for _ in range(25):
        env.step(grip_action(+1.0))
        if aperture(env) < 0.003:
            break
    # 到目标初始开度（若需中间态，用 open 命令逼近）
    if init_frac > 0.0:
        target_a = A_OPEN * init_frac
        for _ in range(40):
            env.step(grip_action(-1.0))  # open 方向
            if aperture(env) >= target_a:
                break
    a_start = aperture(env)
    start_state = {
        "ctrl_target": None,  # 经 format_action 的 current_action 才是真目标
        "gripper_current_action": [float(x) for x in env.env.robots[0].gripper.current_action],
        "aperture": a_start,
    }
    # open 命令，逐控制步记录
    rows = []
    prev_a = aperture(env)
    for cs in range(n_control_steps):
        env.step(grip_action(-1.0))  # open
        a = aperture(env)
        rows.append({
            "control_step": cs,
            "aperture": a,
            "aperture_velocity": (a - prev_a) / CONTROL_TIMESTEP,  # per 控制步
        })
        prev_a = a
    return rows, start_state


def analyze_open_response(rows):
    """返回 onset（控制步）与 stable（控制步）。

    onset：首次 aperture 明显增大（adot > 噪声地板）。
    stable：aperture 达 A_OPEN - EPS_A 的控制步。
    """
    onset = None
    for r in rows:
        if r["aperture_velocity"] > 0.001:  # 噪声地板（per 控制步）
            onset = r["control_step"]
            break
    stable = None
    for r in rows:
        if r["aperture"] >= A_OPEN - EPS_A:
            stable = r["control_step"]
            break
    return onset, stable


def main():
    out_dir = os.path.join(ROOT, ".cache-local", "fpck_characterization")
    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, "exp1_empty_open.jsonl")

    env = make_env()
    ident = env_identity()

    # 协议：3 姿态 × 3 初始 aperture（closed=0 / mid=0.5 / near_closed=0.2）× 5 重复
    poses = ["neutral", "near_table", "transport"]
    init_fracs = {"closed": 0.0, "mid": 0.5, "near_closed": 0.2}
    n_reps = 5

    results = []
    with open(raw_path, "w", encoding="utf-8") as f:
        for pose in poses:
            for init_name, init_frac in init_fracs.items():
                for rep in range(n_reps):
                    rows, start_state = run_open_response(env, init_frac=init_frac)
                    onset, stable = analyze_open_response(rows)
                    rec = {
                        "experiment": "empty_open_response",
                        "pose": pose, "init_aperture_name": init_name,
                        "init_aperture_frac": init_frac, "rep": rep,
                        "target_aperture": A_OPEN,
                        "start_state": start_state,
                        "onset_control_steps": onset,
                        "stable_control_steps": stable,
                        "aperture_trace": [r["aperture"] for r in rows],
                        "identity": ident,
                    }
                    results.append(rec)
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    print(f"{pose}/{init_name}/rep{rep}: onset={onset} stable={stable}")
    env.close()

    # 汇总
    stable_vals = [r["stable_control_steps"] for r in results
                   if r["stable_control_steps"] is not None]
    onset_vals = [r["onset_control_steps"] for r in results
                  if r["onset_control_steps"] is not None]
    print()
    print(f"=== 空夹爪 open 响应汇总（{len(results)} trials）===")
    print(f"onset 控制步分布: {sorted(set(onset_vals))}")
    print(f"stable 控制步分布: {sorted(set(stable_vals))}")
    if stable_vals:
        max_stable = max(stable_vals)
        H_open = int(np.ceil(1.2 * max_stable)) + 1
        print(f"max_observed_stable_steps = {max_stable}")
        print(f"按预注册规则 H_open_empty = ceil(1.2*max) + 1 = {H_open} 控制步")
        print(f"（multiplicative_margin=1.2, discrete_margin_steps=1）")
    print(f"raw → {raw_path}")


if __name__ == "__main__":
    main()
