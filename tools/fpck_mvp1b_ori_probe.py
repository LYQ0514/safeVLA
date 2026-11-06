"""MVP-1b 前置：姿态动作语义探测（验证 OSC_POSE 的 axis-angle 方向）。

发送绕 z 轴的 axis-angle 动作，观察 closing axis（左右 finger 差）如何旋转。
DEVELOPMENT_ONLY。
"""

from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = "/data/safeVLA"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "dataset", "SafeLIBERO", "safelibero"))
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402


def make_env():
    from libero.libero import benchmark
    from tools.g1a_soup_trace import make_env as _make
    ts = benchmark.get_benchmark_dict()["safelibero_long"](safety_level="I")
    task = ts.get_task(0)
    return _make(ts, task, 0)


def finger_pos(env, side):
    d = env.sim.data._data
    m = env.sim.model
    bn = f"gripper0_{side}finger"
    bid = m.body_name2id(bn)
    return np.asarray(d.xpos[bid]).copy()


def closing_axis(env):
    """当前两指闭合轴（世界系单位向量）。"""
    l = finger_pos(env, "left")
    r = finger_pos(env, "right")
    v = r - l
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.zeros(3)


def eef_rot(env):
    d = env.sim.data._data
    m = env.sim.model
    eef_body = env.env.robots[0].robot_model.eef_name
    bid = m.body_name2id(eef_body)
    return np.asarray(d.xmat[bid]).reshape(3, 3)


def main():
    env = make_env()
    env.env.reset()
    for _ in range(15):
        env.step(np.zeros(7))

    print("=== 初始姿态 ===")
    c0 = closing_axis(env)
    R0 = eef_rot(env)
    print(f"closing axis: {c0.round(3)}")
    print(f"approach axis (R z 列): {R0[:,2].round(3)}")

    # 绕 +z 旋转 0.2 rad（axis-angle = [0,0,0.2]，OSC 增量）
    print("\n=== 发送绕 +z axis-angle [0,0,0.2] 若干步 ===")
    for i in range(20):
        act = np.zeros(7)
        act[5] = 0.4  # 绕 z：0.4 * 0.5 = 0.2 rad 目标
        act[-1] = -1.0
        env.step(act)
        if i % 5 == 0 or i == 19:
            c = closing_axis(env)
            R = eef_rot(env)
            print(f"  step{i}: closing={c.round(3)} approach={R[:,2].round(3)}")

    env.close()
    print("\n=== 结论 ===")
    print("观察 closing axis 是否从 y 转向 x（绕 +z 应使 y→x 或 y→-x）")


if __name__ == "__main__":
    main()
