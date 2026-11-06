"""MVP-1b：自由空间姿态对齐（closing axis 对齐 cream_cheese 最小水平轴）。

按协议 v2 §9：用旋转矩阵（data.xmat），axis-angle 闭环，不混 quaternion 顺序。
目标：closing axis 对齐 world x（cream_cheese 最小水平轴），approach axis 保持 -z。
DEVELOPMENT_ONLY。
"""

from __future__ import annotations

import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = "/data/safeVLA"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "dataset", "SafeLIBERO", "safelibero"))
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

ORI_TOL = 0.035  # ~2°
MAX_ROT = 0.05   # rad/step
ACTION_ORI_CLIP = 0.1  # 0.05/0.5
HARD_CAP = 150


def make_env():
    from libero.libero import benchmark
    from tools.g1a_soup_trace import make_env as _make
    ts = benchmark.get_benchmark_dict()["safelibero_long"](safety_level="I")
    task = ts.get_task(0)
    return _make(ts, task, 0)


def finger_pos(env, side):
    d = env.sim.data._data
    m = env.sim.model
    bid = m.body_name2id(f"gripper0_{side}finger")
    return np.asarray(d.xpos[bid]).copy()


def closing_axis(env):
    v = finger_pos(env, "right") - finger_pos(env, "left")
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.zeros(3)


def eef_rot(env):
    d = env.sim.data._data
    m = env.sim.model
    eef_body = env.env.robots[0].robot_model.eef_name
    bid = m.body_name2id(eef_body)
    return np.asarray(d.xmat[bid]).reshape(3, 3)


def mat2axisangle(R):
    """旋转矩阵 → axis-angle（log map）。返回 (axis, angle)。"""
    R = np.clip(R, -1.0, 1.0)
    cos = (np.trace(R) - 1.0) / 2.0
    cos = np.clip(cos, -1.0, 1.0)
    angle = np.arccos(cos)
    if angle < 1e-8:
        return np.zeros(3), 0.0
    if angle > np.pi - 1e-6:
        # 180° 特例
        axis = np.sqrt(np.clip((np.diag(R) + 1) / 2, 0, None))
        return axis / (np.linalg.norm(axis) + 1e-9), angle
    # 反对称部分
    so3 = (R - R.T) / (2 * np.sin(angle))
    axis = np.array([so3[2, 1], so3[0, 2], so3[1, 0]])
    n = np.linalg.norm(axis)
    return axis / (n + 1e-9), angle


def build_target_rot(closing_world, approach_world=np.array([0.0, 0.0, -1.0])):
    """由 closing axis + approach axis 构造目标旋转矩阵。

    关键事实（运行时实测）：closing axis（手指张开方向）= body 局部 **-y** 轴。
    approach axis = body 局部 z 轴 ≈ 世界 -z（top-down）。
    所以 R_target 应满足：
      R_target 的列1（body y 世界方向）= -closing_world
      R_target 的列2（body z 世界方向）= approach_world
      R_target 的列0 = 列1 × 列2（右手系）
    """
    c = closing_world / np.linalg.norm(closing_world)
    a = approach_world / np.linalg.norm(approach_world)
    # body y 轴（世界方向）= -closing
    body_y = -c
    body_z = a
    body_x = np.cross(body_y, body_z)
    body_x = body_x / np.linalg.norm(body_x)
    # 重新正交化 body_y
    body_y = np.cross(body_z, body_x)
    body_y = body_y / np.linalg.norm(body_y)
    R = np.column_stack([body_x, body_y, body_z])
    return R


def align_orientation(env, closing_target, max_steps=HARD_CAP):
    """axis-angle 闭环对齐 closing axis 到 closing_target。"""
    for i in range(max_steps):
        c_cur = closing_axis(env)
        # closing axis 误差角
        dot = np.clip(np.dot(c_cur, closing_target), -1.0, 1.0)
        ang_err = np.arccos(dot)
        if ang_err < ORI_TOL:
            return "CONVERGED", i, ang_err
        # 目标旋转矩阵
        R_target = build_target_rot(closing_target)
        R_cur = eef_rot(env)
        R_err = R_target @ R_cur.T
        axis, angle = mat2axisangle(R_err)
        # 发送 axis-angle delta（clip 到 MAX_ROT）
        delta = axis * min(angle, MAX_ROT)
        act = np.zeros(7)
        act[3:6] = np.clip(delta / 0.5, -ACTION_ORI_CLIP, ACTION_ORI_CLIP)
        act[-1] = -1.0
        env.step(act)
    return "TIMEOUT", max_steps, ang_err


def main():
    env = make_env()
    env.env.reset()
    for _ in range(15):
        env.step(np.zeros(7))

    print("=== MVP-1b 姿态对齐 ===")
    c0 = closing_axis(env)
    print(f"初始 closing axis: {c0.round(3)}")

    # 目标：closing axis 对齐 world x（cream_cheese 最小水平轴）
    closing_target = np.array([1.0, 0.0, 0.0])
    label, steps, ang_err = align_orientation(env, closing_target)

    c_final = closing_axis(env)
    R_final = eef_rot(env)
    print(f"结果: {label} steps={steps}")
    print(f"最终 closing axis: {c_final.round(3)}（目标 {closing_target}）")
    print(f"最终 approach axis: {R_final[:,2].round(3)}（应接近 [0,0,-1]）")
    dot = np.clip(np.dot(c_final, closing_target), -1.0, 1.0)
    print(f"closing 对齐误差角: {np.degrees(np.arccos(dot)):.2f}°")

    # 正交性检查
    R = R_final
    print(f"det(R_final) = {np.linalg.det(R):.4f}（应≈+1）")
    env.close()


if __name__ == "__main__":
    main()
