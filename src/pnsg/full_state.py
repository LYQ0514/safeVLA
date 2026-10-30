"""PNSG 完整状态快照与恢复。

对应 docs/设计.md 第 10 节：认证前必须 save_full_state / restore_full_state，
每个候选从完全相同状态开始。

用 MuJoCo 原生 qpos+qvel（set_state_from_flattened）实现，验证过恢复后差 0.0。
"""

from __future__ import annotations

import numpy as np


def save_full_state(env):
    """保存完整仿真状态（qpos + qvel 展平），返回 (flattened, meta)。"""
    sim = env.sim
    state = sim.get_state()
    flattened = np.asarray(state.flatten(), dtype=np.float64).copy()
    return {"flattened": flattened}


def restore_full_state(env, snapshot):
    """从快照恢复仿真状态。"""
    sim = env.sim
    sim.set_state_from_flattened(snapshot["flattened"])
    sim.forward()
