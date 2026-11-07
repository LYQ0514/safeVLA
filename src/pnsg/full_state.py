"""PNSG 完整状态快照与恢复。

对应 docs/设计.md 第 10 节：认证前必须 save_full_state / restore_full_state，
每个候选从完全相同状态开始。

用 MuJoCo 原生 qpos+qvel（set_state_from_flattened）实现，验证过恢复后差 0.0。

除 MuJoCo 物理状态外，还保存/恢复 episode 级状态（robosuite env 的
`timestep`/`done`/`cur_time`）。否则反事实分支的 `env.step` 会推进 timestep 而
restore 不回退它，导致：
- 同一 anchor 的 sibling 之间 timestep 起点累积漂移；
- anchor 采集结束后 nominal 主循环继续 `env.step` 时 timestep 超 horizon、
  `done=True` → 抛 `ValueError("executing action in terminated episode")`。
"""

from __future__ import annotations

import numpy as np

_EPISODE_KEYS = ("timestep", "done", "cur_time")


def _episode_sources(env):
    """返回 (env, env.env) 中携带 robosuite episode 状态的候选对象序列。"""
    inner = getattr(env, "env", None)
    return [s for s in (env, inner) if s is not None]


def _read_episode_state(env):
    for src in _episode_sources(env):
        if all(hasattr(src, k) for k in _EPISODE_KEYS):
            return {
                "timestep": int(src.timestep),
                "done": bool(src.done),
                "cur_time": float(src.cur_time),
            }
    return None


def _write_episode_state(env, state):
    for src in _episode_sources(env):
        if all(hasattr(src, k) for k in _EPISODE_KEYS):
            src.timestep = int(state["timestep"])
            src.done = bool(state["done"])
            src.cur_time = float(state["cur_time"])
            return True
    return False


def save_full_state(env):
    """保存完整仿真状态（qpos + qvel 展平）+ episode 级状态（timestep/done/cur_time）。"""
    sim = env.sim
    state = sim.get_state()
    flattened = np.asarray(state.flatten(), dtype=np.float64).copy()
    snap = {"flattened": flattened}
    ep = _read_episode_state(env)
    if ep is not None:
        snap["episode"] = ep
    return snap


def restore_full_state(env, snapshot):
    """从快照恢复仿真状态（含 episode 级状态）。"""
    sim = env.sim
    sim.set_state_from_flattened(snapshot["flattened"])
    sim.forward()
    if "episode" in snapshot:
        _write_episode_state(env, snapshot["episode"])
