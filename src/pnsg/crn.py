"""Common Random Numbers（CRN）调度器（Step 5 prospective audit 协议）。

对应想法v10.md 第 9 节 + Step 5 pilot 协议：
- 显式、无状态、可复现的 z 调度
- nominal stream：三种方法共享同一个 z（按 episode/query 配对）
- auxiliary stream：方法额外的候选调用用独立子流，不推进 nominal
- 保存实际 z 数组 + sha256（不只存 seed）

关键：不用 Python 内置 hash()（跨进程不稳定），用 SHA-256 生成 seed。
z 采样用 numpy.random.default_rng（PCG64，NumPy 版本稳定）。
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

EXPERIMENT_VERSION = "step5_v1"


def stable_hash(*args):
    """SHA-256 稳定哈希 → 32 位整数 seed。

    args 会被字符串化拼接，顺序敏感。跨进程、跨平台稳定。
    """
    s = "|".join(str(a) for a in args)
    digest = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)


def nominal_seed(suite, task_id, episode_id, replicate_id, query_index):
    """nominal z 的 seed（三种方法共享）。"""
    return stable_hash(EXPERIMENT_VERSION, "nominal", suite, task_id,
                       episode_id, replicate_id, query_index)


def aux_seed(method, task_id, episode_id, query_index, candidate_id, attempt_id):
    """方法辅助流的 seed（独立，不推进 nominal）。"""
    return stable_hash(EXPERIMENT_VERSION, "aux", method, task_id,
                       episode_id, query_index, candidate_id, attempt_id)


def sample_z(seed, shape=(10, 32), dtype=np.float32):
    """从 seed 生成标准高斯 z ~ N(0, I)。

    返回 (z, meta)，meta 含 generator 信息用于身份记录。
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(shape).astype(dtype)
    meta = {
        "generator_name": "numpy.random.PCG64",
        "numpy_version": np.__version__,
        "shape": list(shape),
        "dtype": str(np.dtype(dtype)),
        "seed": int(seed),
    }
    return z, meta


def z_sha256(z):
    """z 数组的 sha256（字节级身份）。"""
    return hashlib.sha256(np.asarray(z).tobytes()).hexdigest()


def action_sha256(action):
    """action 数组的 sha256。"""
    return hashlib.sha256(np.asarray(action).tobytes()).hexdigest()


def observation_sha256(obs, keys=("agentview_image", "robot0_eef_pos", "robot0_gripper_qpos")):
    """观测的身份 hash（选关键字段）。"""
    h = hashlib.sha256()
    for k in keys:
        if k in obs:
            h.update(str(k).encode())
            h.update(np.asarray(obs[k]).tobytes())
    return h.hexdigest()


def initial_state_hash(env):
    """仿真初始状态（qpos+qvel）的 sha256。"""
    state = np.asarray(env.sim.get_state().flatten(), dtype=np.float64)
    return hashlib.sha256(state.tobytes()).hexdigest()
