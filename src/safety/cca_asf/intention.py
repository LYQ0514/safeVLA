"""CCA-ASF 任务意图提取（从冻结动作块推断任务方向 d_z）。

规格书 §3.1：不使用训练标签，从动作块前 L 条臂部归一化增量加权求和。
eta 加权：omega_k = eta^k / sum_j eta^j，0 < eta <= 1。
退化回退：d_bar 模长 <= eps_dir 时用首步动作；再退化则 None。
"""

from __future__ import annotations

import numpy as np


def chunk_direction(chunk_actions, eta=0.8, eps_dir=1e-6, intent_len=None):
    """从动作块提取 6D 任务方向 d_z（世界系，单位向量或 None）。

    chunk_actions: [K, 7] 或 [K, 6]（前 6 维为臂部归一化动作）。
    只取前 intent_len（默认全取）条。
    """
    chunk = np.asarray(chunk_actions, dtype=float)
    if chunk.ndim == 1:
        chunk = chunk[None, :]
    if chunk.shape[0] == 0:
        return None
    arm = chunk[:, :6]  # 前 6 维
    L = len(arm) if intent_len is None else min(int(intent_len), len(arm))
    if L <= 0:
        return None
    arm = arm[:L]

    weights = eta ** np.arange(L)
    wsum = weights.sum()
    if wsum <= 0:
        return None
    weights = weights / wsum

    d_bar = weights @ arm  # (6,)
    n = float(np.linalg.norm(d_bar))
    if n > eps_dir:
        return d_bar / n

    z0 = arm[0]
    n0 = float(np.linalg.norm(z0))
    if n0 > eps_dir:
        return z0 / n0
    return None
