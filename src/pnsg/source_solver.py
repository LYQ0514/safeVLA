"""PNSG 源空间求解器：H-Fast 最小范数 + Farkas 证书。

对应 docs/设计.md 第 7、8 节。纯 numpy，无 JAX/robosuite 依赖。
"""

from __future__ import annotations

import itertools

import numpy as np


def minimum_norm_halfspaces(A, b, tol=1e-9):
    """min ||Δz||_2 s.t. A Δz ≥ b。枚举所有非空 active set（Fast ≤3 约束）。

    返回 (Δz*, 范数) 或 (None, None) 若不可行。

    对 active set A_a：
        λ = (A_a A_a^T)^+ b_a
        Δz = A_a^T λ
    接受条件：λ ≥ 0 且 A Δz ≥ b - tol，取范数最小者。
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = A.shape[0]

    if m == 0:
        return np.zeros(A.shape[1]), 0.0

    if np.all(b <= tol):
        return np.zeros(A.shape[1]), 0.0

    best = None
    best_norm = np.inf

    indices = list(range(m))
    for r in range(1, m + 1):
        for active in itertools.combinations(indices, r):
            active = list(active)
            Aa = A[active]
            ba = b[active]
            # λ = (Aa Aa^T)^+ ba
            gram = Aa @ Aa.T
            lam = np.linalg.pinv(gram) @ ba
            if np.any(lam < -tol):
                continue
            dz = Aa.T @ lam
            if np.all(A @ dz >= b - tol):
                norm = float(np.linalg.norm(dz))
                if norm < best_norm:
                    best = dz
                    best_norm = norm

    if best is None:
        return None, None
    return best, best_norm


def farkas_certificate(A, b, null_tol=1e-8, margin_tol=1e-8):
    """搜索 Farkas witness：y ≥ 0, 1^T y = 1, ||A^T y|| ≤ null_tol, b^T y > margin_tol。

    返回 dict {found, y, aty_norm, bty}。witness 存在 ⟹ 线性系统不可行。
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = A.shape[0]
    if m == 0:
        return {"found": False, "y": None, "aty_norm": 0.0, "bty": 0.0}

    # 简单方法：对每对约束 i,j 求凸组合 y = α e_i + (1-α) e_j，
    # 使 A^T y ≈ 0。也尝试单约束（A 的行接近 0 时）。
    # 这是一个启发式 Farkas 搜索，不是完备的。

    # 归一化 A 的行，找行组合接近 0
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)

    # 单行接近零（A 某行本身接近 0 且 b>0）→ 不可行
    for i in range(m):
        if np.linalg.norm(A[i]) <= null_tol and b[i] > margin_tol:
            y = np.zeros(m); y[i] = 1.0
            return {"found": True, "y": y, "aty_norm": 0.0, "bty": float(b[i])}

    # 两行组合
    best = None
    for i, j in itertools.combinations(range(m), 2):
        # 找 α 使 α A_i + (1-α) A_j 的范数最小
        # min ||α a_i + (1-α) a_j||^2 → 解析解
        ai, aj = A[i], A[j]
        # α* = (aj-ai)·aj / ||ai-aj||^2
        d = ai - aj
        dd = d @ d
        if dd < 1e-16:
            alpha = 0.5
        else:
            alpha = float(((aj - ai) @ aj) / dd)
        alpha = np.clip(alpha, 0.0, 1.0)
        aty = alpha * ai + (1 - alpha) * aj
        if np.linalg.norm(aty) <= null_tol:
            bty = alpha * b[i] + (1 - alpha) * b[j]
            if bty > margin_tol:
                y = np.zeros(m); y[i] = alpha; y[j] = 1 - alpha
                return {"found": True, "y": y, "aty_norm": float(np.linalg.norm(aty)), "bty": float(bty)}

    return {"found": False, "y": None, "aty_norm": None, "bty": None}


def compute_chi_lin(delta_z, r_budget=12.0):
    """线性可修复半径归一化难度 χ_lin = r_crit / r_budget。"""
    if delta_z is None:
        return float("inf")
    return float(np.linalg.norm(delta_z)) / r_budget
