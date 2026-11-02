"""CCA-ASF 冲突感知各向异性度量（纯数学，无 MuJoCo 依赖）。

实现规格书 §4 + 2026-11-02 裁决的 D2 修正版：

- AEGIS 原生 9D 速度空间 u（平移局部系 3 + 角速度 3 + z 方向演化 3）。
- 白化矩阵 B = diag([0.2]*6 + [1,1,1])（= W_0^{1/2}，因 W_0 = diag([1/25]*6 + [1,1,1])）。
- 白化任务方向 d_y = [R1^T d_p, d_omega, 0_3]（单位向量）。
- 正确度量 W_t = B^T (I + beta d_y d_y^T) B = W_0 + beta (B d_y)(B d_y)^T。
- 标准 QP 形式：P_new = P_0 + DeltaP, q_new = q_0 - DeltaP @ u_nom,
  DeltaP = beta (B d_y)(B d_y)^T。

关键修正（裁决强调）：不可用字面 W_0 + beta d_y d_y^T（会得 4.04 而非 0.20）。
"""

from __future__ import annotations

import numpy as np

# AEGIS 白化矩阵 B = W_0^{1/2}（W_0 = diag([1/25]*6 + [1,1,1])）
B_DIAG = np.array([0.2] * 6 + [1.0] * 3, dtype=float)


def map_dz_to_dy(dz, R1):
    """6D 任务方向（世界系）→ 9D 白化方向 d_y（单位向量或 None）。

    dz ∈ R^6 = [d_p(世界系平移), d_omega(axis-angle姿态)]，||dz||_2 = 1。
    R1 为末端朝向旋转矩阵（3x3）。
    返回 d_y = [R1^T d_p, d_omega, 0_3] 归一化；退化返回 None。
    """
    dz = np.asarray(dz, dtype=float).reshape(-1)
    if dz.shape[0] != 6:
        raise ValueError(f"dz must be 6-dim, got {dz.shape[0]}")
    n = float(np.linalg.norm(dz))
    if n <= 1e-12:
        return None
    dz = dz / n
    R1 = np.asarray(R1, dtype=float)
    if R1.shape != (3, 3):
        raise ValueError(f"R1 must be (3,3), got {R1.shape}")
    dy = np.concatenate([R1.T @ dz[:3], dz[3:6], np.zeros(3)])
    ny = float(np.linalg.norm(dy))
    if ny <= 1e-12:
        return None
    return dy / ny


def compute_conflict_beta(A, b, u_nom, d_y, B_diag=B_DIAG, beta_max=4.0,
                          violation_tol=1e-7, rank_tol=1e-8):
    """白化空间冲突门控 → beta ∈ [0, beta_max]。

    A u >= b 为 AEGIS 原生 9D 约束（m x 9），u_nom 为名义 9D 变量。
    d_y 为白化任务方向（9D 单位向量或 None）。

    步骤（规格书 §4.1 + 裁决 §6.2）：
      1. 违反集 V = {i: b_i - (A u_nom)_i > violation_tol}；
      2. 白化法向 Atilde_V = A[V] * (1/B_diag)（= A B^{-1}，B 对角）；
      3. 行归一化 → SVD 行空间投影 P_row；
      4. xi = ||P_row d_y||^2, rho = 4 xi (1-xi), beta = beta_max * rho。
    退化（V 空 / d_y None / 无秩）→ 0。
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    u_nom = np.asarray(u_nom, dtype=float).reshape(-1)
    if A.shape[0] != b.shape[0] or A.shape[1] != u_nom.shape[0]:
        raise ValueError(f"dim mismatch: A{A.shape} b{b.shape} u_nom{u_nom.shape}")
    if d_y is None:
        return 0.0

    v = b - A @ u_nom  # >0 表示违反
    V = np.where(v > violation_tol)[0]
    if len(V) == 0:
        return 0.0

    Atilde = A[V] / B_diag[None, :]  # 白化法向（B 对角）
    norms = np.linalg.norm(Atilde, axis=1)
    Atilde = Atilde / np.maximum(norms[:, None], 1e-12)

    # SVD 行空间投影（不用显式逆）
    _, s, Vt = np.linalg.svd(Atilde, full_matrices=False)
    rank = int(np.sum(s > rank_tol))
    if rank == 0:
        return 0.0
    Vr = Vt[:rank]            # (rank, 9)
    P = Vr.T @ Vr             # (9, 9) 行空间正交投影
    xi = float(d_y @ P @ d_y)
    xi = float(np.clip(xi, 0.0, 1.0))
    rho = 4.0 * xi * (1.0 - xi)
    return float(beta_max * rho)


def build_anisotropic_W(W0, d_y, beta, B_diag=B_DIAG):
    """W_t = W_0 + beta (B d_y)(B d_y)^T。

    W0 为 AEGIS 原生 9x9 对角度量 diag([1/25]*6 + [1,1,1])。
    beta <= 0 或 d_y None 时返回 W0 原样（严格退化）。
    """
    W0 = np.asarray(W0, dtype=float)
    if W0.shape != (9, 9):
        raise ValueError(f"W0 must be (9,9), got {W0.shape}")
    if d_y is None or beta <= 0.0:
        return W0.copy()
    Bd = B_diag * np.asarray(d_y, dtype=float)
    return W0 + beta * np.outer(Bd, Bd)


def build_cca_asf_objective(W0, q0, u_nom, A, b, dz, R1, B_diag=B_DIAG,
                            beta_max=4.0, violation_tol=1e-7, rank_tol=1e-8):
    """构造 CCA-ASF 的标准 QP 目标 (P_new, q_new, beta)。

    目标约定：min 1/2 u^T P u + q^T u（cvxpy quad_form 无 1/2，此处供单测用）。
    P_new = W0 + DeltaP, q_new = q0 - DeltaP @ u_nom, DeltaP = beta (B d_y)(B d_y)^T。

    W0 为 AEGIS 原二次项矩阵；q0 为 AEGIS 原线性项；u_nom 为名义 9D 变量；
    A u >= b 为原生约束（仅用于门控，不修改）；dz 为 6D 任务方向（可 None）；R1 旋转阵。
    """
    W0 = np.asarray(W0, dtype=float)
    q0 = np.asarray(q0, dtype=float).reshape(-1)
    u_nom = np.asarray(u_nom, dtype=float).reshape(-1)
    if W0.shape != (9, 9) or q0.shape[0] != 9 or u_nom.shape[0] != 9:
        raise ValueError("W0/q0/u_nom 维度必须为 9")

    if dz is None:
        return W0.copy(), q0.copy(), 0.0

    d_y = map_dz_to_dy(dz, R1)
    beta = compute_conflict_beta(A, b, u_nom, d_y, B_diag=B_diag,
                                 beta_max=beta_max, violation_tol=violation_tol,
                                 rank_tol=rank_tol)
    if beta <= 0.0 or d_y is None:
        return W0.copy(), q0.copy(), 0.0

    Bd = B_diag * d_y
    DeltaP = beta * np.outer(Bd, Bd)
    P_new = W0 + DeltaP
    q_new = q0 - DeltaP @ u_nom
    return P_new, q_new, float(beta)
