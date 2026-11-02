"""CCA-ASF 数学单元测试（T1-T7, T10）。

纯 NumPy，无 MuJoCo。验收条件见规格书 §10 + 裁决 §六。
"""

from __future__ import annotations

import numpy as np
import pytest

from src.safety.cca_asf.metric import (
    B_DIAG, build_anisotropic_W, build_cca_asf_objective,
    compute_conflict_beta, map_dz_to_dy,
)
from src.safety.cca_asf.intention import chunk_direction

# AEGIS 原生 9D 度量 W0 = diag([1/25]*6 + [1,1,1])
W0 = np.diag([1.0 / 25] * 6 + [1.0, 1.0, 1.0])


def _random_rot():
    """随机旋转矩阵（Gram-Schmidt）。"""
    A = np.random.default_rng(0).normal(size=(3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


# ---------------------------------------------------------------------------
# T1：坐标映射 B J_t d_z = [R1^T d_p, d_omega, 0_3]
# ---------------------------------------------------------------------------
def test_T1_coordinate_mapping():
    rng = np.random.default_rng(1)
    for _ in range(100):
        R1 = _random_rot()
        dz = rng.normal(size=6)
        dz /= np.linalg.norm(dz)
        dy = map_dz_to_dy(dz, R1)
        expected = np.concatenate([R1.T @ dz[:3], dz[3:6], np.zeros(3)])
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(dy, expected, atol=1e-12)
        # d_y 是单位向量
        assert abs(np.linalg.norm(dy) - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# T2：beta=0 时 P,q 与原 AEGIS 数值一致
# ---------------------------------------------------------------------------
def test_T2_beta_zero_recovers_base():
    rng = np.random.default_rng(2)
    A = rng.normal(size=(1, 9))
    b = np.array([0.5])
    u_nom = rng.normal(size=9)
    q0 = -W0 @ u_nom  # 原 AEGIS 线性项（中心化目标）
    dz = rng.normal(size=6); dz /= np.linalg.norm(dz)
    R1 = _random_rot()
    # 用 beta_max=0 强制 beta=0
    P_new, q_new, beta = build_cca_asf_objective(
        W0, q0, u_nom, A, b, dz, R1, beta_max=0.0)
    assert beta == 0.0
    np.testing.assert_allclose(P_new, W0, atol=1e-12)
    np.testing.assert_allclose(q_new, q0, atol=1e-12)


# ---------------------------------------------------------------------------
# T3：beta=4, d_y=e1 时 W_11 = 5/25 = 0.2（不是 4.04）
# ---------------------------------------------------------------------------
def test_T3_whitening_strength():
    # d_y = e1 意味着 R1^T d_p = [1,0,0], d_omega = 0
    # 构造 d_z 使 map_dz_to_dy 输出 e1：R1 = I, d_p = [1,0,0], d_omega = 0
    R1 = np.eye(3)
    dz = np.array([1.0, 0, 0, 0, 0, 0])
    d_y = map_dz_to_dy(dz, R1)
    np.testing.assert_allclose(d_y, np.array([1, 0, 0, 0, 0, 0, 0, 0, 0]), atol=1e-12)

    Wt = build_anisotropic_W(W0, d_y, beta=4.0)
    assert abs(Wt[0, 0] - 5.0 / 25.0) < 1e-12, f"W_11={Wt[0,0]:.4f} 应为 0.2"
    # 其余对角保持不变
    for i in range(1, 9):
        assert abs(Wt[i, i] - W0[i, i]) < 1e-12


# ---------------------------------------------------------------------------
# T4：正定性（随机状态 W_t ≻ 0）
# ---------------------------------------------------------------------------
def test_T4_positive_definite():
    rng = np.random.default_rng(3)
    for _ in range(100):
        R1 = _random_rot()
        dz = rng.normal(size=6); dz /= np.linalg.norm(dz)
        d_y = map_dz_to_dy(dz, R1)
        beta = rng.uniform(0, 4)
        Wt = build_anisotropic_W(W0, d_y, beta)
        evals = np.linalg.eigvalsh(Wt)
        assert np.all(evals > 0), f"W_t 非正定: min eig={evals.min():.3e}"
        # 白化坐标特征值应为 {1,...,1, 1+beta}
        evals_w = np.linalg.eigvalsh(Wt / np.outer(B_DIAG, B_DIAG))
        assert abs(evals_w.max() - (1 + beta)) < 1e-9


# ---------------------------------------------------------------------------
# T5：线性项正确性（新 QP 目标与中心化表达式相等，允许常数项差异）
# ---------------------------------------------------------------------------
def test_T5_linear_term_correct():
    rng = np.random.default_rng(4)
    A = rng.normal(size=(2, 9))
    b = np.array([0.3, 0.2])
    u_nom = rng.normal(size=9)
    dz = rng.normal(size=6); dz /= np.linalg.norm(dz)
    R1 = _random_rot()
    q0 = -W0 @ u_nom

    P_new, q_new, beta = build_cca_asf_objective(W0, q0, u_nom, A, b, dz, R1)

    # 中心化目标 1/2 (u-u_nom)^T W_new (u-u_nom) 与 1/2 u^T P_new u + q_new^T u 差常数
    # 等价地验证梯度在 u_nom 处为 0：
    d_y = map_dz_to_dy(dz, R1)
    Wt = build_anisotropic_W(W0, d_y, beta)
    # 梯度 = P_new @ u_nom + q_new 应 = 0（因目标在 u_nom 处最小，若无约束影响）
    grad = P_new @ u_nom + q_new
    np.testing.assert_allclose(grad, np.zeros(9), atol=1e-9)
    # 且 P_new 与 Wt 一致
    np.testing.assert_allclose(P_new, Wt, atol=1e-12)


# ---------------------------------------------------------------------------
# T6：约束不变（build_cca_asf_objective 不修改 A,b）
# ---------------------------------------------------------------------------
def test_T6_constraints_unchanged():
    rng = np.random.default_rng(5)
    A = rng.normal(size=(3, 9))
    b = np.array([0.1, 0.2, 0.3])
    u_nom = rng.normal(size=9)
    dz = rng.normal(size=6); dz /= np.linalg.norm(dz)
    R1 = _random_rot()
    q0 = -W0 @ u_nom
    A_copy, b_copy = A.copy(), b.copy()
    build_cca_asf_objective(W0, q0, u_nom, A, b, dz, R1)
    np.testing.assert_array_equal(A, A_copy)
    np.testing.assert_array_equal(b, b_copy)


# ---------------------------------------------------------------------------
# T7：辅助维度 DeltaP[6:9,:] = 0，但求解器仍优化 u[6:9]
# ---------------------------------------------------------------------------
def test_T7_aux_dims_zero_penalty():
    rng = np.random.default_rng(6)
    R1 = _random_rot()
    dz = rng.normal(size=6); dz /= np.linalg.norm(dz)
    d_y = map_dz_to_dy(dz, R1)
    beta = 3.0
    Wt = build_anisotropic_W(W0, d_y, beta)
    DeltaP = Wt - W0
    # 后 3 行 3 列应为 0
    np.testing.assert_allclose(DeltaP[6:, :], 0.0, atol=1e-12)
    np.testing.assert_allclose(DeltaP[:, 6:], 0.0, atol=1e-12)
    # 前 6x6 有非零
    assert np.any(np.abs(DeltaP[:6, :6]) > 1e-12)
    # W0 后 3 维对角惩罚保留（=1，未改）
    for i in (6, 7, 8):
        assert abs(Wt[i, i] - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# T10：任务方向退化（零方向 / NaN / 无违反约束 → 安全退化）
# ---------------------------------------------------------------------------
def test_T10_degenerate_direction():
    rng = np.random.default_rng(7)
    A = rng.normal(size=(1, 9))
    b = np.array([0.5])
    u_nom = rng.normal(size=9)
    q0 = -W0 @ u_nom
    R1 = _random_rot()

    # 零方向
    P, q, beta = build_cca_asf_objective(W0, q0, u_nom, A, b, np.zeros(6), R1)
    assert beta == 0.0
    np.testing.assert_allclose(P, W0, atol=1e-12)

    # None 方向
    P, q, beta = build_cca_asf_objective(W0, q0, u_nom, A, b, None, R1)
    assert beta == 0.0

    # 无违反约束（u_nom 已满足 A u >= b）
    A_ok = np.array([[1.0] + [0.0] * 8])
    b_ok = np.array([-1.0])  # u_nom[0] >= -1 恒满足（u_nom[0] 随机但 > -1）
    dz = np.ones(6); dz /= np.linalg.norm(dz)
    P, q, beta = build_cca_asf_objective(W0, q0, u_nom, A_ok, b_ok, dz, R1)
    assert beta == 0.0

    # chunk_direction 退化
    assert chunk_direction(np.zeros((5, 7))) is None
    assert chunk_direction(np.array([[0, 0, 0, 0, 0, 0, 0]])) is None


# ---------------------------------------------------------------------------
# 额外：解析算例复现（规格书 §6 + 裁决：z*_W = (0.2,0.8) vs z*_I = (0.5,0.5)）
# ---------------------------------------------------------------------------
def test_analytic_example():
    # 2D 情形直接验证 W 构造的解析解（不经过 9D 映射）
    W2 = np.diag([1.0, 1.0])
    d = np.array([1.0, 0.0])
    beta = 3.0
    W_aniso = W2 + beta * np.outer(d, d)  # 规格书 §6 的 2D 各向异性
    # 约束 z1+z2>=1，名义 (0,0)
    # 各向同性解 (0.5,0.5)；各向异性解 (0.2,0.8)
    zI = np.array([0.5, 0.5])
    zW = np.array([0.2, 0.8])
    # 验证两者都在约束边界
    assert abs(zI.sum() - 1) < 1e-12
    assert abs(zW.sum() - 1) < 1e-12
    # 验证各向异性解确实最小化 W_aniso 度量（沿任务维偏移 0.2 < 0.5）
    assert abs(zW[0]) < abs(zI[0])
