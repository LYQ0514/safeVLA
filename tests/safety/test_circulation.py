"""Circulation 严格实现的数学单元测试（对应论文性质）。

验证 Gonçalves 等 T-CST 2024 的 circulation inequality 关键数学性质：
- Ω skew-symmetric（Ω^T = -Ω）
- Ω orthonormal（2D 子空间内 Ω^T Ω = I）
- T ⊥ N（Ω 反对称 ⇒ N^T Ω N = 0）
- β 递减、β(0)>0、β→-∞
- N = a_v/‖a_v‖ 单位化
纯 NumPy，无 MuJoCo。
"""

from __future__ import annotations

import numpy as np
import pytest

from src.safety.circulation import (
    circulation_beta,
    circulation_normal_local,
    circulation_omega_2d,
    circulation_omega_3d_xy,
    circulation_tangent_world,
)


def _random_rot():
    A = np.random.default_rng(1).normal(size=(3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


# ---------------------------------------------------------------------------
# Ω 矩阵性质（论文 §IV-B 式 (6) + §IV-C workaround）
# ---------------------------------------------------------------------------

def test_omega_2d_skew_symmetric():
    for sign in (+1.0, -1.0):
        O = circulation_omega_2d(sign)
        assert np.allclose(O.T, -O), "Ω 必须反对称 Ω^T = -Ω"


def test_omega_2d_orthonormal():
    for sign in (+1.0, -1.0):
        O = circulation_omega_2d(sign)
        assert np.allclose(O.T @ O, np.eye(2)), "n=2 Ω 必须正交"


def test_omega_2d_sign_negation():
    # 式 (6)：Ω1 = [[0,-1],[1,0]]（逆时针），Ω2 = -Ω1 = [[0,1],[-1,0]]（顺时针）
    # 2×2 反对称矩阵 det = 上三角元素平方，恒为正，不能用 det 区分方向；
    # 正确验证是「顺时针 = 逆时针的负矩阵」。
    assert np.allclose(circulation_omega_2d(-1.0), -circulation_omega_2d(+1.0))
    assert np.allclose(circulation_omega_2d(+1.0), [[0, -1], [1, 0]])


def test_omega_3d_skew_symmetric():
    for sign in (+1.0, -1.0):
        O = circulation_omega_3d_xy(sign)
        assert np.allclose(O.T, -O), "3D Ω 必须反对称"


def test_omega_3d_xy_orthonormal_z_singular():
    # §IV-C：xy 子空间 orthonormal，z 维奇异
    for sign in (+1.0, -1.0):
        O = circulation_omega_3d_xy(sign)
        # xy 子空间正交
        assert np.allclose(O[:2, :2].T @ O[:2, :2], np.eye(2))
        # z 维奇异（第三行全零）
        assert np.allclose(O[2, :], 0.0)
        assert np.allclose(O[:, 2], 0.0)
        # 奇数维反对称 ⇒ det=0（论文脚注 1）
        assert np.linalg.det(O) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 法向 N 与切向 T（论文 §III-A：N=∇D/‖∇D‖，T=ΩN，T⊥N）
# ---------------------------------------------------------------------------

def test_normal_local_is_unit():
    a_v = np.array([0.3, -0.4, 0.5])
    N = circulation_normal_local(a_v)
    assert N is not None
    assert np.linalg.norm(N) == pytest.approx(1.0)
    # 方向与 a_v 相同
    assert np.dot(N, a_v / np.linalg.norm(a_v)) == pytest.approx(1.0)


def test_normal_degenerate_returns_none():
    assert circulation_normal_local(np.zeros(3)) is None
    assert circulation_normal_local(np.array([1e-12, 0, 0])) is None


def test_tangent_orthogonal_to_normal():
    R1 = _random_rot()
    a_v = np.array([1.0, 2.0, 0.5])
    T_world, T_local = circulation_tangent_world(R1, a_v, +1.0)
    assert T_world is not None
    N_world = R1 @ circulation_normal_local(a_v)
    # T ⊥ N（世界系）
    assert np.abs(np.dot(T_world, N_world)) < 1e-12, "T 必须正交于 N"
    # T 是单位向量（Ω orthonormal 保范数，在 xy 子空间投影后可能小于 1）
    assert np.linalg.norm(T_world) <= 1.0 + 1e-12


def test_tangent_preserves_norm_when_n_horizontal():
    # 若 N 在水平面（无 z 分量），Ω（xy 旋转）保范数，T 仍是单位向量
    R1 = np.eye(3)
    a_v = np.array([1.0, 1.0, 0.0])  # ∇h 水平
    T_world, _ = circulation_tangent_world(R1, a_v, +1.0)
    assert np.linalg.norm(T_world) == pytest.approx(1.0)


def test_tangent_sign_controls_direction():
    # circ_sign=+1 逆时针 vs -1 顺时针：T 方向相反
    R1 = np.eye(3)
    a_v = np.array([1.0, 0.0, 0.0])  # N = +x
    T_ccw, _ = circulation_tangent_world(R1, a_v, +1.0)
    T_cw, _ = circulation_tangent_world(R1, a_v, -1.0)
    # N=+x，Ω1 N = [0,1,0]（+y，逆时针）；Ω2 N = [0,-1,0]
    assert np.allclose(T_ccw, [0, 1, 0])
    assert np.allclose(T_cw, [0, -1, 0])


def test_tangent_degenerate_returns_none():
    T_w, T_l = circulation_tangent_world(np.eye(3), np.zeros(3), +1.0)
    assert T_w is None and T_l is None


# ---------------------------------------------------------------------------
# β(D) 性质（论文 Assumption 1-7）
# ---------------------------------------------------------------------------

def test_beta_decreasing():
    # β(h) = gain·(activate - h)，dβ/dh = -gain < 0
    gain, activate = 50.0, 0.02
    h_lo, h_hi = 0.0, 0.5
    assert circulation_beta(h_lo, gain, activate) > circulation_beta(h_hi, gain, activate)


def test_beta_zero_positive():
    # β(0) = gain·activate > 0
    assert circulation_beta(0.0, 50.0, 0.02) > 0


def test_beta_diverges_negative():
    # β(h)→-∞ 当 h→∞
    assert circulation_beta(1e6, 50.0, 0.02) < -1e6


def test_beta_negative_near_goal():
    # 目标处（h 大）β<0，约束平凡化
    assert circulation_beta(0.5, 50.0, 0.02) < 0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
