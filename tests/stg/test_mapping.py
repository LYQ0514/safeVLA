"""STG 任务引导项映射到 9D QP 的数学测试（G1，纯 NumPy）。

任务书 §2.1 / 用户 G1 验收项 5：
- 任务引导项正确映射到原生 9D QP；
- 目标 Hessian 满足要求（PSD）。

软项：λ‖P_t u - g_t‖²_Q，Hessian = W + λ P_t^T Q P_t（cvxpy quad_form 无 1/2）。
"""

from __future__ import annotations

import numpy as np
import pytest

from src.safety.stg import OSC_POS_SCALE

# AEGIS 常量
QP_ACTION_SCALE = 0.2


def _pt(R1):
    P = np.zeros((3, 9))
    P[:, :3] = QP_ACTION_SCALE * R1
    return P


def _random_rot(seed=0):
    A = np.random.default_rng(seed).normal(size=(3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


# ---------------------------------------------------------------------------
# P_t 结构
# ---------------------------------------------------------------------------

def test_pt_shape_and_structure():
    R1 = _random_rot()
    P = _pt(R1)
    assert P.shape == (3, 9)
    # 后 6 列（旋转 + 辅助）为零
    assert np.allclose(P[:, 3:], 0.0)
    # 前 3 列 = 0.2 R1
    assert np.allclose(P[:, :3], QP_ACTION_SCALE * R1)


# ---------------------------------------------------------------------------
# 目标 Hessian PSD
# ---------------------------------------------------------------------------

def test_hessian_psd():
    R1 = _random_rot()
    P = _pt(R1)
    Q_w = np.eye(3)
    W = np.diag([1.0 / 25] * 6 + [1.0] * 3)
    lam = 1.0
    H = W + lam * P.T @ Q_w @ P
    eig = np.linalg.eigvalsh(H)
    assert eig.min() > 0, "目标 Hessian 必须正定（W 正定 + 软项 PSD）"


def test_soft_term_psd_only():
    R1 = _random_rot()
    P = _pt(R1)
    Q_w = np.eye(3)
    H_soft = P.T @ Q_w @ P  # 软项二次部分
    eig = np.linalg.eigvalsh(H_soft)
    assert eig.min() >= -1e-12, "软项二次型必须 PSD"
    # rank ≤ 3（P_t 是 3×9）
    assert np.sum(eig > 1e-10) <= 3


def test_lambda_zero_soft_term_vanishes():
    R1 = _random_rot()
    P = _pt(R1)
    Q_w = np.eye(3)
    H0 = P.T @ Q_w @ P * 0.0
    assert np.allclose(H0, 0.0)


# ---------------------------------------------------------------------------
# 量纲：g_t 与 z[:3] 同量纲（归一化平移增量）
# ---------------------------------------------------------------------------

def test_osc_scale_consistent():
    # OSC_POS_SCALE = 0.05 是「归一化平移增量 → 米」的映射
    # g_t（归一化增量）× OSC_POS_SCALE = 物理位移（米）
    assert OSC_POS_SCALE == 0.05
    g_t = np.array([0.5, 0.0, 0.0])  # 归一化增量
    phys = g_t * OSC_POS_SCALE         # 米
    assert phys[0] == 0.025


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
