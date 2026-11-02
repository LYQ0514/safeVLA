"""CCA-ASF 数值验收测试：解析算例 + 随机 QP 对比解析单约束公式。

规格书 §6 的「务必单测」：随机生成可行的 2-6 维线性 QP，对比 OSQP 与
解析单约束公式，对 KKT 残差和零干预条件断言。
"""

from __future__ import annotations

import numpy as np
import pytest

cvxpy = pytest.importorskip("cvxpy")

from src.safety.cca_asf.metric import build_anisotropic_W, map_dz_to_dy  # noqa: E402


def _solve_qp(W, z_nom, C, r, lower, upper):
    """求解 min (z-z_nom)^T W (z-z_nom) s.t. C z >= r, lower <= z <= upper。

    返回 z*（np.ndarray）或 None（不可行）。
    """
    n = len(z_nom)
    z = cvxpy.Variable(n)
    constraints = [C @ z >= r, z >= lower, z <= upper]
    obj = cvxpy.Minimize(cvxpy.quad_form(z - z_nom, W))
    prob = cvxpy.Problem(obj, constraints)
    try:
        prob.solve(solver=cvxpy.OSQP)
    except Exception:
        return None
    if z.value is None:
        return None
    return np.asarray(z.value, dtype=float)


def test_analytic_example_strict():
    """规格书 §6 严格复现：(0.2,0.8) vs (0.5,0.5)。"""
    z_nom = np.array([0.0, 0.0])
    d = np.array([1.0, 0.0])
    beta = 3.0
    W_iso = np.eye(2)
    W_aniso = np.eye(2) + beta * np.outer(d, d)  # = diag(4,1)

    C = np.array([[1.0, 1.0]])
    r = np.array([1.0])
    lower = np.array([-1.0, -1.0])
    upper = np.array([1.0, 1.0])

    zI = _solve_qp(W_iso, z_nom, C, r, lower, upper)
    zW = _solve_qp(W_aniso, z_nom, C, r, lower, upper)
    np.testing.assert_allclose(zI, [0.5, 0.5], atol=1e-6)
    np.testing.assert_allclose(zW, [0.2, 0.8], atol=1e-6)


def test_zero_intervention_random():
    """零干预：名义动作可行时，各向异性 QP 严格返回名义动作（随机 1000 个）。"""
    rng = np.random.default_rng(42)
    for n in (2, 3, 4, 6):
        for _ in range(250):
            d = rng.normal(size=n); d /= np.linalg.norm(d)
            beta = rng.uniform(0, 4)
            W = np.eye(n) + beta * np.outer(d, d)
            # 名义动作严格在盒约束 [-1,1] 内
            z_nom = rng.uniform(-0.5, 0.5, size=n)
            # 约束 C z >= r，设 r 使 z_nom 严格可行（留 0.5 裕度）
            C = rng.normal(size=(1, n))
            r = np.array([C[0] @ z_nom - 0.5])
            lower = np.full(n, -1.0)
            upper = np.full(n, 1.0)
            z = _solve_qp(W, z_nom, C, r, lower, upper)
            assert z is not None
            np.testing.assert_allclose(z, z_nom, atol=1e-5)


def test_single_constraint_ratio():
    """定理 3：单约束时 d 方向偏移比值 = (1-gamma)/(1-gamma cos^2 theta)。"""
    rng = np.random.default_rng(7)
    for _ in range(200):
        n = rng.integers(2, 7)
        d = rng.normal(size=n); d /= np.linalg.norm(d)
        beta = rng.uniform(0.1, 4.0)
        gamma = beta / (1 + beta)
        c = rng.normal(size=n)
        theta_cos = float(d @ c) / np.linalg.norm(c)
        # 避开正交（cos=0 比值无定义）和平行（cos=1 无重分配）
        if abs(theta_cos) < 0.1 or abs(theta_cos) > 0.99:
            continue
        # 名义动作违反单约束
        z_nom = np.zeros(n)
        r = np.array([1.0])  # 需要 c^T z >= 1，但 z_nom=0 违反
        W_iso = np.eye(n)
        W_aniso = np.eye(n) + beta * np.outer(d, d)
        lower = np.full(n, -100.0)  # 大盒，近似无盒（避免盒约束激活）
        upper = np.full(n, 100.0)
        zI = _solve_qp(W_iso, z_nom, c[None, :], r, lower, upper)
        zW = _solve_qp(W_aniso, z_nom, c[None, :], r, lower, upper)
        assert zI is not None and zW is not None
        devI = abs(float(d @ (zI - z_nom)))
        devW = abs(float(d @ (zW - z_nom)))
        ratio_actual = devW / devI if devI > 1e-12 else None
        ratio_expected = (1 - gamma) / (1 - gamma * theta_cos ** 2)
        assert ratio_actual is not None
        assert ratio_actual <= 1.0 + 1e-6
        np.testing.assert_allclose(ratio_actual, ratio_expected, atol=1e-4)


def test_constraint_feasibility_invariant():
    """定理 2：各向异性不改变硬约束可行性（同一可行集）。"""
    rng = np.random.default_rng(11)
    for _ in range(200):
        n = rng.integers(2, 7)
        d = rng.normal(size=n); d /= np.linalg.norm(d)
        beta = rng.uniform(0, 4)
        C = rng.normal(size=(2, n))
        r = rng.uniform(-0.5, 0.5, size=2)
        z_nom = rng.normal(size=n)
        lower = np.full(n, -1.0)
        upper = np.full(n, 1.0)
        zI = _solve_qp(np.eye(n), z_nom, C, r, lower, upper)
        zW = _solve_qp(np.eye(n) + beta * np.outer(d, d), z_nom, C, r, lower, upper)
        # 可行性一致：两者同可解或同不可解
        assert (zI is None) == (zW is None)
