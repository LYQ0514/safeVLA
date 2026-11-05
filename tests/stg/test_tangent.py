"""STG 切向性数学单测（G1，纯 NumPy）。

任务书 §5 / 用户 G1 验收项 3：
- 切向方向满足几何正交性 |n^T t| ≤ 1e-6；
- 退化几何（平行/反向/零动作）；
- 方向切换连续性；
- 旋转等变性。
"""

from __future__ import annotations

import numpy as np
import pytest

from src.safety.stg import normal_world, tangent_world


def _random_rot(seed=0):
    A = np.random.default_rng(seed).normal(size=(3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


# ---------------------------------------------------------------------------
# 法向 n = ∇h 方向
# ---------------------------------------------------------------------------

def test_normal_world_is_unit():
    R1 = _random_rot()
    a_v = np.array([0.3, -0.4, 0.5])
    n = normal_world(R1, a_v)
    assert n is not None
    assert np.linalg.norm(n) == pytest.approx(1.0)


def test_normal_degenerate():
    assert normal_world(np.eye(3), np.zeros(3)) is None
    assert normal_world(np.eye(3), np.array([1e-12, 0, 0])) is None


# ---------------------------------------------------------------------------
# 切向正交性
# ---------------------------------------------------------------------------

def test_tangent_orthogonal_to_normal():
    for seed in range(5):
        n = np.random.default_rng(seed).normal(size=3)
        n = n / np.linalg.norm(n)
        d = np.random.default_rng(seed + 100).normal(size=3)
        d = d / np.linalg.norm(d)
        t = tangent_world(n, d)
        if t is not None:
            assert abs(n @ t) < 1e-9, "切向必须正交于法向"


def test_tangent_is_unit():
    n = np.array([1.0, 0, 0])
    d = np.array([0.0, 1.0, 0.0])
    t = tangent_world(n, d)
    assert np.linalg.norm(t) == pytest.approx(1.0)


def test_tangent_projects_out_normal_component():
    # t = (I - n n^T) d 归一化：d 的切向分量方向正确
    n = np.array([1.0, 0, 0])
    d = np.array([0.7, 0.7, 0.0])
    t = tangent_world(n, d)
    # 切向 = d 去掉 x 分量 = [0, 0.7, 0] 归一化 = [0,1,0]
    assert np.allclose(t, [0.0, 1.0, 0.0])


# ---------------------------------------------------------------------------
# 退化几何
# ---------------------------------------------------------------------------

def test_tangent_degenerate_parallel():
    # d 与 n 平行（纯径向意图）→ 无切向
    n = np.array([1.0, 0, 0])
    d = np.array([1.0, 0, 0])  # d ∥ n
    assert tangent_world(n, d) is None


def test_tangent_degenerate_antiparallel():
    n = np.array([1.0, 0, 0])
    d = np.array([-1.0, 0, 0])  # d 反向 n
    assert tangent_world(n, d) is None


def test_tangent_degenerate_zero_intent():
    n = np.array([1.0, 0, 0])
    assert tangent_world(n, np.zeros(3)) is None


# ---------------------------------------------------------------------------
# 方向切换：法向变化时切向连续变化（非跳变）
# ---------------------------------------------------------------------------

def test_tangent_continuous_under_normal_rotation():
    # 法向缓慢旋转，切向方向角连续变化（无符号跳变）
    d = np.array([1.0, 0, 0])
    prev_t = None
    for theta in np.linspace(0, np.pi / 2, 50):
        n = np.array([np.cos(theta), np.sin(theta), 0.0])
        t = tangent_world(n, d)
        if prev_t is not None and t is not None:
            assert np.dot(prev_t, t) > 0.5, "切向不应发生符号跳变"
        prev_t = t


# ---------------------------------------------------------------------------
# 旋转等变性
# ---------------------------------------------------------------------------

def test_tangent_rotation_equivariance():
    # 整体旋转 n 和 d，切向也应旋转同样角度
    n = np.array([1.0, 0, 0])
    d = np.array([0.0, 1.0, 0.0])
    t = tangent_world(n, d)
    R = _random_rot(1)
    n_rot = R @ n
    d_rot = R @ d
    t_rot = tangent_world(n_rot, d_rot)
    assert np.allclose(t_rot, R @ t, atol=1e-9)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
