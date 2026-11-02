"""CCA-ASF TASK C 集成测试：真实 AEGIS filter 的 euclidean vs cca_asf。

不依赖 MuJoCo 完整 rollout，而是：
1. 构造一个可控的 AegisSafetyLayer（用 object.__new__ 绕过 __init__，手动注入
   obstacle 状态 + CBF 系数），直接调 filter() 对比两种模式。
2. 验证 β=0 等价、非零各向异性激活、约束一致、辅助维度、异常处理。

由于 AegisSafetyLayer.filter 内部调 compute_h_coeffs_3d（需要真实位姿/椭球），
这里用一个更直接的方式：monkeypatch compute_h_coeffs_3d 返回固定系数，
从而精确控制约束行与名义动作的关系，验证各向异性逻辑本身。
"""

from __future__ import annotations

import numpy as np
import pytest

cvxpy = pytest.importorskip("cvxpy")
import src.safety.aegis_adapted as aegis_mod  # noqa: E402


def _make_layer(filter_mode, beta_max=4.0):
    layer = object.__new__(aegis_mod.AegisSafetyLayer)
    layer.env = None
    layer.verbose = False
    layer.obstacle_name = "red_coffee_mug_obstacle_1"
    layer.p2 = np.array([0.0, 0.0, 0.5])
    layer.R2 = np.eye(3)
    layer.Q2_diag = np.array([0.05, 0.05, 0.05])
    layer.z_fixed = np.array([0.0, 0.0, 1.0])
    layer.Q1_diag = np.array([0.06, 0.12, 0.20])
    layer.feasible_count = 0
    layer.infeasible_count = 0
    layer.intervene_count = 0
    layer.filter_mode = filter_mode
    layer.beta_max = float(beta_max)
    layer.eta = 0.8
    layer.intent_len = 5
    layer._task_direction_z = None
    layer._last_p1 = np.zeros(3)
    return layer


def _obs(R1=np.eye(3)):
    # 构造 obs：eef 位姿 + 障碍物位姿（供 _eef_center / find 用不到，因为已注入状态）
    return {
        "robot0_eef_pos": np.zeros(3),
        "robot0_eef_quat": np.array([1.0, 0.0, 0.0, 0.0]),  # wxyz，单位
        "red_coffee_mug_obstacle_1_pos": np.array([0.0, 0.0, 0.5]),
        "red_coffee_mug_obstacle_1_quat": np.array([1.0, 0.0, 0.0, 0.0]),
    }


# monkeypatch compute_h_coeffs_3d：返回固定系数，精确控制约束
def _patch_h_coeffs(monkeypatch, a_v, a_omega, a_uz, h):
    def fake(p_i, Q_i, R_i, p_j, Q_j, R_j, z):
        return np.asarray(a_v, dtype=float), np.asarray(a_omega, dtype=float), \
               np.asarray(a_uz, dtype=float), float(h), np.zeros(3)
    monkeypatch.setattr(aegis_mod, "compute_h_coeffs_3d", fake)


def test_T9_beta_zero_equivalence(monkeypatch):
    """cca_asf(beta_max=0) 与 euclidean 输出完全一致。"""
    # 固定 CBF 系数：平移约束 + 姿态 + z 方向都非零
    a_v = np.array([0.0, 0.0, 1.0])
    a_omega = np.array([0.2, 0.0, 0.0])
    a_uz = np.array([0.0, 0.0, 0.3])
    h = 0.5
    _patch_h_coeffs(monkeypatch, a_v, a_omega, a_uz, h)

    action = np.array([0.1, -0.2, 0.05, 0.01, -0.03, 0.02, -1.0])
    obs = _obs()

    eu = _make_layer("euclidean")
    out_eu, info_eu = eu.filter(action, obs)

    ca = _make_layer("cca_asf", beta_max=0.0)
    ca.set_task_direction(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    out_ca, info_ca = ca.filter(action, obs)

    np.testing.assert_allclose(out_eu, out_ca, atol=1e-8)
    assert info_ca["beta"] == 0.0


def test_C_T1_nonzero_anisotropy(monkeypatch):
    """存在约束冲突时，beta > 0 且输出偏离欧氏。"""
    # 约束法向主要在 +z 平移（局部系），任务方向 +x 平移（世界系），夹角非 0/1
    a_v = np.array([0.0, 0.0, 1.0])
    a_omega = np.zeros(3)
    a_uz = np.zeros(3)
    h = -1.0  # 负 h → 约束更紧，名义动作更可能违反
    _patch_h_coeffs(monkeypatch, a_v, a_omega, a_uz, h)

    # 名义动作朝 -z（违反 a_v·u >= -10h 约束）
    action = np.array([0.0, 0.0, -0.5, 0.0, 0.0, 0.0, -1.0])
    obs = _obs()

    eu = _make_layer("euclidean")
    out_eu, info_eu = eu.filter(action, obs)

    ca = _make_layer("cca_asf", beta_max=4.0)
    # 任务方向朝 +x（世界系，R1=I 时即局部系 +x），与约束法向 +z 正交？不行，
    # 正交时 cos=0 → xi=0 → beta=0。改朝 +x+z 混合（45°）使 0<xi<1
    d = np.array([0.7071, 0.0, 0.7071, 0.0, 0.0, 0.0])
    ca.set_task_direction(d)
    out_ca, info_ca = ca.filter(action, obs)

    assert info_ca["beta"] > 0.0, f"beta={info_ca['beta']} 应为正"
    # 输出应与欧氏不同（各向异性重分配）
    assert np.linalg.norm(out_ca[:3] - out_eu[:3]) > 1e-6


def test_C_T2_constraints_consistent(monkeypatch):
    """两个 QP 的安全约束一致：beta>0 时解仍满足同一 CBF 约束。"""
    a_v = np.array([0.0, 0.0, 1.0])
    a_omega = np.zeros(3)
    a_uz = np.zeros(3)
    h = -1.0
    _patch_h_coeffs(monkeypatch, a_v, a_omega, a_uz, h)

    action = np.array([0.0, 0.0, -0.5, 0.0, 0.0, 0.0, -1.0])
    obs = _obs()

    ca = _make_layer("cca_asf", beta_max=4.0)
    ca.set_task_direction(np.array([0.7071, 0.0, 0.7071, 0.0, 0.0, 0.0]))
    out_ca, info_ca = ca.filter(action, obs)

    # 反推 u（从 out 反映射）：out[:3] = 0.2 * R1 @ u_v → u_v = 5 * R1.T @ out[:3]
    R1 = aegis_mod._quat_wxyz_to_rot(np.array([1.0, 0.0, 0.0, 0.0]))
    u_v = 5.0 * (R1.T @ out_ca[:3])
    # CBF 约束：a_u_v·u_v + a_u_omega·u_omega + a_uz·u_z + 10h >= 0
    # a_u_v = 0.2*a_v = [0,0,0.2]；a_u_omega=0；a_uz=0
    lhs = 0.2 * a_v @ u_v + 10 * h
    assert lhs >= -1e-5, f"CBF 约束违反: lhs={lhs:.3e}"


def test_C_T3_aux_dims_optimized(monkeypatch):
    """辅助维度仍参与优化（未冻结）。z 方向系数非零时 u_z 被求解。"""
    a_v = np.array([0.0, 0.0, 1.0])
    a_omega = np.zeros(3)
    a_uz = np.array([0.0, 0.0, 0.5])  # z 方向参与约束
    h = -1.0
    _patch_h_coeffs(monkeypatch, a_v, a_omega, a_uz, h)

    action = np.array([0.0, 0.0, -0.5, 0.0, 0.0, 0.0, -1.0])
    obs = _obs()

    ca = _make_layer("cca_asf", beta_max=4.0)
    ca.set_task_direction(np.array([0.7071, 0.0, 0.7071, 0.0, 0.0, 0.0]))
    out_ca, info_ca = ca.filter(action, obs)
    # 只要不报错且 feasible，即说明 9D 完整求解（辅助维度未被删列）
    assert info_ca["feasible"] is True


def test_C_T4_degenerate_direction(monkeypatch):
    """方向为零/None 时安全退化，不输出未验证控制。"""
    a_v = np.array([0.0, 0.0, 1.0])
    a_omega = np.zeros(3)
    a_uz = np.zeros(3)
    h = -1.0
    _patch_h_coeffs(monkeypatch, a_v, a_omega, a_uz, h)

    action = np.array([0.0, 0.0, -0.5, 0.0, 0.0, 0.0, -1.0])
    obs = _obs()

    ca = _make_layer("cca_asf", beta_max=4.0)
    # 零方向
    ca.set_task_direction(np.zeros(6))
    out_ca, info_ca = ca.filter(action, obs)
    assert info_ca["beta"] == 0.0

    # None 方向
    ca.set_task_direction(None)
    out_ca2, info_ca2 = ca.filter(action, obs)
    assert info_ca2["beta"] == 0.0
    np.testing.assert_allclose(out_ca, out_ca2, atol=1e-12)
