"""Circulation Inequalities for CBF-QP（严格对应论文数学形式）。

论文：Gonçalves, Krishnamurthy, Tzes, Khorrami,
      "Control Barrier Functions With Circulation Inequalities",
      IEEE Trans. Control Systems Technology, 32(4): 1426-1441, 2024.
      （arXiv:2307.09901 为 CDC 扩展版，含完整推导。）

CCBF-QP 公式（论文式 (3)）：

    u(q) = argmin_μ  ‖μ - u_d(q)‖²
    s.t.  N(q)^T μ ≥ α(D(q))       (CBF 约束，法向)
          T(q)^T μ ≥ β(D(q))       (circulation 约束，切向)
          A(q) μ ≥ b(q)            (其他控制约束)

关键定义（论文 Assumption 1 + §III-A）：
    - D(q)：closeness 度量（D>0 ⇒ q∉C，D 大 = 远离障碍物）。
    - N(q) = ∇D(q)/‖∇D(q)‖：法向单位向量。
    - Ω ∈ R^{n×n}：skew-symmetric orthonormal 矩阵，Ω^T = -Ω = Ω^{-1}。
      只存在于偶数维（hairy ball theorem 推论）；见 §IV-C 的奇数维 workaround。
    - T(q) = Ω N(q)：切向（自动正交于 N，因 N^T T = N^T Ω N = 0，Ω 反对称）。
    - β(D)：递减、β(0)>0、β(D)→-∞（如 β(D)=a-bD）；且 β(D(q_g))<0。
    - α(D)：α(0)=0、α(D)<0 for D>0（如 α(D)=-ηD）。

本模块在 AEGIS 中的适配（严格记录）：
    1. D ≡ h（AEGIS 椭圆安全距离，h 大 = 远离障碍物）。
    2. ∇h（末端局部系）= a_v（AEGIS 的 compute_h_coeffs_3d 返回 a_v = η_row·R1，
       数值验证 ∇h_world = R1 @ a_v，故 N_local = a_v/‖a_v‖）。
       ⚠️ 注意：法向必须用 ∇h = a_v，而非「末端→障碍物连线方向 z_fixed」。
       各向异性椭球下二者方向不重合（实测夹角约 28°），用 z_fixed 会引入
       系统性切向误差。
    3. 末端平移是 3D（奇数维）→ 采用论文 §IV-C workaround：Ω 在 xy 水平面（2D）
       是 orthonormal 的（n=2 的 Ω1/Ω2 嵌入），z 维奇异（不参与 circulation）。
       这对应「绕障碍物的水平切向」，适合桌面操作（末端主要在水平面绕行）。
    4. μ 对应末端局部平移速度 v_local；AEGIS 的 QP 变量 u[:3] = V_REF_GAIN·v_local。
       因此约束在 QP 变量空间写成 T_local^T·u[:3] ≥ V_REF_GAIN·β(h)，与 CBF 约束
       的缩放（QP_ACTION_SCALE × V_REF_GAIN = 1）保持一致。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "circulation_omega_2d",
    "circulation_omega_3d_xy",
    "circulation_tangent_world",
    "circulation_beta",
    "circulation_normal_local",
    "circulation_constraint",
]


def circulation_omega_2d(sign):
    """n=2 的 skew-symmetric orthonormal 矩阵（论文式 (6)）。

    Ω1 = [[0,-1],[1,0]] 对应逆时针（counter-clockwise，sign=+1）；
    Ω2 = -Ω1 对应顺时针（sign=-1）。

    验证性质：Ω^T = -Ω（反对称）；Ω^T Ω = I（正交）。
    """
    s = 1.0 if float(sign) >= 0 else -1.0
    return np.array([[0.0, -s], [s, 0.0]], dtype=np.float64)


def circulation_omega_3d_xy(sign):
    """末端平移 3D 的 Ω（论文 §IV-C 奇数维 workaround）。

    在 xy 水平面内嵌入 n=2 的 Ω（orthonormal），z 维奇异（零行零列）。
    满足：Ω^T = -Ω（反对称）；在 xy 子空间 Ω^T Ω = I（该子空间 orthonormal）；
    z 维奇异（det Ω = 0），符合 §IV-C「奇数维需牺牲 orthonormal 或可逆性」，
    这里牺牲了可逆性（z 维不 circulate），保留反对称性以保证 T ⊥ N。
    """
    o2 = circulation_omega_2d(sign)
    Omega = np.zeros((3, 3), dtype=np.float64)
    Omega[:2, :2] = o2
    return Omega


def circulation_normal_local(a_v):
    """法向 N_local = ∇h/‖∇h‖ = a_v/‖a_v‖（末端局部系，单位向量）。

    a_v 来自 AEGIS 的 compute_h_coeffs_3d，是 ∇h 在末端局部系的表示。
    返回单位向量；若 a_v 退化（‖a_v‖≈0），返回 None（无切向，circulation 关闭）。
    """
    a_v = np.asarray(a_v, dtype=np.float64).ravel()
    n = float(np.linalg.norm(a_v))
    if n <= 1e-10:
        return None
    return a_v / n


def circulation_tangent_world(R1, a_v, sign):
    """切向 T_world = Ω_world @ N_world（世界系单位切向）。

    - N_world = R1 @ (a_v/‖a_v‖)（∇h 世界系方向）；
    - Ω_world = circulation_omega_3d_xy(sign)（世界系 xy 平面旋转）；
    - T_world = Ω_world @ N_world。

    返回 (T_world, T_local)，其中 T_local = R1^T @ T_world（QP 变量 u[:3] 的局部系）。
    a_v 退化时返回 (None, None)。
    """
    N_local = circulation_normal_local(a_v)
    if N_local is None:
        return None, None
    R1 = np.asarray(R1, dtype=np.float64)
    N_world = R1 @ N_local
    T_world = circulation_omega_3d_xy(sign) @ N_world
    T_local = R1.T @ T_world
    return T_world, T_local


def circulation_beta(h, gain, activate):
    """β(D) = gain·(activate - h)，递减、β(0)>0、β→-∞（论文 Assumption 1-7）。

    这里 D ≡ h。β(0) = gain·activate > 0；dβ/dh = -gain < 0（递减）；
    h→∞ 时 β→-∞（约束平凡化）。满足 β(D(q_g))<0 当 h(q_g)>activate。
    """
    return float(gain) * (float(activate) - float(h))


def circulation_constraint(R1, a_v, h, sign, gain, activate, v_ref_gain):
    """构造 circulation 约束的系数与右端（论文式 (3) 的 T^T μ ≥ β(D)）。

    返回 (T_local, rhs)，使约束写为 T_local @ u[:3] >= rhs，
    其中 u[:3] 是 AEGIS 的 QP 变量（= v_ref_gain · v_local）。

    量纲说明（严格）：
      论文约束 T^T μ ≥ β(D)，μ = v_local（局部速度）。
      AEGIS QP 变量 u[:3] = v_ref_gain · v_local，故
      T_local^T u[:3] = v_ref_gain · T_local^T v_local ≥ v_ref_gain · β(D)。
      即 rhs = v_ref_gain · β(D)。

    返回 None（a_v 退化，关闭 circulation）或 (T_local, rhs)。
    """
    _, T_local = circulation_tangent_world(R1, a_v, sign)
    if T_local is None:
        return None
    rhs = v_ref_gain * circulation_beta(h, gain, activate)
    return T_local, rhs
