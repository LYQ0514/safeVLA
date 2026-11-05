"""STG（Spatiotemporal Task-Guided Governor）引导模块。

研究原型：冻结 VLA + AEGIS CBF-QP 的时空联合安全控制。**不是已验证算法，不是
ERG 安全证书**（首轮 governor 仅 ERG 启发的 dynamic reference heuristic）。

核心（任务书 §2.2）：
- 法向 n_t = ∇h/‖∇h‖（AEGIS 代理，远离障碍物方向）；
- 意图 d_t（世界系单位方向，来自当前未过滤动作 z[:3]）；
- 切向 t̂_t = (I - n n^T) d_t 归一化（意图的切向投影，保留任务方向去掉径向分量）；
- 动态参考 r_t（世界系米），含迟滞记忆；
- 引导 g_t = (r_{t+1} - p_ee)/OSC_POS_SCALE（归一化平移增量，与 z[:3] 同量纲）；
- 软项：min_{u∈U_AEGIS} J_0(u) + λ_t‖P_t u + d - g_t‖²_Q，不改变硬约束 U_AEGIS。

2×2 因子：
- B0 = 无环流无参考（λ=0，原生 AEGIS）
- B1 = circulation_only：瞬时切向引导，无 r_t 记忆
- B2 = governor_only：动态参考 r_t 迟滞，无切向
- B3 = joint_stg：r_t 迟滞 + 切向耦合

坐标/量纲（已由 G0 审计）：
- z[:3] ∈ [-1,1] 归一化平移增量（世界系），OSC 映射 0.05 m；
- P_t = 0.2 R1 [I_3, 0, 0]，z_p,out = P_t u + (z_p,ref - P_t u_ref)；
- g_t 与 z[:3] 同量纲（归一化平移增量）。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "OSC_POS_SCALE",
    "normal_world",
    "tangent_world",
    "STGGuide",
]

# OSC_POSE 位置尺度：归一化平移增量 → 米（robosuite osc.py output_max[0]）
OSC_POS_SCALE = 0.05


def normal_world(R1, a_v):
    """世界系法向 n_world = R1 @ (a_v/‖a_v‖)（远离障碍物方向，单位向量）。

    a_v 是 AEGIS compute_h_coeffs_3d 的返回值，即 ∇h 在末端局部系的表示
    （G0 数值验证：∇h_world = R1 @ a_v）。退化（‖a_v‖≈0）返回 None。
    """
    a_v = np.asarray(a_v, dtype=np.float64).ravel()
    n = float(np.linalg.norm(a_v))
    if n <= 1e-10:
        return None
    R1 = np.asarray(R1, dtype=np.float64)
    return R1 @ (a_v / n)


def tangent_world(n_world, d_world):
    """切向 t̂ = (I - n n^T) d 归一化（单位向量）。

    d_world 是意图方向（世界系单位向量）。切向是 d 在 n 的切平面上的投影：
    去掉「顶向/远离障碍物」的径向分量，保留「沿障碍物切面滑动」的分量。
    退化（‖d‖≈0 或 ‖t̄‖≈0，即 d 与 n 平行）返回 None。
    """
    n_world = np.asarray(n_world, dtype=np.float64).ravel()
    d_world = np.asarray(d_world, dtype=np.float64).ravel()
    if np.linalg.norm(d_world) <= 1e-10:
        return None
    d_world = d_world / np.linalg.norm(d_world)
    t_bar = d_world - n_world @ d_world * n_world  # (I - n n^T) d
    tn = float(np.linalg.norm(t_bar))
    if tn <= 1e-10:
        return None  # d 与 n 平行（径向意图），无切向
    return t_bar / tn


class STGGuide:
    """STG 引导器（B0/B1/B2/B3），纯 NumPy，无 MuJoCo 依赖。

    参数（开发集预注册，测试集冻结）：
        mode        : "B0" | "B1" | "B2" | "B3"
        lam_max     : 软项权重 λ 上限（风险内固定为 lam_max，首轮不做动态裕度）
        h_enter     : 风险进入阈值（h < h_enter 进入风险）
        h_exit      : 风险退出阈值（h > h_exit 退出，h_exit > h_enter 迟滞更宽）
        alpha       : 参考吸引速率（r_t → r_des），∈ (0,1]
        gamma       : 切向引导幅度（米/步），仅 B1/B3 用
    """

    def __init__(self, mode="B0", lam_max=1.0, h_enter=0.01, h_exit=0.02,
                 alpha=0.5, gamma=0.01):
        assert mode in ("B0", "B1", "B2", "B3"), f"未知 mode: {mode}"
        assert h_exit >= h_enter, "退出阈值必须 ≥ 进入阈值（迟滞）"
        self.mode = mode
        self.lam_max = float(lam_max)
        self.h_enter = float(h_enter)
        self.h_exit = float(h_exit)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self._in_risk = False
        self._r = None          # 虚拟参考位置（世界系米）
        self._prev_tangent = None  # 前一步切向（防抖动诊断）

    def reset(self):
        """episode 开始：清除全部状态，防止跨 episode 泄漏。"""
        self._in_risk = False
        self._r = None
        self._prev_tangent = None

    def in_risk(self):
        return self._in_risk

    def step(self, p_ee, a_v, R1, d_world, h):
        """一步引导计算。

        参数：
            p_ee     : 末端位置（世界系米，AEGIS 的 eef_pos，非椭球中心 p1）
            a_v      : AEGIS ∇h 系数（末端局部系 3D）
            R1       : 末端姿态 3×3
            d_world  : 意图方向（世界系 3D，可非单位，来自当前动作 z[:3]）
            h        : AEGIS CBF 值（椭圆归一化距离，h>0 安全）

        返回 (g_t, lam_t, info)：
            g_t   : 归一化平移增量（世界系 3D，与 z[:3] 同量纲）或 None（无引导）
            lam_t : 软项权重（0 = 回归 B0）
            info  : 诊断 dict
        """
        info = {"mode": self.mode, "in_risk": self._in_risk,
                "tangent_degenerate": False, "lambda": 0.0}

        # 1. 风险迟滞判定
        if not self._in_risk:
            if h < self.h_enter:
                self._in_risk = True
                self._r = np.asarray(p_ee, dtype=np.float64).copy()  # 进入时初始化
                self._prev_tangent = None
        else:
            if h > self.h_exit:
                self._in_risk = False
                self._r = None
                self._prev_tangent = None

        if not self._in_risk or self.mode == "B0":
            info["in_risk"] = self._in_risk
            return None, 0.0, info

        # 2. 法向与切向
        n_world = normal_world(R1, a_v)
        d_norm = np.asarray(d_world, dtype=np.float64).ravel()
        dn = float(np.linalg.norm(d_norm))
        d_unit = d_norm / dn if dn > 1e-10 else np.zeros(3)
        t_hat = tangent_world(n_world, d_unit) if n_world is not None else None

        # 3. 参考演化 + 引导
        g_t = None
        if self.mode == "B1":
            # 瞬时切向引导，无 r_t 记忆
            if t_hat is not None:
                g_t = (self.gamma / OSC_POS_SCALE) * t_hat
            else:
                info["tangent_degenerate"] = True
        elif self.mode in ("B2", "B3"):
            # 动态参考 r_t 演化
            if self._r is None:
                self._r = np.asarray(p_ee, dtype=np.float64).copy()
            # r_des = 意图位移目标（名义位移幅度 × 方向）
            step_size = dn * OSC_POS_SCALE  # 名义平移幅度（米）
            r_des = np.asarray(p_ee, dtype=np.float64) + step_size * d_unit
            r_next = self._r + self.alpha * (r_des - self._r)
            if self.mode == "B3" and t_hat is not None:
                r_next = r_next + self.gamma * t_hat  # 切向环流耦合
            self._r = r_next
            g_t = (r_next - np.asarray(p_ee, dtype=np.float64)) / OSC_POS_SCALE
        else:
            return None, 0.0, info

        # 4. λ 调度（首轮固定 lam_max，无动态裕度）
        lam_t = self.lam_max if g_t is not None else 0.0

        info.update({
            "in_risk": True,
            "lambda": lam_t,
            "tangent": (None if t_hat is None else t_hat.copy()),
            "normal": (None if n_world is None else n_world.copy()),
            "reference_r": (None if self._r is None else self._r.copy()),
            "g_t": (None if g_t is None else np.asarray(g_t).copy()),
        })
        return g_t, lam_t, info
