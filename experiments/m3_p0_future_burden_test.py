"""M3-P0：Intervention-Induced Future Burden Test（单积分器版）。

独立 2D 单积分器（速度即控制）+ 圆形障碍（CBF 相对度 1 仿射约束）+
冻结 PD 朝目标策略。

系统：
  s = [x, y]，控制 u = [vx, vy]（速度）
  s' = s + u * dt

安全约束（CBF 相对度 1，仿射对 u 线性）：
  h = ||p - c||^2 - (r+margin)^2
  h' = 2(p-c)^T u
  要求 h' >= -alpha h  →  2(p-c)^T u >= -alpha h
  即 n^T u >= b，n = (p-c)/||p-c||，b = -alpha * (||p-c||^2 - (r+margin)^2) / (2||p-c||)

  这是标准的、有物理意义的 CBF：当 h 小（接近障碍），b 变大，约束收紧；
  当 h 大（远离），b 为负（无约束力）。

冻结策略（名义速度，定期重规划）：
  u^π = -K * (p - goal)（朝目标比例控制）

候选：
  - u_QP：argmin ||u - u^π||^2 s.t. n^T u >= b（当前修正最小，贴 CBF 边界）
  - u_away：沿法向额外远离（留安全余量，n^T u > b）

未来负担 D_H：模拟 H 个重规划周期，每周期重算 π 和 QP，累加折扣 ||u_safe - u^π||^2。

核心问题：u_QP（当前修正最小）是否系统性导致更大的 D_H（未来负担）？
"""

from __future__ import annotations

import numpy as np


class SingleIntegrator2D:
    def __init__(self, dt=0.1):
        self.dt = dt

    def step(self, s, u):
        return s + u * self.dt


class CircleObstacle:
    """圆形障碍，CBF 相对度 1：n^T u >= b。"""

    def __init__(self, center, radius, alpha=2.0, margin=0.05):
        self.c = np.asarray(center, dtype=float)
        self.r = float(radius)
        self.alpha = alpha
        self.margin = margin

    def constraint(self, s):
        """返回 (n, b) 使 n^T u >= b，或 None（无约束激活）。

        n 为单位法向（指向远离障碍），b = -alpha*(||p-c|| - (r+margin))。
        当 ||p-c|| <= r+margin（已进入危险带），b >= 0，约束要求远离；
        当 ||p-c|| > r+margin，b < 0（有安全余量，约束可能无约束力）。
        """
        p = s[:2]
        dvec = p - self.c
        dist = float(np.linalg.norm(dvec))
        if dist < 1e-9:
            n = np.array([1.0, 0.0])
        else:
            n = dvec / dist
        # CBF: h = dist - (r+margin)；h' = n^T u；要求 h' >= -alpha h
        h = dist - (self.r + self.margin)
        b = -self.alpha * h
        # 只有当「名义速度会进一步深入」或「已危险」时约束才有意义。
        # 返回约束供 QP 处理；无约束力时（b 很负）QP 返回名义。
        return n, float(b)

    def violates(self, s):
        return float(np.linalg.norm(s[:2] - self.c)) < self.r


class FrozenPolicy:
    """冻结 PD 朝目标（速度控制），定期重规划。"""

    def __init__(self, goal, K=1.0):
        self.goal = np.asarray(goal, dtype=float)
        self.K = K

    def nominal(self, s):
        return -self.K * (s[:2] - self.goal)


def qp_projection(u_nom, n, b):
    """min ||u - u_nom||^2 s.t. n^T u >= b。解析解（单约束）。

    u = u_nom + max(0, (b - n^T u_nom)/||n||^2) * n
    """
    u_nom = np.asarray(u_nom, dtype=float)
    n = np.asarray(n, dtype=float)
    lhs = n @ u_nom
    if lhs >= b:
        return u_nom.copy()
    return u_nom + (b - lhs) / (n @ n) * n


def future_burden(s0, u0, policy, obstacle, dyn, H=4, gamma=0.9):
    """从 (s0,u0) 出发，未来 H 个重规划周期累加折扣干预负担。"""
    s = s0.copy()
    burden = 0.0
    s = dyn.step(s, u0)
    for k in range(H):
        u_nom = policy.nominal(s)
        n, b = obstacle.constraint(s)
        u_safe = qp_projection(u_nom, n, b)
        burden += (gamma ** k) * np.sum((u_safe - u_nom) ** 2)
        s = dyn.step(s, u_safe)
    return burden, s


def task_progress(s0, u0, policy, obstacle, dyn, H=8):
    """执行候选后机器人朝目标靠近的距离（正 = 靠近）。"""
    s = s0.copy()
    start = float(np.linalg.norm(s0[:2] - policy.goal))
    s = dyn.step(s, u0)
    for k in range(H):
        u_nom = policy.nominal(s)
        n, b = obstacle.constraint(s)
        u_safe = qp_projection(u_nom, n, b)
        s = dyn.step(s, u_safe)
    end = float(np.linalg.norm(s[:2] - policy.goal))
    return start - end


def main():
    rng = np.random.default_rng(0)
    dyn = SingleIntegrator2D(dt=0.1)
    obstacle = CircleObstacle(center=[0.0, 0.0], radius=0.3, margin=0.05)

    n_scene = 500
    reversals = 0
    n_valid = 0
    n_side_no_task_loss = 0

    for i in range(n_scene):
        # 初始位置贴障碍边缘，目标在障碍中心「另一侧」（直线必穿障碍）
        ang = rng.uniform(0, 2 * np.pi)
        dist0 = obstacle.r + obstacle.margin + rng.uniform(0.01, 0.08)
        p0 = obstacle.c + dist0 * np.array([np.cos(ang), np.sin(ang)])
        # 目标在障碍中心另一侧：沿「朝障碍中心」方向穿过障碍
        n_out = np.array([np.cos(ang), np.sin(ang)])  # 从中心指向外的法向
        goal = obstacle.c - 1.5 * n_out  # 障碍另一侧
        s0 = np.array([p0[0], p0[1]], dtype=float)
        policy = FrozenPolicy(goal)

        u_nom = policy.nominal(s0)
        n, b = obstacle.constraint(s0)
        u_qp = qp_projection(u_nom, n, b)
        dev_qp = float(np.linalg.norm(u_qp - u_nom))

        # 只有 QP 真的介入了（名义动作违反约束）才有效
        if dev_qp < 1e-9:
            continue

        # 绕行候选：沿切向绕行（主动绕过障碍，而非贴表面滑行）
        # 切向 = 垂直法向的两个方向，选「朝目标切向分量更大」的那个
        t1 = np.array([-n[1], n[0]])
        t2 = np.array([n[1], -n[0]])
        # 朝目标的切向分量：目标在 -n 方向，切向分量 = t·(goal方向投影到切向)
        # 简化：选使「下一步切向移动后离目标更近」的方向
        u_tan1 = qp_projection(u_nom + t1 * float(np.linalg.norm(u_nom)) * 0.8, n, b)
        u_tan2 = qp_projection(u_nom + t2 * float(np.linalg.norm(u_nom)) * 0.8, n, b)
        # 选任务进度更好的切向方向
        if task_progress(s0, u_tan1, policy, obstacle, dyn, H=8) >= task_progress(s0, u_tan2, policy, obstacle, dyn, H=8):
            u_side = u_tan1
        else:
            u_side = u_tan2
        dev_side = float(np.linalg.norm(u_side - u_nom))

        n_valid += 1
        b_qp, _ = future_burden(s0, u_qp, policy, obstacle, dyn, H=4)
        b_side, _ = future_burden(s0, u_side, policy, obstacle, dyn, H=4)
        prog_qp = task_progress(s0, u_qp, policy, obstacle, dyn, H=8)
        prog_side = task_progress(s0, u_side, policy, obstacle, dyn, H=8)

        if b_qp > b_side + 1e-9:
            reversals += 1
            if prog_side >= prog_qp - 0.01:
                n_side_no_task_loss += 1

    print("=== M3-P0 排序反转测试（单积分器）===")
    print(f"有效场景（QP 介入 + side 修正更大）: {n_valid}")
    print(f"排序反转: {reversals}/{n_valid}")
    if n_valid > 0:
        print(f"反转率: {reversals / n_valid:.3f}")
        print(f"反转且不损任务进度: {n_side_no_task_loss}/{reversals}")


if __name__ == "__main__":
    main()
