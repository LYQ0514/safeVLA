"""VLPSA 安全层（方案 B：核心思想复现）。

在我们的 20Hz OSC 协议下复现 VLPSA 的两个核心创新：
1. **Poisson Safety Function（PSF）**：从 occupancy grid 解 Poisson 方程得平滑 SDF，
   替代 AEGIS 的椭球近似，可表达任意形状。
2. **全身采样**：在机械臂多个 link 上采样点（而非 AEGIS 的单个末端椭球），
   每个采样点各产生一条 CBF 约束。

与论文的差异（协议对齐，见 docs）：
- 论文 100Hz joint-velocity 控制，我们用 20Hz OSC 末端速度（与基线/AEGIS 一致）
- 论文从深度传感器建 occupancy，我们用 MuJoCo 几何 oracle
- 碰撞判定用统一框架的位移法（与基线/AEGIS 一致）

CBF-QP（末端速度空间）：
    min  ∥v − v_nom∥²
    s.t. ∇h(y_i)ᵀ (J_i J_ee⁺) v ≥ −γ h(y_i),  ∀ 采样点 i
其中 v_nom = 策略输出的末端速度（action[:6]），J_i 是采样点位置雅可比，
J_ee 是末端全雅可比。
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import cvxpy as cp  # noqa: E402
import mujoco  # noqa: E402
import torch  # noqa: E402

from src.safety.psf import PSF, build_occupancy_grid, solve_psf  # noqa: E402

# 全身采样 body（8 link 原点 + eef + 双指）
SAMPLE_BODIES = [f"robot0_link{n}" for n in range(8)] + [
    "gripper0_eef", "gripper0_leftfinger", "gripper0_rightfinger"
]


class VlpsaLayer:
    def __init__(self, env, spacing=0.03, cbf_gain=1.0, device="cuda",
                 verbose=False, surface_scale=1.0, clf_gain=1.0, slack_penalty=1.0,
                 buffer=0.0, grasp_protection=True, grasp_radius=0.06):
        self.env = env
        self.spacing = spacing
        self.cbf_gain = cbf_gain  # γ in α(h) = γ h
        self.clf_gain = clf_gain  # γ_c in V̇ ≤ -γ_c V + δ
        self.slack_penalty = slack_penalty  # p in min ∥v-v_nom∥² + p δ²
        self.buffer = buffer  # occupancy ε-膨胀（米）
        self.device = device
        self.verbose = verbose
        self.grasp_protection = grasp_protection
        self.grasp_radius = grasp_radius  # 被抓物体包围球半径（米）

        # 原生 mujoco 对象（robosuite 用 wrapper，原生对象在 ._model/.​_data）
        self._m = env.sim.model._model
        self._d = env.sim.data._data
        self._qidx = list(env.robots[0].controller.qvel_index)  # 机器人关节列

        # 采样点：(body_id, local_point)。local_point 是 body 局部坐标。
        self._samples = self._build_samples(surface_scale)
        self.psf = None
        self._jacp = np.zeros((3, self._m.nv), dtype=np.float64)
        self._jacr = np.zeros((3, self._m.nv), dtype=np.float64)

        # 持物保护状态
        self._gripper_open_width = None  # 初始张开宽度（用于抓取检测）
        self._grasped = False
        self._grasp_sample = None  # (body_id=eef, local_point)

        # 统计
        self.qp_calls = 0
        self.feasible = 0
        self.intervened = 0
        self.grasp_events = 0

    def _build_samples(self, surface_scale):
        """对每个 link 的碰撞 mesh，采样其 AABB 的 8 角点 + 6 面心 + 中心。

        mesh 的 geom_size 是 scale，作为 AABB 半轴近似（实测 mesh 局部 AABB
        与 scale 吻合）。这样每个 link 采样 ~几十点，覆盖 link 实体体积。
        """
        samples = []
        for n in range(8):
            bid = mujoco.mj_name2id(self._m, mujoco.mjtObj.mjOBJ_BODY, f"robot0_link{n}")
            # 该 link 的所有 mesh geom 的 AABB 角点
            for g in range(self._m.ngeom):
                if self._m.geom_bodyid[g] != bid:
                    continue
                half = self._m.geom_size[g].copy() * surface_scale
                pos = self._m.geom_pos[g].copy()
                # 8 角
                for sx in (-1, 1):
                    for sy in (-1, 1):
                        for sz in (-1, 1):
                            lp = pos + np.array([sx * half[0], sy * half[1], sz * half[2]])
                            samples.append((bid, lp))
                # 6 面心
                for axis in range(3):
                    for s in (-1, 1):
                        lp = pos.copy()
                        lp[axis] += s * half[axis]
                        samples.append((bid, lp))
        # 末端 + 夹爪全部 body（含指尖 tip！之前漏掉指尖，导致 CBF 感知不到指尖碰撞）
        gripper_bodies = [
            "gripper0_right_gripper", "gripper0_eef",
            "gripper0_leftfinger", "gripper0_rightfinger",
            "gripper0_finger_joint1_tip", "gripper0_finger_joint2_tip",
        ]
        for bn in gripper_bodies:
            try:
                bid = mujoco.mj_name2id(self._m, mujoco.mjtObj.mjOBJ_BODY, bn)
            except Exception:
                continue
            samples.append((bid, np.zeros(3)))
        if self.verbose:
            print(f"[vlpsa] 全身采样点 {len(samples)} 个")
        return samples

    # ---- episode 初始化 ----
    def reset(self, obs, prompt=""):
        occ = build_occupancy_grid(self.env, spacing=self.spacing, buffer=self.buffer)
        h = solve_psf(occ, max_iter=200, device=self.device)
        torch.cuda.synchronize()
        self.psf = PSF(occ, h)
        # 重置计数器（episode 间复用 layer，否则统计跨 episode 累计）
        self.qp_calls = 0
        self.feasible = 0
        self.intervened = 0
        # 持物保护状态：抓取检测基准 = 初始（张开）宽度
        gq = np.asarray(obs.get("robot0_gripper_qpos", [0.04, -0.04]), dtype=np.float64)
        self._gripper_open_width = abs(gq[0] - gq[1])
        self._grasped = False
        self._grasp_sample = None
        self.grasp_events = 0
        if self.verbose:
            print(f"[vlpsa] occupancy {occ.shape} 占用 {occ.occupied.sum()} "
                  f"体素, h∈[{h.min().item():.4f},{h.max().item():.4f}], "
                  f"gripper 张开 {self._gripper_open_width:.4f}")
        return True

    # ---- 持物保护：抓取检测与采样点注入 ----
    def _update_grasp(self, obs):
        """检测抓取状态，注入/移除被抓物体采样点（论文 III-B）。"""
        if not self.grasp_protection:
            return
        gq = np.asarray(obs.get("robot0_gripper_qpos", [0.04, -0.04]), dtype=np.float64)
        width = abs(gq[0] - gq[1])
        # 抓取：宽度明显小于初始张开宽度（两指闭合夹住物体）
        threshold = 0.5 * self._gripper_open_width
        grasped = width < threshold
        if grasped and not self._grasped:
            self._grasped = True
            self.grasp_events += 1
            # 被抓物体包围球，固定在末端（eef body 局部原点，半径 grasp_radius）
            eef_bid = mujoco.mj_name2id(self._m, mujoco.mjtObj.mjOBJ_BODY, "gripper0_eef")
            self._grasp_sample = (eef_bid, np.array([0.0, 0.0, 0.0]))
        elif not grasped:
            self._grasped = False
            self._grasp_sample = None

    def _active_samples(self):
        """返回当前有效采样集（机器人全身 + 可能的被抓物体包围球）。

        被抓物体用 grasp_radius 的球面采样（6 轴向点），固定在 eef 局部坐标。
        """
        samples = list(self._samples)
        if self._grasped and self._grasp_sample is not None:
            bid, _ = self._grasp_sample
            r = self.grasp_radius
            for axis in range(3):
                for s in (-1, 1):
                    lp = np.zeros(3)
                    lp[axis] = s * r
                    samples.append((bid, lp))
        return samples

    # ---- 采样点世界坐标 ----
    def _sample_points(self, samples=None):
        if samples is None:
            samples = self._active_samples()
        pts = np.zeros((len(samples), 3))
        for i, (bid, lp) in enumerate(samples):
            R = self._d.xmat[bid].reshape(3, 3)
            pts[i] = self._d.xpos[bid] + R @ lp
        return pts

    # ---- 采样点位置雅可比 (3,7) ----
    def _sample_jacobians(self, samples=None):
        if samples is None:
            samples = self._active_samples()
        J = np.zeros((len(samples), 3, 7))
        for i, (bid, lp) in enumerate(samples):
            mujoco.mj_jac(self._m, self._d, self._jacp, self._jacr, lp, bid)
            J[i] = self._jacp[:, self._qidx]
        return J

    # ---- 末端全雅可比 (6,7) ----
    def _eef_jacobian(self):
        bid = mujoco.mj_name2id(self._m, mujoco.mjtObj.mjOBJ_BODY, "gripper0_eef")
        mujoco.mj_jacBody(self._m, self._d, self._jacp, self._jacr, bid)
        J_full = np.vstack([self._jacp[:, self._qidx], self._jacr[:, self._qidx]])
        return J_full

    def filter(self, action, obs, x_desired=None, action_scale=0.009, joint_space=True):
        """按论文 MC-CBF-QP 解安全关节速度，返回 (safe_action, info)。

        论文在**关节速度空间**（v ∈ R⁷）解 QP，保留 Franka 的冗余自由度
        （7 关节 vs 6 末端），这是之前末端空间复现的根本错误所在：

            min  ∥v − v_nom∥² + p δ²
            s.t. ∇hᵀ Jᵢ v ≥ −γ_cbf hᵢ     (CBF, 全身硬约束，Jᵢ∈R³ˣ⁷)
                 eᵀ J_ee v ≥ γ_clf V − δ   (CLF, 追踪软约束)

        action 是 π0.5 的末端速度格式（[dx,dy,dz,rx,ry,rz,grip]）。
        返回 8 维关节动作（7 关节速度 + 夹爪），供 JOINT_VELOCITY 控制器；
        若 joint_space=False，映射回 6 维末端速度（旧逻辑，供对照）。
        """
        info = {"h_min": None, "feasible": None, "intervened": False}

        if self.psf is None:
            return np.asarray(action, dtype=np.float64), info

        # 持物保护：检测抓取，更新采样集
        self._update_grasp(obs)
        samples = self._active_samples()

        pts = self._sample_points(samples)
        Js = self._sample_jacobians(samples)   # (N, 3, 7) 关节位置雅可比
        J_ee = self._eef_jacobian()            # (6, 7) 末端全雅可比

        # 查询 h 与 ∇h（GPU）
        pts_t = torch.as_tensor(pts, dtype=torch.float32, device=self.device)
        h_t, grad_t = self.psf.query(pts_t)
        h = h_t.cpu().numpy()
        grad = grad_t.cpu().numpy()

        info["h_min"] = float(h.min())

        # ---- nominal 关节速度 ----
        # π0.5 末端速度 → 物理末端速度 → 关节速度（最小范数伪逆，作为 QP 初始猜测）
        x_dot_des = np.zeros(6)
        x_dot_des[:3] = np.asarray(action[:3]) * action_scale   # 平移 m/s
        x_dot_des[3:6] = np.asarray(action[3:6]) * action_scale  # 旋转（近似）
        v_nom = np.linalg.pinv(J_ee) @ x_dot_des                 # (7,)

        if joint_space:
            # ---- 关节空间 QP（论文 MC-CBF-QP）----
            v = cp.Variable(7)
            constraints = []
            objective_expr = cp.sum_squares(v - v_nom)
            # CBF：∇hᵀ Jᵢ v ≥ −γ hᵢ（Jᵢ 直接 3×7，不经过伪逆 → 保留冗余度）
            A_rows, b_rows = [], []
            for i in range(len(samples)):
                A_rows.append(grad[i] @ Js[i])   # (7,)
                b_rows.append(-self.cbf_gain * h[i])
            A = np.vstack(A_rows)
            b = np.array(b_rows)
            constraints.append(A @ v >= b)
            # CLF：追踪期望轨迹（软约束）
            use_clf = x_desired is not None and self.clf_gain > 0
            delta = None
            if use_clf:
                delta = cp.Variable(1)
                objective_expr = objective_expr + self.slack_penalty * cp.square(delta)
                x_ee = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
                # 位置误差用物理米（不除 action_scale）：V 单位 m²，
                # eᵀ J_ee_pos v 单位 m·(m/s)=m²/s，γV 单位 m²/s —— 量纲一致。
                e = np.asarray(x_desired, dtype=np.float64) - x_ee
                V = 0.5 * float(e @ e)
                J_ee_pos = J_ee[:3]  # (3,7)
                constraints.append(e @ J_ee_pos @ v >= self.clf_gain * V - delta)
                info["V"] = V
                info["e_norm"] = float(np.linalg.norm(e))

            prob = cp.Problem(cp.Minimize(objective_expr), constraints)
            self.qp_calls += 1
            try:
                prob.solve(solver=cp.OSQP)
            except Exception:
                prob = None

            if prob is not None and v.value is not None:
                v_safe = np.asarray(v.value, dtype=np.float64)
                self.feasible += 1
                info["feasible"] = True
                if np.linalg.norm(v_safe - v_nom) > 1e-4:
                    self.intervened += 1
                    info["intervened"] = True
            else:
                v_safe = v_nom
                info["feasible"] = False

            # 输出 8 维：7 关节速度 + 夹爪
            out = np.zeros(8)
            out[:7] = v_safe
            out[7] = action[6]  # 夹爪（JOINT_VELOCITY 下 action[7]=+1 闭合）
            return out, info

        else:
            # ---- 末端空间 QP（旧逻辑，供对照，不推荐）----
            v_nom_ee = x_dot_des
            J_ee_pinv = np.linalg.pinv(J_ee)
            A_rows, b_rows = [], []
            for i in range(len(samples)):
                A_rows.append(grad[i] @ Js[i] @ J_ee_pinv)
                b_rows.append(-self.cbf_gain * h[i])
            A = np.vstack(A_rows)
            b = np.array(b_rows)
            v = cp.Variable(6)
            constraints = [A @ v >= b]
            prob = cp.Problem(cp.Minimize(cp.sum_squares(v - v_nom_ee)), constraints)
            self.qp_calls += 1
            try:
                prob.solve(solver=cp.OSQP)
            except Exception:
                prob = None
            if prob is not None and v.value is not None:
                v_safe = np.asarray(v.value, dtype=np.float64)
                self.feasible += 1
                info["feasible"] = True
                if np.linalg.norm(v_safe - v_nom_ee) > 1e-4:
                    self.intervened += 1
                    info["intervened"] = True
            else:
                v_safe = v_nom_ee
                info["feasible"] = False
            out = np.asarray(action, dtype=np.float64).copy()
            out[:6] = v_safe
            return out, info

    def stats(self):
        return {
            "qp_calls": self.qp_calls,
            "feasible": self.feasible,
            "intervened": self.intervened,
            "infeasible_rate": (self.qp_calls - self.feasible) / max(self.qp_calls, 1),
            "grasp_events": self.grasp_events,
        }
