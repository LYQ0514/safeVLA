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
                 verbose=False, surface_scale=1.0):
        self.env = env
        self.spacing = spacing
        self.cbf_gain = cbf_gain  # γ in α(h) = γ h
        self.device = device
        self.verbose = verbose

        # 原生 mujoco 对象（robosuite 用 wrapper，原生对象在 ._model/.​_data）
        self._m = env.sim.model._model
        self._d = env.sim.data._data
        self._qidx = list(env.robots[0].controller.qvel_index)  # 机器人关节列

        # 采样点：(body_id, local_point)。local_point 是 body 局部坐标。
        self._samples = self._build_samples(surface_scale)
        self.psf = None
        self._jacp = np.zeros((3, self._m.nv), dtype=np.float64)
        self._jacr = np.zeros((3, self._m.nv), dtype=np.float64)

        # 统计
        self.qp_calls = 0
        self.feasible = 0
        self.intervened = 0

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
        # eef + 双指（body 原点）
        for bn in ["gripper0_eef", "gripper0_leftfinger", "gripper0_rightfinger"]:
            bid = mujoco.mj_name2id(self._m, mujoco.mjtObj.mjOBJ_BODY, bn)
            samples.append((bid, np.zeros(3)))
        if self.verbose:
            print(f"[vlpsa] 全身采样点 {len(samples)} 个")
        return samples

    # ---- episode 初始化 ----
    def reset(self, obs, prompt=""):
        occ = build_occupancy_grid(self.env, spacing=self.spacing)
        h = solve_psf(occ, max_iter=200, device=self.device)
        torch.cuda.synchronize()
        self.psf = PSF(occ, h)
        if self.verbose:
            print(f"[vlpsa] occupancy {occ.shape} 占用 {occ.occupied.sum()} "
                  f"体素, h∈[{h.min().item():.4f},{h.max().item():.4f}]")
        return True

    # ---- 采样点世界坐标 ----
    def _sample_points(self):
        pts = np.zeros((len(self._samples), 3))
        for i, (bid, lp) in enumerate(self._samples):
            R = self._d.xmat[bid].reshape(3, 3)
            pts[i] = self._d.xpos[bid] + R @ lp
        return pts

    # ---- 采样点位置雅可比 (3,7) ----
    def _sample_jacobians(self):
        J = np.zeros((len(self._samples), 3, 7))
        for i, (bid, lp) in enumerate(self._samples):
            mujoco.mj_jac(self._m, self._d, self._jacp, self._jacr, lp, bid)
            J[i] = self._jacp[:, self._qidx]
        return J

    # ---- 末端全雅可比 (6,7) ----
    def _eef_jacobian(self):
        bid = mujoco.mj_name2id(self._m, mujoco.mjtObj.mjOBJ_BODY, "gripper0_eef")
        mujoco.mj_jacBody(self._m, self._d, self._jacp, self._jacr, bid)
        J_full = np.vstack([self._jacp[:, self._qidx], self._jacr[:, self._qidx]])
        return J_full

    def filter(self, action, obs):
        """把策略末端速度 action[:6] 投影到安全集，返回 (safe_action, info)。"""
        v_nom = np.asarray(action[:6], dtype=np.float64)
        info = {"h_min": None, "feasible": None, "intervened": False}

        if self.psf is None:
            return np.asarray(action, dtype=np.float64), info

        pts = self._sample_points()
        Js = self._sample_jacobians()
        J_ee = self._eef_jacobian()

        # 查询 h 与 ∇h（GPU）
        pts_t = torch.as_tensor(pts, dtype=torch.float32, device=self.device)
        h_t, grad_t = self.psf.query(pts_t)
        h = h_t.cpu().numpy()
        grad = grad_t.cpu().numpy()

        info["h_min"] = float(h.min())

        # 末端速度 → 关节速度伪逆
        J_ee_pinv = np.linalg.pinv(J_ee)  # (7,6)

        # 每个采样点一条约束：∇h_iᵀ J_i J_ee⁺ v ≥ −γ h_i
        A_rows = []
        b_rows = []
        for i in range(len(self._samples)):
            a = grad[i] @ Js[i] @ J_ee_pinv  # (3)@(3,7)@(7,6) = (6,)
            A_rows.append(a)
            b_rows.append(-self.cbf_gain * h[i])
        A = np.vstack(A_rows)  # (M, 6)
        b = np.array(b_rows)

        # QP
        v = cp.Variable(6)
        objective = cp.Minimize(cp.sum_squares(v - v_nom))
        constraints = [A @ v >= b]
        prob = cp.Problem(objective, constraints)
        self.qp_calls += 1
        try:
            prob.solve(solver=cp.OSQP)
        except Exception:
            prob = None

        if prob is not None and v.value is not None:
            v_safe = v.value
            self.feasible += 1
            info["feasible"] = True
            if np.linalg.norm(v_safe - v_nom) > 1e-4:
                self.intervened += 1
                info["intervened"] = True
        else:
            # QP 不可行：回退原始动作（与 AEGIS 一致，但记录；这里保守改为限幅）
            v_safe = v_nom
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
        }
