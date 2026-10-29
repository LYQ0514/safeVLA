"""AEGIS 安全约束层适配（对比实验用）。

上游代码：`third_party/aegis/main_aegis.py` + `utils.py`（MIT，原样保留）。
本文件只做**接口适配**，不修改算法：CBF 的 h 计算与 QP 求解逻辑全部沿用上游。

改动清单（必须写进报告，因为影响可比性）：
1. `eef_marker` body 删除 —— 上游靠它做可视化，公开版 SafeLIBERO 没有这个 body，
   直接调用 `body_name2id("eef_marker")` 会抛 ValueError。
2. 障碍物识别改为 **oracle 真值** —— 上游用智谱 GLM-4.5v 云 API（api_key 为空）
   + GroundingDINO + 深度点云反投影 + 最小体积椭球拟合。
   我们直接从 `sim.model` 读障碍物的位姿与 geom 尺寸。
3. π0.5 通过本项目 `Pi05Policy` 调用（文件 IPC），不走上游的 websocket server。
4. 几何与椭球尺寸从 MuJoCo geom 读取，其余（p1/p2、Q1/Q2、R1/R2）喂给同一套函数。

**因此我们跑出的 AEGIS 数字不能直接与论文的 77.85% CAR 对比**，
只能作为「AEGIS 控制层在本机条件下的表现」。
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, "third_party", "aegis"))

# 上游 utils 里有 evaluate 分支依赖，直接 import 我们要用的数学部分
from utils import compute_h_coeffs_3d, compute_h_ij, project_matrix  # noqa: E402

import cvxpy as cp  # noqa: E402

# ---------------------------------------------------------------------------
# 上游超参数，原样保留（见 main_aegis.py 顶部常量）
# ---------------------------------------------------------------------------
ALPHA = 1.0            # CBF gain（上游定义但实际用下面的 10*h 系数）
MAX_VEL = 1.0
DT = 0.05              # 上游 dt = 0.05
QP_H_GAIN = 10         # 上游约束里用 10 * h
QP_ACTION_SCALE = 0.2  # 上游 action_input = 0.2 * R1 @ u_v
V_REF_GAIN = 5         # 上游 u_v_ref = 5 * v_ref
OMEGA_REF_GAIN = 5
U_Z_NOM_GAIN = 10

# 末端椭球半轴（上游按任务类型二选一，见 main_aegis.py ~line 155）
EEF_Q_DEFAULT = np.array([0.06, 0.12, 0.11])
EEF_Q_LONG_OBJECT = np.array([0.06, 0.12, 0.20])
EEF_OFFSET_LOCAL = np.array([0.0, 0.0, -0.08])


def set_h_gain(gain):
    """消融用：调整 CBF 约束里的 h 增益（上游固定 10）。

    调小 → 约束更松，更少干预、TSR 更高但 CAR 更低。
    """
    global QP_H_GAIN
    QP_H_GAIN = gain


def set_use_small_eef(flag):
    """消融用：是否忽略上游「大椭球」规则（orange juice/milk/alphabet soup）。

    True = 所有任务都用 [0.06,0.12,0.11] 小椭球，用来验证
    "大椭球导致过度约束" 这一假设。
    """
    global _USE_SMALL_EEF
    _USE_SMALL_EEF = flag


_USE_SMALL_EEF = False


def _quat_wxyz_to_rot(quat_wxyz):
    """MuJoCo/robosuite 四元数是 wxyz，转 3x3 旋转矩阵。"""
    from scipy.spatial.transform import Rotation as R
    w, x, y, z = quat_wxyz
    return R.from_quat([x, y, z, w]).as_matrix()


def find_active_obstacle(env, obs):
    """oracle：找出工作空间内的障碍物（上游同款判定逻辑）。

    上游用 VLM + GroundingDINO 猜哪个障碍物会被撞；
    这里有仿真真值，直接取工作空间内第一个 obstacle。
    """
    names = [n.replace("_joint0", "") for n in env.sim.model.joint_names if "obstacle" in n]
    for name in names:
        key = name + "_pos"
        if key not in obs:
            continue
        p = np.asarray(obs[key], dtype=np.float64)
        if p[2] > 0 and abs(p[0]) < 0.5 and abs(p[1]) < 0.5:
            return name
    return None


def obstacle_ellipsoid(env, obs, name):
    """oracle：从 MuJoCo geom 直接取障碍物的位姿与包围椭球半轴。

    上游是对深度点云做最小体积椭球拟合（mvee_cvxpy）。我们用真值：
    - 位姿：obs[f"{name}_pos"] / `_quat`
    - 半轴：该 body 所有 geom 的 size 取最大（保守，保证包住物体）
    """
    pos = np.asarray(obs[name + "_pos"], dtype=np.float64)
    quat = np.asarray(obs[name + "_quat"], dtype=np.float64)
    R = _quat_wxyz_to_rot(quat)

    model = env.sim.model
    body_name = name + "_main"
    try:
        bid = model.body_name2id(body_name)
    except Exception:
        # 兜底：按前缀找
        bid = next(
            (i for i, b in enumerate(model.body_names) if b.startswith(name)),
            None,
        )
        if bid is None:
            return None
    gids = [g for g in range(model.ngeom) if model.geom_bodyid[g] == bid]
    if not gids:
        return None
    # 每个 geom 的 size 是半轴（box/ellipsoid/capsule 含义不同），取最大最保守
    half_axes = np.max(np.array([model.geom_size[g] for g in gids]), axis=0)
    half_axes = np.maximum(half_axes, 1e-3)
    return pos, R, half_axes


class AegisSafetyLayer:
    """AEGIS 的 CBF-QP 安全层（算法沿用上游）。

    用法：
        layer = AegisSafetyLayer(env)
        layer.reset(obs)                       # 每个 episode 开始时
        safe_action = layer.filter(action, obs)

    `filter()` 内部与上游 main_aegis.py 第 ~300-360 行同构：
      v_ref = R1.T @ action[:3];  u_v_ref = 5 * v_ref
      a_v, a_omega, a_uz, h, mu = compute_h_coeffs_3d(...)
      解 QP: min ||u - u_ref||^2_W  s.t. a_v·u_v + a_omega·u_w + a_uz·u_z + 10h >= 0
      action_out[:3] = 0.2 * R1 @ u_v ; [:6] = 0.2 * u_omega ; [6] 保持夹爪
    """

    def __init__(self, env, verbose=False):
        self.env = env
        self.verbose = verbose
        self.obstacle_name = None
        self.p2 = None
        self.R2 = None
        self.Q2_diag = None
        self.z_fixed = None
        self.Q1_diag = EEF_Q_DEFAULT.copy()
        self.feasible_count = 0
        self.infeasible_count = 0
        self.intervene_count = 0

    def reset(self, obs, task_language=""):
        """episode 开始：定位障碍物、算初始椭球与 z 方向。"""
        # 末端椭球半轴按任务类型选（上游同款规则）
        lang = task_language.lower()
        if _USE_SMALL_EEF:
            self.Q1_diag = EEF_Q_DEFAULT.copy()
        elif any(k in lang for k in ("orange juice", "milk", "alphabet soup")):
            self.Q1_diag = EEF_Q_LONG_OBJECT.copy()
        else:
            self.Q1_diag = EEF_Q_DEFAULT.copy()

        name = find_active_obstacle(self.env, obs)
        if name is None:
            self.obstacle_name = None
            if self.verbose:
                print("[aegis] 未找到工作空间内障碍物 → 安全层关闭")
            return False

        ell = obstacle_ellipsoid(self.env, obs, name)
        if ell is None:
            self.obstacle_name = None
            return False

        self.obstacle_name = name
        self.p2, self.R2, self.Q2_diag = ell

        p1 = self._eef_center(obs)
        z = self.p2 - p1
        nz = np.linalg.norm(z)
        self.z_fixed = z / nz if nz > 1e-8 else np.array([1.0, 0, 0])
        self._last_p1 = p1

        if self.verbose:
            print(f"[aegis] 障碍物={name} 半轴={np.round(self.Q2_diag,4)} "
                  f"距离={nz:.3f}m")
        return True

    @staticmethod
    def _eef_center(obs):
        """末端椭球中心：eef 位姿沿局部 z 偏移 -0.08（上游同款）。"""
        eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
        R1 = _quat_wxyz_to_rot(np.asarray(obs["robot0_eef_quat"], dtype=np.float64))
        return eef_pos + R1 @ EEF_OFFSET_LOCAL

    def filter(self, action, obs):
        """把策略动作投影到安全集内。返回 (safe_action, info)。"""
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        info = {"active": False, "h": None, "feasible": None}

        if self.obstacle_name is None:
            return action, info

        p1 = self._eef_center(obs)
        R1 = _quat_wxyz_to_rot(np.asarray(obs["robot0_eef_quat"], dtype=np.float64))

        v_ref = R1.T @ action[:3]
        u_v_ref = V_REF_GAIN * v_ref
        omega_ref = action[3:6]
        u_omega_ref = OMEGA_REF_GAIN * omega_ref

        a_v, a_omega, a_uz, h, mu_row = compute_h_coeffs_3d(
            p1, self.Q1_diag, R1,
            self.p2, self.Q2_diag, self.R2,
            self.z_fixed,
        )
        a_u_v = QP_ACTION_SCALE * a_v
        a_u_omega = QP_ACTION_SCALE * a_omega
        u_z_nom = U_Z_NOM_GAIN * mu_row

        u = cp.Variable(9)
        W = np.diag([1.0 / 25] * 6 + [1.0, 1.0, 1.0])
        u_ref_vec = np.hstack([u_v_ref, u_omega_ref, u_z_nom])
        objective = cp.Minimize(cp.quad_form(u - u_ref_vec, W))
        constraints = [a_u_v @ u[:3] + a_u_omega @ u[3:6] + a_uz @ u[6:] + QP_H_GAIN * h >= 0]
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.OSQP)
        except Exception:
            prob = None

        info["active"] = True
        info["h"] = float(h)

        if prob is not None and u.value is not None:
            self.feasible_count += 1
            u_v = u.value[:3]
            u_omega = u.value[3:6]
            u_z = u.value[6:]
            info["feasible"] = True
            # 判断是否真的介入了（与参考解不同）
            if np.linalg.norm(u_v - u_v_ref) > 1e-3:
                self.intervene_count += 1
                info["intervened"] = True
        else:
            # ⚠️ 上游在不可行时**回退到原始动作**（最危险）。
            # 这里保留上游行为以保证"复现"，但单独计数以便报告。
            self.infeasible_count += 1
            u_v = action[:3]
            u_omega = action[3:6]
            u_z = u_z_nom
            info["feasible"] = False
            info["fallback_to_raw"] = True

        # z 方向（椭球朝向）演化，上游同款
        Id = np.eye(3)
        dz = (Id - np.outer(self.z_fixed, self.z_fixed)) @ u_z
        self.z_fixed = self.z_fixed + dz * DT
        self.z_fixed = self.z_fixed / max(np.linalg.norm(self.z_fixed), 1e-8)

        out = np.zeros(7)
        out[:3] = QP_ACTION_SCALE * R1 @ u_v
        out[3:6] = QP_ACTION_SCALE * u_omega
        out[6] = action[6]  # 夹爪不变

        self._last_p1 = p1
        # 障碍物可能被碰动：每步刷新位姿（上游只初始化一次，见文档）
        if self.obstacle_name:
            key = self.obstacle_name + "_pos"
            if key in obs:
                self.p2 = np.asarray(obs[key], dtype=np.float64)

        return out, info

    def stats(self):
        total = self.feasible_count + self.infeasible_count
        return {
            "qp_calls": total,
            "feasible": self.feasible_count,
            "infeasible": self.infeasible_count,
            "intervened": self.intervene_count,
            "infeasible_rate": (self.infeasible_count / total) if total else 0.0,
        }
