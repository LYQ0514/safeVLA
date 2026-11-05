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

# CCA-ASF 白化度量复用（避免与 metric.py 重复实现，保持一致）
from src.safety.cca_asf.metric import (  # noqa: E402
    B_DIAG, compute_conflict_beta, map_dz_to_dy,
)

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


def _quat_wxyz_to_rot(quat):
    """四元数 → 3x3 旋转矩阵（xyzw 约定，兼容历史调用名）。

    ⚠️ 修复记录（2026-11-04）：robosuite 的 `robot0_eef_quat` 是 **xyzw** 约定
    （robosuite/robots/single_arm.py: `convert_quat(..., to="xyzw")`），不是 wxyz。
    之前误当 wxyz 做 `[x,y,z,w]=[q1,q2,q3,q0]` 重排，导致 R1 的 y/z 轴符号翻转、
    末端椭球中心 p1 偏移方向反了（16cm 偏差）。现已改为 scipy 默认 xyzw 直通。
    函数名保留以免破坏调用点，语义已纠正。
    """
    from scipy.spatial.transform import Rotation as R
    return R.from_quat(np.asarray(quat, dtype=np.float64)).as_matrix()


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


def _geom_half_axes(geom_type, geom_size):
    """按 MuJoCo geom type 把 geom_size 解释为「半轴」（世界系前，局部系后）。"""
    s = np.asarray(geom_size, dtype=np.float64)
    if geom_type == 2:  # sphere: [radius]
        return np.array([s[0], s[0], s[0]])
    if geom_type == 3:  # capsule: [radius, half_length]，z 半轴 = radius + half_length
        return np.array([s[0], s[0], s[0] + s[1]])
    if geom_type == 5:  # cylinder: [radius, half_length]
        return np.array([s[0], s[0], s[1]])
    if geom_type in (4, 6, 7):  # ellipsoid / box / mesh: 3 半轴（mesh 的 size 视为半轴）
        return s.copy()
    return np.array([s[0], s[0], s[0]]) if s.shape[0] == 1 else s.copy()


def _box_vertices(center, Rmat, half):
    """box/椭球 geom 的世界系 8 顶点（对角顶点）。"""
    verts = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                local = np.array([sx * half[0], sy * half[1], sz * half[2]])
                verts.append(center + Rmat @ local)
    return np.array(verts)


def obstacle_ellipsoid(env, obs, name):
    """oracle：对「碰撞几何顶点」拟合最小外接椭球（MVEE），保证包围性。

    修复（审计 P0-B）：之前用「body 原点 + geom_size 逐轴 max」或「AABB 半长」，
    两者都**不能保证包住**碰撞几何（AABB 半长做椭球半轴时 8 顶点里 6 个越界）。
    正确做法：
    - 遍历该 body 所有 contype>0（碰撞几何）的 geom；
    - 用 geom_xpos/geom_xmat 算每个 geom 的世界系 8 顶点；
    - 对全体顶点拟合 MVEE（与上游对点云拟合 MVEE 同一数学操作，输入换成真值顶点）。

    返回 (center, R, half_axes)，R 是 MVEE 主轴（列向量），half_axes 是半轴长。
    """
    model = env.sim.model
    data = env.sim.data
    body_name = name + "_main"
    try:
        bid = model.body_name2id(body_name)
    except Exception:
        bid = next(
            (i for i, b in enumerate(model.body_names) if b.startswith(name)), None
        )
        if bid is None:
            return None

    has_world = hasattr(data, "geom_xpos") and hasattr(data, "geom_xmat")
    verts = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != bid or model.geom_contype[g] <= 0:
            continue  # 只取碰撞几何（contype>0），跳过纯视觉 mesh
        half = _geom_half_axes(int(model.geom_type[g]), model.geom_size[g])
        if has_world:
            c = np.asarray(data.geom_xpos[g], dtype=np.float64)
            Rm = np.asarray(data.geom_xmat[g], dtype=np.float64).reshape(3, 3)
        else:
            c = (np.asarray(data.xpos[bid], dtype=np.float64)
                 + np.asarray(model.geom_pos[g], dtype=np.float64))
            Rm = _quat_wxyz_to_rot(model.geom_quat[g])
        verts.append(_box_vertices(c, Rm, half))

    if not verts:
        return None
    P = np.vstack(verts)

    # 对顶点拟合 MVEE（复用上游 mvee_cvxpy + 特征分解，同 fit_ellipse 逻辑）
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(P)
        hull_pts = P[hull.vertices]
    except Exception:
        hull_pts = P
    from utils import mvee_cvxpy
    center, A = mvee_cvxpy(hull_pts)
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.clip(eigvals, 1e-15, None)
    axes = 1.0 / np.sqrt(eigvals)
    sort_idx = np.argsort(axes)[::-1]
    axes = axes[sort_idx]
    R = eigvecs[:, sort_idx]
    # 保守放大：MVEE 的 SCS 数值解有 ~1e-4 量级的越界，放大 0.1% 保证严格包围
    half_axes = np.maximum(axes, 1e-3) * 1.001
    return np.asarray(center, dtype=np.float64), R, half_axes


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

    def __init__(self, env, verbose=False, filter_mode="euclidean",
                 beta_max=4.0, eta=0.8, intent_len=5,
                 extra_obstacle_names=None, auto_support=False,
                 use_circulation=False, circ_gain=50.0, circ_activate=0.02,
                 circ_sign=1.0, eef_q_override=None, grasp_aware=False,
                 grasped_object_body=None, grasped_q=None,
                 keep_small_when_grasped=False, perception_mode="oracle",
                 stg_mode="B0", stg_lam_max=1.0, stg_h_enter=0.01,
                 stg_h_exit=0.02, stg_alpha=0.5, stg_gamma=0.01):
        self.env = env
        # perception_mode: "oracle"=MuJoCo 真值 AABB；"perception"=VLM+GroundingDINO+MVEE
        # （与上游论文一致）。两者是不同的方法变体，须分开报告。
        assert perception_mode in ("oracle", "perception"), \
            f"未知 perception_mode: {perception_mode}"
        self.perception_mode = perception_mode
        # suite_name 用于 perception 模式的 filtering_points（不同套件的 z 过滤范围不同）
        self.suite_name = "safelibero_long"
        self.verbose = verbose
        self.obstacle_name = None
        self.p2 = None
        self.R2 = None
        self.Q2_diag = None
        self.z_fixed = None
        self.Q1_diag = EEF_Q_DEFAULT.copy()
        # 传递链扩展：额外纳入保护的障碍物（如支撑 hazard 的底座）。
        # 每个是 dict{name, p2, R2, Q2, z}，filter 里各加一条 CBF 约束。
        self.extra_obstacle_names = list(extra_obstacle_names or [])
        self.auto_support = bool(auto_support)
        self.extra_obstacles = []
        self.feasible_count = 0
        self.infeasible_count = 0
        self.intervene_count = 0
        # CCA-ASF 配置（filter_mode ∈ {euclidean, cca_asf}）
        assert filter_mode in ("euclidean", "cca_asf"), f"未知 filter_mode: {filter_mode}"
        self.filter_mode = filter_mode
        self.beta_max = float(beta_max)
        self.eta = float(eta)
        self.intent_len = int(intent_len)
        # 供外部每步注入的 6D 任务方向（世界系）；None = 未注入 → 退化为欧氏
        self._task_direction_z = None
        # E0 审计：是否在 info 里记录 QP 快照（u_nom/A_row/r/d_y/u_sol），默认关
        self.record_qp_snapshot = False
        # ---- Circulation CBF（CBF with Circulation Inequalities，T-CST 2024）----
        # 缓解 CBF-QP 的稳定虚假平衡点（spurious equilibrium）：当末端被障碍物
        # 径向推离卡死时，强制一个切向（环绕）速度分量，让末端绕开障碍物。
        # 严格形式（论文式 (3)）：T(q)^T μ ≥ β(D(q))，T = Ω N，N = ∇D/‖∇D‖。
        #   - N = ∇h = a_v（末端局部系），D ≡ h（椭圆安全距离）；
        #   - Ω skew-symmetric orthonormal（末端平移 3D 奇数维 → §IV-C workaround，
        #     在 xy 水平面 2D orthonormal，z 维奇异）；
        #   - β(h) = circ_gain * (circ_activate - h)，递减、β(0)>0、h 大时平凡化；
        #   - circ_sign 控制绕向（+1 逆时针 / -1 顺时针）。
        # 实现见 src/safety/circulation.py（含量纲与坐标变换的严格说明）。
        self.use_circulation = bool(use_circulation)
        self.circ_gain = float(circ_gain)
        self.circ_activate = float(circ_activate)
        self.circ_sign = 1.0 if float(circ_sign) >= 0 else -1.0
        # ---- STG（Spatiotemporal Task-Guided Governor）软引导 ----
        # 研究原型：在原生 AEGIS QP 目标中加软引导项，不改变硬约束 U_AEGIS。
        # 2×2 因子：B0 无引导 / B1 切向 / B2 参考迟滞 / B3 联合。
        # 软项：min J_0(u) + λ‖P_t u - g_t‖²_Q，P_t=0.2·R1·[I3,0,0]（G0 已验证）。
        # λ=0 时精确回归原生 AEGIS（B0）。详见 src/safety/stg.py。
        self.stg_mode = stg_mode
        self.stg_lam_max = float(stg_lam_max)
        self.stg_h_enter = float(stg_h_enter)
        self.stg_h_exit = float(stg_h_exit)
        self.stg_alpha = float(stg_alpha)
        self.stg_gamma = float(stg_gamma)
        from src.safety.stg import STGGuide
        self.stg_guide = STGGuide(
            mode=stg_mode, lam_max=stg_lam_max, h_enter=stg_h_enter,
            h_exit=stg_h_exit, alpha=stg_alpha, gamma=stg_gamma)
        # 末端椭球尺寸覆盖（轴敏感性扫描 / grasp-aware 豁免用）。None = 用任务默认。
        self.eef_q_override = None if eef_q_override is None else np.asarray(
            eef_q_override, dtype=np.float64).copy()
        # grasp-aware 椭球：接近（未抓，夹爪张开）用小椭球，抓住后（夹爪闭合）用
        # 大椭球（包住被抓物体）。gripper_qpos 绝对值和 < grasp_close_thresh 判闭合。
        self.grasp_aware = bool(grasp_aware)
        self.grasp_q_open = None   # 未抓时的椭球（接近阶段，仅夹爪本体）
        self.grasp_q_grasped = None  # 已抓时的椭球（夹爪 + 被抓物体）
        self.grasp_close_thresh = 0.035
        self._grasped = False
        # 被抓物体分离约束：抓取阶段加「被抓物体椭球 vs 障碍物椭球」的 CBF，
        # 而非扩大末端椭球。grasped_object_body = 被抓物体的 joint 名（如 alphabet_soup_1）。
        self.grasped_object_body = grasped_object_body
        self.grasped_q = None if grasped_q is None else np.asarray(grasped_q, dtype=np.float64)
        # keep_small_when_grasped：抓取后不切大椭球（保持小椭球），只靠 sep 约束保护
        # 被抓物体。避免大椭球「末端 vs mug」约束支配「soup vs mug」分离约束。
        self.keep_small_when_grasped = bool(keep_small_when_grasped)

    def reset(self, obs, task_language=""):
        """episode 开始：定位障碍物、算初始椭球与 z 方向。"""
        # 修复（2026-11-04）：计数器在 __init__ 初始化，但 layer 跨 episode 复用，
        # 必须在 reset 时清零，否则 stats() 返回的是跨 episode 累计值。
        self.feasible_count = 0
        self.infeasible_count = 0
        self.intervene_count = 0
        # STG：每个 episode 重置引导器状态（防跨 episode 泄漏）
        self.stg_guide.reset()
        # CCA-ASF：每次 reset 清除注入的任务方向
        self._task_direction_z = None
        # 末端椭球半轴按任务类型选（上游同款规则）
        lang = task_language.lower()
        if _USE_SMALL_EEF:
            self.Q1_diag = EEF_Q_DEFAULT.copy()
        elif any(k in lang for k in ("orange juice", "milk", "alphabet soup")):
            self.Q1_diag = EEF_Q_LONG_OBJECT.copy()
        else:
            self.Q1_diag = EEF_Q_DEFAULT.copy()

        if self.eef_q_override is not None:
            self.Q1_diag = self.eef_q_override.copy()

        # grasp-aware：接近阶段（未抓）用夹爪本体小椭球，抓住后用大椭球。
        # 默认：open=[0.06,0.04,0.20]（夹爪本体），grasped=任务默认（包住被抓物体）。
        if self.grasp_aware:
            self.grasp_q_grasped = self.Q1_diag.copy()
            self.grasp_q_open = np.array([0.06, 0.04, self.Q1_diag[2]])
            self._grasped = False
            self.Q1_diag = self.grasp_q_open.copy()

        name = find_active_obstacle(self.env, obs)
        if name is None:
            self.obstacle_name = None
            if self.verbose:
                print("[aegis] 未找到工作空间内障碍物 → 安全层关闭")
            return False

        # 障碍物椭球来源：oracle（真值 AABB）vs perception（VLM+GroundingDINO+MVEE）
        if self.perception_mode == "perception":
            from src.safety.aegis_perception import fit_obstacle_from_perception
            ell, vlm_name = fit_obstacle_from_perception(
                self.env, obs, task_language, self.suite_name)
            if self.verbose:
                print(f"[aegis] perception 识别: {vlm_name}")
        else:
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

        # 额外障碍物（传递链保护）：初始化椭球与分离方向。
        # auto_support 模式：自动检测 hazard 的支撑物并加入保护（P1 端到端）。
        extra_names = list(self.extra_obstacle_names)
        if self.auto_support and name is not None:
            try:
                from src.safety.support_detection import detect_supports, supports_as_joint_names
                hazard_body = name + "_main"
                supports, _ = detect_supports(self.env, hazard_body)
                auto = supports_as_joint_names(supports)
                if self.verbose and auto:
                    print(f"[aegis] 自动检测支撑物: {auto}")
                extra_names += auto
            except Exception:
                pass

        self.extra_obstacles = []
        seen_extra = set()
        for ename in extra_names:
            if ename in seen_extra:
                continue
            seen_extra.add(ename)
            eell = obstacle_ellipsoid(self.env, obs, ename)
            if eell is None:
                continue
            ep2, eR2, eQ2 = eell
            ez = ep2 - p1
            enz = np.linalg.norm(ez)
            ez = ez / enz if enz > 1e-8 else np.array([1.0, 0, 0])
            self.extra_obstacles.append({
                "name": ename, "p2": ep2, "R2": eR2, "Q2": eQ2, "z": ez,
            })

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

    def set_task_direction(self, d_z):
        """注入 6D 任务方向（世界系，单位向量或 None）。

        d_z = [d_p(世界系平移, 3), d_omega(axis-angle姿态, 3)]。
        None = 清除注入，下一 filter 退化为欧氏。
        """
        if d_z is None:
            self._task_direction_z = None
            return
        d_z = np.asarray(d_z, dtype=np.float64).reshape(-1)
        if d_z.shape[0] != 6:
            raise ValueError(f"d_z 必须 6 维，得到 {d_z.shape[0]}")
        n = float(np.linalg.norm(d_z))
        if n <= 1e-12:
            self._task_direction_z = None
            return
        self._task_direction_z = (d_z / n).copy()

    def filter(self, action, obs):
        """把策略动作投影到安全集内。返回 (safe_action, info)。"""
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        info = {"active": False, "h": None, "feasible": None}

        if self.obstacle_name is None:
            return action, info

        p1 = self._eef_center(obs)
        R1 = _quat_wxyz_to_rot(np.asarray(obs["robot0_eef_quat"], dtype=np.float64))

        # grasp-aware：按夹爪状态切换末端椭球。夹爪闭合（|qpos| 均值 < 阈值）= 已抓。
        if self.grasp_aware:
            gq = np.asarray(obs.get("robot0_gripper_qpos"), dtype=np.float64)
            grasped = bool(np.abs(gq).mean() < self.grasp_close_thresh)
            if grasped != self._grasped:
                self._grasped = grasped
                if not self.keep_small_when_grasped:
                    self.Q1_diag = (self.grasp_q_grasped if grasped
                                    else self.grasp_q_open).copy()
                # keep_small_when_grasped=True：始终小椭球，抓取保护交给 sep 约束
            info["grasped"] = bool(grasped)

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

        # 额外障碍物（传递链）系数：各算一组，各加一条约束
        extra_coeffs = []
        for eo in self.extra_obstacles:
            ea_v, ea_omega, ea_uz, eh, _ = compute_h_coeffs_3d(
                p1, self.Q1_diag, R1, eo["p2"], eo["Q2"], eo["R2"], eo["z"],
            )
            extra_coeffs.append((ea_v, ea_omega, ea_uz, eh))

        u = cp.Variable(9)
        W = np.diag([1.0 / 25] * 6 + [1.0, 1.0, 1.0])
        u_ref_vec = np.hstack([u_v_ref, u_omega_ref, u_z_nom])
        # u_ref（QP 目标中心）与 u_nom（门控名义输入）当前数值相同，
        # 规格 §3.1 要求分别命名以防后续误用。
        u_nom = u_ref_vec

        # ---- CCA-ASF：白化各向异性低秩更新（仅替换目标二次项） ----
        W_used = W
        beta = 0.0
        d_y = None
        if self.filter_mode == "cca_asf" and self._task_direction_z is not None:
            d_y = map_dz_to_dy(self._task_direction_z, R1)
            if d_y is not None:
                # 冲突门控（白化空间）
                A_row = np.hstack([a_u_v, a_u_omega, a_uz])[None, :]  # (1,9)
                r_val = np.array([-QP_H_GAIN * h])
                beta = compute_conflict_beta(A_row, r_val, u_nom, d_y,
                                             beta_max=self.beta_max)
                if beta > 0.0:
                    Bd = B_DIAG * d_y
                    DeltaP = beta * np.outer(Bd, Bd)
                    W_used = W + DeltaP

        info["beta"] = float(beta)
        info["task_direction"] = (
            None if self._task_direction_z is None
            else np.asarray(self._task_direction_z, dtype=np.float64).copy()
        )
        if self.record_qp_snapshot:
            info["qp_snapshot"] = {
                "u_nom": np.asarray(u_ref_vec, dtype=np.float64).copy(),
                "A_row": np.hstack([a_u_v, a_u_omega, a_uz]).copy(),
                "r": float(-QP_H_GAIN * h),
                "d_y": (None if d_y is None else d_y.copy()),
                "beta": float(beta),
            }

        # ---- STG 软引导（B1/B2/B3）：min J_0 + λ‖P_t u - g_t‖²，不改变硬约束 ----
        # P_t = 0.2·R1·[I3,0,0]（G0 已验证），g_t 为归一化平移增量（与 z[:3] 同量纲）。
        # λ=0（无风险/退化）时精确回归原生 AEGIS（B0）。详见 src/safety/stg.py。
        stg_g = None
        stg_lam = 0.0
        if self.stg_mode in ("B1", "B2", "B3"):
            p_ee = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
            d_world = action[:3].copy()  # 意图方向（世界系归一化平移增量）
            stg_g, stg_lam, stg_info = self.stg_guide.step(p_ee, a_v, R1, d_world, h)
            info["stg_in_risk"] = bool(stg_info.get("in_risk", False))
            info["stg_tangent_degenerate"] = bool(stg_info.get("tangent_degenerate", False))
        info["stg_lambda"] = float(stg_lam)

        objective = cp.Minimize(cp.quad_form(u - u_ref_vec, W_used))
        if stg_lam > 0.0 and stg_g is not None:
            P_t = np.zeros((3, 9))
            P_t[:, :3] = QP_ACTION_SCALE * R1  # 0.2 * R1
            Q_w = np.eye(3)
            # 软项 λ‖P_t u - g_t‖² 展开为 9D 二次型 + 线性项（避免 3D 中间表达式，
            # 降低 CVXPY 规范化开销，G1 微基准显示 p95 比值 1.22→1.14）
            P_QP = stg_lam * (P_t.T @ Q_w @ P_t)       # 9×9 PSD
            lin = -2.0 * stg_lam * (stg_g @ Q_w @ P_t)  # 9D
            objective = cp.Minimize(
                cp.quad_form(u - u_ref_vec, W_used)
                + cp.quad_form(u, P_QP)
                + lin @ u
            )

        constraints = [a_u_v @ u[:3] + a_u_omega @ u[3:6] + a_uz @ u[6:] + QP_H_GAIN * h >= 0]
        for (ea_v, ea_omega, ea_uz, eh) in extra_coeffs:
            constraints.append(
                QP_ACTION_SCALE * ea_v @ u[:3]
                + QP_ACTION_SCALE * ea_omega @ u[3:6]
                + ea_uz @ u[6:]
                + QP_H_GAIN * eh >= 0
            )
        # ---- Circulation 约束（严格对应论文 CCBF-QP 式 (3)：T(q)^T μ ≥ β(D(q))）----
        # T = Ω N，N = ∇D/‖∇D‖，D ≡ h（椭圆安全距离），Ω skew-symmetric orthonormal。
        # 法向用 ∇h = a_v（局部系），而非「末端→障碍物连线方向 z_fixed」——
        # 各向异性椭球下二者方向不重合（实测夹角约 28°），用 z_fixed 是启发式近似。
        # 末端平移 3D 是奇数维 → 论文 §IV-C workaround：Ω 在 xy 水平面 orthonormal，
        # z 维奇异。详见 src/safety/circulation.py。
        circ_active = False
        if self.use_circulation:
            from src.safety.circulation import circulation_constraint
            c = circulation_constraint(R1, a_v, h, self.circ_sign,
                                       self.circ_gain, self.circ_activate,
                                       V_REF_GAIN)
            if c is not None:
                T_local, rhs = c
                constraints.append(T_local @ u[:3] >= rhs)
                circ_active = True
        info["circ_active"] = circ_active
        # ---- 被抓物体分离约束（grasped object vs obstacle）----
        # 抓取阶段：保护「被抓 soup 不撞 mug」，正确约束是 soup 椭球 vs mug 椭球的
        # 分离，而非扩大末端椭球。soup 被抓时速度 ≈ 末端速度（刚性跟随），故约束
        # 施加在 u[:3] 上。soup 作为「被保护椭球 i」，mug 作为「障碍物 j」。
        grasped_sep_active = False
        if self.grasped_object_body and self._grasped and self.obstacle_name:
            try:
                gobj_name = self.grasped_object_body.replace("_main", "")
                g_ell = obstacle_ellipsoid(self.env, obs, gobj_name)
                if g_ell is not None:
                    g_p, g_R, g_Q = g_ell
                    g_z = self.p2 - g_p  # soup → mug 方向（保护 soup 远离 mug）
                    g_nz = np.linalg.norm(g_z)
                    if g_nz > 1e-6:
                        g_z = g_z / g_nz
                        # soup 椭球朝向用单位阵（soup 近似不旋转）
                        g_R_eye = np.eye(3)
                        ga_v, ga_omega, ga_uz, gh, _ = compute_h_coeffs_3d(
                            g_p, (g_Q if self.grasped_q is None else self.grasped_q),
                            g_R_eye, self.p2, self.Q2_diag, self.R2, g_z,
                        )
                        # soup 速度 = 末端速度：约束 a_v · u_v + a_omega · u_omega
                        # 但 soup 平移由末端平移驱动，姿态近似不随末端姿态变。
                        # 简化：只约束平移分量（soup 分离主要由平移决定）。
                        constraints.append(
                            QP_ACTION_SCALE * ga_v @ u[:3] + QP_H_GAIN * gh >= 0
                        )
                        grasped_sep_active = True
            except Exception:
                grasped_sep_active = False
        info["grasped_sep_active"] = bool(grasped_sep_active)
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.OSQP)
        except Exception:
            prob = None

        info["active"] = True
        info["h"] = float(h)

        # 记录每条 CBF 约束的对偶变量（>0 表示该约束在最优解处 binding = 激活）。
        # 用于「约束激活」的精确实证（命题 7.2）。
        if prob is not None:
            try:
                duals = [float(c.dual_value) for c in constraints]
                info["constraint_duals"] = {
                    "hazard": duals[0],
                    "extra": duals[1:],
                }
            except Exception:
                pass

        if prob is not None and u.value is not None:
            self.feasible_count += 1
            u_v = u.value[:3]
            u_omega = u.value[3:6]
            u_z = u.value[6:]
            info["feasible"] = True
            if self.record_qp_snapshot:
                info["qp_snapshot"]["u_sol"] = np.asarray(u.value, dtype=np.float64).copy()
            # 判断是否真的介入了（与参考解不同）。
            # 修复（2026-11-04）：不只比 u_v（平移），还比 u_omega（旋转），
            # 否则「只改旋转不改平移」的过滤会被漏计。
            if (np.linalg.norm(u_v - u_v_ref) > 1e-3
                    or np.linalg.norm(u_omega - u_omega_ref) > 1e-3):
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
        # 障碍物位姿：严格复现上游——fit_ellipse 一次后 episode 内固定，不刷新。
        # （之前每步刷新 obs 位姿是「跟踪移动障碍物」的保守化，偏离上游，已移除。）
        # 额外障碍物（传递链保护，支撑感知扩展）仍每步刷新——它不是上游 AEGIS 的一部分，
        # 是本项目的扩展，保留其位姿跟踪以正确保护静态支撑物。
        for eo in self.extra_obstacles:
            key = eo["name"] + "_pos"
            if key in obs:
                eo["p2"] = np.asarray(obs[key], dtype=np.float64)
            ez = eo["p2"] - p1
            enz = np.linalg.norm(ez)
            eo["z"] = ez / enz if enz > 1e-8 else np.array([1.0, 0, 0])

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

    # ------------------------------------------------------------------
    # 状态保存 / 恢复：AEGIS 是有状态控制器（z_fixed 演化、p2 更新、计数器）。
    # 只恢复 MuJoCo 状态无法复现一个候选分支对 filter 的推进，必须连同这些字段
    # 一起保存/恢复，候选分支才能从相同 AEGIS 状态开始。
    # ------------------------------------------------------------------
    def save_state(self):
        """保存 filter 输出的全部有状态字段（含缓存与计数器）。"""
        return {
            "obstacle_name": self.obstacle_name,
            "p2": None if self.p2 is None else np.asarray(self.p2).copy(),
            "R2": None if self.R2 is None else np.asarray(self.R2).copy(),
            "Q2_diag": None if self.Q2_diag is None else np.asarray(self.Q2_diag).copy(),
            "z_fixed": None if self.z_fixed is None else np.asarray(self.z_fixed).copy(),
            "Q1_diag": np.asarray(self.Q1_diag).copy(),
            "feasible_count": self.feasible_count,
            "infeasible_count": self.infeasible_count,
            "intervene_count": self.intervene_count,
            "_last_p1": None if getattr(self, "_last_p1", None) is None
                        else np.asarray(self._last_p1).copy(),
            "extra_obstacles": [
                {
                    "name": eo["name"],
                    "p2": np.asarray(eo["p2"]).copy(),
                    "R2": np.asarray(eo["R2"]).copy(),
                    "Q2": np.asarray(eo["Q2"]).copy(),
                    "z": np.asarray(eo["z"]).copy(),
                }
                for eo in self.extra_obstacles
            ],
        }

    def restore_state(self, st):
        """恢复 filter 状态（深拷贝，避免共享引用）。"""
        self.obstacle_name = st["obstacle_name"]
        self.p2 = None if st["p2"] is None else np.asarray(st["p2"]).copy()
        self.R2 = None if st["R2"] is None else np.asarray(st["R2"]).copy()
        self.Q2_diag = None if st["Q2_diag"] is None else np.asarray(st["Q2_diag"]).copy()
        self.z_fixed = None if st["z_fixed"] is None else np.asarray(st["z_fixed"]).copy()
        self.Q1_diag = np.asarray(st["Q1_diag"]).copy()
        self.feasible_count = st["feasible_count"]
        self.infeasible_count = st["infeasible_count"]
        self.intervene_count = st["intervene_count"]
        self._last_p1 = None if st.get("_last_p1") is None else np.asarray(st["_last_p1"]).copy()
        self.extra_obstacles = [
            {
                "name": eo["name"],
                "p2": np.asarray(eo["p2"]).copy(),
                "R2": np.asarray(eo["R2"]).copy(),
                "Q2": np.asarray(eo["Q2"]).copy(),
                "z": np.asarray(eo["z"]).copy(),
            }
            for eo in st.get("extra_obstacles", [])
        ]
