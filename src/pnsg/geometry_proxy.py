"""PNSG 几何代理：控制点球覆盖 + 障碍物球 + 线性约束。

对应 docs/设计.md 第 4、5 节。在 .venv（py3.8 + robosuite）环境运行。

控制点 p_k^0, B_k(3×6), r_k：
- p_k^0：当前状态控制点世界位置
- B_k：控制点位置对 6 维 pose action 的 Jacobian（解析：J_i J_ee^+ @ scale）
- r_k：控制点球半径

离散轨迹：p_{t,k} = p_k^0 + Σ_{l≤t} B_k a_l^pose
净空：g = |p_{t,k} - o_q| - r_k - R_q
"""

from __future__ import annotations

import numpy as np
import mujoco

# OSC_POSE action 尺度（实测）：action=1 → 平移 0.01039 m/步，旋转 0.09197 rad/步
ACTION_TRANS_SCALE = 0.01039
ACTION_ROT_SCALE = 0.09197

# 采样 body（8 link + 指尖）
_SAMPLE_BODIES = [f"robot0_link{n}" for n in range(8)] + [
    "gripper0_eef", "gripper0_leftfinger", "gripper0_rightfinger",
    "gripper0_finger_joint1_tip", "gripper0_finger_joint2_tip",
]


class ControlPoint:
    __slots__ = ("p0", "B", "r")

    def __init__(self, p0, B, r):
        self.p0 = np.asarray(p0, dtype=np.float64)   # (3,)
        self.B = np.asarray(B, dtype=np.float64)     # (3,6)
        self.r = float(r)


class ObstaclePrimitive:
    __slots__ = ("center", "radius")

    def __init__(self, center, radius):
        self.center = np.asarray(center, dtype=np.float64)  # (3,)
        self.radius = float(radius)


def _body_name2id(m, name):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)


def build_control_points(env, obs, control_radius=0.02):
    """构造控制点球覆盖（body 原点 + 轴向偏移 + 指尖），带 B_k 和 r_k。"""
    m = env.sim.model._model
    d = env.sim.data._data
    qidx = list(env.robots[0].controller.qvel_index)  # [0..6]

    # 末端全雅可比 (6,7) 与伪逆 (7,6)
    eef_bid = _body_name2id(m, "gripper0_eef")
    jacp = np.zeros((3, m.nv)); jacr = np.zeros((3, m.nv))
    mujoco.mj_jacBody(m, d, jacp, jacr, eef_bid)
    J_ee = np.vstack([jacp[:, qidx], jacr[:, qidx]])  # (6,7)
    J_ee_pinv = np.linalg.pinv(J_ee)  # (7,6)
    # action 尺度
    S = np.diag([ACTION_TRANS_SCALE] * 3 + [ACTION_ROT_SCALE] * 3)

    cps = []
    for bn in _SAMPLE_BODIES:
        try:
            bid = _body_name2id(m, bn)
        except Exception:
            continue
        R = d.xmat[bid].reshape(3, 3)
        p_body = d.xpos[bid]

        # 采样局部点：body 原点 + 6 轴向偏移 0.03
        local_pts = [np.zeros(3)]
        for axis in range(3):
            for s in (-0.03, 0.03):
                lp = np.zeros(3); lp[axis] = s
                local_pts.append(lp)

        for lp in local_pts:
            # 世界位置
            pw = p_body + R @ lp
            # 雅可比 (3,7)
            mujoco.mj_jac(m, d, jacp, jacr, lp, bid)
            J_i = jacp[:, qidx]  # (3,7)
            B = J_i @ J_ee_pinv @ S  # (3,6)
            cps.append(ControlPoint(pw, B, control_radius))
    return cps


def build_obstacles(env, obs):
    """构造障碍物球（工作空间内的 obstacle objects，用 geom 最大半轴当半径）。"""
    m = env.sim.model._model
    obstacle_names = set(
        n.replace("_joint0", "") for n in env.sim.model.joint_names if "obstacle" in n
    )
    obstacles = []
    for name in obstacle_names:
        key = name + "_pos"
        if key not in obs:
            continue
        p = np.asarray(obs[key], dtype=np.float64)
        # 只取工作空间内
        if not (p[2] > 0 and abs(p[0]) < 0.5 and abs(p[1]) < 0.5):
            continue
        # 半径 = 该 body 所有 geom 最大半轴
        try:
            bid = _body_name2id(m, name + "_main")
        except Exception:
            continue
        gids = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid]
        if not gids:
            continue
        r = max(float(np.max(m.geom_size[g])) for g in gids)
        obstacles.append(ObstaclePrimitive(p, r))
    return obstacles


def predict_control_points(cps, actions_pose):
    """预测控制点轨迹 p_{t,k}，actions_pose 形状 (T, 6)。"""
    T = actions_pose.shape[0]
    K = len(cps)
    traj = np.zeros((T, K, 3))
    # 累积位移
    cum = np.zeros((K, 3))
    for t in range(T):
        for k, cp in enumerate(cps):
            cum[k] += cp.B @ actions_pose[t]
            traj[t, k] = cp.p0 + cum[k]
    return traj


def compute_clearance(cps, obstacles, actions_pose):
    """计算每个 (t,k,q) 的净空，返回 (T, K, Q) 数组。"""
    traj = predict_control_points(cps, actions_pose)  # (T,K,3)
    T, K, _ = traj.shape
    Q = len(obstacles)
    clear = np.zeros((T, K, Q))
    for t in range(T):
        for k, cp in enumerate(cps):
            for q, ob in enumerate(obstacles):
                dist = np.linalg.norm(traj[t, k] - ob.center)
                clear[t, k, q] = dist - cp.r - ob.radius
    return clear


def selected_constraints(cps, obstacles, actions_pose, h_floor=0.005,
                         max_constraints=3, margin=0.01):
    """选净空最小的约束，返回列表 [(covector(60,), b)]。

    covector 第 l 步 pose 维度（l*6:(l+1)*6）为 B_k^T n，l≤t；l>t 为 0。
    b = h_floor - clearance。
    """
    H = actions_pose.shape[0]  # 10
    traj = predict_control_points(cps, actions_pose)  # (H,K,3)
    rows = []
    for t in range(H):
        for k, cp in enumerate(cps):
            for q, ob in enumerate(obstacles):
                dvec = traj[t, k] - ob.center
                dist = np.linalg.norm(dvec)
                g = dist - cp.r - ob.radius
                if g < h_floor + margin:
                    n = dvec / (dist + 1e-12)
                    # covector：B_k^T n，对 l<=t 的 pose 维度
                    cov = np.zeros(H * 6)
                    grad = cp.B.T @ n  # (6,)
                    for l in range(t + 1):
                        cov[l * 6:(l + 1) * 6] = grad
                    b = h_floor - g
                    rows.append((cov, b, g))
    # 按净空排序，取前 max_constraints
    rows.sort(key=lambda x: x[2])
    return [(np.asarray(c), float(b)) for c, b, _ in rows[:max_constraints]]
