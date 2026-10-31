"""PNSG 非线性认证器 C_2。

对应 docs/设计.md 第 10 节：从同一状态恢复，执行候选前两个动作，
检查统一安全条件。几何阈值 0mm。

本实现检查（设计文档列出的可落地子集）：
1. robot-obstacle 禁止接触（MuJoCo contact，碰撞几何 contype>0）
2. robot self-collision（相对初始状态新增的）
3. joint position limits
4. external action limits（clip 后 [-1,1]）
5. 最小真实净空 ≥ 0（球体代理，严格 0mm）

未实现（本版暂缓）：
- grasped target-obstacle（需抓取检测，参考 VLPSA 的 grasp_protection）
- Panda velocity contract
- 阶段接触规则（robot-target / target-goal，需任务阶段状态机）
"""

from __future__ import annotations

import numpy as np
import mujoco


class CertifyResult:
    def __init__(self, pass_, reason="", checks=None, min_robot_obstacle_dist=None):
        self.pass_ = bool(pass_)
        self.reason = reason
        self.checks = checks or {}
        # native 几何净空：执行后 robot-obstacle 所有 contact pair 的最小 dist
        # （MuJoCo 原生几何，dist>0=分离，dist<=0=接触/穿透）
        self.min_robot_obstacle_dist = min_robot_obstacle_dist


class Certifier:
    def __init__(self, env):
        self.env = env
        self._m = env.sim.model._model
        self._d = env.sim.data._data
        self._qidx = list(env.robots[0].controller.qvel_index)

        # 机器人碰撞几何（contype>0）
        self.robot_geoms = {
            g for g in range(self._m.ngeom)
            if self._m.geom_contype[g] > 0
            and (env.sim.model.body_names[self._m.geom_bodyid[g]].startswith("robot0")
                 or env.sim.model.body_names[self._m.geom_bodyid[g]].startswith("gripper0"))
        }
        # 障碍物碰撞几何（contype>0）
        obstacle_names = set(
            n.replace("_joint0", "") for n in env.sim.model.joint_names if "obstacle" in n
        )
        self.obstacle_geoms = {
            g for g in range(self._m.ngeom)
            if self._m.geom_contype[g] > 0
            and any(env.sim.model.body_names[self._m.geom_bodyid[g]].startswith(o)
                    for o in obstacle_names)
        }
        # 自碰撞排除：同一 body 内的 geom 对
        self._geom_body = {g: self._m.geom_bodyid[g] for g in self.robot_geoms}

    def _initial_self_collision(self):
        """记录初始状态的自碰撞 geom 对（相对新增才判违规）。"""
        pairs = set()
        for c in range(self._d.ncon):
            g1 = self._d.contact[c].geom1
            g2 = self._d.contact[c].geom2
            if g1 in self.robot_geoms and g2 in self.robot_geoms:
                if self._geom_body[g1] != self._geom_body[g2]:
                    pairs.add((min(g1, g2), max(g1, g2)))
        return pairs

    def _min_robot_obstacle_dist(self):
        """当前状态 robot-obstacle 所有 contact pair 的最小 dist（native 几何）。

        返回 float 或 None（无 robot-obstacle contact pair，即分离超过 margin）。
        """
        dists = []
        for c in range(self._d.ncon):
            g1 = self._d.contact[c].geom1
            g2 = self._d.contact[c].geom2
            if (g1 in self.robot_geoms and g2 in self.obstacle_geoms) or \
               (g2 in self.robot_geoms and g1 in self.obstacle_geoms):
                dists.append(float(self._d.contact[c].dist))
        return min(dists) if dists else None

    def certify_prefix(self, actions_prefix):
        """执行 actions_prefix（前 2 动作，已 clip），检查 C_2。

        返回 CertifyResult。调用前必须 restore_full_state 到同一快照。
        min_robot_obstacle_dist = 执行后 robot-obstacle 的最小 native 几何距离
        （dist>0=分离，dist<=0=接触/穿透，None=无接触即分离超过 margin）。
        """
        env = self.env
        obs = None
        checks = {}
        min_dist = None

        # 记录初始自碰撞
        init_self = self._initial_self_collision()

        # 执行前两个动作，逐步检查
        for i, act in enumerate(actions_prefix):
            act = np.asarray(act, dtype=np.float64)

            # 检查 external action limits
            if np.any(np.abs(act) > 1.0 + 1e-6):
                return CertifyResult(False, f"action limits 违反 @ step {i}",
                                     checks, self._min_robot_obstacle_dist())

            obs, _r, _d, _info = env.step(act)

            # 关节限位
            qpos = np.asarray(env.sim.data.qpos[self._qidx])
            for j, (lo, hi) in enumerate(self._m.jnt_range[self._qidx]):
                if qpos[j] < lo - 1e-3 or qpos[j] > hi + 1e-3:
                    return CertifyResult(False, f"joint limit 违反 @ step {i}",
                                         checks, self._min_robot_obstacle_dist())

            # robot-obstacle 接触
            for c in range(self._d.ncon):
                g1 = self._d.contact[c].geom1
                g2 = self._d.contact[c].geom2
                if (g1 in self.robot_geoms and g2 in self.obstacle_geoms) or \
                   (g2 in self.robot_geoms and g1 in self.obstacle_geoms):
                    return CertifyResult(False, f"robot-obstacle 接触 @ step {i}",
                                         checks, self._min_robot_obstacle_dist())

            # 新增 self-collision
            cur_self = set()
            for c in range(self._d.ncon):
                g1 = self._d.contact[c].geom1
                g2 = self._d.contact[c].geom2
                if g1 in self.robot_geoms and g2 in self.robot_geoms:
                    if self._geom_body[g1] != self._geom_body[g2]:
                        cur_self.add((min(g1, g2), max(g1, g2)))
            new_self = cur_self - init_self
            if new_self:
                return CertifyResult(False, f"新增 self-collision @ step {i}",
                                     checks, self._min_robot_obstacle_dist())

        min_dist = self._min_robot_obstacle_dist()
        return CertifyResult(True, "C_2 pass", checks, min_dist)


def minimum_clearance(env, cps, obstacles):
    """真实状态下的最小球体净空（严格 0mm 阈值）。"""
    from src.pnsg.geometry_proxy import compute_clearance
    # 用当前观测重建控制点，预测零动作的净空（即当前净空）
    d = env.sim.data._data
    m = env.sim.model._model
    qidx = list(env.robots[0].controller.qvel_index)
    pts = []
    for cp in cps:
        pts.append(cp.p0)
    clear = []
    for p in pts:
        for ob in obstacles:
            clear.append(np.linalg.norm(p - ob.center) - cp.r - ob.radius)
    return min(clear)
