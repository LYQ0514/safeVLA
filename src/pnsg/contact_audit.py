"""碰撞事件身份审计基础设施（对应 docs/想法v3.md）。

三层指标：
1. 主碰撞事件：substep 级 forbidden robot/target-obstacle contact（contact.dist ≤ 0）
2. 场景不稳定事件：zero-action 自发位移、初始穿透
3. 严重度指标：同轨迹 L1/L2 位移

关键：substep recorder —— 不能在 env.step() 返回后看 data.contact（会漏掉
25 个 physics substep 中间的瞬时接触），必须复刻 step 循环在每个 substep 检查。
"""

from __future__ import annotations

import numpy as np
import mujoco


class ContactEventRecorder:
    """substep 级 forbidden contact 记录器。

    复刻 robosuite env.step 的 substep 循环，在每次 sim.step() 后检查 contact，
    记录 robot-obstacle 与 target-obstacle 的 forbidden contact。
    """

    def __init__(self, env):
        self.env = env
        self.inner = env.env  # 真正的 robosuite env
        self._m = env.sim.model._model
        self._d = env.sim.data._data

        # robot 碰撞几何（contype>0）
        self.robot_geoms = {
            g for g in range(self._m.ngeom)
            if self._m.geom_contype[g] > 0
            and (env.sim.model.body_names[self._m.geom_bodyid[g]].startswith("robot0")
                 or env.sim.model.body_names[self._m.geom_bodyid[g]].startswith("gripper0"))
        }
        # obstacle 碰撞几何（contype>0）
        obstacle_names = set(
            n.replace("_joint0", "") for n in env.sim.model.joint_names if "obstacle" in n
        )
        self.obstacle_geoms = {
            g for g in range(self._m.ngeom)
            if self._m.geom_contype[g] > 0
            and any(env.sim.model.body_names[self._m.geom_bodyid[g]].startswith(o)
                    for o in obstacle_names)
        }
        # target 几何（被操作物体：bowl/plate 等，非 obstacle 的 movable 物体）
        # 简化：所有 body 名含 bowl/plate/milk/cream/soup/tomato/mug 等的 movable 物体
        self.target_geoms = self._build_target_geoms()

    def _build_target_geoms(self):
        """目标物体几何：movable 且非 obstacle 的非机器人物体。"""
        tg = set()
        for g in range(self._m.ngeom):
            bn = self.env.sim.model.body_names[self._m.geom_bodyid[g]]
            if bn.startswith(("robot0", "gripper0", "table", "world", "mount0",
                              "wooden_cabinet", "flat_stove")):
                continue
            if "obstacle" in bn:
                continue
            if self._m.geom_contype[g] <= 0:
                continue
            tg.add(g)
        return tg

    def _check_contacts(self, substep, policy_step):
        """检查当前 data.contact，返回本 substep 发现的 forbidden contact 列表。"""
        events = []
        for c in range(self._d.ncon):
            if self._d.contact[c].dist > 0:  # 只记录实际穿透/接触
                continue
            g1 = self._d.contact[c].geom1
            g2 = self._d.contact[c].geom2
            pair = {g1, g2}
            typ = None
            if pair & self.robot_geoms and pair & self.obstacle_geoms:
                typ = "robot-obstacle"
            elif pair & self.target_geoms and pair & self.obstacle_geoms:
                typ = "target-obstacle"
            if typ:
                b1 = self.env.sim.model.body_names[self._m.geom_bodyid[g1]]
                b2 = self.env.sim.model.body_names[self._m.geom_bodyid[g2]]
                events.append({
                    "type": typ,
                    "geom1": g1, "geom2": g2,
                    "body1": b1, "body2": b2,
                    "dist": float(self._d.contact[c].dist),
                    "substep": substep, "policy_step": policy_step,
                })
        return events

    def substep_step(self, action):
        """复刻 env.step 的 substep 循环，返回 (obs, events)。

        每个 substep 后检查 contact，捕获瞬时接触。
        """
        inner = self.inner
        events = []
        policy_step = True
        n_substeps = int(inner.control_timestep / inner.model_timestep)
        obs = None
        for i in range(n_substeps):
            inner.sim.forward()
            inner._pre_action(action, policy_step)
            inner.sim.step()
            inner._update_observables()
            ev = self._check_contacts(substep=i, policy_step=policy_step)
            events.extend(ev)
            policy_step = False
        obs = inner._get_observations(force_update=True) if hasattr(inner, '_get_observations') else None
        return obs, events

    def first_contact(self, action):
        """执行 action，返回第一个 forbidden contact（或 None）。"""
        obs, events = self.substep_step(action)
        if events:
            return events[0]
        return None


class DisplacementRecorder:
    """同轨迹 L1/L2 位移记录器（一次 rollout 同时记录两种范数）。"""

    def __init__(self, env, body_names):
        self.env = env
        self._m = env.sim.model._model
        self._d = env.sim.data._data
        self.body_names = body_names
        self.initial = {b: self._body_pos(b).copy() for b in body_names}

    def _body_pos(self, name):
        bid = mujoco.mj_name2id(self._m, mujoco.mjtObj.mjOBJ_BODY, name + "_main")
        return self._d.xpos[bid]

    def record(self, step):
        """记录当前步各 body 的 L1/L2 位移，返回 list[dict]。"""
        rows = []
        for b in self.body_names:
            cur = self._body_pos(b)
            delta = cur - self.initial[b]
            l1 = float(np.linalg.norm(delta, ord=1))
            l2 = float(np.linalg.norm(delta, ord=2))
            # 数学不变量校验
            assert l2 <= l1 + 1e-9, f"L2={l2} > L1={l1}"
            assert l1 <= np.sqrt(3) * l2 + 1e-9, f"L1={l1} > sqrt3*L2={np.sqrt(3)*l2}"
            rows.append({
                "step": step, "body": b,
                "initial": self.initial[b].copy(),
                "current": cur.copy(),
                "delta_xyz": delta.copy(),
                "l1": l1, "l2": l2,
            })
        return rows


def zero_action_stability(env, body_names, n_steps=100):
    """零动作稳定性检查：返回各 body 的峰值 L2 位移。"""
    rec = DisplacementRecorder(env, body_names)
    for step in range(n_steps):
        env.step(np.zeros(7, dtype=np.float32))  # 假设 OSC 7 维
    peak = {}
    for b in body_names:
        bid = mujoco.mj_name2id(env.sim.model._model, mujoco.mjtObj.mjOBJ_BODY, b + "_main")
        cur = env.sim.data._data.xpos[bid]
        peak[b] = float(np.linalg.norm(cur - rec.initial[b], ord=2))
    return peak
