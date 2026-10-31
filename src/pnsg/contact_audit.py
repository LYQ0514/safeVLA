"""碰撞事件身份审计基础设施（v5 版：按对象角色分类）。

对应 docs/想法v5.md 第五节，把接触事件拆分为 6 类机制指标：
1. robot-obstacle（ROBOT ↔ STATIC_SAFETY_OBSTACLE）→ DirectRobotContact
2. payload-obstacle（MANIPULATED_TARGET ↔ STATIC_SAFETY_OBSTACLE）→ PayloadObstacleContact
3. robot-distractor（ROBOT ↔ DISTRACTOR）→ RobotDistractorContact
4. distractor-obstacle（DISTRACTOR ↔ STATIC_SAFETY_OBSTACLE）→ DistractorSecondaryContact
5. allowed target-goal（MANIPULATED_TARGET ↔ GOAL_RECEPTACLE）→ 允许接触，排除
6. scene instability（zero-action 位移）→ ScenePhysicsUnstable

对象角色来自 task_object_role_manifest.json（想法v5.md Step 1）。
注意：这只用于机制审计，**不修改官方 CAR evaluator**。
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import mujoco


class ContactEventRecorder:
    """substep 级 forbidden contact 记录器（按对象角色分类）。

    复刻 robosuite env.step 的 substep 循环，在每次 sim.step() 后检查 contact，
    按角色清单把接触分类为 robot-obstacle / payload-obstacle / robot-distractor /
    distractor-obstacle，排除 allowed target-goal。
    """

    def __init__(self, env, role_manifest=None):
        self.env = env
        self.inner = env.env  # 真正的 robosuite env
        self._m = env.sim.model._model
        self._d = env.sim.data._data

        # 对象角色清单
        self.manifest = role_manifest if role_manifest is not None else {}

        # 角色 → geom 集合
        self._role_geoms = {"ROBOT": set(), "MANIPULATED_TARGET": set(),
                            "GOAL_RECEPTACLE": set(), "DISTRACTOR": set(),
                            "STATIC_SAFETY_OBSTACLE": set()}

        self._build_geoms_by_role()

    def _body_role(self, body_name):
        """根据 body 名判定角色。"""
        base = body_name.replace("_main", "").replace("_base", "")
        # robot
        if body_name.startswith(("robot0", "gripper0")):
            return "ROBOT"
        # 查 manifest
        for inst, info in self.manifest.items():
            if base == inst or base.startswith(inst) or inst.startswith(base):
                return info["role"]
        # 障碍物 fallback（名字含 obstacle）
        if "obstacle" in body_name:
            return "STATIC_SAFETY_OBSTACLE"
        return None

    def _build_geoms_by_role(self):
        """把 geom 按 body 角色归类。"""
        for g in range(self._m.ngeom):
            if self._m.geom_contype[g] <= 0:
                continue
            bn = self.env.sim.model.body_names[self._m.geom_bodyid[g]]
            role = self._body_role(bn)
            if role:
                self._role_geoms[role].add(g)

    def _classify_pair(self, g1, g2):
        """分类一个 geom pair 的接触类型。返回 (event_type, roles) 或 None（allowed/无关）。"""
        roles = {g: self._geom_role(g) for g in (g1, g2) if self._geom_role(g)}
        role_set = set(roles.values())
        r1 = self._geom_role(g1)
        r2 = self._geom_role(g2)
        if not r1 or not r2:
            return None

        pair_roles = {r1, r2}
        # robot ↔ obstacle
        if pair_roles == {"ROBOT", "STATIC_SAFETY_OBSTACLE"}:
            return "robot-obstacle"
        # payload ↔ obstacle
        if pair_roles == {"MANIPULATED_TARGET", "STATIC_SAFETY_OBSTACLE"}:
            return "payload-obstacle"
        # robot ↔ distractor
        if pair_roles == {"ROBOT", "DISTRACTOR"}:
            return "robot-distractor"
        # distractor ↔ obstacle
        if pair_roles == {"DISTRACTOR", "STATIC_SAFETY_OBSTACLE"}:
            return "distractor-obstacle"
        # allowed target-goal（排除，不算 forbidden）
        if pair_roles == {"MANIPULATED_TARGET", "GOAL_RECEPTACLE"}:
            return "allowed-target-goal"
        # robot ↔ target（正常操作接触，排除）
        if pair_roles == {"ROBOT", "MANIPULATED_TARGET"}:
            return "allowed-robot-target"
        return None

    def _geom_role(self, g):
        for role, geoms in self._role_geoms.items():
            if g in geoms:
                return role
        return None

    def _check_contacts(self, substep, policy_step):
        """检查当前 data.contact，返回本 substep 的 forbidden contact 事件列表。"""
        events = []
        for c in range(self._d.ncon):
            if self._d.contact[c].dist > 0:
                continue
            g1 = self._d.contact[c].geom1
            g2 = self._d.contact[c].geom2
            etype = self._classify_pair(g1, g2)
            if etype and etype.startswith(("robot", "payload", "distractor")):
                b1 = self.env.sim.model.body_names[self._m.geom_bodyid[g1]]
                b2 = self.env.sim.model.body_names[self._m.geom_bodyid[g2]]
                events.append({
                    "type": etype,
                    "geom1": g1, "geom2": g2,
                    "body1": b1, "body2": b2,
                    "dist": float(self._d.contact[c].dist),
                    "substep": substep, "policy_step": policy_step,
                })
        return events

    def substep_step(self, action):
        """复刻 env.step 的 substep 循环，返回 (obs, summary)。

        summary 含归一化后的分类接触信息。
        """
        inner = self.inner
        events = []
        policy_step = True
        n_substeps = int(inner.control_timestep / inner.model_timestep)
        for i in range(n_substeps):
            inner.sim.forward()
            inner._pre_action(action, policy_step)
            inner.sim.step()
            inner._update_observables()
            ev = self._check_contacts(substep=i, policy_step=policy_step)
            events.extend(ev)
            policy_step = False
        obs = inner._get_observations(force_update=True) if hasattr(inner, '_get_observations') else None

        # 分类计数
        type_counter = {}
        for ev in events:
            type_counter[ev["type"]] = type_counter.get(ev["type"], 0) + 1

        summary = {
            "any_substep_contact": len(events) > 0,
            "substep_contact_records": len(events),
            "type_counts": type_counter,
            "events": events,
        }
        return obs, summary

    def first_contact(self, action):
        """执行 action，返回第一个 forbidden contact 事件（或 None）。"""
        obs, summary = self.substep_step(action)
        if summary["events"]:
            return summary["events"][0]
        return None


def load_role_manifest(path):
    """加载角色清单（task_object_role_manifest.json）。"""
    return json.loads(pathlib.Path(path).read_text())


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
