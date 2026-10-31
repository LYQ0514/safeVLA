"""碰撞事件身份审计（v7 contract 版）。

对应 docs/想法v7.md 的 contact classification contract：
- 语义分类（semantic_class）与允许状态（allowance）分离
- 角色顺序无关（对称性）
- 完整字段（geom/body/instance/role/dist/preexisting/task_stage/allowance/hashes）
- 三级接触（native / geometric dist<=0 / proximity dist>0）
- 初始接触集合（preexisting）
- 连续事件合并（contiguous_contact_event，gap=0）
- 五级统计单位

只用于机制审计，不修改官方 CAR evaluator。
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import mujoco

from src.pnsg.contact_contract import (
    classify_role_pair,
    allowance_for,
    temporal_allowance,
    CONTRACT_VERSION,
)


# 环境 body 前缀（fixture，非 movable object）
_ENV_BODY_PREFIXES = ("world", "table", "mount0")


class ContactEventRecorder:
    """substep 级接触记录器（按 contract 分类，不静默丢弃 allowed contact）。"""

    def __init__(self, env, role_manifest=None, fixtures=None, contract_hash=""):
        self.env = env
        self.inner = env.env
        self._m = env.sim.model._model
        self._d = env.sim.data._data
        self.contract_hash = contract_hash

        self.manifest = role_manifest if role_manifest is not None else {}
        self.fixtures = fixtures if fixtures is not None else {}
        self._fixture_names = set(self.fixtures.keys())

        # geom → role（惰性构建）
        self._geom_role_cache = {}
        # 初始接触集合（执行前 snapshot）
        self.initial_contact_pairs = set()

    # ---- body 名 → role ----
    def _body_role(self, body_name):
        if body_name.startswith(("robot0", "gripper0")):
            return "ROBOT"
        if body_name.startswith(_ENV_BODY_PREFIXES):
            return "ENVIRONMENT"
        # fixture（最长前缀匹配）
        for fix in sorted(self._fixture_names, key=len, reverse=True):
            if body_name == fix or body_name.startswith(fix + "_"):
                return "ENVIRONMENT"
        # manifest objects
        for inst in sorted(self.manifest.keys(), key=len, reverse=True):
            if body_name == inst or body_name.startswith(inst + "_"):
                return self.manifest[inst]["roles"][0] if self.manifest[inst]["roles"] else None
        if "obstacle" in body_name:
            return "STATIC_SAFETY_OBSTACLE"
        return None

    def _body_instance(self, body_name):
        """body 名 → instance 名（用于记录 instance1/instance2）。"""
        if body_name.startswith(("robot0", "gripper0")):
            return body_name
        # 去掉 _main/_base 后缀
        base = body_name
        for suf in ("_main", "_base"):
            if base.endswith(suf):
                base = base[: -len(suf)]
                break
        return base

    def _geom_role(self, g):
        if g in self._geom_role_cache:
            return self._geom_role_cache[g]
        bn = self.env.sim.model.body_names[self._m.geom_bodyid[g]]
        role = self._body_role(bn)
        self._geom_role_cache[g] = role
        return role

    def record_initial_contacts(self, worsen_eps=0.003):
        """记录当前状态（执行前 snapshot）的 geometric contact 集合 + 初始深度。

        初始深度用于「worsened」判定：运行中接触深度比初始深超过 worsen_eps
        即标为 PREEXISTING_WORSENED（穿透恶化）。
        """
        self.initial_contact_pairs = set()
        self.initial_contact_depth = {}
        self.worsen_eps = worsen_eps
        for c in range(self._d.ncon):
            if self._d.contact[c].dist > 0:
                continue
            g1 = self._d.contact[c].geom1
            g2 = self._d.contact[c].geom2
            pair = (min(g1, g2), max(g1, g2))
            dist = float(self._d.contact[c].dist)
            self.initial_contact_pairs.add(pair)
            # 保留最深穿透（最负的 dist）
            if pair not in self.initial_contact_depth or dist < self.initial_contact_depth[pair]:
                self.initial_contact_depth[pair] = dist

    def _temporal_identity(self, pair, dist):
        """判定时间身份：PREEXISTING / NEW / PREEXISTING_WORSENED。"""
        if pair not in self.initial_contact_pairs:
            return "NEW"
        init_dist = self.initial_contact_depth.get(pair, 0.0)
        # 恶化：当前穿透比初始深超过 worsen_eps
        if dist < init_dist - self.worsen_eps:
            return "PREEXISTING_WORSENED"
        return "PREEXISTING"

    # ---- 分类 ----
    def _classify_contact(self, g1, g2, dist):
        """返回完整分类 dict（语义分类 + 时间身份 + 两阶段 allowance）。"""
        r1 = self._geom_role(g1)
        r2 = self._geom_role(g2)
        b1 = self.env.sim.model.body_names[self._m.geom_bodyid[g1]]
        b2 = self.env.sim.model.body_names[self._m.geom_bodyid[g2]]
        semantic = classify_role_pair(r1, r2) if (r1 and r2) else "OTHER_UNCLASSIFIED"
        pair = (min(g1, g2), max(g1, g2))
        temporal = self._temporal_identity(pair, dist)
        allowance = temporal_allowance(semantic, temporal)
        return {
            "geom1": g1, "geom2": g2,
            "body1": b1, "body2": b2,
            "instance1": self._body_instance(b1),
            "instance2": self._body_instance(b2),
            "role1": r1, "role2": r2,
            "semantic_class": semantic,
            "temporal_identity": temporal,
            "allowance": allowance,
            "contact_dist": float(dist),
            "contact_level": "geometric_contact" if dist <= 0 else "proximity_only",
            "preexisting": temporal != "NEW",
        }

    def _check_contacts(self, substep, policy_step, control_step):
        """检查当前 data.contact，返回本 substep 所有 relevant contact（不丢弃 allowed）。"""
        events = []
        for c in range(self._d.ncon):
            g1 = self._d.contact[c].geom1
            g2 = self._d.contact[c].geom2
            dist = float(self._d.contact[c].dist)
            r1 = self._geom_role(g1)
            r2 = self._geom_role(g2)
            # 只记录至少一边有角色（robot/object/obstacle/environment）
            if not r1 and not r2:
                continue
            cls = self._classify_contact(g1, g2, dist)
            cls["substep"] = substep
            cls["policy_step"] = policy_step
            cls["control_step"] = control_step
            events.append(cls)
        return events

    def substep_step(self, action, control_step=0):
        """复刻 env.step 的 substep 循环，返回 (obs, summary)。"""
        inner = self.inner
        events = []
        policy_step = True
        n_substeps = int(inner.control_timestep / inner.model_timestep)
        for i in range(n_substeps):
            inner.sim.forward()
            inner._pre_action(action, policy_step)
            inner.sim.step()
            inner._update_observables()
            ev = self._check_contacts(substep=i, policy_step=policy_step, control_step=control_step)
            events.extend(ev)
            policy_step = False
        obs = inner._get_observations(force_update=True) if hasattr(inner, '_get_observations') else None
        summary = {"events": events}
        return obs, summary

    def first_contact(self, action, control_step=0):
        obs, summary = self.substep_step(action, control_step=control_step)
        if summary["events"]:
            return summary["events"][0]
        return None


class ContactAggregator:
    """接触事件聚合器：五级统计 + 连续事件合并 + 初始接触标记。

    独立于 recorder，接收事件流，输出聚合结果。
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.raw_records = []  # 所有 native contact records
        self.geometric_records = []  # dist <= 0
        self.proximity_records = []  # dist > 0
        self.contiguous_events = []  # 连续事件（同 pair 连续 substep 合并，gap=0）

    def add_step(self, summary):
        """喂入一个 control step 的 summary。"""
        for ev in summary["events"]:
            self.raw_records.append(ev)
            if ev["contact_level"] == "geometric_contact":
                self.geometric_records.append(ev)
            else:
                self.proximity_records.append(ev)

    def compute_contiguous_events(self):
        """把 geometric contact 记录合并为连续事件（gap tolerance=0）。"""
        # 按 (geom pair) 分组，连续 substep 合并
        # 全局 timeline：按 (control_step, substep) 排序
        ordered = sorted(self.geometric_records, key=lambda e: (e["control_step"], e["substep"]))
        events = []
        active = {}  # (geom1, geom2) -> 当前连续事件
        for ev in ordered:
            key = (ev["geom1"], ev["geom2"])
            # 全局单调时间戳
            ts = ev["control_step"] * 1000000 + ev["substep"]
            if key in active:
                # 检查是否连续（gap=0，即上一个 substep 之后立刻出现）
                if ts == active[key]["end_ts"] + 1:
                    # 连续，延长
                    active[key]["end_ts"] = ts
                    active[key]["duration_substeps"] += 1
                    active[key]["minimum_dist"] = min(active[key]["minimum_dist"], ev["contact_dist"])
                    active[key]["maximum_penetration"] = max(active[key]["maximum_penetration"], -ev["contact_dist"])
                    continue
                else:
                    # 断开，归档
                    events.append(self._finalize_event(active.pop(key)))
            # 新事件
            active[key] = {
                "geom1": ev["geom1"], "geom2": ev["geom2"],
                "semantic_class": ev["semantic_class"],
                "allowance": ev["allowance"],
                "start_ts": ts, "end_ts": ts,
                "start_control_step": ev["control_step"], "start_substep": ev["substep"],
                "duration_substeps": 1,
                "minimum_dist": ev["contact_dist"],
                "maximum_penetration": max(0.0, -ev["contact_dist"]),
                "preexisting": ev["preexisting"],
            }
        for key in active:
            events.append(self._finalize_event(active[key]))
        self.contiguous_events = events
        return events

    @staticmethod
    def _finalize_event(e):
        return {
            "geom1": e["geom1"], "geom2": e["geom2"],
            "semantic_class": e["semantic_class"],
            "allowance": e["allowance"],
            "start_control_step": e["start_control_step"],
            "start_substep": e["start_substep"],
            "duration_substeps": e["duration_substeps"],
            "minimum_dist": e["minimum_dist"],
            "maximum_penetration": e["maximum_penetration"],
            "preexisting": e["preexisting"],
        }

    def five_level_counts(self):
        """五级统计单位。"""
        from collections import Counter
        self.compute_contiguous_events()

        positive_substeps = len(set((e["control_step"], e["substep"]) for e in self.geometric_records))
        positive_control_steps = len(set(e["control_step"] for e in self.geometric_records))
        contiguous = len(self.contiguous_events)
        class_counter = Counter(e["semantic_class"] for e in self.geometric_records)
        return {
            "raw_contact_record_count": len(self.raw_records),
            "contact_positive_substep_count": positive_substeps,
            "contact_positive_control_step_count": positive_control_steps,
            "contiguous_contact_event_count": contiguous,
            "contact_positive_episode_count": 1 if positive_control_steps > 0 else 0,
            "semantic_class_record_counts": dict(class_counter),
        }


def load_role_manifest(path):
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
