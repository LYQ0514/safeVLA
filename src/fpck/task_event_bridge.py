"""TaskEventBridge：统一事件桥（E1a 事件级自动机接入）。

职责（裁决要求）：
- contact contract 语义化；
- goal-clause 独立求值；
- manipulated-target pose/位移；
- grasp/controlled/transport/release 见证；
- 自动机连续更新（跨 PREGRASP→REENTRY 不重置）；
- append-only 原始事件输出。

两个 instrumented 执行循环只调用 bridge，不自行决定标签。

事件频率：
- Joint-PD 阶段：每个 physics substep 记录 contact + task pose + obstacle 位移 + 违规候选；
  每 25 substep 聚合一次控制步（goal truth + lifecycle witness + 自动机状态）；
- OSC/π0.5 阶段：每个环境控制步记录完整事件（contact + goal + native + 自动机转移）。

自动机连续性：bridge.reset() 只在 settled initial state 调用一次；handoff 时
用 snapshot()/restore() 保存/恢复自动机序列化状态，禁止 reset。
"""

from __future__ import annotations

import hashlib
import json
import time

import numpy as np

from src.fpck.task_automaton import (
    TaskAutomaton, TaskEvent, AutomatonState,
    MODE_BAD, MODE_UNKNOWN, MODE_ACC,
    TARGET_MOTION_MM, LIFT_THRESHOLD_MM, RELATIVE_DRIFT_MM,
    CONTROL_WITNESS_STEPS, TRANSPORTED_MM,
    RELEASE_CONTACT_GONE_STEPS, RELEASE_VEL_STEPS, RELEASE_VEL_MPS,
)
from src.fpck.goal_evaluator import eval_goal_clauses, native_success

SPEC_VERSION = "fpck-task-automaton-v1.2"

# 主违规码优先级（同 substep 并发事件）
VIOLATION_PRIORITY = {
    "FORBIDDEN_CONTACT": 0,
    "PAYLOAD_OBSTACLE_CONTACT": 1,
    "UNAUTHORIZED_TARGET_MOTION": 2,
    "STATIC_OBSTACLE_MOVED": 3,
    "NATIVE_FALSE_POSITIVE": 4,
}


class TaskEventBridge:
    """统一事件桥：从物理观测构造 TaskEvent，驱动自动机，输出 append-only 日志。"""

    def __init__(self, automaton: TaskAutomaton, raw_log_path: str,
                 manipulated_targets, goal_keys, obstacle_body=None):
        self.auto = automaton
        self.raw_log_path = raw_log_path
        self.manipulated_targets = list(manipulated_targets)
        self.goal_keys = list(goal_keys)
        self.obstacle_body = obstacle_body  # 如 "red_coffee_mug_obstacle_1_main"

        self.event_index = 0
        self._settled_baseline = None  # {target: pos}（settled 基线）
        self._target_prev = {}         # {target: pos}（上一步，用于见证计数）
        self._lift_hist = {}           # {target: [lift_mm 最近 CONTROL_WITNESS_STEPS 步]}
        self._drift_hist = {}          # {target: [drift_mm ...]}
        self._contact_gone = {}        # {target: 连续无接触步数}
        self._vel_hist = {}            # {target: [vel 最近 RELEASE_VEL_STEPS 步]}
        self._primary_violation = None  # (decision_index, substep, priority, code)
        self._first_decisive_index = None
        self._obstacle_baseline = None
        self._prev_eef = None  # 上一步 EEF 位置（用于相对漂移见证）
        # S04 需要的额外时间线字段（从事件流记录）
        self._first_gt2mm = None      # (event_index, substep) 首次 >2mm
        self._first_gt5mm = None      # 首次 >5mm
        self._first_gt10mm = None     # 首次 >10mm
        self._first_task_contact = None  # 首次 TASK_CONTACT
        self._first_controlled = None    # 首次 CONTROLLED witness
        self._progress_history = []      # [(event_index, mode)] 阶段推进历史
        self._last_progress_index = None  # 最后合法进展 event_index

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def reset(self, env, settled_baseline, target_bodies, obstacle_body):
        """settled initial state 调用一次：记录基线。禁止在 handoff 时再次调用。"""
        self._settled_baseline = {t: settled_baseline[t].copy() for t in target_bodies}
        self._target_prev = {t: settled_baseline[t].copy() for t in target_bodies}
        self._lift_hist = {t: [] for t in target_bodies}
        self._drift_hist = {t: [] for t in target_bodies}
        self._contact_gone = {t: 0 for t in target_bodies}
        self._vel_hist = {t: [] for t in target_bodies}
        self.obstacle_body = obstacle_body
        self._obstacle_baseline = None
        if obstacle_body is not None:
            from tools.g1a_soup_trace import body_state
            self._obstacle_baseline = body_state(env, obstacle_body)["pos"].copy()

    # ------------------------------------------------------------------
    # 观测接口
    # ------------------------------------------------------------------
    def _dynamic_contacts(self, env, target_bodies):
        """用动态 target_bodies/obstacle_body 判定接触（不依赖 contract_objects 硬编码）。

        返回 (robot_target_contact, robot_obstacle_contact, two_finger, contacts_detail)。
        contract_objects 的 PROTECTED_OBJECTS/TASK_OBJECTS 硬编码为 long 套件对象名，
        跨套件（goal/spatial）会误判。这里直接用 bridge 持有的动态 body 名扫描
        mjData.contact。
        """
        m = env.sim.model._model
        d = env.sim.data._data
        names = env.sim.model.body_names
        robot_prefix = ("robot0", "gripper0")
        target_set = set(target_bodies or [])
        obstacle = self.obstacle_body

        robot_target = False
        robot_obstacle = False
        left_finger = False
        right_finger = False
        detail = []

        def is_robot(b):
            return b.startswith(robot_prefix)

        for c in range(d.ncon):
            g1, g2 = d.contact[c].geom1, d.contact[c].geom2
            b1 = names[m.geom_bodyid[g1]]
            b2 = names[m.geom_bodyid[g2]]
            r1, r2 = is_robot(b1), is_robot(b2)
            if not (r1 or r2):
                continue
            other = b2 if r1 else b1
            robot_side = b1 if r1 else b2
            if other in target_set:
                robot_target = True
                if "leftfinger" in robot_side:
                    left_finger = True
                if "rightfinger" in robot_side:
                    right_finger = True
            if obstacle is not None and other == obstacle:
                robot_obstacle = True
            detail.append({"robot": robot_side, "other": other})

        return robot_target, robot_obstacle, (left_finger and right_finger), detail

    def observe_physics_substep(self, env, phase, control_step, physics_substep,
                                target_bodies, target_poses):
        """Joint-PD 阶段：每个 physics substep 记录 contact + 位移 + 违规候选。"""
        robot_target, robot_obstacle, _, contact_detail = self._dynamic_contacts(env, target_bodies)
        forbidden = robot_obstacle  # 仅障碍物接触算 forbidden（不把 unclassified 当 forbidden）
        payload_obstacle = robot_obstacle

        # 障碍物位移
        obstacle_disp = 0.0
        if self.obstacle_body is not None and self._obstacle_baseline is not None:
            from tools.g1a_soup_trace import body_state
            cur = body_state(env, self.obstacle_body)["pos"]
            obstacle_disp = float(np.sum(np.abs(cur - self._obstacle_baseline))) * 1000  # mm

        # target 位移（相对 settled baseline）
        target_disp_mm = {}
        for t in target_bodies:
            cur = target_poses.get(t)
            if cur is not None and t in self._settled_baseline:
                target_disp_mm[t] = float(np.sum(np.abs(
                    np.asarray(cur) - self._settled_baseline[t]))) * 1000

        # 记录位移阈值时间线（S04 需要：first >2/5/10mm）
        max_disp = max(target_disp_mm.values()) if target_disp_mm else 0.0
        if self._first_gt2mm is None and max_disp > 2.0:
            self._first_gt2mm = (self.event_index, physics_substep)
        if self._first_gt5mm is None and max_disp > 5.0:
            self._first_gt5mm = (self.event_index, physics_substep)
        if self._first_gt10mm is None and max_disp > 10.0:
            self._first_gt10mm = (self.event_index, physics_substep)

        # 违规候选（substep 级）
        violation = None
        if forbidden:
            violation = "FORBIDDEN_CONTACT"
        elif payload_obstacle:
            violation = "PAYLOAD_OBSTACLE_CONTACT"
        elif any(v > TARGET_MOTION_MM for v in target_disp_mm.values()) \
                and not self._any_controlled():
            violation = "UNAUTHORIZED_TARGET_MOTION"
        elif obstacle_disp > 1.0:
            violation = "STATIC_OBSTACLE_MOVED"

        # 记录违规（主违规码由最早决定性事件确定）
        if violation is not None and self._primary_violation is None:
            self._primary_violation = (self.event_index, physics_substep,
                                       VIOLATION_PRIORITY.get(violation, 99), violation)
            self._first_decisive_index = self.event_index

        # 精确 substep 时机：检测到违规立即驱动自动机进入 q_bad
        # （否则精确 substep 时机丢失，要等到 control step 才 q_bad）
        if violation is not None and self.auto.state.mode not in ("q_bad", "q_unknown", "q_acc"):
            ev = TaskEvent(
                step=control_step,
                forbidden_contact=forbidden,
                payload_obstacle_contact=payload_obstacle,
                obstacle_displacement_l1_mm=obstacle_disp,
                target_displacement_l1_mm=max(target_disp_mm.values()) if target_disp_mm else 0.0,
            )
            self.auto.process(ev)

        # 输出 substep 事件
        self._emit({
            "phase": phase, "control_step": control_step,
            "physics_substep": physics_substep,
            "event_type": "physics_substep",
            "forbidden_contact": forbidden,
            "payload_obstacle_contact": payload_obstacle,
            "obstacle_displacement_mm": obstacle_disp,
            "target_displacement_mm": target_disp_mm,
            "violation_candidate": violation,
            "automaton_mode": self.auto.state.mode,
        })
        return violation

    def observe_control_step(self, env, obs, phase, control_step,
                             nominal_action, executed_action,
                             chunk_hash=None, buffer_index=None,
                             target_bodies=None, gripper_close=None,
                             eef_pos_world=None):
        """OSC/π0.5 阶段：每个控制步记录完整事件 + 驱动自动机转移。"""
        # 动态接触判定（不依赖 contract_objects 硬编码角色清单）
        robot_target, robot_obstacle, two_finger_contact, _ = \
            self._dynamic_contacts(env, target_bodies)
        forbidden = robot_obstacle  # 仅障碍物接触算 forbidden
        payload_obstacle = robot_obstacle
        clause_truth = eval_goal_clauses(env)
        native = native_success(env)

        # 障碍物位移
        obstacle_disp = 0.0
        if self.obstacle_body is not None and self._obstacle_baseline is not None:
            from tools.g1a_soup_trace import body_state
            cur = body_state(env, self.obstacle_body)["pos"]
            obstacle_disp = float(np.sum(np.abs(cur - self._obstacle_baseline))) * 1000

        # target 位移 + 见证
        target_disp_mm = {}
        task_contact = robot_target  # 动态判定（第 230 行）
        # 见证字段（真实物理量，避免默认值导致见证永不成立）
        target_lifted_mm = 0.0
        eef_target_drift_mm = 0.0
        target_vel_mps = 0.0
        release_cmd = False
        if target_bodies:
            from tools.g1a_soup_trace import body_state, eef_pos
            ee = eef_pos(env)  # EEF 世界系位置
            obs_ = env.env._get_observations()
            gq = np.asarray(obs_.get("robot0_gripper_qpos", [0.0]), dtype=np.float64)
            gripper_open = bool(np.abs(gq).mean() > 0.03)  # |qpos| 均值大 = 张开
            release_cmd = gripper_open

            for t in target_bodies:
                if t not in self._settled_baseline:
                    continue
                st = body_state(env, t)
                cur = st["pos"]
                vel = st["vel"]
                target_disp_mm[t] = float(np.sum(np.abs(
                    cur - self._settled_baseline[t]))) * 1000
                # 抬升（相对 settled baseline 的 z 变化，取正抬升）
                lift_z = float(cur[2] - self._settled_baseline[t][2]) * 1000
                target_lifted_mm = max(target_lifted_mm, lift_z)
                # EEF-目标相对漂移（相对上一步的相对位置变化）
                if t in self._target_prev:
                    rel = np.asarray(cur) - ee
                    if self._prev_eef is not None:
                        prev_rel = np.asarray(self._target_prev[t]) - self._prev_eef
                        drift = float(np.sum(np.abs(rel - prev_rel))) * 1000
                        eef_target_drift_mm = max(eef_target_drift_mm, drift)
                    self._target_prev[t] = cur.copy()
                # 目标速度
                target_vel_mps = max(target_vel_mps, float(np.linalg.norm(vel)))
            self._prev_eef = ee

        # 更新 contact_gone 计数（释放见证需要：机器人-目标接触连续消失步数）
        for t in target_bodies or []:
            if t in self._contact_gone:
                if task_contact:
                    self._contact_gone[t] = 0
                else:
                    self._contact_gone[t] += 1
        contact_gone = max(self._contact_gone.values()) if self._contact_gone else 0

        # 构造 TaskEvent，驱动自动机
        ev = TaskEvent(
            step=control_step,
            goal_clause_truth=clause_truth,
            native_success=native,
            forbidden_contact=forbidden,
            payload_obstacle_contact=payload_obstacle,
            obstacle_displacement_l1_mm=obstacle_disp,
            target_displacement_l1_mm=max(target_disp_mm.values()) if target_disp_mm else 0.0,
            task_contact=task_contact,
            target_lifted_mm=target_lifted_mm,
            eef_target_relative_drift_mm=eef_target_drift_mm,
            gripper_command_close=bool(gripper_close) if gripper_close is not None else (not release_cmd),
            gripper_two_finger_contact=two_finger_contact,
            target_velocity_mps=target_vel_mps,
            contact_gone_steps=contact_gone,
            release_command=release_cmd,
        )
        before = self.auto.state.mode
        self.auto.process(ev)
        after = self.auto.state.mode

        # 记录 TASK_CONTACT / CONTROLLED / progress 时间线
        if task_contact and self._first_task_contact is None:
            self._first_task_contact = (self.event_index, control_step)
        if any(v in ("CONTROLLED", "TRANSPORTED", "RELEASED")
               for v in self.auto.state.lifecycle.values()) and self._first_controlled is None:
            self._first_controlled = (self.event_index, control_step)
        if after != before:
            self._progress_history.append((self.event_index, before, after))
            # 合法进展（mode 前进，非 q_bad/q_unknown）
            if after not in (MODE_BAD, MODE_UNKNOWN):
                self._last_progress_index = self.event_index

        # 违规候选（control step 级，主违规码已在 substep 级优先记录）
        if self._primary_violation is None and after == MODE_BAD:
            code = self.auto.state.violation_code or "UNKNOWN"
            self._primary_violation = (self.event_index, 0,
                                       VIOLATION_PRIORITY.get(code, 99), code)
            self._first_decisive_index = self.event_index

        self._emit({
            "phase": phase, "control_step": control_step,
            "event_type": "control_step",
            "forbidden_contact": forbidden,
            "payload_obstacle_contact": payload_obstacle,
            "obstacle_displacement_mm": obstacle_disp,
            "target_displacement_mm": target_disp_mm,
            "goal_clause_truth": clause_truth,
            "native_success": native,
            "nominal_action": None if nominal_action is None else np.asarray(nominal_action).tolist(),
            "executed_action": None if executed_action is None else np.asarray(executed_action).tolist(),
            "chunk_hash": chunk_hash,
            "buffer_index": buffer_index,
            # 见证字段（审计必需：CONTROLLED/RELEASED 见证为何成立/不成立）
            "task_contact": task_contact,
            "target_lifted_mm": target_lifted_mm,
            "eef_target_drift_mm": eef_target_drift_mm,
            "two_finger_contact": two_finger_contact,
            "gripper_command_close": (bool(gripper_close) if gripper_close is not None else (not release_cmd)),
            "target_velocity_mps": target_vel_mps,
            "contact_gone_steps": contact_gone,
            "release_command": release_cmd,
            "automaton_before": before,
            "automaton_after": after,
            "lifecycle": dict(self.auto.state.lifecycle),
            "achieved_mask": dict(self.auto.state.achieved_mask),
        })
        return after

    # ------------------------------------------------------------------
    # 快照 / 序列化
    # ------------------------------------------------------------------
    def snapshot(self):
        """返回 bridge 完整状态（自动机 + 见证计数器 + 基线）。"""
        return {
            "automaton": self.auto.state.snapshot(),
            "settled_baseline": {t: v.tolist() for t, v in (self._settled_baseline or {}).items()},
            "target_prev": {t: v.tolist() for t, v in self._target_prev.items()},
            "lift_hist": {t: list(v) for t, v in self._lift_hist.items()},
            "drift_hist": {t: list(v) for t, v in self._drift_hist.items()},
            "contact_gone": dict(self._contact_gone),
            "vel_hist": {t: list(v) for t, v in self._vel_hist.items()},
            "primary_violation": self._primary_violation,
            "first_decisive_index": self._first_decisive_index,
            "obstacle_baseline": (None if self._obstacle_baseline is None
                                  else self._obstacle_baseline.tolist()),
            "first_gt2mm": self._first_gt2mm,
            "first_gt5mm": self._first_gt5mm,
            "first_gt10mm": self._first_gt10mm,
            "first_task_contact": self._first_task_contact,
            "first_controlled": self._first_controlled,
            "progress_history": list(self._progress_history),
            "last_progress_index": self._last_progress_index,
        }

    def restore(self, snap):
        """恢复 bridge 状态（含自动机 + 见证计数器）。"""
        self.auto.state = AutomatonState(
            mode=snap["automaton"]["mode"],
            lifecycle=dict(snap["automaton"]["lifecycle"]),
            achieved_mask=dict(snap["automaton"]["achieved_mask"]),
            violation_code=snap["automaton"]["violation_code"],
        )
        self._settled_baseline = {t: np.asarray(v) for t, v in snap["settled_baseline"].items()}
        self._target_prev = {t: np.asarray(v) for t, v in snap["target_prev"].items()}
        self._lift_hist = {t: list(v) for t, v in snap["lift_hist"].items()}
        self._drift_hist = {t: list(v) for t, v in snap["drift_hist"].items()}
        self._contact_gone = dict(snap["contact_gone"])
        self._vel_hist = {t: list(v) for t, v in snap["vel_hist"].items()}
        self._primary_violation = snap["primary_violation"]
        self._first_decisive_index = snap["first_decisive_index"]
        self._obstacle_baseline = (None if snap["obstacle_baseline"] is None
                                   else np.asarray(snap["obstacle_baseline"]))
        self._first_gt2mm = snap.get("first_gt2mm")
        self._first_gt5mm = snap.get("first_gt5mm")
        self._first_gt10mm = snap.get("first_gt10mm")
        self._first_task_contact = snap.get("first_task_contact")
        self._first_controlled = snap.get("first_controlled")
        self._progress_history = list(snap.get("progress_history", []))
        self._last_progress_index = snap.get("last_progress_index")

    def finalize(self):
        """返回最终汇总（引用 trace hash，不代替原始记录）。"""
        return {
            "final_automaton_mode": self.auto.state.mode,
            "violation_code": self.auto.state.violation_code,
            "primary_violation": self._primary_violation,
            "first_decisive_event_index": self._first_decisive_index,
            "final_lifecycle": dict(self.auto.state.lifecycle),
            "final_achieved_mask": dict(self.auto.state.achieved_mask),
            # S04 额外时间线字段
            "first_gt2mm": self._first_gt2mm,
            "first_gt5mm": self._first_gt5mm,
            "first_gt10mm": self._first_gt10mm,
            "first_task_contact": self._first_task_contact,
            "first_controlled_witness": self._first_controlled,
            "progress_history": list(self._progress_history),
            "last_progress_index": self._last_progress_index,
        }

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    def _any_controlled(self):
        return any(v in ("CONTROLLED", "TRANSPORTED", "RELEASED")
                   for v in self.auto.state.lifecycle.values())

    def _emit(self, row):
        row["event_index"] = self.event_index
        row["wall_time"] = time.time()
        self.event_index += 1
        with open(self.raw_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
