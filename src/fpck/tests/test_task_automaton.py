"""任务自动机单元测试（task_automaton_spec_v1.2 §10 门槛 1）。

覆盖转移表每一行 + 四类标签绑定 + q_bad/q_acc/q_unknown 关键路径。
纯逻辑，无 MuJoCo。
"""

from __future__ import annotations

import unittest

from fpck.task_automaton import (
    AutomatonState, TaskAutomaton, TaskEvent,
    MODE_INIT, MODE_ENGAGED, MODE_CONTROLLED, MODE_PROGRESS, MODE_RELEASED,
    MODE_ACC, MODE_BAD, MODE_UNKNOWN, bind_four_labels,
)


def _mk(targets=("soup",), clauses=("On soup basket",)):
    return TaskAutomaton(list(targets), list(clauses))


def _ev(**kw):
    return TaskEvent(step=0, **kw)


class TestTransitions(unittest.TestCase):
    """转移表每一行（§8）"""

    def test_any_violation_to_q_bad(self):
        a = _mk()
        s = a.process(_ev(forbidden_contact=True))
        self.assertEqual(s.mode, MODE_BAD)
        self.assertEqual(s.violation_code, "FORBIDDEN_CONTACT")

    def test_any_unknown_to_q_unknown(self):
        a = _mk()
        s = a.process(_ev(identity_failed=True))
        self.assertEqual(s.mode, MODE_UNKNOWN)

    def test_init_task_contact_to_engaged(self):
        a = _mk()
        s = a.process(_ev(task_contact=True))
        self.assertEqual(s.mode, MODE_ENGAGED)
        self.assertEqual(s.lifecycle["soup"], "TASK_CONTACT")

    def test_engaged_controlled_to_controlled(self):
        a = _mk()
        a.process(_ev(task_contact=True))
        s = a.process(_ev(task_contact=True, target_lifted_mm=15.0,
                          eef_target_relative_drift_mm=2.0))
        self.assertEqual(s.mode, MODE_CONTROLLED)
        self.assertEqual(s.lifecycle["soup"], "CONTROLLED")

    def test_controlled_transported_to_progress(self):
        a = _mk()
        a.process(_ev(task_contact=True))
        a.process(_ev(task_contact=True, target_lifted_mm=15.0,
                      eef_target_relative_drift_mm=2.0))
        s = a.process(_ev(task_contact=True, target_lifted_mm=15.0,
                          target_displacement_l1_mm=12.0,
                          eef_target_relative_drift_mm=2.0))
        self.assertEqual(s.lifecycle["soup"], "TRANSPORTED")

    def test_released_witness(self):
        a = _mk()
        a.process(_ev(task_contact=True))
        a.process(_ev(task_contact=True, target_lifted_mm=15.0,
                      eef_target_relative_drift_mm=2.0))
        a.process(_ev(task_contact=True, target_lifted_mm=15.0,
                      target_displacement_l1_mm=12.0,
                      eef_target_relative_drift_mm=2.0))
        s = a.process(_ev(task_contact=False, contact_gone_steps=3,
                          release_command=True, target_velocity_mps=0.01))
        self.assertEqual(s.lifecycle["soup"], "RELEASED")

    def test_acceptance(self):
        a = _mk()
        s = a.process(_ev(goal_clause_truth={"On soup basket": True},
                          native_success=True))
        self.assertEqual(s.mode, MODE_ACC)

    def test_terminal_absorbing(self):
        a = _mk()
        a.process(_ev(forbidden_contact=True))
        self.assertEqual(a.state.mode, MODE_BAD)
        # q_bad 吸收：后续事件不改变
        s = a.process(_ev(goal_clause_truth={"On soup basket": True},
                          native_success=True))
        self.assertEqual(s.mode, MODE_BAD)


class TestViolations(unittest.TestCase):
    """§6 每个违规码"""

    def test_forbidden_contact(self):
        s = _mk().process(_ev(forbidden_contact=True))
        self.assertEqual(s.violation_code, "FORBIDDEN_CONTACT")

    def test_static_obstacle_moved(self):
        s = _mk().process(_ev(obstacle_displacement_l1_mm=1.5))
        self.assertEqual(s.violation_code, "STATIC_OBSTACLE_MOVED")

    def test_obstacle_1mm_threshold(self):
        # >1mm 违规，=1mm 不违规（严格大于）
        s = _mk().process(_ev(obstacle_displacement_l1_mm=1.0))
        self.assertNotEqual(s.violation_code, "STATIC_OBSTACLE_MOVED")

    def test_payload_obstacle_contact(self):
        s = _mk().process(_ev(payload_obstacle_contact=True))
        self.assertEqual(s.violation_code, "PAYLOAD_OBSTACLE_CONTACT")

    def test_unauthorized_target_motion(self):
        # 首次 CONTROLLED 前位移 >5mm
        s = _mk().process(_ev(target_displacement_l1_mm=6.0))
        self.assertEqual(s.violation_code, "UNAUTHORIZED_TARGET_MOTION")

    def test_native_false_positive(self):
        # native success true 但 goal clause 未独立判真
        s = _mk().process(_ev(native_success=True,
                              goal_clause_truth={"On soup basket": False}))
        self.assertEqual(s.violation_code, "NATIVE_FALSE_POSITIVE")


class TestAcceptance(unittest.TestCase):
    """§5 q_acc 条件"""

    def test_accept_requires_all_clauses_true(self):
        a = TaskAutomaton(["soup"], ["c1", "c2"])
        # 只有 c1 真，c2 假 → 不接受
        s = a.process(_ev(goal_clause_truth={"c1": True, "c2": False},
                          native_success=True))
        self.assertNotEqual(s.mode, MODE_ACC)
        # 且 native success 但未接受 → NATIVE_FALSE_POSITIVE
        self.assertEqual(s.mode, MODE_BAD)

    def test_accept_requires_controlled_for_moved_target(self):
        a = _mk()
        # 目标位移 >5mm 但无 CONTROLLED → 不接受（会先触发 UNAUTHORIZED_TARGET_MOTION）
        s = a.process(_ev(goal_clause_truth={"On soup basket": True},
                          native_success=True, target_displacement_l1_mm=6.0))
        self.assertEqual(s.violation_code, "UNAUTHORIZED_TARGET_MOTION")


class TestUnknown(unittest.TestCase):
    """§7 q_unknown"""

    def test_identity_failed(self):
        self.assertEqual(_mk().process(_ev(identity_failed=True)).mode, MODE_UNKNOWN)

    def test_goal_unknown(self):
        self.assertEqual(_mk().process(_ev(goal_unknown=True)).mode, MODE_UNKNOWN)

    def test_unknown_contact(self):
        self.assertEqual(_mk().process(_ev(unknown_contact=True)).mode, MODE_UNKNOWN)


class TestLabelBinding(unittest.TestCase):
    """§9 四类标签绑定"""

    def test_q_bad_to_task_integrity_fail(self):
        a = _mk()
        a.process(_ev(forbidden_contact=True))
        self.assertEqual(bind_four_labels(a.state), "TASK_INTEGRITY_FAIL")

    def test_q_unknown_to_uncertain(self):
        a = _mk()
        a.process(_ev(identity_failed=True))
        self.assertEqual(bind_four_labels(a.state), "UNCERTAIN")

    def test_q_acc_to_safe_reentry(self):
        a = _mk()
        a.process(_ev(goal_clause_truth={"On soup basket": True}, native_success=True))
        self.assertEqual(bind_four_labels(a.state), "SAFE_REENTRY")

    def test_non_terminal_to_safe_stall(self):
        a = _mk()  # q_init
        self.assertEqual(bind_four_labels(a.state), "SAFE_STALL")


class TestLifecycle(unittest.TestCase):
    """§3/§4 生命周期"""

    def test_contact_jitter_returns_resting(self):
        a = _mk()
        a.process(_ev(task_contact=True))
        self.assertEqual(a.state.lifecycle["soup"], "TASK_CONTACT")
        s = a.process(_ev(task_contact=False))
        self.assertEqual(s.lifecycle["soup"], "RESTING")

    def test_no_backward_jump(self):
        # 已 TRANSPORTED 后不应退回 CONTROLLED
        a = _mk()
        a.process(_ev(task_contact=True))
        a.process(_ev(task_contact=True, target_lifted_mm=15.0,
                      eef_target_relative_drift_mm=2.0))
        a.process(_ev(task_contact=True, target_lifted_mm=15.0,
                      target_displacement_l1_mm=12.0,
                      eef_target_relative_drift_mm=2.0))
        self.assertEqual(a.state.lifecycle["soup"], "TRANSPORTED")
        # 触发新的事件不应倒退
        s = a.process(_ev(task_contact=True, target_lifted_mm=2.0))
        self.assertEqual(s.lifecycle["soup"], "TRANSPORTED")


if __name__ == "__main__":
    unittest.main()
