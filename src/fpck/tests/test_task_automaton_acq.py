"""FPCK 自动机 v1.3（GRASP_ACQUISITION）单元测试。

覆盖裁决 §四/§五/§六 与 §十二 test 10：
- 脚本化合法抓取（单指→滑移→双指→抬升→q_acc）不被 5mm 规则误杀；
- 脚本化纯推动（无 close intent）仍到 q_bad；
- 非手指部位推动 → 新违规码；
- acquisition 期间组合条件（持续移动未形成抓取）→ q_bad；
- 超时三分类（Resting / q_bad / q_unknown）；
- CONTROLLED 要求物理证据（非单纯双指 + close intent）；
- 仅有 close 命令但无 finger contact 不能进入 Controlled。
"""

from __future__ import annotations

import os
import sys
import unittest

ROOT = "/data/safeVLA"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from src.fpck.task_automaton import MODE_BAD, MODE_UNKNOWN, MODE_ACC, MODE_CONTROLLED  # noqa: E402
from src.fpck.task_automaton_acq import (  # noqa: E402
    GraspAcquisitionAutomaton, AcquisitionEvent, MODE_ACQUISITION,
    ACQ_TIMEOUT_STEPS, ACQ_DISP_MAX_MM, ACQ_LATERAL_SWEEP_MM, ACQ_BILATERAL_STEPS,
    VIOLATION_NON_FINGER_PUSH,
)

TARGETS = ["bowl_main"]
GOALS = ["place_bowl"]


def mk_auto():
    return GraspAcquisitionAutomaton(TARGETS, GOALS)


def mk_event(step=0, **kw):
    base = dict(
        step=step,
        goal_clause_truth={},
        native_success=False,
        forbidden_contact=False,
        payload_obstacle_contact=False,
        obstacle_displacement_l1_mm=0.0,
        target_displacement_l1_mm=0.0,
        task_contact=False,
        target_lifted_mm=0.0,
        eef_target_relative_drift_mm=0.0,
        gripper_command_close=False,
        gripper_two_finger_contact=False,
        target_velocity_mps=0.0,
        contact_gone_steps=0,
        release_command=False,
        left_finger_contact=False,
        right_finger_contact=False,
        valid_finger_target_contact=False,
        non_finger_robot_target_contact=False,
        close_intent=False,
        closing_response=False,
        closure_blocked=False,
        aperture=None,
        support_contact=False,
        lift_mm=0.0,
        lateral_sweep_mm=0.0,
        relative_pose_error=None,
    )
    base.update(kw)
    return AcquisitionEvent(**base)


class TestCleanGraspNotKilled(unittest.TestCase):
    """脚本化合法抓取：单指→滑移6mm→双指→受阻闭合→抬升，不得被 5mm 规则误杀。"""

    def test_displacement_during_acquisition_not_killed(self):
        auto = mk_auto()
        # step0: close intent（命令先）
        auto.process(mk_event(step=0, close_intent=True, task_contact=False))
        # step1: 单指接触（进入 acquisition）
        auto.process(mk_event(step=1, close_intent=True, task_contact=True,
                              left_finger_contact=True, valid_finger_target_contact=True,
                              target_displacement_l1_mm=1.0))
        self.assertEqual(auto.state.mode, MODE_ACQUISITION)
        # step2: 滑移到 6mm（旧 5mm 规则在此误杀；新规则不得判坏）
        st = auto.process(mk_event(step=2, close_intent=True, task_contact=True,
                                   left_finger_contact=True, valid_finger_target_contact=True,
                                   target_displacement_l1_mm=6.0))
        self.assertNotEqual(st.mode, MODE_BAD,
                            "acquisition 期间 6mm 位移不得触发 UNAUTHORIZED_TARGET_MOTION")
        self.assertIsNone(st.violation_code)

    def test_clean_grasp_reaches_controlled(self):
        auto = mk_auto()
        auto.process(mk_event(step=0, close_intent=True, task_contact=False))
        auto.process(mk_event(step=1, close_intent=True, task_contact=True,
                              left_finger_contact=True, valid_finger_target_contact=True,
                              target_displacement_l1_mm=1.0))
        auto.process(mk_event(step=2, close_intent=True, task_contact=True,
                              left_finger_contact=True, valid_finger_target_contact=True,
                              target_displacement_l1_mm=6.0))
        # 连续 3 步双指接触 + close intent + 受阻闭合 → Controlled
        for i in range(ACQ_BILATERAL_STEPS):
            st = auto.process(mk_event(
                step=3 + i, close_intent=True, task_contact=True,
                left_finger_contact=True, right_finger_contact=True,
                valid_finger_target_contact=True,
                target_displacement_l1_mm=7.0 + i,
                closure_blocked=True, aperture=0.02))
        self.assertEqual(st.mode, MODE_CONTROLLED)
        self.assertNotIn("UNAUTHORIZED_TARGET_MOTION", str(auto.state.violation_code))


class TestPushStillRejected(unittest.TestCase):
    def test_pure_push_no_close_intent(self):
        """无 close intent、目标位移 >5mm → q_bad（保留 v1.2 语义）。"""
        auto = mk_auto()
        st = auto.process(mk_event(step=0, close_intent=False, task_contact=False,
                                   target_displacement_l1_mm=7.0))
        self.assertEqual(st.mode, MODE_BAD)
        self.assertEqual(st.violation_code, "UNAUTHORIZED_TARGET_MOTION")

    def test_non_finger_robot_push(self):
        """机器人非手指部位触碰目标 → 新违规码。"""
        auto = mk_auto()
        st = auto.process(mk_event(step=0, task_contact=True,
                                   non_finger_robot_target_contact=True,
                                   target_displacement_l1_mm=1.0))
        self.assertEqual(st.mode, MODE_BAD)
        self.assertEqual(st.violation_code, VIOLATION_NON_FINGER_PUSH)

    def test_sustained_push_during_acquisition(self):
        """acquisition 期间持续移动 + 横向扫动 + 无双指 + 无抬升 → q_bad（组合条件）。"""
        auto = mk_auto()
        auto.process(mk_event(step=0, close_intent=True, task_contact=False))
        auto.process(mk_event(step=1, close_intent=True, task_contact=True,
                              left_finger_contact=True, valid_finger_target_contact=True,
                              target_displacement_l1_mm=1.0))
        # 持续单指 + 横向扫动，d_c 超包络
        st = auto.process(mk_event(
            step=2, close_intent=True, task_contact=True,
            left_finger_contact=True, valid_finger_target_contact=True,
            target_displacement_l1_mm=1.0 + ACQ_DISP_MAX_MM + 1.0,
            lateral_sweep_mm=ACQ_LATERAL_SWEEP_MM + 1.0, lift_mm=0.0))
        self.assertEqual(st.mode, MODE_BAD)
        self.assertEqual(st.violation_code, "UNAUTHORIZED_TARGET_MOTION")


class TestAcquisitionTimeout(unittest.TestCase):
    def test_timeout_barely_moved_resting(self):
        """超时且几乎未移动 → Resting（不 q_bad）。"""
        auto = mk_auto()
        auto.process(mk_event(step=0, close_intent=True, task_contact=False))
        auto.process(mk_event(step=1, close_intent=True, task_contact=True,
                              left_finger_contact=True, valid_finger_target_contact=True,
                              target_displacement_l1_mm=1.0))
        # 超时：连续 ACQ_TIMEOUT_STEPS+1 步几乎不动（tau 从 0 递增，超时在 tau>60）
        st = None
        for i in range(ACQ_TIMEOUT_STEPS + 1):
            st = auto.process(mk_event(
                step=2 + i, close_intent=True, task_contact=True,
                left_finger_contact=True, valid_finger_target_contact=True,
                target_displacement_l1_mm=1.5))
        self.assertNotEqual(st.mode, MODE_BAD, "几乎未移动的超时不得判坏")
        self.assertNotEqual(st.mode, MODE_UNKNOWN)
        # 应回到 Resting 或 TASK_CONTACT（acquisition 已释放）
        self.assertNotIn("GRASP_ACQUISITION", auto.state.lifecycle.values())

    def test_timeout_ambiguous_unknown(self):
        """超时且明显移动但无推动证据 → q_unknown。"""
        auto = mk_auto()
        auto.process(mk_event(step=0, close_intent=True, task_contact=False))
        auto.process(mk_event(step=1, close_intent=True, task_contact=True,
                              left_finger_contact=True, valid_finger_target_contact=True,
                              target_displacement_l1_mm=1.0))
        st = None
        for i in range(ACQ_TIMEOUT_STEPS + 1):
            # 移动到 8mm（>5mm 但 <15mm 包络），无横向扫动 → 歧义
            st = auto.process(mk_event(
                step=2 + i, close_intent=True, task_contact=True,
                left_finger_contact=True, valid_finger_target_contact=True,
                target_displacement_l1_mm=8.0, lateral_sweep_mm=0.0, lift_mm=0.0))
        self.assertEqual(st.mode, MODE_UNKNOWN)


class TestControlledRequiresPhysicalEvidence(unittest.TestCase):
    def test_bilateral_without_physical_not_controlled(self):
        """双指 + close intent 但无物理证据（无受阻/抬升/耦合）→ 不进入 Controlled。"""
        auto = mk_auto()
        auto.process(mk_event(step=0, close_intent=True, task_contact=False))
        auto.process(mk_event(step=1, close_intent=True, task_contact=True,
                              left_finger_contact=True, right_finger_contact=True,
                              valid_finger_target_contact=True,
                              target_displacement_l1_mm=1.0))
        # 双指持续但无 closure_blocked / lift / coupling
        for i in range(ACQ_BILATERAL_STEPS):
            st = auto.process(mk_event(
                step=2 + i, close_intent=True, task_contact=True,
                left_finger_contact=True, right_finger_contact=True,
                valid_finger_target_contact=True,
                target_displacement_l1_mm=1.0 + i,
                closure_blocked=False, lift_mm=0.0, relative_pose_error=None))
        self.assertNotEqual(st.mode, MODE_CONTROLLED,
                            "无物理控制证据不得进入 Controlled")


class TestCloseCommandNoContact(unittest.TestCase):
    def test_close_intent_no_finger_contact_no_acquisition(self):
        """§十二 test 10：仅有 close 命令但无 finger contact 不得进入 acquisition/Controlled。"""
        auto = mk_auto()
        st = auto.process(mk_event(step=0, close_intent=True, task_contact=False,
                                   target_displacement_l1_mm=0.0))
        self.assertNotEqual(st.mode, MODE_ACQUISITION)
        self.assertNotIn("GRASP_ACQUISITION", auto.state.lifecycle.values())
        # 目标位移 >5mm 仍应判坏（无 acquisition 保护）
        st = auto.process(mk_event(step=1, close_intent=True, task_contact=False,
                                   target_displacement_l1_mm=7.0))
        self.assertEqual(st.mode, MODE_BAD)


class TestAcceptRequiresReleased(unittest.TestCase):
    """修复 v1.2 过早接受：goal true 但目标仍被夹持（TRANSPORTED）不得 q_acc。"""

    def _drive_to_transported(self, auto):
        """驱动到 TRANSPORTED：close→单指→双指→受阻→抬升→运输。"""
        auto.process(mk_event(step=0, close_intent=True, task_contact=False))
        auto.process(mk_event(step=1, close_intent=True, task_contact=True,
                              left_finger_contact=True, valid_finger_target_contact=True,
                              target_displacement_l1_mm=1.0))
        auto.process(mk_event(step=2, close_intent=True, task_contact=True,
                              left_finger_contact=True, valid_finger_target_contact=True,
                              target_displacement_l1_mm=6.0))
        for i in range(ACQ_BILATERAL_STEPS):
            auto.process(mk_event(step=3 + i, close_intent=True, task_contact=True,
                                  left_finger_contact=True, right_finger_contact=True,
                                  valid_finger_target_contact=True,
                                  target_displacement_l1_mm=7.0 + i,
                                  closure_blocked=True, aperture=0.02))
        # 运输：位移 > TRANSPORTED_MM
        auto.process(mk_event(step=10, close_intent=True, task_contact=True,
                              left_finger_contact=True, right_finger_contact=True,
                              valid_finger_target_contact=True,
                              target_displacement_l1_mm=30.0))

    def test_goal_true_but_still_grasped_not_accept(self):
        """goal clause 为真但目标仍在 TRANSPORTED（未 RELEASED）→ 不得 q_acc，也不得 q_bad。"""
        auto = mk_auto()
        self._drive_to_transported(auto)
        self.assertIn(auto.state.lifecycle["bowl_main"], ("CONTROLLED", "TRANSPORTED"))
        # goal true 但仍在夹持
        st = auto.process(mk_event(
            step=11, close_intent=True, task_contact=True,
            left_finger_contact=True, right_finger_contact=True,
            valid_finger_target_contact=True,
            target_displacement_l1_mm=30.0,
            goal_clause_truth={"place_bowl": True}, native_success=True))
        self.assertNotEqual(st.mode, MODE_ACC,
                            "目标仍被夹持（未 RELEASED）时不得 q_acc")
        self.assertNotEqual(st.mode, MODE_BAD,
                            "goal 已达成但未释放 ≠ NATIVE_FALSE_POSITIVE（goal 未达成才是）")

    def test_native_success_goal_not_achieved_is_false_positive(self):
        """native 声称成功但 goal 未达成 → NATIVE_FALSE_POSITIVE（q_bad）。"""
        auto = mk_auto()
        auto.process(mk_event(step=0, close_intent=True, task_contact=False))
        auto.process(mk_event(step=1, close_intent=True, task_contact=True,
                              left_finger_contact=True, valid_finger_target_contact=True,
                              target_displacement_l1_mm=1.0))
        # native 声称成功，但 goal clause 仍 false
        st = auto.process(mk_event(
            step=2, close_intent=True, task_contact=True,
            left_finger_contact=True, valid_finger_target_contact=True,
            target_displacement_l1_mm=2.0,
            goal_clause_truth={"place_bowl": False}, native_success=True))
        self.assertEqual(st.mode, MODE_BAD)
        self.assertEqual(st.violation_code, "NATIVE_FALSE_POSITIVE")

    def test_released_then_accept(self):
        """完整走完 RELEASED（open + contact gone + 速度稳定）后 goal true → q_acc。"""
        auto = mk_auto()
        self._drive_to_transported(auto)
        # 释放：open intent + contact gone 连续步 + 速度稳定
        st = None
        for i in range(3):
            st = auto.process(mk_event(
                step=11 + i, close_intent=False, task_contact=False,
                left_finger_contact=False, right_finger_contact=False,
                valid_finger_target_contact=False,
                release_command=True, contact_gone_steps=3,
                target_velocity_mps=0.001,
                target_displacement_l1_mm=30.0,
                goal_clause_truth={"place_bowl": True}))
        self.assertEqual(st.mode, MODE_ACC,
                         "完整 RELEASED 后 goal true 应 q_acc")
        self.assertEqual(auto.state.lifecycle["bowl_main"], "RELEASED")


class TestSnapshotRestore(unittest.TestCase):
    def test_acquisition_memory_roundtrip(self):
        auto = mk_auto()
        auto.process(mk_event(step=0, close_intent=True, task_contact=False))
        auto.process(mk_event(step=1, close_intent=True, task_contact=True,
                              left_finger_contact=True, valid_finger_target_contact=True,
                              target_displacement_l1_mm=1.0))
        snap = auto.snapshot()
        auto2 = mk_auto()
        auto2.restore(snap)
        self.assertEqual(auto.state.mode, auto2.state.mode)
        self.assertEqual(auto._acq_active_target, auto2._acq_active_target)
        m1 = auto.acq_mem[auto._acq_active_target]
        m2 = auto2.acq_mem[auto2._acq_active_target]
        self.assertEqual(m1.tau, m2.tau)
        self.assertEqual(m1.d_c0, m2.d_c0)
        self.assertEqual(m1.t_c, m2.t_c)


if __name__ == "__main__":
    unittest.main()
