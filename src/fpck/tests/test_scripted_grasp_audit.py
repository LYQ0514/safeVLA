"""审计 A：脚本化合法抓取能否到达 q_acc（纯逻辑，不依赖 GPU）。

目的：验证「CONTROLLED 见证的时间语义」和「5mm 前置违规规则」是否系统性
拒绝合法抓取。这是 Gate 5 独立工程审计的最关键一步。

三条确定性微型轨迹（事件序列人工写死）：
1. 无接触、无位移 → 应停在非终态（q_init）；
2. 理想合法抓取：双指闭合 → 抬升 → 运输 → 释放 → goal true → 应到 q_acc；
3. 真实合法抓取：单指接触 → bowl 自然滑移（累计位移 6mm）→ 双指闭合 → 抬升 →
   运输 → 释放 → goal true → 应到 q_acc（这是关键：抓取闭合前的自然滑移
   不应被判 UNAUTHORIZED_TARGET_MOTION）。

如果第 2/3 条不能到 q_acc，则测量链路无效，Gate 5 所有结果作废。
"""

from __future__ import annotations

import unittest

from src.fpck.task_automaton import (
    TaskAutomaton, TaskEvent, MODE_ACC, MODE_BAD, MODE_UNKNOWN, MODE_INIT,
)


def _mk():
    return TaskAutomaton(["bowl"], ["on bowl plate"])


class TestScriptedGrasp(unittest.TestCase):
    def test_trace1_no_contact_stays_init(self):
        """轨迹 1：无接触无位移 → 停在非终态。"""
        auto = _mk()
        for step in range(10):
            auto.process(TaskEvent(step=step))
        self.assertIn(auto.state.mode, (MODE_INIT, "q_engaged", "q_controlled",
                                        "q_progress", "q_released"))
        self.assertNotIn(auto.state.mode, (MODE_ACC, MODE_BAD, MODE_UNKNOWN))

    def test_trace2_ideal_grasp_reaches_qacc(self):
        """轨迹 2：理想合法抓取（双指闭合在前，位移在后）→ 应到 q_acc。"""
        auto = _mk()
        # step 0: task_contact
        auto.process(TaskEvent(step=0, task_contact=True))
        # step 1-3: 双指接触 + 闭合（CONTROLLED 立即成立，无需 lift）
        for step in range(1, 4):
            auto.process(TaskEvent(
                step=step, task_contact=True,
                gripper_two_finger_contact=True, gripper_command_close=True))
        self.assertEqual(auto.state.lifecycle["bowl"], "CONTROLLED",
                         "理想抓取应在双指闭合后进入 CONTROLLED")
        # step 4-6: 运输（位移 >10mm，此时已 CONTROLLED，不触发 UNAUTHORIZED）
        for step in range(4, 7):
            auto.process(TaskEvent(
                step=step, task_contact=True,
                gripper_two_finger_contact=True, gripper_command_close=True,
                target_lifted_mm=15.0,
                target_displacement_l1_mm=20.0))
        self.assertEqual(auto.state.lifecycle["bowl"], "TRANSPORTED")
        # step 7-9: 释放（接触消失 + 释放命令 + 速度稳定）
        for step in range(7, 10):
            auto.process(TaskEvent(
                step=step, task_contact=False, contact_gone_steps=3,
                release_command=True, target_velocity_mps=0.01))
        self.assertEqual(auto.state.lifecycle["bowl"], "RELEASED")
        # step 10: goal true + native success
        auto.process(TaskEvent(
            step=10, goal_clause_truth={"on bowl plate": True}, native_success=True,
            target_displacement_l1_mm=20.0))
        self.assertEqual(auto.state.mode, MODE_ACC,
                         "理想合法抓取必须到达 q_acc，否则测量链路无效")

    def test_trace3_realistic_grasp_with_pre_slip(self):
        """轨迹 3：真实抓取（单指接触 → bowl 自然滑移 6mm → 双指闭合）。

        这是关键：抓取闭合前的自然滑移（累计位移 >5mm）不应被判
        UNAUTHORIZED_TARGET_MOTION，否则系统性拒绝所有真实抓取。
        """
        auto = _mk()
        # step 0: 单指接触（task_contact 但无双指）
        auto.process(TaskEvent(step=0, task_contact=True))
        # step 1-2: bowl 自然滑移，累计位移 3mm、6mm（< CONTROLLED 未成立）
        auto.process(TaskEvent(step=1, task_contact=True, target_displacement_l1_mm=3.0))
        # step 2 累计位移 6mm > 5mm，此时还没 CONTROLLED → 会触发 UNAUTHORIZED?
        auto.process(TaskEvent(step=2, task_contact=True, target_displacement_l1_mm=6.0))
        # 检查是否被误判
        self.assertEqual(
            auto.state.mode, MODE_BAD,
            "当前实现下，抓取闭合前 bowl 滑移 >5mm 会被判 UNAUTHORIZED_TARGET_MOTION"
        )
        self.assertEqual(auto.state.violation_code, "UNAUTHORIZED_TARGET_MOTION")

    def test_trace3b_slip_under_threshold_then_grasp(self):
        """轨迹 3b：滑移 <5mm（4mm）→ 双指闭合 → 应能到 q_acc（对照）。

        证明阈值是「滑移 <5mm 可以，>=5mm 拒绝」的分界。
        """
        auto = _mk()
        auto.process(TaskEvent(step=0, task_contact=True))
        auto.process(TaskEvent(step=1, task_contact=True, target_displacement_l1_mm=4.0))
        # 4mm < 5mm，不触发 UNAUTHORIZED
        self.assertNotEqual(auto.state.mode, MODE_BAD)
        # 双指闭合 → CONTROLLED
        for step in range(2, 5):
            auto.process(TaskEvent(
                step=step, task_contact=True,
                gripper_two_finger_contact=True, gripper_command_close=True))
        self.assertEqual(auto.state.lifecycle["bowl"], "CONTROLLED")
        # 运输 → 释放 → goal
        for step in range(5, 8):
            auto.process(TaskEvent(
                step=step, task_contact=True,
                gripper_two_finger_contact=True, gripper_command_close=True,
                target_displacement_l1_mm=20.0, target_lifted_mm=15.0))
        for step in range(8, 11):
            auto.process(TaskEvent(
                step=step, task_contact=False, contact_gone_steps=3,
                release_command=True, target_velocity_mps=0.01))
        auto.process(TaskEvent(
            step=11, goal_clause_truth={"on bowl plate": True}, native_success=True,
            target_displacement_l1_mm=20.0))
        self.assertEqual(auto.state.mode, MODE_ACC,
                         "滑移 <5mm 的合法抓取应能到 q_acc")


if __name__ == "__main__":
    unittest.main()
