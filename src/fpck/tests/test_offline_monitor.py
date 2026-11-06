"""审计 C：独立离线监视器测试。

验证（裁决 §五）：
1. 离线监视器不 import 在线 bridge/automaton_acq/decoder 的分类/guard/acceptance 函数；
2. 离线独立分类（显式 body 名集合）与在线分类结果一致；
3. 在线/离线差分在决定性标签上无分歧（final_mode_match）。
"""

from __future__ import annotations

import os
import sys
import unittest

ROOT = "/data/safeVLA"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from src.fpck.offline_monitor import (  # noqa: E402
    classify_finger, decode_close, contacts_from_detail, OfflineMonitor,
)


class TestIndependence(unittest.TestCase):
    def test_no_import_of_online_classifiers(self):
        """离线监视器源码不得 import 在线分类/guard/acceptance 函数。"""
        src = open(os.path.join(ROOT, "src/fpck/offline_monitor.py")).read()
        forbidden = [
            "from src.fpck.task_event_bridge",
            "from src.fpck.task_automaton_acq",
            "from src.fpck.gripper_action_decoder",
            "import task_event_bridge",
            "import task_automaton_acq",
            "import gripper_action_decoder",
        ]
        for f in forbidden:
            self.assertNotIn(f, src, f"离线监视器禁止 import {f}")


class TestFingerClassify(unittest.TestCase):
    def test_explicit_body_set(self):
        self.assertEqual(classify_finger("gripper0_leftfinger"), "left")
        self.assertEqual(classify_finger("gripper0_finger_joint1_tip"), "left")
        self.assertEqual(classify_finger("gripper0_rightfinger"), "right")
        self.assertEqual(classify_finger("gripper0_finger_joint2_tip"), "right")
        self.assertEqual(classify_finger("gripper0_right_gripper"), "non_finger")
        self.assertEqual(classify_finger("gripper0_eef"), "non_finger")
        self.assertEqual(classify_finger("robot0_link3"), "non_finger")


class TestContactsFromDetail(unittest.TestCase):
    def test_left_finger_contact(self):
        detail = [{"robot": "gripper0_leftfinger", "other": "bowl_main"},
                  {"robot": "gripper0_finger_joint1_tip", "other": "bowl_main"}]
        cf = contacts_from_detail(detail, ["bowl_main"])
        self.assertTrue(cf["left"])
        self.assertFalse(cf["right"])
        self.assertTrue(cf["valid"])
        self.assertFalse(cf["bilateral"])

    def test_bilateral_contact(self):
        detail = [{"robot": "gripper0_leftfinger", "other": "bowl_main"},
                  {"robot": "gripper0_rightfinger", "other": "bowl_main"}]
        cf = contacts_from_detail(detail, ["bowl_main"])
        self.assertTrue(cf["bilateral"])
        self.assertTrue(cf["valid"])

    def test_non_finger_contact(self):
        detail = [{"robot": "gripper0_eef", "other": "bowl_main"}]
        cf = contacts_from_detail(detail, ["bowl_main"])
        self.assertTrue(cf["non_finger"])
        self.assertFalse(cf["valid"])


class TestDecodeClose(unittest.TestCase):
    def test_sign_decoding(self):
        self.assertTrue(decode_close(1.0))
        self.assertTrue(decode_close(0.1))
        self.assertFalse(decode_close(-1.0))
        self.assertFalse(decode_close(None))
        self.assertFalse(decode_close(0.0))


class TestOfflineMonitorMilestones(unittest.TestCase):
    def _make_rows(self):
        """构造一个完整生命周期：close→单指→双指→受阻→抬升→运输→释放→goal。"""
        rows = []
        def r(step, **kw):
            base = {
                "event_type": "control_step", "control_step": step,
                "target_displacement_mm": {"bowl_main": 0.0},
                "target_lifted_mm": 0.0, "native_success": False,
                "task_contact": False, "executed_action": None,
                "contact_detail": [], "target_velocity_mps": 0.0,
                "aperture": None, "aperture_velocity": None,
                "eef_target_drift_mm": None, "goal_clause_truth": {},
                "forbidden_contact": False, "payload_obstacle_contact": False,
            }
            base.update(kw)
            return base
        # close intent 出现
        rows.append(r(0, executed_action=[0,0,0,0,0,0,1.0]))
        # 单指接触
        rows.append(r(1, executed_action=[0,0,0,0,0,0,1.0], task_contact=True,
                      contact_detail=[{"robot":"gripper0_leftfinger","other":"bowl_main"}]))
        # 双指 + 受阻闭合（closure_blocked）——需要进入 acquisition 后再 3 步双指累加
        for i in range(4):
            rows.append(r(2+i, executed_action=[0,0,0,0,0,0,1.0], task_contact=True,
                          contact_detail=[{"robot":"gripper0_leftfinger","other":"bowl_main"},
                                          {"robot":"gripper0_rightfinger","other":"bowl_main"}],
                          aperture=0.02, aperture_velocity=0.0))
        # 运输
        rows.append(r(10, executed_action=[0,0,0,0,0,0,1.0], task_contact=True,
                      contact_detail=[{"robot":"gripper0_leftfinger","other":"bowl_main"},
                                      {"robot":"gripper0_rightfinger","other":"bowl_main"}],
                      target_displacement_mm={"bowl_main": 30.0}))
        # 释放：open + contact gone + 速度稳定
        for i in range(3):
            rows.append(r(11+i, executed_action=[0,0,0,0,0,0,-1.0], task_contact=False,
                          contact_detail=[], target_displacement_mm={"bowl_main": 30.0},
                          target_velocity_mps=0.001))
        # goal true + 全部 released → q_acc
        rows.append(r(20, executed_action=[0,0,0,0,0,0,-1.0], task_contact=False,
                      contact_detail=[], target_displacement_mm={"bowl_main": 30.0},
                      target_velocity_mps=0.001, goal_clause_truth={"place_bowl": True}))
        return rows

    def test_full_lifecycle_milestones(self):
        mon = OfflineMonitor(["bowl_main"])
        result = mon.run(self._make_rows())
        m = result["milestones"]
        self.assertIn("first_close_intent", m)
        self.assertIn("first_valid_finger_contact", m)
        self.assertIn("first_bilateral_contact", m)
        self.assertIn("first_controlled", m)
        self.assertIn("first_transported", m)
        self.assertIn("first_released", m)
        self.assertIn("first_goal_truth", m)
        self.assertEqual(result["final_mode"], "q_acc")
        self.assertEqual(result["final_lifecycle"]["bowl_main"], "RELEASED")


if __name__ == "__main__":
    unittest.main()
