"""手指 body 分类单元测试（fix：指尖 finger_joint*_tip 属于手指，不得误判为非手指）。"""

from __future__ import annotations

import os
import sys
import unittest

ROOT = "/data/safeVLA"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from src.fpck.task_event_bridge import classify_robot_body  # noqa: E402


class TestClassifyRobotBody(unittest.TestCase):
    def test_left_finger(self):
        self.assertEqual(classify_robot_body("gripper0_leftfinger"), "left")

    def test_left_finger_tip(self):
        # 关键回归：指尖属于手指
        self.assertEqual(classify_robot_body("gripper0_finger_joint1_tip"), "left")

    def test_right_finger(self):
        self.assertEqual(classify_robot_body("gripper0_rightfinger"), "right")

    def test_right_finger_tip(self):
        self.assertEqual(classify_robot_body("gripper0_finger_joint2_tip"), "right")

    def test_housing_is_non_finger(self):
        self.assertEqual(classify_robot_body("gripper0_right_gripper"), "non_finger")

    def test_eef_is_non_finger(self):
        self.assertEqual(classify_robot_body("gripper0_eef"), "non_finger")

    def test_robot_link_is_non_finger(self):
        self.assertEqual(classify_robot_body("robot0_link3"), "non_finger")
        self.assertEqual(classify_robot_body("robot0_right_hand"), "non_finger")


if __name__ == "__main__":
    unittest.main()
