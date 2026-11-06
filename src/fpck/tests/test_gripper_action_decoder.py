"""夹爪动作解码器单元测试（裁决 §十二：实现 GRASP_ACQUISITION 前的动作语义门禁）。

纯逻辑测试（快，不依赖 env）：
- §4  deployed_close_intent 在命令下发时立即成立；
- §8  governor 改写夹爪动作时 nominal 与 deployed intent 不同；
- §9  snapshot/restore 后 intent 状态完全一致；
- sign 解码正确性（>0=close，<0=open，None=不臆造）；
- 迟滞（θ_enter > θ_exit + 连续步）；
- closing_response / closure_blocked 判定。

物理测试（需 env，单独文件 test_gripper_action_physics.py）：
- §1/2/3  open/close 命令方向、aperture 变化（探针已确认，此处用脚本化正/反例回归）；
- §5  closing_response 只在 aperture 实际减小时成立；
- §6/7  物体阻挡闭合 vs 空夹爪机械限位区分；
- §10 仅有 close 命令但无 finger contact 不能进入 Controlled（自动机层，见 test_task_automaton）。
"""

from __future__ import annotations

import os
import sys
import unittest

ROOT = "/data/safeVLA"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from src.fpck.gripper_action_decoder import (  # noqa: E402
    GripperActionDecode,
    CloseIntentTracker,
    decode_close_intent,
    compute_closing_response,
    compute_closure_blocked,
    DECODER_VERSION,
)


class TestDecodeCloseIntent(unittest.TestCase):
    def test_positive_is_close(self):
        self.assertTrue(decode_close_intent(1.0))
        self.assertTrue(decode_close_intent(0.1))
        self.assertTrue(decode_close_intent(0.0001))

    def test_negative_is_open(self):
        self.assertFalse(decode_close_intent(-1.0))
        self.assertFalse(decode_close_intent(-0.1))

    def test_none_is_not_close(self):
        # 命令缺失不得臆造意图
        self.assertFalse(decode_close_intent(None))

    def test_zero_boundary_not_close(self):
        # sign 边界在 0：>0 才 close，=0 不 close
        self.assertFalse(decode_close_intent(0.0))


class TestDeployedIntentImmediate(unittest.TestCase):
    """§4：deployed_close_intent 在命令下发时立即成立（不依赖 qpos）。"""

    def test_close_command_sets_intent_immediately(self):
        d = GripperActionDecode()
        d.deployed_gripper_command = 1.0
        d.deployed_close_intent = decode_close_intent(d.deployed_gripper_command)
        self.assertTrue(d.deployed_close_intent)

    def test_open_command_does_not_set_intent(self):
        d = GripperActionDecode()
        d.deployed_gripper_command = -1.0
        d.deployed_close_intent = decode_close_intent(d.deployed_gripper_command)
        self.assertFalse(d.deployed_close_intent)


class TestNominalVsDeployed(unittest.TestCase):
    """§8：governor 改写夹爪动作时，nominal 与 deployed intent 必须不同。"""

    def test_governor_override_detected(self):
        # π0.5 想 close（nominal=+1），governor 改为 open（deployed=-1）
        d = GripperActionDecode()
        d.policy_gripper_raw = 1.0
        d.nominal_close_intent = decode_close_intent(d.policy_gripper_raw)
        d.deployed_gripper_command = -1.0
        d.deployed_close_intent = decode_close_intent(d.deployed_gripper_command)
        self.assertTrue(d.nominal_close_intent)
        self.assertFalse(d.deployed_close_intent)
        self.assertNotEqual(d.nominal_close_intent, d.deployed_close_intent)


class TestHysteresis(unittest.TestCase):
    def test_close_latches_immediately(self):
        t = CloseIntentTracker()
        self.assertTrue(t.update(1.0, 0))
        self.assertTrue(t.latched)
        self.assertEqual(t.start_step, 0)

    def test_single_open_noise_does_not_clear(self):
        t = CloseIntentTracker()
        t.update(1.0, 0)
        # 单步 open 噪声（-0.5）不应清除锁定（n_exit_steps=2）
        self.assertTrue(t.update(-0.5, 1))
        self.assertTrue(t.latched)

    def test_continuous_open_clears(self):
        t = CloseIntentTracker()
        t.update(1.0, 0)
        t.update(-0.5, 1)
        # 连续 2 步 open → 清除
        self.assertFalse(t.update(-0.5, 2))
        self.assertFalse(t.latched)

    def test_deadband_holds_state(self):
        t = CloseIntentTracker()
        t.update(1.0, 0)
        # 死区 [-0.05, 0]：不累积 open，不清除
        self.assertTrue(t.update(-0.02, 1))
        self.assertTrue(t.latched)
        self.assertTrue(t.update(-0.02, 2))
        self.assertTrue(t.latched)


class TestClosingResponse(unittest.TestCase):
    def test_close_and_decreasing_is_closing(self):
        self.assertTrue(compute_closing_response(True, -0.001))

    def test_close_but_static_is_not_closing(self):
        self.assertFalse(compute_closing_response(True, 0.0))

    def test_no_intent_not_closing(self):
        self.assertFalse(compute_closing_response(False, -0.001))

    def test_none_velocity_not_closing(self):
        self.assertFalse(compute_closing_response(True, None))


class TestClosureBlocked(unittest.TestCase):
    def test_blocked_by_object(self):
        # close intent + 双指接触 + aperture 停住 + aperture 未到空夹爪限位
        self.assertTrue(compute_closure_blocked(True, 0.0, True, 0.03))

    def test_air_gripper_limit_not_blocked(self):
        # 空夹爪到机械限位（aperture≈0）≠ 物体阻挡
        self.assertFalse(compute_closure_blocked(True, 0.0, True, 0.003))

    def test_no_bilateral_contact_not_blocked(self):
        self.assertFalse(compute_closure_blocked(True, 0.0, False, 0.03))


class TestSnapshotRestore(unittest.TestCase):
    """§9：snapshot/restore 后 intent 状态完全一致。"""

    def test_decode_roundtrip(self):
        d = GripperActionDecode()
        d.policy_gripper_raw = 1.0
        d.deployed_gripper_command = 1.0
        d.controller_gripper_target = (0.5, -0.5)
        d.nominal_close_intent = True
        d.deployed_close_intent = True
        d.close_intent_latched = True
        d.close_intent_start_step = 7
        d.aperture = 0.02
        d.aperture_velocity = -0.0005
        d.closing_response = True
        d.closure_blocked_by_object = False

        r = GripperActionDecode.restore(d.snapshot())
        self.assertEqual(r.decoder_version, d.decoder_version)
        self.assertEqual(r.policy_gripper_raw, 1.0)
        self.assertEqual(r.deployed_gripper_command, 1.0)
        self.assertEqual(r.controller_gripper_target, (0.5, -0.5))
        self.assertTrue(r.nominal_close_intent)
        self.assertTrue(r.deployed_close_intent)
        self.assertTrue(r.close_intent_latched)
        self.assertEqual(r.close_intent_start_step, 7)
        self.assertAlmostEqual(r.aperture, 0.02)
        self.assertAlmostEqual(r.aperture_velocity, -0.0005)
        self.assertTrue(r.closing_response)
        self.assertFalse(r.closure_blocked_by_object)

    def test_tracker_roundtrip(self):
        t = CloseIntentTracker()
        t.update(1.0, 3)
        t.update(-0.5, 4)  # 累积 1 步 open
        t2 = CloseIntentTracker().restore(t.snapshot())
        self.assertEqual(t.latched, t2.latched)
        self.assertEqual(t.start_step, t2.start_step)
        self.assertEqual(t._open_run, t2._open_run)
        # 再走 1 步 open 应清除（与未恢复前行为一致）
        self.assertEqual(t.update(-0.5, 5), t2.update(-0.5, 5))


if __name__ == "__main__":
    unittest.main()
