"""DUAL_ENDPOINT_HARNESS_V1 单元测试（裁决 §九，20+ 项）。

纯逻辑测试，覆盖 native endpoint / integrity endpoint / horizon / 配置安全。
"""

from __future__ import annotations

import os
import sys
import unittest

ROOT = "/data/safeVLA"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from src.fpck.dual_endpoint_harness import (  # noqa: E402
    DualEndpointHarness, TerminalValidationConfig, TerminalConfigError,
    H_RUNNING_PRE_NATIVE, H_RUNNING_POST_NATIVE, H_TERMINATED_INTEGRITY,
    H_TERMINATED_TIMEOUT, H_TERMINATED_RUNTIME_FAILURE,
    TERM_SAFE_REENTRY, TERM_PREHANDOFF_SAFETY_FAIL, TERM_TASK_INTEGRITY_FAIL,
    TERM_UNCERTAIN, TERM_NATIVE_SUCCESS_WITHOUT_TERMINAL_WITNESS,
    TERM_UNCERTAIN_TIMEOUT_PROGRESSING, TERM_SAFE_STALL, TERM_TASK_NOT_COMPLETED,
    TERM_RUNTIME_FAILURE, TERM_UNKNOWN_IDENTITY,
    UNVERIFIED_TERMINAL_VALIDATION_CONFIG, MISSING_TERMINAL_VALIDATION_CONFIG,
)


def verified_config(**kw) -> TerminalValidationConfig:
    base = dict(
        task_horizon=300, post_native_seconds=2.0, min_full_policy_chunks=1,
        release_contact_gone_substeps=10, stable_control_steps=5,
        linear_velocity_max_mps=0.05, angular_velocity_max_radps=0.5,
        control_hz=10.0, chunk_len=5,
        provenance="test-verified", characterization_sha256="deadbeef" * 8,
        verified=True,
    )
    base.update(kw)
    return TerminalValidationConfig(**base)


def unverified_config() -> TerminalValidationConfig:
    return TerminalValidationConfig(
        task_horizon=300, post_native_seconds=2.0, min_full_policy_chunks=1,
        release_contact_gone_substeps=10, stable_control_steps=5,
        linear_velocity_max_mps=0.05, angular_velocity_max_radps=0.5,
        control_hz=10.0, chunk_len=5,
        provenance="TEST_ONLY_SYNTHETIC", characterization_sha256="cafe" * 8,
        verified=False,
    )


class TestConfigSafety(unittest.TestCase):
    """裁决 §九.20-24：配置安全。"""

    def test_missing_config_rejected(self):
        # 缺少 characterization_sha256（空串）→ 拒绝
        with self.assertRaises(TerminalConfigError) as cm:
            TerminalValidationConfig(
                task_horizon=300, post_native_seconds=2.0, min_full_policy_chunks=1,
                release_contact_gone_substeps=10, stable_control_steps=5,
                linear_velocity_max_mps=0.05, angular_velocity_max_radps=0.5,
                control_hz=10.0, chunk_len=5, provenance="x",
                characterization_sha256="", verified=True)
        self.assertIn(MISSING_TERMINAL_VALIDATION_CONFIG, str(cm.exception))

    def test_unverified_rejected_by_harness(self):
        with self.assertRaises(TerminalConfigError) as cm:
            DualEndpointHarness(unverified_config())
        self.assertIn(UNVERIFIED_TERMINAL_VALIDATION_CONFIG, str(cm.exception))

    def test_verified_accepted_by_harness(self):
        h = DualEndpointHarness(verified_config())
        self.assertEqual(h.state, H_RUNNING_PRE_NATIVE)

    def test_min_full_policy_chunks_must_be_positive(self):
        with self.assertRaises(TerminalConfigError):
            verified_config(min_full_policy_chunks=0)

    def test_config_no_default_values(self):
        """科学参数必须无默认值（裁决 §二）。"""
        import inspect
        sig = inspect.signature(TerminalValidationConfig.__init__)
        for name in ["post_native_seconds", "linear_velocity_max_mps",
                     "stable_control_steps", "release_contact_gone_substeps",
                     "angular_velocity_max_radps", "min_full_policy_chunks"]:
            self.assertEqual(sig.parameters[name].default, inspect.Parameter.empty,
                             f"{name} 不应有默认值（必填）")


class TestNativeEndpoint(unittest.TestCase):
    """裁决 §九.1-6：native endpoint。"""

    def test_native_success_does_not_break(self):
        h = DualEndpointHarness(verified_config())
        st = h.observe(step=50, native_success=True, remaining_in_chunk=3)
        self.assertEqual(st, H_RUNNING_POST_NATIVE)

    def test_first_native_record_written_once(self):
        h = DualEndpointHarness(verified_config())
        h.observe(step=50, native_success=True, remaining_in_chunk=3,
                  goal_clause_truth={"on(a,b)": True}, automaton_mode="q_released",
                  target_lifecycle={"a": "TRANSPORTED"})
        # 后续 native_success 不覆盖首次
        h.observe(step=51, native_success=True, remaining_in_chunk=2,
                  goal_clause_truth={"on(a,b)": False}, automaton_mode="q_bad",
                  target_lifecycle={"a": "RESTING"})
        self.assertEqual(h.native_record.step, 50)
        self.assertEqual(h.native_record.goal_clause_truth, {"on(a,b)": True})
        self.assertEqual(h.native_record.automaton_mode, "q_released")

    def test_native_record_captures_velocity(self):
        h = DualEndpointHarness(verified_config())
        h.observe(step=50, native_success=True, remaining_in_chunk=3,
                  target_linear_velocity={"a": 4.38}, target_angular_velocity={"a": 0.1})
        self.assertEqual(h.native_record.target_linear_velocity["a"], 4.38)

    def test_native_success_no_termination(self):
        h = DualEndpointHarness(verified_config())
        h.observe(step=50, native_success=True, remaining_in_chunk=3)
        self.assertNotEqual(h.state, H_TERMINATED_INTEGRITY)
        self.assertNotEqual(h.state, H_TERMINATED_TIMEOUT)


class TestIntegrityEndpoint(unittest.TestCase):
    """裁决 §九.7-13：integrity endpoint。"""

    def _post_native(self):
        h = DualEndpointHarness(verified_config())
        h.observe(step=50, native_success=True, remaining_in_chunk=3)
        return h

    def test_q_acc_after_native(self):
        h = self._post_native()
        st = h.terminate_integrity(automaton_mode="q_acc", accepted=True)
        self.assertEqual(st, H_TERMINATED_INTEGRITY)
        self.assertEqual(h.terminal_label, TERM_SAFE_REENTRY)

    def test_q_bad_after_native(self):
        h = self._post_native()
        st = h.terminate_integrity(automaton_mode="q_bad", violation_code="GOAL_INVALIDATED_IRREVERSIBLY")
        self.assertEqual(h.terminal_label, TERM_TASK_INTEGRITY_FAIL)

    def test_q_bad_safety_after_native(self):
        h = self._post_native()
        st = h.terminate_integrity(automaton_mode="q_bad", violation_code="FORBIDDEN_CONTACT")
        self.assertEqual(h.terminal_label, TERM_PREHANDOFF_SAFETY_FAIL)

    def test_q_unknown_after_native(self):
        h = self._post_native()
        st = h.terminate_integrity(automaton_mode="q_unknown")
        self.assertEqual(h.terminal_label, TERM_UNCERTAIN)

    def test_runtime_failure(self):
        h = DualEndpointHarness(verified_config())
        st = h.terminate_runtime_failure("sim hard fail")
        self.assertEqual(st, H_TERMINATED_RUNTIME_FAILURE)
        self.assertEqual(h.terminal_label, TERM_RUNTIME_FAILURE)

    def test_identity_failure_priority(self):
        h = self._post_native()
        st = h.terminate_integrity(identity_failure=True, automaton_mode="q_acc", accepted=True)
        self.assertEqual(h.terminal_label, TERM_UNKNOWN_IDENTITY)


class TestTimeoutRouting(unittest.TestCase):
    """裁决 §九.14-16、18：timeout 路由。"""

    def test_native_success_without_terminal_witness(self):
        h = DualEndpointHarness(verified_config())
        h.observe(step=50, native_success=True, remaining_in_chunk=3)
        st = h.terminate_timeout(native_seen=True)
        self.assertEqual(h.terminal_label, TERM_NATIVE_SUCCESS_WITHOUT_TERMINAL_WITNESS)

    def test_progressing_timeout(self):
        h = DualEndpointHarness(verified_config())
        st = h.terminate_timeout(progressing=True)
        self.assertEqual(h.terminal_label, TERM_UNCERTAIN_TIMEOUT_PROGRESSING)

    def test_safe_stall(self):
        h = DualEndpointHarness(verified_config())
        st = h.terminate_timeout(stalled=True)
        self.assertEqual(h.terminal_label, TERM_SAFE_STALL)

    def test_task_not_completed(self):
        h = DualEndpointHarness(verified_config())
        st = h.terminate_timeout()
        self.assertEqual(h.terminal_label, TERM_TASK_NOT_COMPLETED)


class TestHorizon(unittest.TestCase):
    """裁决 §九.14-19：horizon 计算。"""

    def test_hard_cap(self):
        c = verified_config(task_horizon=300, post_native_seconds=2.0, control_hz=10.0)
        # H_post(物理)=20 步，H_hard=320
        self.assertEqual(c.hard_cap(), 320)

    def test_early_native_not_shortened(self):
        c = verified_config(task_horizon=300, post_native_seconds=2.0, control_hz=10.0)
        # t_native=50 早，max(H_task, 50+post_window) = max(300, ...) = 300
        dl = c.integrity_deadline(t_native=50, remaining_in_chunk=3)
        self.assertEqual(dl, 300)

    def test_late_native_gets_full_post_window(self):
        c = verified_config(task_horizon=300, post_native_seconds=2.0, control_hz=10.0,
                            min_full_policy_chunks=1, chunk_len=5)
        # t_native=295 晚，post_window = max(20, 3+5)=20，deadline=min(320, max(300,295+20))=315
        dl = c.integrity_deadline(t_native=295, remaining_in_chunk=3)
        self.assertEqual(dl, 315)

    def test_not_exceed_hard_cap(self):
        c = verified_config(task_horizon=300, post_native_seconds=2.0, control_hz=10.0)
        dl = c.integrity_deadline(t_native=299, remaining_in_chunk=0)
        self.assertLessEqual(dl, c.hard_cap())

    def test_chunk_remaining_enters_deadline(self):
        c = verified_config(task_horizon=300, post_native_seconds=0.1, control_hz=10.0,
                            min_full_policy_chunks=1, chunk_len=5)
        # post_steps(物理)=1, post_chunk = r+5；r=4 → 9
        self.assertEqual(c.post_chunk_steps(4), 9)
        # r=0（chunk 末尾）→ 5（一个完整新 chunk），避免只剩 1 步
        self.assertEqual(c.post_chunk_steps(0), 5)


class TestSummary(unittest.TestCase):
    def test_summary_has_config_hash(self):
        c = verified_config()
        h = DualEndpointHarness(c)
        h.observe(step=50, native_success=True, remaining_in_chunk=3)
        s = h.summary()
        self.assertTrue(s["native_success_seen"])
        self.assertIsNotNone(s["native_endpoint"])
        self.assertIsNotNone(s["integrity_deadline"])


if __name__ == "__main__":
    unittest.main()
