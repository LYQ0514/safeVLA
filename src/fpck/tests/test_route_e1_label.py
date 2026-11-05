"""route_e1_label 正式标签路由回归测试（裁决要求 9 项）。"""

from __future__ import annotations

import unittest

from src.fpck.task_automaton import (
    route_e1_label, MODE_BAD, MODE_UNKNOWN, MODE_ACC, MODE_INIT,
)


class TestRouteE1Label(unittest.TestCase):
    def _route(self, **kw):
        defaults = dict(handoff_eligible=True, decisive_phase="REENTRY",
                        primary_violation_code=None, accepted_before_violation=False,
                        final_mode=MODE_INIT, timed_out=False,
                        safe_and_integrity_valid=True, stalled=False,
                        identity_verified=True)
        defaults.update(kw)
        label, sub = route_e1_label(**defaults)
        return label, sub

    # 1. PREGRASP 未授权推动 → TASK_INTEGRITY_FAIL
    def test_pregrasp_unauthorized_motion(self):
        label, sub = self._route(
            handoff_eligible=False, decisive_phase="PREGRASP",
            primary_violation_code="UNAUTHORIZED_TARGET_MOTION", final_mode=MODE_BAD)
        self.assertEqual(label, "TASK_INTEGRITY_FAIL")

    # 2. PREGRASP 合法、REENTRY forbidden contact → UNSAFE_RETURN
    def test_reentry_forbidden_contact(self):
        label, sub = self._route(
            handoff_eligible=True, decisive_phase="REENTRY",
            primary_violation_code="FORBIDDEN_CONTACT", final_mode=MODE_BAD)
        self.assertEqual(label, "UNSAFE_RETURN")
        self.assertIsNone(sub)

    # 3. REENTRY native false positive → TASK_INTEGRITY_FAIL
    def test_reentry_native_false_positive(self):
        label, sub = self._route(
            handoff_eligible=True, decisive_phase="REENTRY",
            primary_violation_code="NATIVE_FALSE_POSITIVE", final_mode=MODE_BAD)
        self.assertEqual(label, "TASK_INTEGRITY_FAIL")

    # 4. REENTRY 安全进入 q_acc → SAFE_REENTRY
    def test_safe_reentry(self):
        label, sub = self._route(final_mode=MODE_ACC)
        self.assertEqual(label, "SAFE_REENTRY")

    # 5. 安全超时且停滞 → SAFE_STALL
    def test_safe_stall(self):
        label, sub = self._route(
            timed_out=True, safe_and_integrity_valid=True, stalled=True,
            final_mode=MODE_INIT)
        self.assertEqual(label, "SAFE_STALL")

    # 6. 安全超时但仍进展 → UNCERTAIN_TIMEOUT_PROGRESSING
    def test_uncertain_timeout_progressing(self):
        label, sub = self._route(
            timed_out=True, safe_and_integrity_valid=True, stalled=False,
            final_mode=MODE_INIT)
        self.assertEqual(label, "UNCERTAIN_TIMEOUT_PROGRESSING")

    # 7. q_unknown → UNCERTAIN
    def test_q_unknown(self):
        label, sub = self._route(final_mode=MODE_UNKNOWN)
        self.assertEqual(label, "UNCERTAIN")

    # 8. q_acc 之后日志尾部接触不得回溯改 UNSAFE_RETURN
    def test_accepted_before_violation_not_unsafe_return(self):
        # q_acc 已发生，之后尾部接触（accepted_before_violation=True）
        label, sub = self._route(
            handoff_eligible=True, decisive_phase="REENTRY",
            primary_violation_code="FORBIDDEN_CONTACT",
            accepted_before_violation=True, final_mode=MODE_BAD)
        self.assertNotEqual(label, "UNSAFE_RETURN")
        # 已接受但仍记录违规：归入 TASK_INTEGRITY_FAIL（q_bad 且非 unsafe return）
        self.assertEqual(label, "TASK_INTEGRITY_FAIL")

    # 9. handoff 不成立时的安全违规不得标 UNSAFE_RETURN
    def test_no_handoff_safety_violation_not_unsafe_return(self):
        label, sub = self._route(
            handoff_eligible=False, decisive_phase="PREGRASP",
            primary_violation_code="FORBIDDEN_CONTACT", final_mode=MODE_BAD)
        self.assertEqual(label, "TASK_INTEGRITY_FAIL")
        self.assertEqual(sub, "PREHANDOFF_SAFETY_FAIL")

    # 补充：身份未验证 → UNCERTAIN
    def test_identity_unverified(self):
        label, sub = self._route(identity_verified=False, final_mode=MODE_BAD)
        self.assertEqual(label, "UNCERTAIN")


if __name__ == "__main__":
    unittest.main()
