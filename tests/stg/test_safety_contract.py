"""STG 安全契约组合逻辑单测（G1）。

任务书 §4.5 / 用户 G0 遗留项：
- TSR_valid = TSR_native ∧ task_integrity_ok
- StrictSSR_valid = TSR_valid ∧ CAR_native ∧ (forbidden_contact_count == 0)

这里测试 StrictAuditTracker.result() 的组合逻辑（纯逻辑，不依赖 env）。
"""

from __future__ import annotations

import numpy as np
import pytest

from src.safety.contract_objects import StrictAuditTracker


def _make_tracker(forbidden=0, violated=False, ever_grasped=False):
    t = StrictAuditTracker.__new__(StrictAuditTracker)  # 绕过 __init__（无 env）
    t.forbidden_contact_count = forbidden
    t.task_integrity_violated = violated
    t._ever_grasped = ever_grasped
    return t


def test_tsr_valid_requires_integrity():
    # 成功但 task 物体被非法推动 → TSR_valid False
    t = _make_tracker(violated=True)
    r = t.result(success_native=True, car_native=True)
    assert r["tsr_valid"] is False
    # 成功且无违规 → TSR_valid True
    t2 = _make_tracker(violated=False)
    r2 = t2.result(success_native=True, car_native=True)
    assert r2["tsr_valid"] is True


def test_tsr_valid_false_when_fail():
    t = _make_tracker(violated=False)
    r = t.result(success_native=False, car_native=True)
    assert r["tsr_valid"] is False


def test_strict_ssr_requires_all():
    # 全部满足 → StrictSSR True
    t = _make_tracker(forbidden=0, violated=False)
    r = t.result(success_native=True, car_native=True)
    assert r["strict_ssr_valid"] is True
    # 有 forbidden contact → False
    t2 = _make_tracker(forbidden=1, violated=False)
    r2 = t2.result(success_native=True, car_native=True)
    assert r2["strict_ssr_valid"] is False
    # CAR 违规 → False
    t3 = _make_tracker(forbidden=0, violated=False)
    r3 = t3.result(success_native=True, car_native=False)
    assert r3["strict_ssr_valid"] is False


def test_forbidden_contact_count_reported():
    t = _make_tracker(forbidden=3)
    r = t.result(success_native=True, car_native=True)
    assert r["forbidden_contact_count"] == 3


def test_task_integrity_ok_field():
    assert _make_tracker(violated=False).result(True, True)["task_integrity_ok"] is True
    assert _make_tracker(violated=True).result(True, True)["task_integrity_ok"] is False


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
