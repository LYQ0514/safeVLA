"""确定性启动器测试（Commit 1，裁决 §一/§二）。

覆盖：
- 硬门禁：缺 XLA_FLAGS / 缺确定性 flag → DETERMINISM_CONFIGURATION_MISSING；
- 正确配置通过；
- PnsgPolicy 默认语义不被静默改变（xla_flags=None 仍是非确定配置）；
- determinism_gate_record 字段完整。
"""

from __future__ import annotations

import os
import sys
import unittest

ROOT = "/data/safeVLA"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from src.pnsg.deterministic_factory import (  # noqa: E402
    DETERMINISTIC_XLA_FLAGS, GATE_VERSION,
    DETERMINISM_CONFIGURATION_MISSING, DeterminismGateError,
    _check_xla_flags, determinism_gate_record,
)


class TestCheckXlaFlags(unittest.TestCase):
    def test_missing_xla_flags_raises(self):
        with self.assertRaises(DeterminismGateError) as cm:
            _check_xla_flags(None)
        self.assertIn(DETERMINISM_CONFIGURATION_MISSING, str(cm.exception))

    def test_missing_deterministic_flag_raises(self):
        # 只有部分 flag
        with self.assertRaises(DeterminismGateError) as cm:
            _check_xla_flags("--xla_gpu_autotune_level=0")
        self.assertIn(DETERMINISM_CONFIGURATION_MISSING, str(cm.exception))
        self.assertIn("--xla_gpu_exclude_nondeterministic_ops", str(cm.exception))

    def test_empty_string_raises(self):
        with self.assertRaises(DeterminismGateError):
            _check_xla_flags("")

    def test_full_config_passes(self):
        # 不抛异常即通过
        _check_xla_flags(DETERMINISTIC_XLA_FLAGS)

    def test_full_config_with_extra_passes(self):
        _check_xla_flags(DETERMINISTIC_XLA_FLAGS + " --xla_force_host_platform_device_count=1")


class TestPnsgPolicyDefaultUnchanged(unittest.TestCase):
    def test_default_xla_flags_is_none(self):
        """PnsgPolicy 默认 xla_flags=None，不被确定性 factory 静默改变。"""
        from src.pnsg.policy_adapter import PnsgPolicy
        import inspect
        sig = inspect.signature(PnsgPolicy.__init__)
        self.assertIn("xla_flags", sig.parameters)
        self.assertIsNone(sig.parameters["xla_flags"].default,
                          "PnsgPolicy 默认 xla_flags 必须仍为 None（不静默设确定性）")


class TestGateRecord(unittest.TestCase):
    def test_record_fields(self):
        gate = {
            "gate_version": GATE_VERSION, "passed": True,
            "xla_flags": DETERMINISTIC_XLA_FLAGS,
            "xla_flags_at_import": DETERMINISTIC_XLA_FLAGS,
            "xla_flags_env": DETERMINISTIC_XLA_FLAGS,
            "jax_version": "0.5.3", "jaxlib_version": "0.5.3",
            "xla_client_version": "0.5.3", "default_backend": "gpu",
            "devices": ["CudaDevice(id=0)"],
        }
        rec = determinism_gate_record(gate)
        self.assertEqual(rec["determinism_gate_version"], GATE_VERSION)
        self.assertTrue(rec["determinism_gate_passed"])
        self.assertIn("--xla_gpu_autotune_level=0", rec["xla_flags"])
        self.assertIn("--xla_gpu_exclude_nondeterministic_ops", rec["xla_flags"])


class TestCreateDeterministicPolicyIntegration(unittest.TestCase):
    """集成测试：真实 worker + 硬门禁（慢，~1min）。"""

    def test_create_policy_and_gate(self):
        from src.pnsg.deterministic_factory import create_deterministic_policy
        policy, gate = create_deterministic_policy()
        try:
            self.assertTrue(gate["passed"])
            self.assertEqual(gate["gate_version"], GATE_VERSION)
            self.assertIn("--xla_gpu_autotune_level=0", gate["xla_flags_at_import"])
            self.assertIn("--xla_gpu_exclude_nondeterministic_ops", gate["xla_flags_at_import"])
            self.assertEqual(gate["default_backend"], "gpu")
        finally:
            policy.close()


if __name__ == "__main__":
    unittest.main()
