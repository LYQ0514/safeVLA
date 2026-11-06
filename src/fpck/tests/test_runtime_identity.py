"""审计 D：运行时源码身份门禁测试。

验证（裁决 §四）：
- 8 个 fpck 模块从预期 install 路径（/data/safeVLA/src/）导入；
- 运行时文件 hash 与 git HEAD 一致（不是旧 install / 不同代码）；
- v1.2 与 v1.3 是不同模块（task_automaton.py vs task_automaton_acq.py），只在监视器逻辑上不同。
"""

from __future__ import annotations

import os
import sys
import unittest

ROOT = "/data/safeVLA"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from src.fpck.runtime_identity import (  # noqa: E402
    capture_runtime_identity, verify_runtime_identity, IDENTITY_MODULES,
)


class TestRuntimeIdentity(unittest.TestCase):
    def test_all_modules_imported_from_expected_install(self):
        ident = capture_runtime_identity()
        for mn in IDENTITY_MODULES:
            rec = ident["modules"][mn]
            self.assertTrue(rec.get("imported"), f"{mn} 未导入：{rec.get('import_error')}")
            self.assertTrue(rec.get("path_ok"),
                            f"{mn} 路径不属于预期 install：{rec.get('file')}")
            self.assertTrue(rec["file"].endswith(".py"),
                            f"{mn} 的 __file__ 不是 .py：{rec.get('file')}")

    def test_runtime_file_matches_git_head(self):
        ident = capture_runtime_identity()
        for v in ident["module_verify"]:
            self.assertTrue(v.get("matches_git_head"),
                            f"{v['module']} 运行时文件 hash 与 git HEAD 不一致")

    def test_verify_passes(self):
        ident = capture_runtime_identity()
        verify = verify_runtime_identity(ident)
        self.assertTrue(verify["pass"], f"审计 D 失败：{verify['failures']}")

    def test_v12_and_v13_are_distinct_modules(self):
        """v1.2 与 v1.3 是不同文件，只在监视器逻辑上不同。"""
        ident = capture_runtime_identity()
        f12 = ident["modules"]["src.fpck.task_automaton"]["file"]
        f13 = ident["modules"]["src.fpck.task_automaton_acq"]["file"]
        self.assertNotEqual(f12, f13)
        self.assertTrue(f12.endswith("task_automaton.py"))
        self.assertTrue(f13.endswith("task_automaton_acq.py"))


if __name__ == "__main__":
    unittest.main()
