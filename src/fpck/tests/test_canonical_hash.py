"""canonical_hash 单元测试。"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from src.fpck.canonical_hash import (
    semantic_trace_hash, raw_trace_hash, normalize_row,
    SEMANTIC_HASH_VERSION, EXCLUDED_FIELDS,
)


class TestCanonicalHash(unittest.TestCase):
    def test_wall_time_excluded(self):
        # 同一事件，wall_time 不同 → semantic hash 一致
        row1 = {"event_index": 0, "phase": "REENTRY", "wall_time": 100.0,
                "goal_truth": {"a": True}}
        row2 = {"event_index": 0, "phase": "REENTRY", "wall_time": 999.0,
                "goal_truth": {"a": True}}
        self.assertEqual(normalize_row(row1), normalize_row(row2))

    def test_semantic_field_included(self):
        # 事件语义不同 → semantic hash 不同
        row1 = {"event_index": 0, "phase": "REENTRY", "forbidden": True}
        row2 = {"event_index": 0, "phase": "REENTRY", "forbidden": False}
        self.assertNotEqual(normalize_row(row1), normalize_row(row2))

    def test_key_order_normalized(self):
        # key 顺序不同 → 规范化后一致（sort_keys）
        row1 = {"b": 1, "a": 2, "wall_time": 3}
        row2 = {"wall_time": 3, "a": 2, "b": 1}
        self.assertEqual(normalize_row(row1), normalize_row(row2))

    def test_semantic_trace_hash_stable_across_wall_time(self):
        with tempfile.TemporaryDirectory() as d:
            p1 = os.path.join(d, "t1.jsonl")
            p2 = os.path.join(d, "t2.jsonl")
            # 相同事件，不同 wall_time
            for path, wt in [(p1, 1.0), (p2, 999.0)]:
                with open(path, "w") as f:
                    for i in range(3):
                        f.write(json.dumps({"event_index": i, "phase": "X",
                                            "wall_time": wt + i}) + "\n")
            h1 = semantic_trace_hash(p1)
            h2 = semantic_trace_hash(p2)
            self.assertEqual(h1, h2)
            # 但 raw hash 不同（含 wall_time）
            self.assertNotEqual(raw_trace_hash(p1), raw_trace_hash(p2))

    def test_excluded_fields_set(self):
        self.assertIn("wall_time", EXCLUDED_FIELDS)
        self.assertIn("pid", EXCLUDED_FIELDS)
        self.assertEqual(SEMANTIC_HASH_VERSION, "fpck-event-v1")


if __name__ == "__main__":
    unittest.main()
