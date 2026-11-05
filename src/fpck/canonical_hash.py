"""Canonical trace hash（semantic_trace_sha256）。

对 raw_events.jsonl 每行做确定性规范化，排除易变字段（wall_time 等），
保留事件语义字段，得到可跨 run 对比的 semantic hash。

排除字段：wall_time, logged_at, 绝对输出路径, runtime duration, process id, host id。
规范化：JSON key 排序、固定 UTF-8、浮点沿用日志序列化值、每行统一分隔符。

版本：fpck-event-v1
"""

from __future__ import annotations

import hashlib
import json

SEMANTIC_HASH_VERSION = "fpck-event-v1"

# 排除的易变字段
EXCLUDED_FIELDS = {
    "wall_time", "logged_at", "output_path", "absolute_path",
    "runtime_duration", "process_id", "host_id", "pid",
}


def normalize_row(row: dict) -> str:
    """单行事件 → 规范化字符串（排除易变字段 + key 排序）。"""
    cleaned = {k: v for k, v in row.items() if k not in EXCLUDED_FIELDS}
    # key 排序 + 紧凑 JSON + 确保 UTF-8
    return json.dumps(cleaned, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def semantic_trace_hash(raw_log_path: str) -> str:
    """计算 raw_events.jsonl 的 semantic hash。

    每行规范化后以换行连接，整体 sha256。
    """
    lines = []
    with open(raw_log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            lines.append(normalize_row(row))
    content = "\n".join(lines).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def raw_trace_hash(raw_log_path: str) -> str:
    """原始 trace hash（含 wall_time，用于完整性校验）。"""
    with open(raw_log_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()
