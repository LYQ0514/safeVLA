"""G2 冻结病例表（手动选定，附选择理由与排除原因）。

从 B0 收集的 32 episode 中冻结配对病例。选择规则：
- 全部 S1 停滞病例（9 个，STG 核心靶点：CBF 卡住末端、无碰撞、任务未完成）；
- 全部 S3 VLA 无能病例（2 个，负对照：控制器无法挽回）；
- 3 个成功负对照（正常完成，验证 STG 不破坏正常轨迹）；
- S4 碰撞病例**不纳入 G2 配对**（碰撞是几何/接触问题，非环流/参考管理可改善，
  任务书 §6 明确 S4 是「审计限制而非绕障改进」）；纳入会稀释机制信号。

冻结后不再改动；测试集（G4）使用独立的 state/stream。
"""

from __future__ import annotations
import csv
import os

ROOT = "/data/safeVLA"
OUT = os.path.join(ROOT, "experiments", "stg_v1", "outputs", "G2_FROZEN_CASES.csv")

# (task_id, state_id, stream, 选择理由, 预期机制)
FROZEN = [
    # --- S1 停滞（STG 核心靶点）---
    (1, 0, 1, "S1: mug=0, tail_interv=0.87, CBF 卡住末端", "deadlock"),
    (1, 2, 0, "S1: mug=0, tail_interv=1.0, 完全停滞", "deadlock"),
    (1, 2, 1, "S1: mug=0, tail_interv=1.0, 完全停滞", "deadlock"),
    (1, 3, 0, "S1: mug=0, tail_interv=0.99, 停滞", "deadlock"),
    (1, 3, 1, "S1: mug=0, tail_interv=0.97, 停滞", "deadlock"),
    (1, 4, 0, "S1: mug=0, tail_interv=1.0, 完全停滞", "deadlock"),
    (1, 5, 1, "S1: mug=0, tail_interv=0.96, 停滞", "deadlock"),
    (1, 6, 0, "S1: mug=0, tail_interv=0.96, 停滞", "deadlock"),
    (1, 6, 1, "S1: mug=0, tail_interv=1.0, 完全停滞", "deadlock"),
    # --- S3 VLA 无能（负对照）---
    (1, 7, 0, "S3: mug=0, tail_interv=0.0, VLA 无有效动作", "vla_limit"),
    (1, 7, 1, "S3: mug=0, tail_interv=0.31, VLA 弱动作", "vla_limit"),
    # --- 成功负对照 ---
    (0, 0, 0, "NEG: success, mug=13mm, 正常完成", "success"),
    (0, 4, 0, "NEG: success, mug=7mm, 正常完成", "success"),
    (1, 5, 0, "NEG: success, mug=0, 正常完成", "success"),
]

# 排除原因（写入表）
EXCLUDED_REASON = {
    "S4_collision": "S4 碰撞病例：几何/接触问题，非环流/参考管理可改善（任务书 §6）",
}

with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["task_id", "state_id", "stream", "selection_reason", "expected_mechanism"])
    for row in FROZEN:
        w.writerow(row)

print(f"冻结病例表已写 {len(FROZEN)} 例 → {OUT}")
print(f"  S1 停滞: {sum(1 for r in FROZEN if r[4]=='deadlock')}")
print(f"  S3 负对照: {sum(1 for r in FROZEN if r[4]=='vla_limit')}")
print(f"  成功负对照: {sum(1 for r in FROZEN if r[4]=='success')}")
