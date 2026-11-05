# FPCK/CKG S04 run0 验收报告

> 日期：2026-11-05。判定：S04 run0 验收通过，TASK_INTEGRITY_FAIL（门槛 4a 经验证据）。

## 结果

| 项 | 值 |
|---|---|
| case / run | S04 / run0 |
| final_e1_label | **TASK_INTEGRITY_FAIL** |
| violation_code | UNAUTHORIZED_TARGET_MOTION |
| decisive_phase | PREGRASP |
| termination | pregrasp_q_bad（正常终止，非失败） |
| handoff_eligible | false |
| identity_verified | true |
| wall_s | 720s |

## 事件链（门槛 4a 的核心证据）

```
settled (q_init)
→ PREGRASP: Joint-PD 路径执行
→ substep 40: soup 首次 >2mm（event 41）
→ substep 52: soup 首次 >5mm（event 54）→ UNAUTHORIZED_TARGET_MOTION → q_bad
→ substep 96: soup 首次 >10mm（event 99）
→ handoff 不成立（pregrasp_q_bad），不执行 P-V
→ TASK_INTEGRITY_FAIL
```

## 关键时间线字段（S04 特别记录）

| 字段 | 值 | 含义 |
|---|---|---|
| first_gt2mm | (event 41, substep 40) | soup 首次越 2mm |
| first_gt5mm | (event 54, substep 52) | soup 首次越 5mm → 违规触发 |
| first_gt10mm | (event 99, substep 96) | soup 首次越 10mm |
| first_task_contact | None | 从未有合法任务接触 |
| first_controlled_witness | None | 从未有 CONTROLLED 见证 |
| progress_history | [] | 无合法阶段推进 |
| last_progress_index | None | 无进展 |
| Dmax_pregrasp_mm | 153.85mm | soup 被推 153mm |

## 关键判定的正确性

1. **CONTROLLED witness 未成立**（first_controlled_witness=None）；
2. **soup 位移 >5mm**（substep 52）；
3. → 自动机正确判定 **UNAUTHORIZED_TARGET_MOTION**（substep 级即时驱动 q_bad）；
4. → handoff_eligible=false，**不执行 P-V**（裁决要求）；
5. → TASK_INTEGRITY_FAIL（正确标签，非 SAFE_STALL）。

这证明：**任务完整性违规（未授权目标运动）能够被事件自动机经验识别**——不是靠碰撞，而是靠「CONTROLLED 见证前的目标位移 >5mm」这条完整性规则。

## 与历史的对照

| 字段 | 历史 | 本次 run0 |
|---|---|---|
| 最终 soup 位移 | 247.8mm | 153.85mm（同量级） |
| 首次越过 5mm | substep ~150 | substep 52（更早） |
| 性质 | UNAUTHORIZED_TARGET_MOTION | UNAUTHORIZED_TARGET_MOTION ✅ |

具体数值不同（RRT 路径微差），但事件性质一致：**soup 在 CONTROLLED 见证前被大幅推动 → 完整性违规**。

## 对门槛 4 的裁决

按裁决拆分：

- **Gate 4a（unauthorized target motion）**：**PASS**——S04 run0 经验复现了「CONTROLLED
  见证前 soup 位移 >5mm → UNAUTHORIZED_TARGET_MOTION → q_bad」，事件自动机正确识别；
- **Gate 4b（native false positive）**：**not reproduced**——State02 三次 + S04 一次
  都未复现历史 step-0 假成功（native success 但自动机不接受）。

S04 的完整性证据**不能**描述为 State02 假成功的直接复现，这是两类不同的事件。

## 结论

S04 run0 验收通过：TASK_INTEGRITY_FAIL（UNAUTHORIZED_TARGET_MOTION），门槛 4a 获得
经验证据。按裁决，S04 run0 完成后停止。
