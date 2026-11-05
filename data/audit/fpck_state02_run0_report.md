# FPCK/CKG E1a State02 run0 验收报告

> 日期：2026-11-05。判定：**事件级标签正确，验收通过，可进入 run1**

## 结果

| 项 | 值 |
|---|---|
| case / run | State02 / run0 |
| final_e1_label | **TASK_INTEGRITY_FAIL**（q_bad / FORBIDDEN_CONTACT） |
| implementation_status | EVENT_LEVEL_V1_2 |
| legacy_diagnostic_used_for_label | false |
| identity_verified | true |
| wall_s | 650s |

## 事件链（关键机制）

```
settled initial (q_init)
→ PREGRASP: Joint-PD 路径执行（3692 个 physics_substep 事件）
→ HANDOFF: q_init（预抓取合法，handoff eligible）
→ REENTRY: 冻结 π0.5 rollout
→ control_step=19: mug ↔ robot0_link7 碰撞（FORBIDDEN_CONTACT）
→ q_bad (event_index=3712)
→ TASK_INTEGRITY_FAIL
```

## 验收检查（逐项）

| 检查项 | 结果 |
|---|---|
| 身份 identity_verified | ✅ true（q_start_hash / selected_path_hash / handoff_env_hash / terminal_env_hash 均有值） |
| 事件总数非空 | ✅ 4243 |
| event_index 全局连续（不归零） | ✅ 0→4242 严格递增 |
| 首事件 mode=q_init | ✅ PREGRASP physics_substep |
| 自动机跨 PREGRASP→REENTRY 不重置 | ✅ HANDOFF(3692)→REENTRY(3693) 连续 |
| HANDOFF_PRE_ACTION_CHECK 存在 | ✅ 1 次（goal 全 False、native False、mode=q_init） |
| 首次 q_bad 位置 | ✅ REENTRY control_step=19，FORBIDDEN_CONTACT |
| 主违规码 | ✅ FORBIDDEN_CONTACT（priority 0） |
| raw_trace_sha256 | ✅ 20663eae... |
| summary 引用真实文件 | ✅ |

## 与历史 State02 的关系（裁决允许的差异）

历史 State02 是「native success@step0」（RRT 把 soup 撞进 basket 形成假成功）。
本次 run0 是 **REENTRY 阶段 mug 碰撞 → FORBIDDEN_CONTACT → TASK_INTEGRITY_FAIL**。

裁决明确：「State02 run0 不要求复现假成功，任何符合事件链的结果都接受」。
本次结果是符合事件链的合法结果：预抓取合法 → 交还 → π0.5 rollout 中 mug 碰撞。

关键：这证明**自动机能正确识别 REENTRY 阶段的 forbidden contact 并进入 q_bad**，
不依赖历史假成功路径。

## 文件清单

| 文件 | 状态 |
|---|---|
| raw_events.jsonl | ✅ 4243 事件 |
| snapshots/handoff.json | ✅ 存在 |
| run_manifest.json | ✅ 存在 |
| run_summary.json | ✅ 存在 |
| terminal snapshot | ✅ terminal_env_hash 已记录（MuJoCo 状态 hash，未存大二进制） |

## 基础设施错误记录

- run0 首次启动因 `set_arm_q` 缺失 import 而 RUNTIME_FAILURE（基础设施错误 1/2），
  已修复（commit 17e6f35）；
- 重跑成功，无第二次基础设施错误。

## 结论

State02 run0 验收通过：事件级标签正确、自动机连续、身份完整、trace hash 匹配。
按裁决，可继续 State02 run1（先汇报验收表，不自动启动）。
