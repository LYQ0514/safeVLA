# FPCK/CKG E1a 事件级接入：TaskEventBridge + instrumented 循环

> 日期：2026-11-05。实现事件级自动机接入（裁决要求）。

## 实现

### 1. TaskEventBridge（src/fpck/task_event_bridge.py）

统一事件桥，职责：
- contact contract 语义化（monitor_contacts → forbidden / payload_obstacle）；
- goal-clause 独立求值（eval_goal_clauses）；
- manipulated-target pose/位移（相对 settled baseline）；
- 见证计数器（lift_hist / drift_hist / contact_gone / vel_hist）；
- 自动机连续更新（跨 PREGRASP→REENTRY 不重置）；
- append-only 原始事件输出（event_index 严格递增）；
- snapshot/restore 完整状态（自动机 + 见证计数器 + 基线）；
- 主违规码优先级（同 substep 并发事件按 VIOLATION_PRIORITY 定主码）。

### 2. instrumented 循环（src/fpck/instrumented_loops.py）

复制历史函数的最小执行内核，接入 bridge：
- `execute_joint_path_instrumented` ← execute_joint_path（Joint-PD，逐 substep 观测 +
  每 25 substep 聚合控制步）；
- `run_pv_branch_instrumented` ← run_branch 的 P-V 分支（OSC，逐控制步观测）。

源声明（复制不改行为）：
- source_file = tools/m1_p4_p1_reentry.py
- source_commit = 85a0d1c
- source_sha256 = a98e558f78ac0818834b0f06ba5b0247
- local_modification = instrumentation_only

## 测试（37 + 1 等价 = 38 tests 全过）

| 测试文件 | 用例数 | 覆盖 |
|---|---|---|
| test_task_automaton.py | 25 | 转移表/违规码/q_acc/标签绑定 |
| test_task_event_bridge.py | 12 | 自动机连续性、5mm 越界、goal+违规同现、snapshot 序列化、witness 恢复、append-only、trace hash、优先级、q_unknown 保护 |
| test_instrumentation_equivalence.py | 1 | 历史 vs instrumented 执行等价 |

## 关键验证结果

1. **自动机跨阶段不重置**：snapshot/restore 后 mode/lifecycle/achieved_mask 完全一致 ✅；
2. **5mm 越界**：首次 CONTROLLED 前 target 位移 >5mm 触发 UNAUTHORIZED_TARGET_MOTION ✅；
3. **goal true + 违规同现**：违规检查优先 → q_bad（FORBIDDEN_CONTACT）✅；
4. **执行等价**：Dmax_hist == Dmax_inst（1.766e-12），qpos max-err = 0.00e+00 ✅；
5. **q_unknown 保护**：q_unknown → UNCERTAIN，不会变 SAFE_REENTRY ✅。

## 实现状态

- implementation_status = EVENT_LEVEL_V1_2（事件级，非 LEGACY_DIAGNOSTIC）；
- classify_from_history（粗粒度）已不再用于正式标签，仅保留在 runner 骨架中且
  必须标记 LEGACY_DIAGNOSTIC。

## 下一步

重型运行前已满足裁决的全部前置测试（含执行等价）。按裁决「先跑 State02 run0，
检查完整事件日志正确后再继续」，下一步是 State02 run0 的重型执行 + 事件日志检查。
