# FPCK/CKG E1a 门槛 3 审计：独立 goal evaluator 对照

> 日期：2026-11-05。判定：**PASS**

## 结果

| 项 | 值 |
|---|---|
| 配置 | safelibero_long / I / task0，init_state 0-19，frozen π0.5，无干预 |
| episodes | 20 |
| native success | 12/20 |
| evaluator vs native 分歧 | **0/20** |
| native success 的 final_mode | **12/12 全部 q_acc** ✅ |
| native fail 的 final_mode | q_init ×7 + q_progress ×1 |

## 核心结论

1. **门槛 3 PASS**：独立 goal evaluator（`_eval_predicate` 逐 clause 求值）与
   `env.check_success()` 在 20 条无干预 episode 上**完全一致（0 分歧）**；
2. **门槛 5 正向证据**：12 个 native success 全部正确进入 q_acc，无 NATIVE_FALSE_POSITIVE
   误判——自动机能识别正常成功。

## 发现并修复的实现 bug（如实记录）

### canonical_clause 规范化不一致

- **现象**：初版 gate3 里，所有 native success 的 final_mode 都是 q_bad
  （NATIVE_FALSE_POSITIVE），无一 q_acc；
- **根因**：`role_manifest.py` 的 goal_clauses 是 `('In', ['a','b'])`（大写谓词 +
  list args），而 `_eval_predicate` 用的是 `['in','a','b']`（小写谓词）。两者经
  `canonical_clause` 生成的 key 不同：
  - manifest → `"In ['a', 'b']"`（str(list) 形式）
  - evaluator → `"in a b"`（空格连接）
  - 导致 `achieved_mask` 的 key 永远匹配不上 `goal_clause_truth` 的 key；
- **修复**：`goal_evaluator.canonical_clause` 统一小写谓词 + 空格连接 args，
  并兼容 `('In', ['a','b'])` 与 `['in','a','b']` 两种输入（后者扁平化）；
- **验证**：修复后 12 个 native success 全部 q_acc，0 分歧。

## 诚实边界

1. gate3 是「无干预 baseline」，`TaskEvent` 只传 goal_clause_truth + native_success，
   **未传 contact/位移/见证信息**——因此 q_acc 的「CONTROLLED 见证」检查因
   target_displacement=0 默认通过，属聚焦 evaluator 对照的合理简化；
2. 完整见证（TASK_CONTACT/CONTROLLED/TRANSPORTED/RELEASED）的 q_acc 验证留给门槛 5，
   那里会用带完整接触/位姿记录的正规采集；
3. state17 的 q_progress（goal 部分为 true 但未全 true）合理，非 bug。

## 输出

- `data/audit/fpck_e1a_gate3.json`（20 episode 逐步 + 汇总）
- 脚本 `tools/fpck_e1a_gate3.py`
- 修复 `src/fpck/goal_evaluator.py`（canonical_clause 规范化）
