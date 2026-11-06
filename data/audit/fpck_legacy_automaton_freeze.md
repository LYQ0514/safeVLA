# FPCK/CKG 旧自动机版本冻结声明

> 日期：2026-11-06。冻结有问题的自动机版本及其产生的标签，作为审计证据。

## 冻结版本

```text
AUTOMATON_V1_2_LEGACY_INVALID_FOR_GRASP_ACQUISITION
```

## 冻结内容（不可覆盖/删除）

1. **代码**：`src/fpck/task_automaton.py` 的 5mm 累计位移规则（commit `468c200` 审计 A +
   `a33ace5` 审计 B 时的版本）；
2. **原始轨迹**：`data/audit/fpck_e1a_positive_search/safelibero_spatial_t1_s1/raw_events.jsonl`；
3. **旧标签**：spatial/task1/state1 的 `UNAUTHORIZED_TARGET_MOTION`（降级为
   `INVALID_OLD_MONITOR_LABEL / PENDING_PHYSICS_SUBSTEP_REAUDIT`）。

## 已确认的结论

1. **审计 A**（纯逻辑）：人工合法抓取序列（单指→滑移6mm→双指闭合）被 5mm 规则误杀；
2. **审计 B**（真实轨迹）：spatial/task1/state1 的 step 37 位移 7.81mm 是「夹爪闭合
   自然顺应」（close 命令 step35 已下达、单指接触 step36、lift=0、双指 step39、抬升 step41），
   被 5mm 累计位移规则提前误判。

## 已撤回的结论

- 「spatial/task1 三个候选是 UNAUTHORIZED_TARGET_MOTION」；
- 「7/7 native success 都是假成功」；
- 「π0.5 缺乏干净抓取能力」；
- 「SAFE_REENTRY support not established 是可靠策略结论」。

## Gate 状态

| Gate | 状态 |
|---|---|
| Gate 4a | DISPUTED |
| Gate 5 | INVALID |
| Gate 6 | 暂保留（待 contact-contract 独立审计） |
| E1a | ENGINEERING AUDIT IN PROGRESS |
| E1b / value training / CKG 主实验 | BLOCKED |

## 下一步（按裁决顺序）

1. ✅ 冻结旧代码、原始轨迹、旧标签；
2. 新增 physics-substep 只读采集器；
3. 透明采集等价门禁；
4. 重采 spatial/task1/state1（带 physics substep + 原始 contact pair）；
5. 原始 contact-pair 真值表；
6. 审计 C（事件桥差分验证）+ D（运行时源码身份）；
7. 设计 GRASP_ACQUISITION；
8. 新旧监视器离线对比；
9. 重跑所有受影响候选。
