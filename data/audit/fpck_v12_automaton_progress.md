# FPCK/CKG v1.2 + 任务自动机实现进度

> 日期：2026-11-05。方案 A 执行中。E1a 六项门槛，已完成 1-2 项，3-6 待 GPU rollout。

## v1.2 交付核对

| 项 | 状态 |
|---|---|
| v1.2 包 SHA-256 | ✅ `1261397039890980A6C4E8AC196972F2963E91256DB6D7157E0EA2AACF5378EA` |
| `task_automaton_spec_v1.2.md` | ✅ 已解压 + install 到 `docs/method/` |
| Python 3.8 兼容修复 | ✅ v1.2 已内置（`from __future__ import annotations` + src path 逻辑） |
| budget 下溢修复 | ✅ 已含（继承 v1.1） |
| 回归测试 | ✅ 5 tests OK |

## 任务自动机实现（方案 A 核心交付）

### 已完成

1. **`src/fpck/task_automaton.py`**：统一规范监视器
   - 状态 q = (mode, lifecycle, achieved_mask, violation_code)
   - mode 8 态（q_init/engaged/controlled/progress/released/acc/bad/unknown）
   - lifecycle 5 态（RESTING/TASK_CONTACT/CONTROLLED/TRANSPORTED/RELEASED）
   - 8 类 q_bad 违规码 + 7 类 q_unknown + 5 项 q_acc 条件
   - 转移表完整实现（§8）

2. **`src/fpck/tests/test_task_automaton.py`**：25 个单元测试
   - 覆盖转移表每一行（E1a 门槛 1）✅
   - 8 类违规码逐一测试
   - q_acc 5 项条件测试
   - 四类标签绑定测试（§9）

3. **`src/fpck/goal_evaluator.py`**：独立 goal evaluator
   - 复用 SafeLIBERO 的 `_eval_predicate`（逐 clause 求值，绕过 done 包装）
   - `eval_goal_clauses()` 返回 {canonical_clause: bool|None}
   - 关键：`_check_success` 只是 `_eval_predicate` 的 AND，独立性在于「逐 clause + 不依赖 done」

### 已确认（无需新实现）

- **manifest 角色覆盖**（E1a 门槛 2）：`data/task_object_role_manifest.json` 已覆盖
  17 任务（spatial 4 / goal 5 / object 4 / long 4），角色含 MANIPULATED_TARGET /
  GOAL_RECEPTACLE / STATIC_SAFETY_OBSTACLE / DISTRACTOR；
- **role_manifest.py**（v6）：已有 BDDL → 角色清单 + goal clauses + sha256 生成器；
- **契约文件**：`collision_event_contract_v2.json`（roles/allowance/temporal identity）。

## E1a 六项门槛进度

| # | 门槛 | 状态 |
|---|---|---|
| 1 | 自动机单元测试覆盖转移表 | ✅ 25 tests OK |
| 2 | manifest 对象角色覆盖 | ✅ 已确认 17 任务 |
| 3 | 独立 goal evaluator 与 native 对照 20 ep | ⏳ 待 GPU rollout（evaluator 已实现） |
| 4 | State02 假成功进入 q_bad | ⏳ 待 GPU |
| 5 | 正常成功进入 q_acc | ⏳ 待 GPU |
| 6 | S01/S04 事件链复查 | ⏳ 待 GPU |

## 关键决策记录

- **自动机状态机是纯逻辑**（不依赖 MuJoCo），`TaskEvent` 由外部（env 接触记录聚合 +
  goal evaluator + pose 观测）构造，便于单元测试；
- **独立 goal evaluator 复用 `_eval_predicate`**：这是 SafeLIBERO 的谓词求值器，
  独立性在于「逐 clause + 绕过 done」，非重造谓词（已在代码注释说明）；
- **NATIVE_FALSE_POSITIVE 判定顺序修正**：必须在 achieved_mask 更新后判（初版 bug：
  在更新前判导致 goal 未更新时误报，已修复，测试覆盖）。

## 下一步（待 GPU）

门槛 3-6 需要 GPU rollout（π0.5 基线 + 反事实恢复）。这些需要：
1. 写 E1a 采集脚本（nominal + 反事实，记录完整状态）；
2. 跑 20 个无干预基线 episode 做 goal evaluator 对照（门槛 3）；
3. 定位 State02 假成功 + S01/S04 witness（门槛 4/6）。

这是后续的实测工作，需先确认 witness 定位策略（S01/S04 的具体 task/init/stream）。
