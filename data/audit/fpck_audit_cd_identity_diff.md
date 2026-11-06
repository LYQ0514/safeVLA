# FPCK/CKG 审计 D（运行时身份）+ 审计 C（独立离线差分）结论

> 日期：2026-11-06。裁决 §四/§五执行完成。两个审计均 PASS，是后续批量重跑的必要前置。

## 一、审计 D：运行时源码身份（PASS）

**工具**：`src/fpck/runtime_identity.py` + `src/fpck/tests/test_runtime_identity.py`（4 测试）。

**方法**：从 `sys.modules` 取每个模块的实际 `__file__` + 文件 sha256（非仓库目录读 commit），
与 git HEAD 该路径内容 hash 对比。

**验证结果（8 个 fpck 模块）**：

| 模块 | 导入路径 | path_ok | hash 匹配 git HEAD |
|---|---|---|---|
| task_automaton | /data/safeVLA/src/fpck/task_automaton.py | ✅ | ✅ |
| task_automaton_acq | /data/safeVLA/src/fpck/task_automaton_acq.py | ✅ | ✅ |
| task_event_bridge | /data/safeVLA/src/fpck/task_event_bridge.py | ✅ | ✅ |
| gripper_action_decoder | /data/safeVLA/src/fpck/gripper_action_decoder.py | ✅ | ✅ |
| substep_collector | /data/safeVLA/src/fpck/substep_collector.py | ✅ | ✅ |
| collector | /data/safeVLA/src/fpck/collector.py | ✅ | ✅ |
| goal_evaluator | /data/safeVLA/src/fpck/goal_evaluator.py | ✅ | ✅ |
| canonical_hash | /data/safeVLA/src/fpck/canonical_hash.py | ✅ | ✅ |

**价值验证**：审计 D 在开发过程中**正确捕捉到** `task_event_bridge.py` 未提交改动
（加 contact_detail 字段后未 commit），证明身份门禁能防「旧 install / 未提交改动」混入评测。

**v1.2/v1.3 分离确认**：`task_automaton.py`（v1.2，冻结）与 `task_automaton_acq.py`
（v1.3）是不同文件，只在监视器逻辑上不同。

## 二、审计 C：独立离线差分（PASS）

**工具**：`src/fpck/offline_monitor.py` + `src/fpck/tests/test_offline_monitor.py`（7 测试）。

**独立性验证**：offline_monitor.py 只 import `json`/`typing`，**不 import** 在线 bridge 的
`classify_robot_body`/`_dynamic_contacts`、automaton_acq 的 guard/router/acceptance、
decoder 的 close intent 解码。

**独立实现**：
- finger 分类：显式 body 名集合（`LEFT_FINGER_BODIES`/`RIGHT_FINGER_BODIES`），非字符串启发式；
- close intent：`sign(executed_action[6]) > 0` 独立解码；
- 时序状态机：独立重写 acquisition→controlled→transported→released→q_acc/q_bad。

**差分结果（spatial/task1/state1，92 control steps）**：

| 里程碑 | 在线 v1.3 | 离线独立 | 一致 |
|---|---|---|---|
| first_close_intent | 29 | 29 | ✅ |
| first_valid_finger_contact | 36 | 36 | ✅ |
| first_bilateral_contact | 39 | 39 | ✅ |
| first_controlled | 41 | 41 | ✅ |
| first_lift | 42 | 42 | ✅ |
| first_transported | 42 | 42 | ✅ |
| first_goal_truth | 89 | 89 | ✅ |
| **final_mode** | q_bad | q_bad | ✅ |
| violation | NON_FINGER_ROBOT_TARGET_CONTACT | 同 | ✅ |

`diff = []`（零差异），`final_mode_match = true`。

**结论**：在线 v1.3 的分类（字符串启发式）与离线独立实现（显式集合）在决定性标签上
无分歧。finger 分类、close intent 解码、时序状态机三者的独立复现一致。

## 三、过程中的离线实现修正（非在线 bug）

1. `first_controlled` 差异（41 vs 42）根因：离线初版只认 lift≥10mm 作为物理证据，
   少了 closure_blocked / coupling 两个来源。补齐后一致（都 41）。
2. `first_transported` 差异（42 vs 38）根因：离线里程碑标记时机 bug（在 acquisition 阶段
   disp>10 就标记，而非 lifecycle 真正推进到 TRANSPORTED 时）。修正后一致（都 42）。
3. `first_goal_truth` 语义统一：改用 goal_clause_truth 全真（独立 goal evaluator），
   而非 native_success（env check_success，两者在真实日志同步但不总等价）。

## 四、Gate 状态

| Gate | 状态 |
|---|---|
| 审计 D（运行时身份） | **PASS** |
| 审计 C（独立离线差分） | **PASS** |
| Gate 4a | DISPUTED（state1 已澄清：假阴性成立但非干净正类） |
| Gate 5 | 仍 INVALID / 未恢复（state1 非完整 SAFE_REENTRY） |
| Gate 6 | 暂保留 |

## 五、剩余前置（裁决 §九）

1. ✅ 审计 D（运行时身份）；
2. ✅ 审计 C（独立离线差分）；
3. ⏳ 统一 7.81/10.19 mm 位移定义（审计 B 冻结轨迹 vs 重跑差异）；
4. ⏳ 双终止 harness 写入 v1.3 规格（native success 只记录不终止）；
5. ⏳ task manifest 加入 terminal_lifecycle_requirement；
6. ⏳ 固定 post-success 窗口 H_post + 稳定条件；
7. ⏳ 冻结 v1.3 spec / factory hash；
8. ⏳ 从 spatial/state3 开始逐条重跑。
