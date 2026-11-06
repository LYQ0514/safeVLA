# FPCK/CKG 全仓控制路径审计 — 直接 data.ctrl 写入的分类

> 日期：2026-11-06。裁决 §五执行。
> 结论：**没有发现「写 ctrl 后调用 env.step（会被 OSC 控制器覆盖）」的高风险模式**。
> 唯一受影响的是已修复的实验 1 初版（直接写 ctrl + 手动 sim.step，但 ctrl 语义错误）。

## 一、robosuite 控制链（正式定性）

```
env.step(action) → _pre_action → robot.control → run_controller → sim.data.ctrl
```

`sim.data.ctrl` 是**控制器输出层**，不是策略动作接口。在 OSC 闭环里外部写 ctrl 会被
`run_controller` 覆盖。

## 二、全仓扫描结果

### 类别 1：Joint-PD 阶段手动 torque 控制（合法，不经过 OSC）

这些脚本写 `d.ctrl[:7]=tau` 后**直接 `sim.step()`**，不调 `_pre_action`/`robot.control`，
所以 ctrl 不被覆盖。这是 Joint-PD 阶段（P-H 对照）的设计：手动计算关节 torque 绕过 OSC。

| 文件 | ctrl 写入 | 后续调用 | 判定 |
|---|---|---|---|
| src/fpck/instrumented_loops.py:73 | d.ctrl[:7]=tau | sim.step() | 合法 |
| tools/m1_p4_p0e_r_execution.py:134 | d.ctrl[:7]=tau | sim.step() | 合法 |
| tools/m1_p4_p0e_joint_execution.py:289 | d.ctrl[:7]=tau | sim.step() | 合法 |
| tools/m1_p4_p0e_1b_gate0.py:151 | d.ctrl[:7]=tau | sim.step() | 合法 |
| tools/m1_p4_p1_r0_probe.py:59 | d.ctrl[:7]=tau | sim.step() | 合法 |
| tools/m1_p4_p0e_1b_smoothing.py:150 | d.ctrl[:7]=tau | sim.step() | 合法 |
| tools/m1_p4_p1_reentry.py:76 | d.ctrl[:7]=tau | sim.step() | 合法 |
| tools/m1_t0_task_integrity.py:71 | d.ctrl[:7]=tau | sim.step() | 合法 |
| tools/s0f4_actuation_audit.py:69 | ctrl[...]=tau | sim.step()（审计） | 合法 |

### 类别 2：状态 teleport（初始化，非动作执行）

| 文件 | 写入 | 用途 |
|---|---|---|
| tools/m1_p4_p0_reachability.py:67 | d.qpos[:7]=q（仅臂关节） | RRT reachability 采样初始化 |
| tools/m1_p1c_*.py | qdata.qpos[:]=d.qpos | 反事实状态恢复/一致性检查 |

这些是**状态设置**（teleport），用于初始化或状态恢复，不是「动作执行」。qpos 直接
写是 MuJoCo 状态操作的标准做法，不涉及 ctrl 覆盖问题。

### 类别 3：正确路径（env.step + grip action）

| 文件 | 方式 | 判定 |
|---|---|---|
| src/fpck/tests/test_gripper_action_physics.py | env.step(act), act[6]=grip | 合法 ✅ |
| src/fpck/tests/test_gripper_qpos.py | 只读 d.qpos[-2:]（诊断） | 只读 |
| 实验 1（修复后） | env.step(grip_action(±1)) | 合法 ✅ |

### 类别 4：纯逻辑测试（不驱动 env）

- `test_scripted_grasp_audit.py`（审计 A）：构造 TaskEvent 对象，无 env 驱动；
- `test_task_automaton_acq.py`（v1.3）：纯逻辑；
- `test_dual_endpoint_harness.py`（Commit 2）：纯逻辑。

这些不受 ctrl bug 影响。

## 三、特别核查项（裁决 §五）

| 检查项 | 结论 |
|---|---|
| 审计 A 脚本化合法抓取 | 纯逻辑测试，不驱动 env，**不受影响** |
| 脚本化纯推动反例 | 纯逻辑（test_task_automaton_acq），**不受影响** |
| gripper decoder 3 个物理测试 | 用 env.step + act[6]=grip，**正确路径** |
| state1 dual-monitor 重跑 | 走 policy action 路径（env.step），**不受影响** |
| positive-search open/close | 走 policy action 路径，**不受影响** |
| snapshot round-trip | 状态恢复（teleport），非 ctrl 覆盖问题 |
| Joint-PD 循环 | sim.step 直接调用，ctrl 不被覆盖，**合法** |

## 四、唯一受影响项

实验 1 初版（`tools/fpck_characterize_exp1_open.py` 早期版本）：用 `sim.data.ctrl[act]=target`
写夹爪位置目标 + 手动 `sim.step()`。虽然不会被 OSC 覆盖（因为没调 env.step），但：
- 语义错误：把 actuator ctrl 当「位置目标」直接写，跳过了 `grip_action → format_action`
  的 sticky 累积；
- 且与正式部署路径不一致，导致夹爪实际不动（Δ=0）。

已修复为 `env.step(grip_action(±1))`。旧结果（H_open_empty=9）标记 INVALID。

## 五、对实验 2/3 的约束（裁决 §十）

实验 2/3/5 的脚本夹持必须走正式 action 接口 `env.step(action)`，并加物理生效断言：
- 闭合建立：close command 存在 + aperture 达标 + 双指接触 + 目标稳定；
- open 执行：open intent 成立 + controller target 变化 + aperture 增加；
- 释放发生：双指 contact 消失 + 目标脱离耦合。

任何门禁失败 → trial 标记 `CONTROL_COMMAND_NO_PHYSICAL_EFFECT`，不用于阈值统计。

## 六、结论

全仓**无**「写 ctrl 后调 env.step」高风险模式。可直接继续对象清单生成 + 实验 2/3。
