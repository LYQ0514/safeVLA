# FPCK/CKG Commit 4 实验 1 — 空夹爪 open 响应（含一个工程 bug 的发现与修复）

> 日期：2026-11-06。protocol_sha256=02936a14...。

## 一、发现的工程问题：直接写 sim.data.ctrl 被控制器覆盖

**现象**：初版脚本用 `sim.data.ctrl[actuator_id] = target` 直接写夹爪位置目标，
结果夹爪完全不动（aperture Δ=0）。

**根因**：robosuite 的 OSC 控制器在每个 physics substep 的 `_pre_action` →
`robot.control()` → `run_controller()` 里**覆盖**所有 ctrl（含夹爪 actuator），
直接写 `sim.data.ctrl` 的值在下一个 substep 被冲掉。必须走 `env.step(action)` →
`grip_action()` → `format_action()`（sticky 累积）路径才能真正驱动夹爪。

**修复**：改用 `env.step(grip_action(±1.0))`（close=+1, open=-1），经完整
`_pre_action → robot.control → grip_action → format_action` 路径。

**影响**：此前「T_open_stable=6 步」是无效数据（夹爪从未响应）。修复后重跑，
正确结果为 `T_open_stable ∈ {5, 7, 11}`，H_open_empty=15 控制步。

**教训**：表征实验也必须验证「命令确实改变了物理状态」，不能只看统计量。
此 bug 在第一个版本（直接写 ctrl）通过了语法与运行，但物理上无效。

## 二、实验 1 最终结果（45 trials，修复后）

| 维度 | 覆盖 |
|---|---|
| 姿态 | neutral / near_table / transport |
| 初始 aperture | closed(0) / near_closed(0.2) / mid(0.5) |
| 重复 | 5 |

结果（stable 控制步）：
- closed 起始 → 11（从 0.003 开到 0.077）
- near_closed(0.2) 起始 → 7
- mid(0.5) 起始 → 5
- 姿态无影响（3 姿态一致，符合「空夹爪 open 只受关节响应限制」）

onset：全 0（首个控制步即响应，Panda gripper sticky speed 使命令立即生效）。

按预注册规则：
```
max_observed_stable_steps = 11
multiplicative_margin = 1.2
discrete_margin_steps = 1
H_open_empty = ceil(1.2 * 11) + 1 = 15 控制步
```

## 三、Panda gripper 物理参数确认

- joint: `gripper0_finger_joint1/2`，actuator: `gripper0_gripper_finger_joint1/2`；
- actuator ctrlrange: left `[0, 0.04]`，right `[-0.04, 0]`；
- open 状态: ctrl(0.04,-0.04) → aperture≈0.077；
- closed 状态: ctrl(0,0) → aperture≈0.001；
- 初始状态即 open（aperture≈0.077）。

## 四、对象资产元数据（供对象清单）

`akita_black_bowl`（GoogleScannedObject，mesh 0.7 scale）：
- density=100，friction=(0.95, 0.3, 0.1)；
- 主体是 mesh bowl + 大量碰撞 box 近似（~40 个 box geom）；
- 抓取宽度（bowl 开口）约 0.07m 量级（mesh 0.7 scale）。

## 五、状态

| 项 | 状态 |
|---|---|
| 实验 1 空夹爪 open | ✅ PASS（45 trials，H_open_empty=15 步） |
| 对象清单 manifest | ⏳ 待提交 |
| 实验 2/3 载物 open + contact gone | ⏳ |
| 实验 4 静止速度噪声 | ⏳ |
| 实验 5 正常 settling | ⏳ |
| 实验 6 高速反例 | ⏳ |
