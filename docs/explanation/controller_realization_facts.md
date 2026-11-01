# Controller-Realization-Aware Safety：控制接口事实核查

> 日期：2026-11-01
> 状态：新研究线的第一块实证地基 —— 核对 π0.5 部署环境的低层控制器代码，
> 决定中心假设能否落到当前系统，而非想象中的控制接口。
> 代码：`.venv/lib/python3.8/site-packages/robosuite/controllers/{osc.py, base_controller.py}`、
> `robosuite/environments/base.py`。

---

## 0. 结论先行

SafeLIBERO 的 OSC_POSE 控制器**确实存在**「外部状态不可见、但影响力矩输出的内部
记忆状态」——这直接支持本研究的中心假设，不必再建立在想象中的控制接口上。

三个决定性问题的事实答案：

| 问题 | 事实答案 |
|---|---|
| VLA 动作被解释成什么 | **末端位姿增量**（`control_delta=True`），平移 ±0.05m、旋转 ±0.5rad |
| 内部控制器是否有动态状态 | **有**：`goal_pos` / `goal_ori` 是跨 substep 保持的目标位姿记忆 |
| 真实执行器约束在哪生效 | OSC 阻抗控制（kp=150 临界阻尼），力矩 = JᵀF + 重力补偿 + nullspace，**无显式力矩饱和** |

---

## 1. 动作解释：位姿增量，不是绝对位姿也不是速度

`OperationalSpaceController` 配置（`use_delta=True`）：

- 动作 `(x, y, z, ax, ay, az)` 是**末端位姿增量**；
- `output_max = (0.05, 0.05, 0.05, 0.5, 0.5, 0.5)`，`output_min` 取负——
  即一个动作最多平移 ±0.05m、旋转 ±0.5rad；
- `scale_action` 把 [-1,1] 输入映射到 output 范围，再作为目标位姿**增量**叠加。

所以 VLA 的 7 维输出（前 6 维位姿 + 夹爪）中，前 6 维是「相对当前末端的位姿增量」，
第 7 维夹爪直接下发。

## 2. 内部状态：`goal_pos` / `goal_ori` 是跨 substep 的记忆

关键机制（`osc.py` 的 `set_goal` + `base.py` 的 `_pre_action`）：

- 每个控制步（0.05s）内含 25 个物理 substep（0.002s）；
- **只有第 1 个 substep 是 `policy_step=True`**，此时才调用 `set_goal(action)` 更新
  `self.goal_pos` / `self.goal_ori`；
- 其余 24 个 substep 复用**同一个** `goal_pos`（`policy_step=False`，不更新 goal）。

因此 `goal_pos` 是**跨 substep 保持的内部记忆状态**——它是「当前正在追的目标位姿」，
由历史动作累积决定，**不在外部观测 (q, q̇) 里**。

这正是中心假设里的 $\xi_t$：

$$
\xi_t = (\text{goal\_pos}_t,\ \text{goal\_ori}_t,\ \text{relative\_ori}_t,\ \text{ori\_ref}_t)
$$

两个外部状态相同的实例 `(q, q̇, r)`，只要历史动作不同（→ 不同的 goal_pos），
当前控制步产生的力矩就不同：

$$
\tau_t = J^\top \big( kp\,(\text{goal\_pos} - \text{ee\_pos}) - kd\,\text{ee\_vel} \big) + \text{gravity\_comp} + \text{nullspace}
$$

因为 `goal_pos` 不同 → `position_error` 不同 → `desired_force` 不同 → `τ` 不同。

## 3. 执行器约束：无显式力矩饱和

- OSC 是**阻抗控制**：`kp=150`，`kd = 2√(kp) × damping_ratio`（`damping_ratio=1` 临界阻尼）。
- 力矩 = `J_fullᵀ · decoupled_wrench + torque_compensation + nullspace_torques`。
- `actuator_range` 参数存在，但在 OSC 的力矩计算里**不用于钳位输出力矩**，
  主要影响 nullspace 项（`nullspace_torques` 用 joint 参考）。
- **没有**「力矩饱和 sat(·)」——即中心假设里 $\tau = \text{sat}(\cdot)$ 的 sat
  在当前 SafeLIBERO OSC 里**不生效**（输出力矩无上限钳位）。

> 这意味着「力矩饱和」不是本环境的约束点；真实的风险来源是「goal_pos 记忆 +
> 阻抗增益」如何把 VLA 位姿增量转成实际接触速度与冲量。

## 4. 对中心假设的直接支持

中心假设：

> 仅根据 VLA 参考动作和简化外部状态做安全过滤，可能遗漏内环控制状态和接触模式
> 对实际物理响应的影响。

**当前系统支持该假设的第一层**：`goal_pos`（内环目标记忆）是 (q, q̇, r) 之外的
隐藏状态，且直接进入力矩计算。两个「外部不可区分」的实例可以产生不同力矩。

但要注意**边界**（诚实声明）：

1. `goal_pos` 是否**真不可观测**，取决于安全过滤器能否访问控制器对象。若过滤器
   直接读 `controller.goal_pos`，则它不是隐藏状态。
2. `goal_pos` 的**影响量级**未测——它可能被 kp 增益放大，也可能在 25 substep 内
   快速收敛到无差异。需要实验 A 量化。
3. 本环境无积分项（无 `∫error dt`），`goal_pos` 是「目标记忆」而非「误差积分」。
   二者的可观测性含义不同。

## 5. 下一步（实验 A 的最小形式）

**Controller-State Identifiability Test**（已获代码事实支持，可执行）：

- 从同一 (q, q̇) 出发，构造两个不同的 `goal_pos`（对应不同历史动作），
  执行同一条 VLA 指令 r；
- 比较：力矩、接触前末端速度、首次接触时间、接触冲量。

若差异存在且具有安全意义，命题 A 获得实验动机；若差异可忽略（goal 快速收敛），
则「goal_pos 隐藏状态」假说在本环境不成立，需转向「接触模式切换」方向。

---

## 6. 与文献的差异边界（诚实）

- `goal_pos` 作为内环目标记忆，**不是**新发现——阻抗/OSC 控制的「setpoint 记忆」
  是标准实现。
- 本研究的**潜在差异**在于：把「内环目标记忆 + VLA 动作块 + 接触模式切换」放进
  **同一个可验证安全模型**，研究「何时仅凭 VLA 动作无法判断安全」。
- 这是否构成 T-RO 创新，取决于实验 A/B 是否发现**强基线（PACS/IHCBF）未处理的
  可观测机制**。当前尚无此证据，不宣称创新。

## 7. 状态

- 控制接口事实已核对，中心假设的第一层获得代码支持。
- 未启动任何实验；下一步是实验 A 的「同状态不同 goal_pos」最小反事实。
- 不接入 π0.5 完整 SafeLIBERO，不写新控制器，不改冻结策略。
