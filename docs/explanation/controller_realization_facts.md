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

## 3. 执行器约束：力矩饱和**存在**（修正）

- OSC 是**阻抗控制**：`kp=150`，`kd = 2√(kp) × damping_ratio`（临界阻尼）。
- 力矩 = `J_fullᵀ · decoupled_wrench + torque_compensation + nullspace_torques`。
- **力矩饱和存在**：`SingleArm.control()` 里 `self.torques = np.clip(torques, low, high)`，
  其中 `low, high = self.torque_limits`。

> 更正：本文件初版写「无显式力矩饱和」是错的。实际执行力矩被 clip 到 torque_limits。
> 实验须区分 $\tau_{requested}$（OSC 公式输出）与 $\tau_{applied}$（clip 后）。

## 4. 对中心假设的直接支持（**已被 A0 修正**）

中心假设：

> 仅根据 VLA 参考动作和简化外部状态做安全过滤，可能遗漏内环控制状态和接触模式
> 对实际物理响应的影响。

**A0 核实结果**（`docs/explanation/a0_goal_persistence.md`）：`goal_pos` 在 VLA 动作
更新边界（`policy_step=True`）用 **`achieved` 语义完全覆盖**（`goal = ee_pos + delta`），
**不保留历史**。因此「不同历史 goal_pos → 同一新指令产生不同力矩」在动作更新边界
**不成立**。

> 本文件初版主张「goal_pos 是 (q,q̇,r) 之外的隐藏状态 ξ_t」**已被 A0 证伪**——
> 在动作更新边界，旧 goal 被覆盖，历史不影响新目标。

**仍成立的边界事实**（见 §2）：goal_pos 在**同一控制步的 25 substep 内**确实保持不变
（这是「执行周期内目标不变」，不是「跨控制步历史记忆」）。若安全层能在伺服周期
中途介入，则「介入相位」而非「历史目标」才可能成为风险源——这是转向后的新命题。

## 5. 下一步（已由 A0 转向）

原「实验 A：不同 goal_pos 反事实」**不再执行**（前提被 A0 证伪）。

转向「安全介入相位」命题：

> 相同安全指令 + 不同介入相位（新目标更新前 vs 伺服周期中途），是否产生不同物理
> 安全结果？

先核实：安全层是否允许伺服周期中途介入，修正动作在哪个控制相位生效。

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
