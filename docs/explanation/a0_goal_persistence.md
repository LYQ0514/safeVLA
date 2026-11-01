# A0：Goal Persistence Test 结果

> 日期：2026-11-01
> 状态：新研究线第二个门禁，决定「目标记忆 vs 介入相位」两个方向的分岔。
> 前置：`docs/explanation/controller_realization_facts.md`（含两处需修正的事实）

---

## 0. 结论先行

**A0 判决：新指令完全覆盖旧目标（`achieved` 语义）。**

`goal_pos` 在 VLA 动作更新边界（`policy_step=True`）被**完全覆盖**，不保留历史信息：

$$g_t^+ = p_{eef}(q_t) + \Delta p(r_t)$$

因此「不同历史 goal_pos → 同一新指令产生不同力矩」在**动作更新边界不成立**。
上一轮把「25 substep 内 goal_pos 不变」误当成「历史目标记忆」是错的——这是两个
不同性质（你早已指出）。

---

## 1. 代码证据（`set_goal` → `set_goal_position`）

`osc.py` 第 264-266 行：

```python
self.goal_pos = set_goal_position(
    scaled_delta[:3], self.ee_pos, position_limit=self.position_limits, set_pos=set_pos)
```

`control_utils.py` 第 136 行：

```python
goal_position = current_position + delta   # current_position = self.ee_pos
```

**`current_position` 传入的是 `self.ee_pos`（当前实际末端位置），不是旧 `goal_pos`。**

这就是 `achieved` 语义：新目标 = 当前实际位置 + 增量。旧目标历史被丢弃。

## 2. 两个性质的区分（关键）

| 性质 | 是否成立 |
|---|---|
| 25 substep 内 goal_pos 不变（同一控制步内复用）| ✅ 成立 |
| 新指令到来时 goal_pos 保留历史（跨控制步记忆）| ❌ **不成立**（achieved 覆盖）|

上一轮 `controller_realization_facts.md` 的核心主张（goal_pos 是隐藏状态 ξ_t，两个
外部相同状态会因历史 goal 不同产生不同力矩）**在动作更新边界被证伪**。

## 3. 力矩饱和：上一轮报告有误，修正

`SingleArm.control()` 第 252-254 行：

```python
low, high = self.torque_limits
self.torques = np.clip(torques, low, high)
```

**力矩饱和是存在的**（`clip` 到 `torque_limits`）。上一轮「无显式力矩饱和」错误。

实验 A 因此必须记录 $\tau_{\text{requested}}$（OSC 公式输出）与 $\tau_{\text{applied}}$
（clip 后），区分「理想力矩」与「实际执行力矩」。

## 4. A0 判决表（对应你预设的四格）

| 实验结果 | 实际 | 科研判断 |
|---|---|---|
| 新动作完全覆盖旧目标，输出一致 | ✅ 成立 | 旧目标记忆不构成动作更新边界的独立风险 |
| 新动作保留部分历史目标 | — | 不成立（achieved）|
| 旧目标被覆盖但其他内部状态导致输出不同 | — | 未发现 |
| 只有执行周期中途目标差异才影响力矩 | 待查 | 转向「控制更新相位」，见 §5 |

## 5. 备选命题（你的预判，转向）

既然「goal_pos 隐藏记忆」不成立，转向更精确的：

> **VLA Safety Intervention Timing（安全介入相位）**：
> 相同安全指令 + 不同介入相位（新目标更新前 vs 伺服周期中途），
> 是否产生不同物理安全结果？

前提（必须先核实）：

- 当前安全层是否允许「伺服周期中途介入」？
- 修正动作实际在哪个控制相位生效？
- 若安全层始终在控制步边界同步更新，相位差不存在。

## 6. 下一轮核对清单（未启动）

1. 安全层（AEGIS 等）在 robosuite 的 `env.step` 里，是否与 `policy_step` 同步，
   还是可能在 25 substep 中途被调用。
2. 若存在中途介入可能，构造「同一安全指令、不同介入 substep」的反事实，
   测接触前速度/首次接触时间/冲量差异。

## 7. 状态

- A0 完成，判决「achieved 覆盖」。
- 原实验 A（不同 goal_pos）**不再执行**——其前提已被 A0 证伪。
- 研究方向从「控制器状态记忆」转向「安全介入相位」。
- 不启动完整接触实验、不接入 π0.5、不写新控制器。
