# 控制介入时机接口审计（Intervention Timing Audit）

> 日期：2026-11-01
> 状态：只读代码审计，不采集新轨迹。回答「AEGIS 安全判定时域 vs 低层执行时域」是否错位。
> 前置：`a0_goal_persistence.md`（achieved 覆盖）、`controller_realization_facts.md`。

---

## 0. 决定性回答

> **AEGIS 当前只保障「控制更新时刻」的约束满足，不覆盖 25 个 MuJoCo substep 的
> 整段实际物理执行。**

安全过滤在每个控制步边界调用**一次**，随后 25 个 substep 的力矩持续追踪目标、
且经过力矩饱和，整个过程**没有任何安全判定**。

---

## 1. 实际调用链（代码核实）

```
for step in range(max_steps):
    action = policy.act(obs, prompt)     # ① AEGIS filter 在这里调用一次
        └─ AegisPolicy.act(): layer.filter(action, obs)  → safe
    obs, ... = env.step(safe)            # ② 一次性执行 25 substep
        └─ for i in range(25):           # base.py step 内部循环
               sim.forward()
               _pre_action(safe, policy_step=(i==0))   # SingleArm.control
                   ├─ policy_step=True : set_goal(arm_action)  # 仅 i==0
                   ├─ torques = run_controller()               # 每个 substep 都算
                   ├─ self.torques = clip(torques, low, high)  # 力矩饱和
                   └─ sim.data.ctrl[...] = self.torques
               sim.step()
```

## 2. T1–T4 核查结果

| 项 | 事实 |
|---|---|
| **T1 过滤调用时机** | `AegisPolicy.act()` 在 `env.step()` **之前**调用，每控制步**一次**，只过滤第 1 个动作 |
| **T2 动作保持方式** | OSC 只在第 1 substep 更新 `goal_pos/goal_ori`（`set_goal`），随后 24 substep 复用同一 goal，但**每个 substep 都 `run_controller()` 重算力矩** |
| **T3 中途介入能力** | native 流程**不允许**在 24 个中间 substep 更新目标——`set_goal` 只在 `policy_step=True`（i==0）触发。要中途介入必须修改调度机制 |
| **T4 安全模型 vs 实际执行** | AEGIS 的 CBF-QP 只基于**当前 obs 的单步几何**（椭球距离），**不考虑** 25-substep 执行过程、力矩饱和、离散动力学 |

## 3. 两个时间尺度的错位

- **安全判定时域**：控制步边界的一个点（t_k），基于当前 obs 的静态几何。
- **实际执行时域**：25 个 substep（0.05s），期间力矩持续追踪 goal、经饱和、真实动力学演化。

AEGIS 保证的是：

$$h(x(t_k)) \ge 0$$

**不保证**：

$$h(x(t)) \ge 0, \quad \forall t \in [t_k, t_{k+1}]$$

这正是你提出的 **Intersample Safety（采样间安全）** 缺口。

## 4. 力矩饱和的加入（上一轮修正的延续）

`self.torques = np.clip(torques, low, high)`——实际执行力矩可能被饱和，而 AEGIS 的
CBF 预测基于「动作 → 理想运动」的映射，**不建模力矩饱和后的真实运动**。

因此采样间违规有两条潜在来源：

1. **目标追踪误差**：goal 在 25 substep 内不变，但 ee 可能超调/振荡；
2. **力矩饱和**：大动作导致力矩 clip，实际运动偏离 CBF 预测。

## 5. 情况判定（对应你预设的三格）

**情况 A 成立**：安全过滤只发生在控制步边界，不覆盖执行区间。

由此得到候选研究问题：

> 在冻结 VLA + OSC 执行框架下，如何为整个控制保持区间建立安全保证，
> 而不只是检查更新时刻的约束？

## 6. 理论方向（候选，未验证，不宣称创新）

采样间安全的充分条件：若安全函数变化率有下界

$$\dot h(x(t)) \ge -L_h$$

则保持区间 Δ 内

$$h(x(t_k+s)) \ge h(x(t_k)) - L_h s \;\Rightarrow\; h(x(t_k)) \ge L_h \Delta$$

是该区间安全的充分条件（可能极保守）。

**更强方向**：把力矩饱和 + OSC 目标追踪纳入动态模型，建立
**Execution-Horizon Safety Certificate**（执行区间安全证书）。

> 必须注意：连续时间 CBF 的采样保持安全已有成熟理论（ZOH-CBF、event-triggered
> CBF 等）。本方向是否构成新贡献，取决于能否把「VLA 动作块 + OSC goal 追踪 +
> 力矩饱和」这个具体执行栈的特征纳入，而非重复一般 CBF 理论。

## 7. 下一步（未启动）

用同一物理状态、同一名义动作，分别记录：

- 更新时刻的安全余量 $h(x(t_k))$；
- 执行区间内 25 substep 的 $h(x(t))$ 轨迹；
- 力矩是否饱和、何时饱和。

寻找「$h(x(t_k)) \ge 0$ 但执行区间内 $h < 0$」的真实采样间违规。

## 8. 状态

- T1–T4 审计完成，判定「情况 A：安全只覆盖更新时刻」。
- 不采集新轨迹；下一步是采样间违规的最小反事实观测。
- 不重复已有 ZOH-CBF / event-triggered CBF 理论，先查文献确认缺口。
