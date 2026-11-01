# Safety-Critical VLA Experimental Protocol

> Physical Ground Truth, Counterfactual Attribution, and Violation Timing
> 版本 v1（2026-11-01）。可执行技术规范，约束后续所有安全实验；非研究总结。
> 反面案例引用：`eaa9d7a`（step330）、`0902f4c`（375mm 观测伪影）。

---

## P1 — 安全评价的时间身份

**必须固定**：区分「首次违规前的预防」与「违规后的损害缓解」。

首次违规时间：

$$
\tau_{\text{viol}} = \inf\{ t : D_t > \delta \}, \qquad
D_t = \max_{\tau \le t} \| p(\tau) - p(0) \|_1, \quad \delta = 1\text{mm}
$$

**强制检查**：

- 记录 $\tau_{\text{viol}}$ 与历史违规状态（$D_t$ 是否已 $>\delta$）。
- 对 $t \ge \tau_{\text{viol}}$ 的候选动作，禁止用「能否恢复 CAR 合格」作评价——
  历史违规不可逆，$D_{t+1} \ge D_t$。
- 违规后的评价只针对「后续位移 / 冲击 / 损害」的**减量**。

> 区分：历史安全判定不可恢复 ≠ 当前控制对后续物理状态无作用。

## P2 — 反事实因果归因

**必须固定**：区分「物体实际运动」与「当前控制动作的增量影响」。

反事实效果须指定比较对象：

$$
\Delta_H(a, b \mid x) = Y_H(x, do(a)) - Y_H(x, do(b))
$$

**强制检查**：

- 同状态、同随机性、不同动作对照。
- $\Delta_H = 0$ 只说明「这两个动作在指定状态 / 时域 / 指标下无观测差异」，
  **不得**扩大为「所有动作都没有控制能力」。
- 归因前必须区分「历史动力学 / 当前动作 / 间接传播」三个来源。

## P3 — 物理真值来源

**必须固定**：策略输入与安全评价使用不同数据接口。

**强制检查**：

- 安全指标**只**从同步后的 MuJoCo 物理状态读取
  （`sim.data.xpos / cvel`，或 `_update_observables(force=True)` 后），
  **禁止**用 `env._get_observations()` 缓存替代安全真值。
- 策略推理仍用部署时的真实观测身份，**不得**为安全指标读取一致性而刷新 / 替换 VLA 输入。

---

## 落地

对应回归测试：`tests/test_safety_ground_truth_consistency.py`
（物理真值一致性 / 快照恢复一致性 / 缓存隔离性）。
