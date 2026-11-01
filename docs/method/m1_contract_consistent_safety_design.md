# M1：契约一致的最小干预安全过滤（设计）

> 日期：2026-11-01
> 状态：方法设计 + 数学契约。Oracle 原型，非可部署安全保证。
> 前置：`m0_s63_link6_contact_mechanism_audit.md`、`g1a_r3_g4_closed_loop_execution_attribution.md`。

---

## 0. 方法定位

不分别修补 AEGIS 的椭球/QP/Link6，而是统一机制，同时解决：
- **过度干预**：末端代理排除实际安全的名义动作（soup 接近失败）；
- **保护遗漏**：只保护末端，漏掉 Link6 连杆（Link6–mug 接触）。

**核心命题**：以真实物理安全契约为准，尽可能保留 π0.5 名义动作，
对全身几何 + 整个执行历史做一致的安全检查。

## 1. 数学契约

扩展状态：

$$\xi_t = (x_t, p_0, D_t)$$

候选动作经完整 OSC 控制步物理预测：

$$\hat x_{t+1} = \mathcal F_{OSC}(x_t, a)$$

历史位移预测（k=25 substep，j=全部受保护对象）：

$$\hat D_{t+1}(a) = \max\left[ D_t, \max_{k,j} \|\hat p_{j,t,k}(a) - p_{j,0}\|_1 \right]$$

全身禁止接触：

$$C_{body}(a) = \max_{k,\ell,o} \mathbb I[\text{substep }k \text{ 发生禁止接触}]$$

验证通过条件：

$$\mathcal V(\xi_t, a) = 1 \iff \hat D_{t+1}(a) < \delta_{CAR} - \epsilon \;\wedge\; C_{body}(a) = 0$$

（δ_CAR=1mm，ε 为预设安全裕度，不随结果调参）

## 2. 候选集合（第一版）

$$\mathcal A_t = \{ a_{nom}, a_{AEGIS}, a_{0.25}, a_{0.5}, a_{0.75}, a_{hold} \}$$

其中 $a_\alpha = \alpha a_{nom} + (1-\alpha) a_{AEGIS}$。
插值/保持动作都必须重新验证，不能假设「两安全则插值安全」。

选择：

$$a_t^* = \arg\min_{a \in \mathcal A_t^{verified}} \|a - a_{nom}\|_W^2$$

无候选时记录 `NO_VERIFIED_ACTION`，不能把 AEGIS 当必然安全 fallback。

## 3. 文献边界（不越界）

- 全身 CBF（VLPSA/WBAG）已存在，「全身覆盖」不是创新；
- M1 的候选价值在「native 契约一致 + 最小干预 + 高效预测验证」三者结合。

## 4. 待验证的三项贡献目标

1. Contract-consistent safety（检查与 CAR/禁止接触/历史状态一致）；
2. Task-preserving action selection（保留已验证安全的名义动作）；
3. Efficient predictive verification（高效代理替代 MuJoCo Oracle + 误差界）。

## 5. 实验阶段

| 阶段 | 范围 | 问题 |
|---|---|---|
| M1-P0 | step15/20/62/63 固定状态 | 验证器能否保留安全名义 + 识别 Link6/CAR 风险 |
| M1-P1 | 全时域闭环 | 能否恢复 soup + 保持 CAR + 真正选择性过滤 |
