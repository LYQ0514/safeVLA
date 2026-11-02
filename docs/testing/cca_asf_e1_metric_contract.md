# CCA-ASF E1 指标口径冻结（A0）

> 日期：2026-11-02
> 状态：指标定义固化。只改统计/日志命名，不重跑实验。
> 前置：E0 QP 快照对照 + E1 smoke 的混用发现。

---

## 0. 核心结论

官方 CAR 与「零禁止接触」是**两个不同契约**，不可混用：

$$\mathrm{CAR}_{native} \neq \mathbb I[\mathrm{ForbiddenContacts}=0]$$

E1 早期汇总把两者混在一个判断里，导致：
- euclidean 的 `forbid=712`（大量 gripper 指尖擦 ketchup）被当成「712 次违规」；
- cca 的 `forbid=47` + mug 撞飞 290mm 被当成「接触更少」。

这是错误比较——一个是接触统计量，一个是历史最大物体位移，没有共同尺度。

## 1. 四类指标（正式定义，冻结）

| 指标 | 定义 | 数据来源 |
|---|---|---|
| **Native CAR** | 官方 active obstacle（mug）自 episode 初始以来历史最大 L1 位移 ≤ 1mm | `body_state(mug).pos` 真值，单调不减 |
| **Strict Forbidden Contact** | 按预注册角色规则，robot 与 mug/底座/ketchup 等的禁止接触**事件数**（非 substep 次数）| `monitor_contacts` + 逐 substep `mjData.contact` |
| **Valid TSR** | 原生任务完成 ∧ 必要操作过程合法（夹持+相对稳定+连续抬升）| `check_success` + TASK 完整性监控 |
| **Valid StrictSSR** | Valid TSR ∧ Native CAR ∧ 零禁止接触 | 三者联合 |

## 2. 关键口径澄清

1. **接触事件 vs substep 次数**：一次持续接触会在多个 physics substep 反复出现，
   `monitor_contacts` 逐 substep 计数会虚高。正式报告应区分「接触事件数」
   （一段持续接触 = 1 事件）与「接触 substep 数」。712 是 substep 数，不是
   712 次独立碰撞。

2. **ketchup 是真实障碍物**（workspace 内 [−0.212,−0.132,0.509]），gripper 擦它
   是真实接触，但 Native CAR 只盯 mug（`find_active_obstacle` 返回第一个），
   所以「euclidean 擦 ketchup 但 CAR 通过」是**合法的口径分离**，不是矛盾。

3. **tomato_sauce 在 workspace 外**（x=−10m），不应算活跃障碍物，但按保守
   DISTRACTOR 规则仍记录，单独标注，不并入 StrictSSR。

4. **mug 撞飞 290mm** 是 Native CAR 层面的真实违规，无论 forbid 计数多少都成立。
   必须用「首次 CAR 越界时刻」定位，而不是只看最终位移。

## 3. 历史结果处理

- 保留所有原始字段（`n_forbidden`、`Dmax`、`success`、`n_beta_gt0` 等），不覆盖；
- 新增独立字段：`contact_events`（事件数）、`contact_substeps`（substep 数）、
  `first_car_violation_step`、`strict_ssr_valid`；
- 旧的 `n_forbidden` 语义改为 `contact_substeps`（明确是 substep 计数）。

## 4. 对 E1 的约束

- 主指标 pre-register 为 `Valid StrictSSR`；Native TSR / Native SSR 仍完整报告；
- 报告时四类指标**分列**，不合并成一个「安全」布尔；
- 接触统计用「事件数」而非「substep 数」做安全性判断。

## 5. 工程产物

- 本文档（口径冻结）；
- E1 脚本的指标字段命名修正（后续 B 完成后一并落盘）。
