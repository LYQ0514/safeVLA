# M1-P2-P1a：HVNP 自主闭环 Pilot（step0–99）

> 日期：2026-11-01
> 状态：单案例自主决策 Pilot，通过。
> 前置：`m1_p2_p0_hvnp_audit.md`、`../method/m1_p2_hvnp_design.md`。

---

## 0. 结论（回答决定性问题的答案）

> **是**。HVNP 从 episode 起点自主运行 100 步，全程零禁止接触、历史 Dmax=0、
> CAR 合格；98/100 步保留名义动作，仅在 step52 提前做了一次 L+A 避让（两步），
> 成功避免了名义轨迹在 step61 的 Link6–mug 接触；预测与真实执行零偏差
> （prediction miss=0），无候选集失效。冻结 π0.5 持续推进任务（d_soup 288.9→15.7mm）。

## 1. 结果

| 指标 | 数据 |
|---|---|
| 执行步数 | 100/100，无提前终止 |
| 真实安全 | Dmax=**0**、CAR 合格、首次禁止接触=**None** |
| Nominal 保留 | 98/100（98%），无连续替换 |
| 安全替换 | L+A 1 次（step52–53 两步避让）|
| 预测准确性 | prediction miss=0 |
| 任务进度 | d_soup 288.9→15.7mm；soup/cream 子目标未完成（step99 尚早）|
| 计算成本 | 均值 0.32s/步，P95 0.34s，最大 1.47s |
| 候选集失效 | NO_VERIFIED_SEQUENCE=0 |

## 2. 核心证据：提前预防（而非事后补救）

step52 时，HVNP 预测名义 N 的 H=10 窗口（step52–62）发现：

> 名义 N 会在 **step61** 产生 Link6–mug 接触（Dmax 0.346mm）。

因此 N 不安全 → 验证替代候选 → L+A 安全且干预最小 → 提前在 step52 执行 L+A 避让。
避让后（step53 起）名义 N 再次安全，全部保留。全程零接触。

**对比**：

| 分支 | 接触 | Dmax | 性质 |
|---|---|---|---|
| B（AEGIS 被动过滤）| step59–61 接触 | 0.9–1.15mm（CAR 越界）| 事后，未预测 |
| **HVNP（预测+提前干预）** | **无** | **0** | **提前预防** |

这是 HVNP 相较 AEGIS 的关键区别：**H=10 预测窗口覆盖了接触点，在接触形成前
（step52）就主动避让，而不是等接触发生后再被动过滤**。

## 3. 关键方法学观察

1. **HVNP 未进入 B 分支的危险状态**：HVNP 在 step52 就干预，其后续状态分布与 B 分支
   不同，从未进入 step59/61 的接触状态。这符合用户预判——「HVNP 若未进入此前危险
   物理状态，不是失败，也不强制复现」。三个固定状态（step54/57/59）仍是独立回归
   测试，不要求自主 episode 中出现。

2. **Nominal-first 有效降低开销**：98% 步只需 1 次 N 验证（决策均值 0.32s）；
   只有 step52 需验证全部 5 候选（决策 1.47s）。这印证「安全状态下不必要干预」的
   设计与「Lazy Verification」的计算价值。

3. **预测-执行一致性**：全 100 步 prediction miss=0，scratch env 全状态 fork 的
   预测判决与 live 真实执行一致（de-risk 已验证 fork 保真度）。

## 4. 尚未验证（不过度宣称）

1. 完整 episode（step100+ 的 soup 抓取、cream 放置、最终 TSR/SSR）——soup 子目标
   在 step99 尚未完成；
2. 多案例泛化（当前仅 State02/Stream0 单案例）；
3. 实时部署可行性（决策 0.32s 含 GPU 推理，且为离线 Oracle）；
4. 「几何替代动作选择能力」的独立证明——step52 的 L+A 仍是固定字典候选。

## 5. P1a 通过标准逐项核验

| 通过标准 | 结果 |
|---|---|
| step0–99 无违反安全契约 | ✅ 零接触、Dmax=0 |
| 无未处理候选集失效 | ✅ NO_VERIFIED_SEQUENCE=0 |
| 在线选择与 Oracle 执行一致 | ✅ miss=0 |

## 6. 下一步（待用户裁决）

P1a 通过 → 按用户指令「以同样的规则直接进入完整 episode」（P1b）：运行完整任务
（Long-I 最多 550 步），验收实际 CAR/TSR/SSR、soup/cream 子目标、干预次数及每次
决策耗时。需注意：完整 episode 的决策总开销（约 550 步 × 0.32s ≈ 3 分钟）可接受，
但需观察后半段（soup 抓取/运输）是否有新的接触风险或候选集失效。

## 7. 工程产物

- `tools/m1_p2_p1a_derisk.py`：scratch fork 预测保真度验证。
- `tools/m1_p2_p1a_pilot.py`：HVNP 自主闭环 Pilot。
- `data/audit/m1_p2_p1a_pilot.json`：100 步逐步记录。
