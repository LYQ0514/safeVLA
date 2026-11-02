# CCA-ASF v1 最终判决

> 日期：2026-11-02
> 状态：研究原型归档，终止 E1 扩大实验。
> 前置：E0/B/B1 全部文档。

---

## 0. 总体结论

CCA-ASF v1 是一个**数学机制成立、闭环安全失效**的研究原型。它成功证明了
「各向异性方向重分配」在局部 QP 层面可精确实现（TASK A-D 全通过），但在真实
闭环中，这种重分配以「正交扰动增大」为代价，在单条 CBF 未覆盖的间接传递区
（夹爪碰底座 → 底座带 mug）累积成 CAR 违规。

## 1. 四项正式结论

| 结论 | 判决 |
|---|---|
| **理论结果：成立** | 固定可行集 + 加权 QP 下，CCA 减少任务方向代理偏移（E0 验证 46/46）|
| **实现结果：成立** | 9D 白化映射 / β=0 退化等价 / 真实 QP 接入 / 各向异性激活（TASK A-D）|
| **闭环任务收益：未证实** | 减少代理方向偏移未建立与合法抓取/任务成功/StrictSSR 的可靠关系 |
| **闭环安全风险：已观察** | S01/st3 可重复 27-30mm mug 位移，euclidean 对应复现为 0 |

## 2. 核心科学反例

$$\boxed{D_\parallel\downarrow \quad\not\Rightarrow\quad \mathrm{CAR}\uparrow}$$

甚至：

$$\boxed{D_\parallel\downarrow,\ D_\perp\uparrow,\ D_{\mathrm{mug}}^{\max}\uparrow}$$

这揭示：**任务方向上的动作保留 ≠ 真实机器人安全**。CCA 的低秩加权 QP 保留了
CBF 约束，但单条 CBF 无法刻画「夹爪碰底座 → 底座带 mug」的间接传递链（底座不在
CBF 椭球内）。

## 3. 因果链（已闭合）

step0 各向异性方向重分配（ΔD_⊥>0，E0 定量）→ 300 步闭环分叉 → step219-222
夹爪碰底座 → 底座带 mug → 27-30mm CAR 违规。euclidean 同条件 Dmax≈0。

**证据等级**：已验证「各向异性改变轨迹 + CCA 分支碰撞」；未严格证明「正交扰动
是唯一致因」（未隔离正交分量与其他状态变化）。

## 4. 是否否定各向异性？

**否**。本次失败暴露的是「代理任务方向保护 ⇏ 真实收益/安全」，不代表「任何
各向异性安全修正都无价值」。本次风险同时涉及两个因素：

1. CCA 把控制修正重分配到其他方向；
2. 原 AEGIS 约束未覆盖「夹爪碰底座 → 底座带 mug」的间接传递机制。

但**没有理由继续围绕 S01 修补 CCA-ASF**——此前已多次经历「方法问题 → 基础设施
修复循环」，应坚持止损。

## 5. 真正有价值的收获

不是 CCA 是否成功，而是**把各向异性安全修正的数学收益与实际物理后果清楚分开**：

> 控制空间中更符合代理任务方向的修正，不一定在闭环物理系统中更安全，也不一定
> 更有利于任务完成。

这为后续重新研究各向异性提供了一个具体反例。

## 6. 归档范围

- 保留：`src/safety/cca_asf/`（数学模块）、`AegisSafetyLayer` 的 `filter_mode`、
  全部测试（26 个）、E0/B/B1 文档。
- 终止：E1 正式 16 episodes、E2 扩大、β/方向提取器调参。
- 后续若重立题，需重新定义「真实任务意图」的度量（而非短时动作块方向代理），
  并解决「间接传递链（底座）不在 CBF 内」的安全契约问题。

## 7. 文档索引

- `cca_asf_runtime_manifest.md`（TASK A）
- `cca_asf_task_c_integration_report.md`（TASK C）
- `cca_asf_task_d_runtime_smoke_test.md`（TASK D）
- `cca_asf_task_e0_qp_effect_audit.md`（E0 方向重分配）
- `cca_asf_e1_metric_contract.md`（A0 口径）
- `cca_asf_e1_catastrophic_car_audit.md`（B 归因）
- `cca_asf_e1_trajectory_divergence_closure.md`（B1 证据补全）
