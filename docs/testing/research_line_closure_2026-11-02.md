# 研究线收束：冻结 VLA 安全控制的 M0→M2 全阶段归档

> 日期：2026-11-02
> 状态：研究线终止归档。本文件是整条主线的单一事实源索引。
> 关联：`research_overview_2026-11-01.md`（前期全景）及各阶段 testing 文档。

---

## 0. 终止说明

经用户裁决，M2-P0（Re-entry State Discrimination）在 G1 候选注册表阶段终止。
终止的直接原因：修复碰撞检测的 margin 一致性 bug 后，「真安全预抓取候选」大幅
缩小（S01 从 21 降到 5），且 RRT 连通全部失败、gap 截断导致候选无区分度——碰撞
检测基础设施尚未达到可信状态，继续推进 M2 方法会在地基不可靠的情况下盖楼。

## 1. 完整研究脉络（按时间/逻辑顺序）

| 阶段 | 核心问题 | 结论 |
|---|---|---|
| M0 | AEGIS 末端椭球为何导致 TSR 损失 + link6 碰撞 | 过度干预 + 保护遗漏（link6-mug 接触）|
| M1-P0/P0b | 单步安全候选验证 | 接触入口只有 hold 安全，需短时域预防 |
| M1-P1/P1c | 中心距离梯度 / 表面间隙梯度避让 | 单步梯度无效，闭环预测安全不足 |
| M1-P2 (HVNP) | 短时域候选序列选择 | P1b 单案例完整安全成功（正例）|
| M1-P3 (E1) | HVNP 跨状态泛化 | 0/12 SSR，候选字典覆盖不足 |
| M1-P4 | 安全任务可达性 + 动力学执行 + 接管 | 三层分离（见 §3）|
| M1-Contract-Audit | 安全契约完整性 | 底座漏检（第一 bug）|
| M1-T0 | TASK 物体完整性 | soup 撞飞漏检（第二 bug）|
| M2-P0 | 可接管状态区分 | margin 宽相位漏检（第三 bug），终止 |

## 2. 核心科学结论（三层分离）

整条线的根本发现，是三个「不互推」：

$$\boxed{\begin{aligned}
&\text{Safe Action} &&\not\Rightarrow \text{Task Success}\\
&\text{Safe Geometric Path} &&\not\Rightarrow \text{Task Integrity}\\
&\text{Safe Executable Path} &&\not\Rightarrow \text{Policy Re-entry Success}
\end{aligned}}$$

具体证据：
1. HVNP 能选安全动作，但 E1 中 0/12 任务成功（安全来自提前终止，非安全操作）；
2. P4-P0 找到几何安全路径，但 M1-T0 发现 S04 的路径把 soup 撞飞 248mm（TASK 漏检）；
3. P4-P0e-r 找到动力学可执行的路径，但 P4-P1 中冻结 π0.5 无法安全接续
   （S01 撞 mug 13mm / S04 无法抓取 / HVNP 候选字典姿态失配）。

## 3. 唯一经完整验证的正例：P1b

- State02 / HVNP / 完整闭环，任务成功 @ step275；
- 经 M1-Contract-Audit 完整契约重放：0 禁止接触、Dmax=0；
- 经 M1-T0 操作过程复核：VALID_GRASP（夹爪接触 + eef 相对静止 19.4mm + 稳定抬升）。

**P1b 是「冻结 VLA 通过有限安全干预完成合法任务」的唯一可信正例**，但它是
「mug 不挡主路径」的易案例，不能推广到困难状态。

## 4. 三个评测基础设施 bug（重要教训）

| # | bug | 阶段 | 影响 |
|---|---|---|---|
| 1 | 底座 `box_small_base_1_main` 被字符串过滤漏检 | M1-Contract-Audit | 零接触结论需复核 |
| 2 | TASK 物体（soup/cream）被预抓取路径撞飞未监控 | M1-T0 | 假成功（check_success 纯位置谓词）|
| 3 | `classify_ik` margin=0 宽相位漏检，~76% 假安全解 | M2-P0 G1 | 安全候选池严重高估 |

共同教训：**碰撞检测的 margin / 宽相位 / 角色清单三者必须一致且完整**，且需要
统一的、带回归门禁的碰撞判定函数，消除「各脚本各自实现 margin 语义」的分裂。

## 5. M2-P0 终止时的未解决工程问题

1. **gap 截断**：`exact_gap` 用 margin 上限做 gap 截断（50mm→300mm），候选无区分度；
2. **RRT 连通失败**：修复 margin bug 后，真安全解（S01 仅 5 个）无法用 RRT 从
   q_start 连通；
3. **候选池大幅缩小**：修复后「安全可达 + 连通 + TASK 完整」的候选几乎为空，
   M2 的「状态区分实验」失去前提。

这些是「碰撞检测基础设施未收敛」的表现，而非 M2 方法本身无效。

## 6. 未来方法方向（假设基础设施收敛后）

当前全部证据指向的命题：

$$\text{Feasibility-Aware} + \text{Re-entry-Aware Safety Intervention for Frozen VLAs}$$

即：不只要「安全到达」，还要「到达冻结 VLA 能安全完成任务的状态」。三层联合：

$$\text{Safe Reachability} \cap \text{Task Integrity} \cap \text{Policy Re-entry Viability}$$

但需注意：CoRe（反事实恢复）、Foresight Residual RL、SafeLoop 等已有工作也在研究
「接管状态质量」，此命题本身非无人问津。真正可能的差异化贡献是「全身物理安全
契约 + 任务物体完整性 + 动态可达性 + 冻结策略可接管性」的联合选择机制——但这需要
基础设施收敛后才能验证。

## 7. 关键文档索引

- 机制审计：`m0_s63_link6_contact_mechanism_audit.md`、`g1a_r3_g4_closed_loop_execution_attribution.md`
- HVNP 主结果：`m1_p2_p1b_full_episode_audit.md`（正例）、`m1_p3_e1_same_task_generalization.md`（负例）
- P4 三层：`m1_p4_p0r2_grasp_orientation_reachability.md`、`m1_p4_p0e_r_complete_contract_execution.md`、`m1_p4_p1_frozen_vla_reentry_audit.md`
- 基础设施：`m1_contract_coverage_root_cause_audit.md`、`m1_t0_task_integrity_audit.md`、`m2_p0_g1_margin_consistency_bug.md`

## 8. 资产状态

- `src/safety/contract_objects.py`：角色清单 + 接触监视器 + TASK 完整性监控（保留）；
- `tools/m1_p4_p0r2_reachability.py`：`classify_ik` 已修复（正 margin + 排除相邻 body）；
- 临时诊断脚本（_smoke/_diag/_debug/_verify/_probe）与后台日志已清理。
