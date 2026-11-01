# S63：Link6 接触机制审计 + M0-P1 归因修订

> 日期：2026-11-01
> 状态：M0 机制审计收尾。揭示 AEGIS 的连杆安全覆盖缺口。
> 前置：`m0_physics_grounded_nominal_feasibility.md`。

---

## 0. 结论

> Step63 逐 substep 审计发现：mug 被 `robot0_link6`（机械臂中段连杆）持续接触
> （substep 0-20，穿透 0.9mm→0），伴随 mug 位移渐进增至 0.871mm。
> AEGIS 的末端椭球（围绕 gripper，长轴 0.20m）**不覆盖 link6**（link6 距 eef 0.222m），
> 因此 link6 推动 mug 的运动完全不受 CBF 约束。
> 这是 **Body-Link Safety Coverage Gap（机器人连杆安全覆盖缺口）**。

## 1. 事件重建（step63 回放）

- 接触对：`red_coffee_mug_obstacle_1_main ↔ robot0_link6`，持续 substep 0-20（21/25）；
- 穿透深度：substep0 −0.9mm → substep20 −0.0mm（渐进减小）；
- mug 瞬时位移：substep0 0.381mm → substep20 0.871mm → substep24 0.841mm（回落）。

## 2. 运行身份（重要边界）

| 项 | 值 |
|---|---|
| 本次回放最大瞬时位移 | 0.871mm（**< 1mm，未复现违规**）|
| 原始违规记录 Dmax | ≈1.34mm |
| 关系 | **不是同一条完全复现的轨迹**（跨 run 数值分叉）|

> 本次回放发现强候选物理机制（link6 接触），但**不能宣称「原始 1.34mm 就是这次 link6 接触造成」**，
> 需在真正越界的运行中确认相同接触对。

## 3. AEGIS 保护范围核实

| 命题 | 结论 |
|---|---|
| 代码只为 eef 建 CBF | ✅ 成立 |
| 末端椭球空间不覆盖 link6 | ✅ 成立（link6 距 eef 0.222m > 椭球长轴 0.20m）|
| link6 rbound | 0.133m（真实中段连杆碰撞几何）|

**正式归类：Body-Link Safety Coverage Gap。**

## 4. 归因修订

| 先前解释 | 最新判决 |
|---|---|
| 前 40 步 GVMIS 漏检累计位移 | 不支持（前段未发生 CAR 违规）|
| Step63 突发碰撞冲量 | 不支持（持续接触 + 渐进推动）|
| Mug 被末端 gripper 碰撞 | 不支持（接触几何是 robot0_link6）|
| AEGIS 保护所有连杆 | 不成立（link6 不在保护包络内）|
| Link6 接触导致原始 1.34mm 越界 | 强候选解释，待原始违规轨迹确认 |

## 5. 接触安全 vs 位移安全（分开记录）

- 本次 0.871mm 未超 CAR，但 link6-mug 接触**本身有风险**；
- 持续接触使系统逼近违规边界，不能因未超阈值就视为无风险。

## 6. M0 阶段的完整发现（两类几何安全误差）

| 误差 | 影响 |
|---|---|
| **过度排除**：保守末端椭球限制 soup 接近 | TSR 损失 |
| **保护遗漏**：link6 不在末端约束内，推动 mug | CAR 风险 |

两者必须同时解决。

## 7. 下一步（结束 S63 诊断，进入方法设计）

下一代安全过滤器需满足：
1. Geometry-Faithful（真实碰撞几何，不过度排除）；
2. Whole-Body Collision Coverage（覆盖危险连杆，不只末端）；
3. History-Aware Physical Safety（历史 CAR 契约，全程时域）；
4. 冻结 π0.5 任务成功（CAR + TSR + SSR 联合）。

> 不能简单照搬全身 CBF（需与 whole-body CBF 已有方法区分技术贡献）；
> 不能回到「任务方向感知椭球」（C2 已证无效）。
