# FPCK/CKG 审计 B（physics-substep 最终裁决）：铁证确认 5mm 规则误判

> 日期：2026-11-06。用 physics-substep 原始 contact pair 重采集 spatial/task1/state1，
> 确认首个 5mm 越界的物理机制。

## 最终结论：铁证确认——首个 5mm 越界是「左指单侧接触 + 碗在支撑面滚动」

## physics-substep 原始 contact pair（首个 5mm 越界时刻 cs=37, ps=3）

bowl（akita_black_bowl_1_main）在首个 5mm 越界时刻（disp=5.16mm）的接触对**只有**：

```
akita_black_bowl_1_main ↔ gripper0_leftfinger          （左指接触）
akita_black_bowl_1_main ↔ glazed_rim_porcelain_ramekin_1_main  （碗仍接触支撑面！）
```

**没有**：
- `gripper0_rightfinger`（右指尚未接触）
- 任何 `robot0_link*`（机械臂连杆未接触 bowl）
- 任何夹爪外壳（`gripper0_right_gripper` 未接触 bowl）

## 物理机制判定（对照审计 B §七判据）

| 证据 | 观察 | 判定 |
|---|---|---|
| 位移来自手指接触 | ✅ bowl ↔ leftfinger | 抓取建立 |
| 碗仍在支撑面 | ✅ bowl ↔ ramekin（未抬升） | 自然滚动 |
| 右指尚未接触 | ✅ 无 rightfinger | 单侧接触阶段 |
| 无机械臂横向扫动 | ✅ 无 robot link 接触 | 非推动 |
| 无夹爪外壳碰碗 | ✅ 无 gripper0_right_gripper | 非外壳推 |

**结论**：碗的 5.16mm 位移是「左指单侧接触，碗在 ramekin 支撑面上滚动」的抓取建立
自然顺应。右指尚未接触，CONTROLLED 见证（需双指）尚未成立，被旧 5mm 累计位移规则
**提前误判为 UNAUTHORIZED_TARGET_MOTION**。

这不是「非法推动」（推动应由机械臂连杆或夹爪外壳接触碗），而是「单指接触导致的
支撑面滚动」——完全符合真实抓取建立的事件顺序。

## 三个候选的重新定性

- **state1**：铁证确认误判（左指单侧接触 + 支撑面滚动）；
- **state3/state4**：相同摘要模式（先 >5mm 后合法抓取），高风险疑似误判，需同法重审；
- **S04**：同样由抓取前累计位移 >5mm 触发，需重审。

## 已确认/撤回的结论（最终版）

### 已确认

1. 旧 5mm 规则是**系统性假阴性机制**（审计 A 纯逻辑 + 审计 B physics-substep 铁证）；
2. spatial/task1/state1 的旧标签 `UNAUTHORIZED_TARGET_MOTION` **无效**；
3. 「7/7 假成功」「π0.5 缺乏干净抓取能力」**撤回**；
4. state1 实际表现：闭爪意图 + 单指接触 + 双指接触 + >10mm 抬升（与「能力不足」相反）。

### 数据局限（诚实）

- `gripper_qpos` 采集有 bug（读 `d.qpos[-2:]` 不是夹爪关节），需修正才能补全夹爪
  关节位置/速度的精确时序；
- 但 **contact pair 是铁证**，不依赖 gripper qpos。

## Gate 状态（最终）

| Gate | 状态 |
|---|---|
| Gate 4a | **DISPUTED（铁证：state1 误判，S04 需重审）** |
| Gate 5 | **INVALID** |
| Gate 6 | 暂保留（State02/S01 forbidden contact 非 5mm 规则） |
| E1a | ENGINEERING AUDIT IN PROGRESS |
| E1b / value training / CKG | BLOCKED |

## 下一步

1. 修 gripper qpos 采集（补全夹爪关节时序）；
2. 设计 GRASP_ACQUISITION 状态（Resting→GraspAcquisition→Controlled）；
3. 新旧监视器离线对比同一原始轨迹；
4. 重跑 state1/3/4 + S04。

## 方法层面的深问题（审计收获）

**任务完整性不能通过「累计位移阈值 + 是否已抓取」的瞬时规则定义**，必须对抓取建立
过程做时序建模（单指接触 → 支撑面滚动 → 双指约束 → 抬升）。这反而是 FPCK 论文中
真正有价值的理论与工程设计点。
