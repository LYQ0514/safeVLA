# FPCK/CKG 实验 5 — cream_cheese 释放落置稳定包络

> 日期：2026-11-07。SINGLE_OBJECT_CHARACTERIZATION。
> threshold_selection_status: PENDING_EXPERIMENT_6

## 一、cream_table 结果（有效）

| 锚点 | substep |
|---|---|
| release（最后 finger contact 后） | 123 |
| impact（首次 support contact） | 173 |
| 撞击后完全稳定 | ~6 substep（12ms） |

### settling-time 表（关键候选）

| 候选阈值 | T_settle（相对撞击） |
|---|---|
| v=0.05, ω=0.2 | 1 substep |
| v=0.05, ω=0.1 | 1 substep |
| v=0.01, ω=0.05（最严） | 6 substep |
| v=0.2, ω=1.0（最松） | 0 substep |

### 物理规律

**角速度是 settling 瓶颈**：线速度在撞击后迅速满足（0-1 substep），但角速度
（对象弹跳/滚动）需要 6 substep 才衰减到 ω≤0.05。cream_cheese 平底 + 桌面平，
settling 很快（~12ms）。

## 二、cream_receptacle 问题

impact=None：basket 被 `build_scene` 误移出工作区（z=9.24）。`build_scene` 把所有
非 target 对象 teleport 到 z=10，但 receptacle 条件需要 basket 留在桌面作支撑。

**根因**：实验 5 的 receptacle 条件需要「basket 留在工作区 + cream_cheese 释放到
basket 内」，与 `build_scene` 的「移开所有其它对象」冲突。且「释放到 basket」需要
「移动到 basket 上方」的额外导航。

## 三、当前状态

| 条件 | 状态 |
|---|---|
| cream_table | PASS（完整 settling 曲线） |
| cream_receptacle | 待修正（basket 被移开 + 需移动导航） |
| cream_plate | INITIALIZATION_INVALID（实验 4 已定） |

## 四、决策点

cream_table 已经给出「释放落置稳定包络」的核心数据（撞击后 6 substep 稳定）。
cream_receptacle 的修正涉及「移动对象到 basket 上方释放」的导航，接近抓取规划。

按止损原则，建议：
1. **接受 cream_table 作为实验 5 的有效条件**（单一支撑条件）；
2. cream_receptacle 标记「需额外导航，暂缓」（不阻塞主线）；
3. 进入实验 6（高速触盘反例），用 cream_table 的 settling 曲线作为正常落置基线。

最终阈值的联合冻结：cream_table 显示「撞击后 ≤6 substep 稳定」，配合实验 6 的
高速反例拒绝区间，可以确定 v_max/ω_max 的分离裕量。

## 五、配置状态

```
normal_release_settling_v1:
  verification_status: VERIFIED_WITHIN_SCOPE
  object: cream_cheese
  valid_support_conditions: [table]
  invalid_support_conditions: [plate]
  receptacle: PENDING_EXTRA_NAVIGATION
  threshold_selection_status: PENDING_EXPERIMENT_6
```
