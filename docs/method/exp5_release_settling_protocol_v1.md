# 实验 5 正常释放落置稳定包络协议 v1

> 日期：2026-11-07。**在运行前冻结**。protocol_sha256 记录冻结 hash。
> SINGLE_OBJECT：cream_cheese。scene：CHARACTERIZATION_SCENE_V2。

## 0. 职责

实验 5 产出「后缀速度包络 + settling-time 曲线」，不直接计算单一 `v_settled_max`，
不冻结最终阈值（留实验 6 联合裁决）。

## 1. 支撑条件（只运行实验 4 已通过的两个）

- `cream_table`
- `cream_receptacle`
- `cream_plate` 不参与（实验 4 已 INITIALIZATION_INVALID，不调位置）。

## 2. 流程（固定观察时长，不提前停止）

```
grasp → lift → open → contact gone → fall/impact → 固定观察窗口
```

- 抓取复用冻结门禁（cream_cheese 沿 y 轴，grasp aperture 42.2mm）；
- open 后固定观察 `H_obs = 100` 控制步（= 2500 substep），不检测「看起来稳定」提前停。

## 3. 三个时间锚点（不混淆）

```
t_release = 最后一次 finger-object contact 后的首个 substep
t_impact  = 首次合法 support-object contact
t_end     = 固定观察窗口终点
```

open command、contact gone、首次落地是三个不同时刻。

## 4. 原始测量（逐 physics substep）

线/角速度、COM 位置姿态、finger/object/support contact pair、penetration、
支撑对象、native body velocity、位姿有限差分速度、goal predicate、lifecycle/mode、
control/substep/仿真时间。原生 vs 有限差分一致性核验。

## 5. 后缀包络（不循环定义 settling）

对每条轨迹：
```
M_v(t) = max_{τ∈[t,t_end]} ||v(τ)||_2
M_ω(t) = max_{τ∈[t,t_end]} ||ω(τ)||_2
```

候选阈值网格（预注册）：
```
v_candidates  = [0.01, 0.02, 0.05, 0.10, 0.20]  m/s
ω_candidates  = [0.05, 0.10, 0.20, 0.50, 1.00]  rad/s
```

settling time：
```
T_settle(v,ω) = inf { t ≥ t_impact : M_v(t) ≤ v AND M_ω(t) ≤ ω }
```
固定窗口内不满足 → `RIGHT_CENSORED_NOT_SETTLED`（不记 0 或 max horizon）。

## 6. 四个速度量（区分）

1. 释放瞬间速度；2. 首次撞击前速度；3. 撞击峰值速度；4. 撞击后后缀稳定包络。
只有第 4 项用于候选稳定阈值；前 3 项供实验 6 对照。

## 7. 重复口径

```
technical_replicates: 3
unique_initial_condition_count: 1
independent_object_count: 1
```
table/receptacle 是两个独立物理条件，非独立对象。
两种支撑回弹不同 → 保守 settling 曲线取 max(table, receptacle)。

## 8. 输出状态

```
normal_release_settling_v1:
  verification_status: VERIFIED_WITHIN_SCOPE
  object: cream_cheese
  valid_support_conditions: [table, receptacle]
  invalid_support_conditions: [plate]
  threshold_selection_status: PENDING_EXPERIMENT_6
```
