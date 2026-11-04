# AEGIS 复现严格性审计：发现一个影响全部复现的四元数约定 bug

> 日期：2026-11-04。触发：应要求审视 AEGIS 复现是否严格正确。
> 结论：**发现一个从最初就存在的严重 bug，影响全部 AEGIS 复现**。

---

## 1. Bug：末端姿态四元数约定错误

### 1.1 事实

- robosuite 的 `robot0_eef_quat` 是 **xyzw** 约定（权威依据：`robosuite/robots/single_arm.py`
  第 311 行 `T.convert_quat(self.sim.data.get_body_xquat(...), to="xyzw")`）；
- 上游 AEGIS 用 `R.from_quat(eef_quat)`（scipy 默认 xyzw），**正确**；
- 我的适配 `src/safety/aegis_adapted.py` 的 `_quat_wxyz_to_rot` **错误地假设 wxyz**，
  做 `R.from_quat([x,y,z,w]) = from_quat([q1,q2,q3,q0])`，导致 R1 的 y/z 轴**符号翻转**。

### 1.2 影响量化（实测）

以 long task0 init0 的初始状态为例：

| 量 | 正确（xyzw） | 我的错误（wxyz） |
|---|---|---|
| R1 第 2 行 | [0.001, −1.0, 0.001] | [−0.001, +1.0, −0.001] |
| p1（末端椭球中心） | [−0.060, 0.011, **0.748**] | [−0.060, 0.011, **0.588**] |

**末端椭球中心偏移方向反了**：正确应是末端**下方** 8cm（z=0.748），我的错成末端
**上方** 8cm（z=0.588），偏差 16cm。

### 1.3 影响范围

- `p1 = eef_pos + R1 @ [0,0,-0.08]`：椭球中心错误（16cm 偏差）；
- `v_ref = R1.T @ action[:3]`：末端局部速度参考错误（y/z 轴符号反）；
- `a_v = eta_row @ R1`：CBF 速度系数错误。

**所有走 `aegis_adapted.py` 的复现都受影响**，包括：
- `data/eval/aegis_1024/`（TSR=44.9%，本文档反复引用的 AEGIS 基线）；
- CCA-ASF、circulation、grasp-aware、被抓物体分离、支撑感知 的全部实验。

## 2. 其他已确认的复现偏差（次要，但需记录）

### 2.0 偏差总表

| 偏差 | 上游 | 我的适配 | 严重性 | 状态 |
|---|---|---|---|---|
| 四元数约定 | `R.from_quat`（xyzw） | `_quat_wxyz_to_rot` 误当 wxyz | **严重**（p1 偏移 16cm） | ✅ 已修复 |
| p2 刷新 | 固定（fit_ellipse 一次） | 每步刷新 obs 位姿 | 中（我更保守/准确） | 未改（保留，更正确） |
| num_steps_wait | 20 | 10 | 轻 | 未改 |
| 障碍物几何 | MVEE 点云拟合 | MuJoCo geom 真值 | 中（oracle 化，已声明） | 保留 |
| 障碍物识别 | VLM+GroundingDINO | oracle（工作空间第一个 obstacle） | 中（已声明） | 保留 |

### 2.1 障碍物几何：MVEE 点云 vs geom 真值

### 2.1 障碍物几何：MVEE 点云 vs geom 真值

- 上游：`fit_ellipse`（凸包点云 + MVEE 最小体积椭球）；
- 我的适配：`obstacle_ellipsoid` 从 MuJoCo `geom_size` 直接读半轴（文档已声明此差异）；
- 后果：Q2/R2 与上游不同（上游从视觉点云拟合，我用真值），但方向是「更准的真值」，
  不是 bug，而是「oracle 化」的已声明差异。

### 2.2 障碍物识别：VLM+GroundingDINO vs oracle

- 上游：`obstacle_detection`（VLM 命名 + GroundingDINO 检测）；
- 我的适配：`find_active_obstacle` 直接取工作空间内第一个 "obstacle" joint（文档已声明）。

### 2.3 `_quat2axisangle`（state 输入）—— 经核查是**正确**的

- robosuite 的 `quat2axisangle` 文档明确输入是 `(x,y,z,w)` 即 **xyzw**，`quat[3]`=w；
- `robot0_eef_quat` 是 xyzw，故上游 `_quat2axisangle`、`libero_state`、round3 的
  `quat_to_axisangle` 都是**正确**的（`q[3]`=w 是 xyzw 的正确位置，不是 wxyz）；
- **无 bug**。（初版审计误判 `quat[3]`=w ⟹ wxyz，实为 xyzw 的 w 也在 index 3。）

## 3. 对既有结论的影响

### 3.1 修复后的实测对比

**（a）确定性轨迹（PnsgPolicy，long task0 init 0/1/2）**

| init | 修复前（bug）末端距soup | 修复后末端距soup | 修复前 mug位移 | 修复后 mug位移 |
|---|---|---|---|---|
| 0 | 21.4cm | **1.4cm** | 15.0mm | 12.2mm |
| 1 | 20.5cm | **0.6cm** | 0.0mm | 11.3mm |
| 2 | 11.1cm | **0.7cm** | 22.9mm | **0.1mm** |

**（b）正式评测（Pi05Policy 1024，long task0 level I）**

| 配置 | TSR | 碰撞率 | n |
|---|---|---|---|
| bug 时代（aegis_1024） | **0.0%** | 24.0% | 50 |
| 修复后（aegis_1024_fixed_test） | **45.0%** | 55.0% | 20 |

**结论：bug 严重失真了 AEGIS 行为，且方向是「夸大任务失败」。**
- bug 时代 long task0 TSR=0%（本文档反复引用的「AEGIS 完全无法完成 long 任务」）是
  **错误的**——修复后 TSR=45%（20 episode 里 9 成功）；
- 修复后「末端能接近 soup（0.6-1.4cm）」「能完成任务」，说明「末端椭球悬空（上方
  8cm）」让 bug 时代的 CBF 保护错位，把本该可行的接近判成危险、把本该保护的 mug 判成
  远处，系统性夸大「任务-安全冲突」；
- 但修复后**碰撞率反而升高**（24%→55%）——因为 bug 时代末端椭球悬空、几乎不真正
  介入保护 mug，修复后椭球在正确位置、真正开始约束末端，碰撞率反映真实行为。

### 3.2 受影响最大的是「AEGIS 的绝对指标」

`aegis_1024` 的 TSR=44.9%、CAR=60.4% 等**绝对数字不可信**——它们基于错误的末端椭球
中心（16cm 偏差）。

### 3.3 相对结论（对比实验）部分可能仍有效

- 「vanilla vs AEGIS」的 TSR 对比方向（AEGIS 损害任务）可能仍成立，但具体数值失真；
- CCA-ASF「方向重分配引入副作用」、circulation「打破平衡点」等**机制结论**，其
  定性方向可能仍有效（bug 是系统性偏移，不改变「约束存在」这一事实），但定量
  不可信；
- **尤其「末端椭球禁区 19.3cm」「7.4cm 禁区边界」等几何归因数字全部作废**——它们
  基于错误的 p1。

### 3.4 必须重做

**所有 AEGIS 相关的定量结论都需要在修复 bug 后重做**，至少：
1. ✅ 修复 `_quat_wxyz_to_rot` → xyzw 直通（已完成，验证 p1 正确）；
2. 重跑 AEGIS baseline（`aegis_1024` 全量 1600 episode，或至少 long 套件）；
3. 重新评估之前所有「AEGIS 指标低」的归因（`aegis_metric_attribution.md`）、
   circulation/grasp-aware/分离约束的结论（`circulation_cbf_aegis.md`、
   `grasp_aware_*`、`grasped_sep_result.md`）。

## 4. 立即行动

1. ✅ 修复 bug（已完成）：`_quat_wxyz_to_rot` 改为 `R.from_quat(quat)`（xyzw 直通）；
2. ✅ 验证修复：p1 从错误的 z=0.588 修正为 0.748，R1 与正确值一致；
3. ✅ 实测影响：long task0 末端到 soup 从 11-21cm → 0.6-1.4cm；
4. ⏳ 待做：重跑 AEGIS baseline 全量，量化 TSR/CAR 的真实变化；
5. ⏳ 待做：回看所有 AEGIS 结论，标注「bug 前需重验」。

## 5. 诚实声明

- 这是一个**严重的、从最初就存在的复现错误**，我之前的 AEGIS 定量结论（TSR=44.9%
  等）都不可信；
- 我之前多次「数据驱动归因」（如 mug 挡 soup、末端椭球禁区 19.3cm）都基于错误的
  p1，其几何数字（如 7.4cm 禁区边界）需要修复后重算；
- 相对定性结论（AEGIS 损害任务、circulation 打破平衡点）方向可能仍成立，但需修复
  后重验才能确认。
