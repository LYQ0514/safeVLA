# M1-P4-P0e：Joint-Space Path Execution Feasibility（S01）

> 日期：2026-11-01
> 状态：S01 单路径动态执行，**负结果**（几何路径存在，物理执行超 CAR）。
> 前置：`m1_p4_p0_safe_pregrasp_feasibility.md`、`../method/m1_p4_safe_task_reachability_design.md`。

---

## 0. 结论

> S01 的几何安全路径（RRT 35 路点，8.98 rad）用关节 PD 跟踪（robosuite
> JOINT_POSITION 语义，含重力补偿）执行时，**无硬接触但 mug 历史位移 2.115mm，
> 超 CAR 1mm**。根因是**跟踪滞后**（临界阻尼 PD 固有滞后 `(kd/kp)·v ≈ 141 mrad`）
> 叠加 t=10s 处的**拐角尖峰**（494 mrad），使机械臂偏离无碰撞路径，轻触 mug。

**关键命题**：几何路径无碰撞（P4-P0 已验证）≠ 物理可执行（P0e 证伪）。
这是继 P4-P0 之后的第二层证据，定量揭示了「几何 vs 动力学」可行性差距。

## 1. 结果

| 指标 | 值 |
|---|---|
| 首接触（硬接触）| **None**（重力补偿后消失）|
| 历史 Dmax | **2.115mm**（超 CAR 1mm）|
| 跟踪误差 max / mean / final | 494 / 161 / 4 mrad |
| 终点 pos_err | 4.0 mrad（精确到达）|
| 终点 vel_norm | 0.0016 rad/s（已稳定）|
| max_tau | 45.8 N·m（< 80 限幅）|
| 执行时间 T | 18.46s（路径 8.98 rad）|

## 2. 实现 bug 修复（重力补偿）

第一版执行用了朴素 PD（`tau = kp·err − kd·qvel`），**缺少重力补偿**，机械臂在
重力下下垂，稳态误差 ~g/kp ≈ 400 mrad，导致 hard contact + Dmax 175mm。

修正为 robosuite JOINT_POSITION 语义：

$$\tau = M \cdot (k_p\, e - k_d\,\dot q) + q_{frc\_bias}$$

（惯性缩放 + 重力补偿 `qfrc_bias`），hard contact 消失（首接触 None）。这是实现
bug，不是方法结论。

## 3. 残余跟踪滞后（真实发现，非 bug）

修正后跟踪误差时间线：

- 稳态 ~150–160 mrad：这是临界阻尼 PD 的**固有速度滞后** `err = (k_d/k_p)·v =
  (14.14/50)·0.5 = 141 mrad`，与观测一致。
- t=10s 尖峰 494 mrad：RRT 路径的**拐角**（分段线性路径方向突变）。

根因：**raw RRT 路径是分段线性的、有拐角**，且 JOINT_POSITION 默认增益（kp=50）
在 v_max=0.5 下有 ~141 mrad 跟踪滞后。二者叠加，机械臂在窄通道处偏离路径，
轻触 mug（2.1mm）。

## 4. 验收判定

| 验收条件 | 结果 |
|---|---|
| 无禁止接触 | ⚠️ 无 hard contact，但 mug 位移 2.1mm（轻触）|
| Dmax < 1mm | ❌ 2.115mm |
| 到达预抓取位姿 | ✅ pos_err 4 mrad |
| 关节速度稳定 | ✅ 0.0016 rad/s |
| 力矩限幅 | ✅ max 45.8 N·m |

**判定：FAIL**（Dmax 超 CAR）。

> ⚠️ **归因修正（见 m1_p4_p0e_1b）**：本报告的「跟踪滞后导致轻触」归因有误。
> 后续 1b 审计确认 mug 位移的真实来源是「robot link5 碰到 mug 底座
> box_small_base_1_main（FREE joint）→ 底座经支撑接触带动 mug」，且该接触被
> `build_body_geoms` 漏检（底座 body 名不含 "obstacle"）。跟踪滞后（141 mrad）
> 存在但非本次 CAR 越界的直接原因。

## 5. 科学含义

1. **几何路径 ≠ 物理可执行**：P4-P0 的「安全路径存在」只是几何层；P0e 证明 raw
   RRT 路径 + 有限增益关节 PD 无法在 CAR 约束下执行（跟踪滞后导致偏离）。

2. **两个误差源**：
   - 固有速度滞后（`(kd/kp)·v`，增益/速度决定，非路径问题）；
   - 拐角尖峰（RRT 分段线性路径的拐角，路径质量问题）。

3. **指向的下一步**（非本轮，待用户裁决）：
   - **路径平滑**（shortcut + 样条）：去除拐角，是标准 RRT 后处理，非「调参」；
   - **降低 v_max**：直接减小固有滞后（属保守设定调整，需用户同意）；
   - **提高 kp**：更硬跟踪（属控制器调参，用户已排除）；
   - **动力学规划**（kinodynamic RRT）：从根上约束速度/加速度。

## 6. 工程设定（已记录，统一未调）

- v_max = 0.5 rad/s，a_max = 1.0 rad/s²（模型无显式关节速度上限，保守取值）；
- kp = 50，kd = 14.14（匹配 JOINT_POSITION 默认，临界阻尼）；
- 力矩限幅 = 原 actuator ctrlrange（j1-5 ±80，j6-7 ±12）；
- gripper 执行期保持 NVS 状态。

## 7. CRN 修复（附带验证）

`generate_block` 硬编码 state2/stream0 已修正为显式传参。噪声哈希验证不同 stream
确实产生不同噪声（state1/stream3/4/5 哈希互异）。E1 的「12 组」因此应降级为
「4 个独立状态（stream 维度未生效）」，已在上游文档记录。

## 8. 工程产物

- `tools/m1_p4_p0e_joint_execution.py`：关节 PD 执行（computed-torque + 重力补偿）。
- `data/audit/m1_p4_p0e_s01_execution.json`：S01 执行指标 + 轨迹采样。
