# M1-P4-P0e-1b：轨迹平滑对照 + 障碍物覆盖审计

> 日期：2026-11-01
> 状态：S01 平滑对照 + 根因归因（发现接触检测覆盖 bug）。
> 前置：`m1_p4_p0e_joint_path_execution_audit.md`。

---

## 0. 结论（重要归因修正）

> P0e-1 报告「Dmax 2.1mm 来自跟踪滞后」是**错误归因**。真实根因是：
> **机器人 link5 碰到了 mug 的底座 `box_small_base_1_main`（FREE joint），
> 底座被推动后经支撑接触带动 mug**。而这一接触**被接触检测漏掉了**——因为
> `build_body_geoms` 只收集 body 名含 `"obstacle"` 的 geom，`box_small_base_1_main`
> 的名字不含该后缀，其 geom（215-219）不在 obstacle 集合里。

这是一个**安全契约障碍物清单不完整**的 bug，不是「几何 vs 动力学」问题。

## 1. 三个关键事实（逐步定位）

### 事实 1：mug 是 FREE joint（自由漂浮），不是 fixed

`jnt_type=0 = mjJNT_FREE`。此前我把 `jnt_type=0` 误读为 fixed，导致错误地认为
「mug 不可能位移」。实际 mug、box_small_base、soup、table 全是 FREE joint，
可自由移动。

### 事实 2：mug 位移是真实物理位移，且机制是底座传递

substep5085-5088 接触对（首次超阈值处）：

```
box_small_base_1_main ↔ robot0_link5   (dist=-0.015mm)  ← 机器人碰到底座
living_room_table_col ↔ robot0_link6   (dist=-0.032mm)  ← link6 碰桌腿
red_coffee_mug ↔ box_small_base_1_main (dist=-0.001mm)  ← mug 被底座带动
```

链条：**link5 → box_small_base（FREE）→ 支撑接触 → mug**。全程 0 个 robot↔mug
直接接触，但 mug 通过底座被间接推动。

### 事实 3：接触检测漏掉了底座

`build_body_geoms` 的 obstacle 过滤逻辑：

```python
obstacle_names = set(n.replace("_joint0", "") for n in joint_names if "obstacle" in n)
obstacle_geoms = {g ... if body_name.startswith(o) for o in obstacle_names}
```

`box_small_base_1_main` 的 joint 名是 `box_small_base_1_joint0`（不含 "obstacle"），
其 geom（215-219）**不在 obstacle_geoms 里**。故 `check_contact_data` 检测不到
`robot↔box_small_base` 接触，误报「无接触」。

## 2. 平滑对照结果（受同一 bug 影响）

| 轨迹 | Dmax | 跟踪误差 max/mean | 说明 |
|---|---|---|---|
| Raw RRT | 2.679mm | 521/162 mrad | 未检测到底座接触 |
| Smoothed RRT | 2.675mm | 197/87 mrad | 跟踪误差降 60%，Dmax 几乎不变 |

平滑显著改善跟踪误差（521→197 mrad），但 **Dmax 不变**（2.68mm）——这**再次印证**
跟踪误差不是 mug 位移的原因，底座接触才是。平滑是有益的，但未解决根因。

## 3. 判决

| 预设判决 | 结果 |
|---|---|
| 平滑轨迹几何不安全 | 否（Gate1 通过）|
| 几何安全且实际 CAR 合格 | 否（Dmax 2.68mm）|
| **跟踪峰值下降但 CAR 仍违规** | ✅ **命中：拐角不是唯一原因** |
| 跟踪误差改善不明显 | 否 |

**根因**：接触检测漏掉 `box_small_base_1_main`（底座），导致「安全」是假象。

## 4. 科学含义（比原结论更重要）

1. **这不是方法问题，是评测基础设施 bug**：HVNP / P4 全线的安全契约都依赖
   `build_body_geoms` 的 obstacle 清单，而它漏掉了 mug 底座。这意味着之前所有
   「零接触」结论都需要复核——如果机器人碰到底座但被漏检，实际是违规。

2. **间接传递是真实风险**：mug 不是孤立物体，它有底座、有支撑关系。安全契约
   必须覆盖「受保护对象的支撑/附着部件」（base、handle 等），否则直接接触 mug
   没发生，但底座被推照样让 mug 位移超 CAR。

3. **几何 vs 动力学问题仍然成立但次要**：跟踪滞后确实存在（141 mrad 稳态），
   但本次 CAR 越界的直接原因是漏检的底座接触。

## 5. 需要复核的事项（诚实）

- P0e-1 的「Dmax 2.1mm 来自跟踪滞后」归因**错误**，应修正为「底座接触漏检」；
- 是否此前 HVNP/P4 的「零接触」结论也漏掉了底座接触？需回溯 `build_body_geoms`
  在所有脚本中的使用，确认 obstacle 清单完整性。

## 6. 下一步（待用户裁决）

1. **修正 `build_body_geoms`**：把 `box_small_base_1_main` 等受保护对象的
   支撑/附着 body 纳入 obstacle 清单（或改用「受保护对象 body 及其子 body」的
   完整枚举）；
2. **重跑 P0e-1**：修正接触检测后，重新执行 S01 路径，看是否还能「安全到达」
   （预计会检测到底座接触，Dmax 可能仍超 CAR，但归因会正确）；
3. **回溯 HVNP 结论**：确认 E1/P1b 的「零接触」是否也受此 bug 影响。

## 7. 工程产物

- `tools/m1_p4_p0e_1b_gate0.py`：位移来源审计（静止基线 + 接触链定位）。
- `tools/m1_p4_p0e_1b_smoothing.py`：轨迹平滑对照。
- `data/audit/m1_p4_p0e_1b_gate0.json` / `m1_p4_p0e_1b_smoothing.json`。
