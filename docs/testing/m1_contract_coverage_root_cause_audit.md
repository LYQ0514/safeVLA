# M1-Contract-Audit：受保护对象交互完整性根因审计

> 日期：2026-11-02
> 状态：根因定位完成。安全契约覆盖缺口。
> 前置：`m1_p4_p0e_1b_trajectory_smoothing_audit.md`（底座接触线索）。

---

## 0. 结论

> 安全验证器的障碍物清单**不完整**：`build_body_geoms` 用 body 名含 `"obstacle"`
> 的字符串推断，漏掉了 mug 的底座 `box_small_base_1_main`（独立 FREE joint，
> 4 个碰撞 geom 216-219）。机器人 link5 碰到底座 → 底座（FREE 可动）被推动 →
> 经支撑接触带动 mug，导致 mug 位移超 CAR。这一接触被漏检，使「零禁止接触」成为假象。

这是评测基础设施 bug，不是 HVNP 算法本身失效，也不是「RRT 不能动态执行」。

## 1. 完整证据链

### 1.1 mug 是 FREE joint（自由漂浮），此前误读为 fixed

MuJoCo 枚举 `mjJNT_FREE = 0`。此前把 `jnt_type=0` 误读为 fixed，导致错误地断言
「mug 不可能位移」。实际上 mug、底座、soup、table 全是 FREE joint，可自由移动。

### 1.2 mug 位移机制：底座传递

P0e 执行中首次超 CAR 处（substep5085-5088）的原生接触链：

```
box_small_base_1_main ↔ robot0_link5   (dist=-0.015mm)  ← link5 碰底座
living_room_table_col ↔ robot0_link6   (dist=-0.032mm)  ← link6 碰桌腿
red_coffee_mug ↔ box_small_base_1_main (dist=-0.001mm)  ← 底座带动 mug
```

全程 0 个 robot↔mug 直接接触，但 mug 通过底座被间接推动 2.9mm。

### 1.3 底座被字符串过滤漏掉

`build_body_geoms`：
```python
obstacle_names = set(n.replace("_joint0", "") for n in joint_names if "obstacle" in n)
obstacle_geoms = {g ... if body_name.startswith(o) for o in obstacle_names}
```

`box_small_base_1_main` 的 joint 名是 `box_small_base_1_joint0`（不含 "obstacle"），
其碰撞 geom 216-219 不在 obstacle 集合，`check_contact_data` 检测不到 robot↔底座。

## 2. 场景角色清单（事实枚举）

FREE 环境物体共 12 个，显式分类：

| 角色 | 物体 |
|---|---|
| PROTECTED（CAR 直接评估）| red_coffee_mug / moka_pot / white_storage_box / milk / wine_bottle / yellow_book（均 `_obstacle_` 后缀）|
| **PROTECTED_SUPPORT（漏检）** | **box_small_base_1_main（mug 底座）** |
| TASK | alphabet_soup / cream_cheese / basket |
| DISTRACTOR | tomato_sauce / ketchup |
| FIXED_ENV | floor / living_room_table / table_col |

底座 `box_small_base_1_main`（bid=32）位于 mug 正下方 0.075m，4 个 box 碰撞 geom
在 y 方向延伸覆盖 soup 抓取区域——是 mug 位移的真实物理传递通道。

## 3. 修复方案

见 `src/safety/contract_objects.py`：显式角色清单替代字符串推断，独立接触监视器
`monitor_contacts` 扫描原生 `mjData.contact` 的全部 robot↔environment 接触对，
按角色分类（forbidden / task / fixed_env / unclassified / robot_robot / env_env），
报告未被角色覆盖的事件。

## 4. 回归门禁结果

| 门禁 | 结果 |
|---|---|
| G1 覆盖完整性 | ✅ 12 个 FREE 物体全部有明确角色，无 UNCLASSIFIED |
| G2 事件复现 | ✅ 底座 geom 216-219 已在 forbidden 集合；link5↔底座接触被识别 |
| G3 误报控制 | ✅ 初始 mug↔base 归 env_env（非 robot 接触），无 forbidden 误报 |
| G4 无副作用 | ✅ monitor_contacts 只读，qpos/qvel 哈希不变 |

## 5. 工程产物

- `src/safety/contract_objects.py`：显式角色清单 + 接触监视器。
- `tools/m1_contract_scene_inventory.py`：场景 body/geom/joint 结构枚举。
- `tools/m1_contract_gate.py`：G1-G4 回归门禁。
- `data/audit/m1_contract_scene_inventory.json` / `m1_contract_gate_results.json`。
