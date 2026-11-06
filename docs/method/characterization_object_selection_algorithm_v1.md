# 对象选择算法规范 v1（characterization object selection）

> 日期：2026-11-06。**本文件在运行对象扫描前冻结**。
> 选择算法先于结果确定，人工只能合理性审核，不能手动换结果。
> selection_algorithm_sha256 记录本文件冻结时的 hash。

## 0. 数据源优先级（裁决 §一）

1. 冻结的 task/object role manifest（`data/task_object_role_manifest.json`）；
2. BDDL goal/initial-state 交叉检查；
3. 环境实际 body/geom 身份运行时验证。

不一致 → `ROLE_MANIFEST_BDDL_MISMATCH`，不进候选池，进审计列表。

## 1. 候选池资格（裁决 §一/§八）

`eligible(o)` 当且仅当全部满足：
- 在 ≥1 任务注册为 `MANIPULATED_TARGET`；
- 有自由运动 joint（free joint，记录 joint type）；
- 有有效 collision geom（contype/conaffinity 非零）；
- 可抓取宽度在夹爪物理范围；
- 非 fixture/receptacle/static body。

## 2. 编译属性读取（裁决 §三）

- 质量/惯性：`model.body_mass[bid]`、`model.body_inertia[bid]`、`body_ipos`、`body_iquat`（局部值）；
- 关节：`body_jntnum`、joint name/type、qpos/qvel 地址；
- 摩擦：逐 collision geom 记录 `friction`/`solref`/`solimp`/`condim`；
- 碰撞几何：`geom_count`、collision geom 名、visual-only 名、contype/conaffinity。

## 3. 尺寸与可抓取宽度（裁决 §四/§五）

计算三种尺寸：
- **body-local AABB**（collision geom 转 body 局部坐标的 union AABB）；
- **principal-axis 尺寸**（PCA/OBB，`w1≤w2≤w3`）；
- **world AABB**（仅调试，不参与选择）。

候选抓取宽度 `W(o) = {w1, w2}`（两个较小 principal 维）。排除竖直厚度方向后，
选择与水平双指夹取兼容的最小宽度。无法确定抓取轴 → `GRASP_AXIS_AMBIGUOUS`。

## 4. 夹爪几何筛选裕量（裁决 §六）

预注册（非根据对象列表调整）：
```
closed_aperture_m = 0.001
open_aperture_m = 0.077
w_min = closed + δ_min = 0.001 + 0.005 = 0.006 m
w_max = 0.9 * open = 0.9 * 0.077 = 0.0693 m
```
0.9 是几何筛选裕量（非任务阈值），来源：模型分辨率 + finger pad 几何余量。

## 5. 脚本抓取可行性门禁（裁决 §七）

对每个候选对象执行固定脚本（正式 `env.step(action)` 路径）：
1. 标准化姿态；2. 夹爪接近；3. 闭爪；4. 持续双指接触；5. 低速抬升固定高度；6. 保持。

通过：双指接触持续 + lift 达固定高度 + 相对位姿稳定 + 无障碍接触 + 无 teleport + 无直接写 ctrl。
失败 → `SCRIPTED_GRASP_FEASIBILITY_FAIL`。

## 6. 选择算法（完全确定，裁决 §八）

1. **资格筛选**：排除非 target/无 free joint/无 collision geom/宽度超范围/抓取轴不明/脚本门禁失败；
2. **资产去重**：按 `physical_config_id = H(asset, scale, mass, inertia, friction)`；
3. **排序**：`estimated_grasp_width` 升序，tie-break `physical_config_id` 升序；
4. **分位选择**：`i_q = round(q * (n-1))`，`q ∈ {0.2, 0.5, 0.8}`（固定 rank 公式，不用库插值）；
5. **形状多样性**：若三者同 shape family → 保留 q20/q50，从 q80 附近按宽度距离找不同 family；
   找不到 → 保留原选择 + `SHAPE_DIVERSITY_UNAVAILABLE`；
6. **质量覆盖**：若质量跨度 < 预注册比例（max/min < 1.5），增选 `C4 = argmax mass`（仍须过全部门禁）。

## 7. shape family 映射（裁决 §九，结果前冻结）

按 object_class 到 shape family 的映射（预先冻结）：

```text
box_like        : alphabet_soup, bbq_sauce, chocolate_pudding, cream_cheese,
                  milk, orange_juice, tomato_sauce
bowl_or_cup     : akita_black_bowl
cylindrical     : porcelain_mug, white_yellow_mug
```

（映射来源：对象资产类别 + 几何形状；名称仅作初始分类，扫描后用 PCA 比例校验；
若 PCA 比例与映射冲突 → 记录 `SHAPE_FAMILY_PCA_MISMATCH` 并归 `unknown`。）

## 8. 排除原因枚举（裁决 §十一）

```text
NOT_MANIPULATED_TARGET
MISSING_BODY
NO_FREE_JOINT
NO_COLLISION_GEOM
GRASP_WIDTH_TOO_SMALL
GRASP_WIDTH_TOO_LARGE
GRASP_AXIS_AMBIGUOUS
SCRIPTED_GRASP_FEASIBILITY_FAIL
ROLE_MANIFEST_BDDL_MISMATCH
ENV_INSTANTIATION_FAILURE
```

## 9. 扫描效率（裁决 §十）

每唯一 task 只实例化一次 env，从编译模型提取该任务所有 manipulated targets，按
physical config 去重。相同资产多任务出现保留多实例交叉核对，属性不同拆 config。

## 10. 完整性（裁决 §十一）

报告 `tasks_total / scanned / failed`、`targets_declared / found / missing`、
`role_mismatches`、`assets_unique`、`physical_configs_unique`、`eligible_count`、
`excluded_count`。任何任务失败 → `INCOMPLETE_SCAN`，不生成 verified selected。

## 11. 排除的评测 anchor

```text
spatial/task1/state3
spatial/task1/state4
```

对象资产不是评测单元；排除的是具体 init/策略轨迹/结果，不排除 asset 本身。
