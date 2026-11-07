# 标准物理表征场景协议 v1（CHARACTERIZATION_SCENE_V1）

> **状态：SUPERSEDED_BY_V2（2026-11-07，principal-axis 强制竖直规则被替代）**
>
> superseded_reason: tomato_sauce 长轴横放才是自然稳定姿态，「最长轴强制竖直」导致站不稳
> （CHARACTERIZATION_INITIALIZATION_INVALID）。canonical 姿态应来自资产注册默认姿态。
> superseded_by: characterization_scene_v2
>
> 日期：2026-11-06。**在运行任何标准场景门禁前冻结**。
> 解决「对象抓取可行性对任务布局敏感」问题（裁决：不选 A 不选 B，独立标准场景）。
> scene_protocol_sha256 记录本文件冻结时的 hash。

## 0. 目标

为实验 2（载物 open）/ 3（contact gone）提供**统一、单对象、可复现**的物理表征场景，
使对象差异只来自几何/质量/惯性/摩擦/collision shape，而非任务布局。

## 1. 复用组件（SafeLIBERO 真实物理）

- Panda 机器人 + PandaGripper + OSC_POSE 控制器；
- MuJoCo 2.3.7，control_timestep 0.05，physics_timestep 0.002（25 substeps）；
- 桌面/重力；
- 原始对象 asset（collision geom / 质量 / 惯性 / 摩擦）。

## 2. 移除组件

- distractor / 障碍物 / 任务语言 / VLA / multi-object goal / 随机任务布局。
- 每次只放置**一个被测对象**（+ 一个支撑面）。

## 3. 对象标准位姿（结果前冻结）

### 水平位置
固定居中、可达、非奇异：
```
p_char = (x_c, y_c) = (0.0, 0.0)  # 世界系，桌面中心（各对象统一，不单独调整）
```

### 高度
对象最低 collision point 落在支撑面上（不把 COM 设桌面高度）：
```
z_object = z_table - z_min_local
```
（z_min_local 从对象 collision geom 的最低点算出）

### 姿态（canonical upright，shape-family 规则预先冻结）
```
box_like    : 最长轴竖直（body-local z 对齐世界 z），yaw=0
bowl_or_cup : 开口向上（asset 默认 upright），yaw=0
cylindrical : 对称轴竖直，yaw=0
```
水平 yaw 固定 ψ=0，不为对象手工调整。无明确 upright → `CANONICAL_ORIENTATION_UNDEFINED`。

## 4. 机器人标准初态（所有对象统一）

- arm qpos = 中性位姿（远离奇异/joint limit）；
- gripper open；
- EEF 在对象上方可 reach 的固定 start pose；
- controller/mocap 状态初始化。

## 5. 初始化流程

1. 加载场景（robot + 支撑面 + 被测对象）；
2. teleport 被测对象到标准位姿，其余 free-joint 对象移出工作区；
3. 机器人 teleport 到标准初态；
4. `sim.forward()`；
5. 无动作 settling（确认对象稳定）；
6. 保存 characterization anchor；
7. 从 anchor 开始脚本抓取门禁（之后所有动作走 env.step）。

初始化 teleport 合法（trial 开始前）；计时开始后禁止 teleport。

## 6. settling 门禁

对象初始化后检查：
- 无穿透桌面、无初始机器人接触、无持续滚动、姿态稳定、无 NaN、身份正确。
失败 → `CHARACTERIZATION_INITIALIZATION_INVALID`（标准场景不适合该资产，非对象不可抓）。

## 7. 对象门禁（同 frozen gate）

每对象用完全相同：机器人初态 / 对象水平位置 / 支撑面 / 脚本门禁 / 模板顺序 /
sticky 映射 / 控制参数 / 超时规则。对象差异只来自物理属性。

## 8. 场景 family

```
CHARACTERIZATION_SCENE_V1:
  GRASP_RELEASE    （实验 2/3：抓取-释放）
  ON_SURFACE       （实验 5：plate 放置）
  IN_RECEPTACLE    （实验 5：receptacle 放置）
```
共享机器人/controller/物理参数/身份 hash。

## 9. 场景身份 hash

覆盖：robot XML/hash、gripper hash、controller config、arena/table hash、
object asset hash、support asset hash、初始 robot qpos、初始 object pose、
control frequency、physics timestep、settling procedure、scene builder source hash、
MuJoCo version。

## 10. 止损规则

标准场景运行后不调整对象位置/yaw、不改机器人初态、不增模板、不改抓取参数。
对象无法通过只记录分类。

## 11. 当前结果标记

- `MVP_INIT_STATE_LAYOUT` / `RESET_DEFAULT_LAYOUT` 结果保留为探索性；
- 只有 `CHARACTERIZATION_SCENE_V1` 结果进入正式 object manifest。
