# 标准物理表征场景协议 v2（CHARACTERIZATION_SCENE_V2）

> 日期：2026-11-07。**在运行任何正式对象门禁前冻结**。
> supersedes: characterization_scene_v1（SUPERSEDED：principal-axis 强制竖直被替代）。
> scene_protocol_v2_sha256 记录本文件冻结时的 hash。

## 0. 相对 v1 的变更

| 变更 | 原因 |
|---|---|
| canonical 姿态改为「资产注册默认姿态」 | v1 的「最长轴强制竖直」导致 tomato_sauce 站不稳（它长轴横放） |
| 删除 principal-axis→world-z 强制规则 | principal geometry 不知道底面/COM 偏置/开口方向 |

## 1. canonical orientation 优先级（裁决 §二）

### 第一优先级：资产注册默认姿态
使用 SafeLIBERO 对象类的 `rotation`/`rotation_axis` 字段（资产注册的默认姿态，
把 mesh 从原始方向转到 canonical upright）：
```
orientation_source = ASSET_REGISTERED_DEFAULT
source = 对象类 rotation 字段
```
记录：raw rotation、rotation_axis、source_file、source_sha256、quaternion_order。

### 第二优先级：注册任务姿态集合
资产无默认姿态时：收集该 physical config 在所有注册任务的初始 quaternion →
canonicalize 符号 → 去重 → 固定 task/state 排序 → 标准桌面逐个 settling 验证 →
第一个稳定者 = canonical。
```
orientation_source = FIRST_REGISTERED_STABLE_POSE
```

### 第三优先级：无稳定姿态
```
CANONICAL_STABLE_POSE_NOT_FOUND
```
对象从实验 2/3 排除，但不称物理不可抓。

## 2. 四元数 canonicalization（裁决 §五）

归一化 + wxyz + 第一个非零分量为正（q 与 -q 同姿态，统一符号）。

## 3. 标准位置高度（裁决 §六）

不能复用 reset 后的 z。按 canonical 姿态下的最低 collision point：
```
z_body = z_table - z_min_collision(R_canonical) + ε_clear
```
ε_clear 为数值穿透裕量，预注册（0.001m）。

## 4. settling 门禁（裁决 §七，独立工程门禁）

`SCENE_INITIALIZATION_SETTLING_GATE_V1` 只判断：无持续移动 / 无穿透 / 合法 support
contact / 无翻倒 / 无机器人接触。不依赖实验 4 的终态速度阈值（避免循环依赖）。

## 5. canonical pose manifest（裁决 §四）

生成 `characterization_canonical_pose_manifest_v1.json`，每 physical config 记录：
physical_config_id、object_class、position_rule、orientation_source、
source_reference、quaternion_order、quaternion、settling_gate（status/support_contact/
max_translation/max_rotation）。manifest 在正式 10 对象重跑前冻结。

## 6. 其余（同 v1）

- 标准水平位置 (0,0)，yaw=0；
- 其它对象移出工作区；
- 机器人标准初态；
- 初始化 teleport 合法（trial 前）；
- 止损规则（不调位置/yaw/模板/参数）；
- 场景 family（GRASP_RELEASE / ON_SURFACE / IN_RECEPTACLE）；
- 对象横放可接受（资产稳定自然姿态即可，不强制语义朝向）。

## 7. 结果标记

- `SCENE_V1_DEVELOPMENT_PASS`（cream_cheese）保留为开发，正式 manifest 需 v2 重跑；
- `SCENE_V1_INITIALIZATION_INVALID`（tomato_sauce，reason=FORCED_PRINCIPAL_AXIS_UPRIGHT_UNSTABLE）保留；
- 只有 v2 结果进入正式 object manifest。
