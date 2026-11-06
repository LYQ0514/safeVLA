# Terminal Validation 物理表征协议 v1

> 日期：2026-11-06。**本文件在运行任何表征实验前冻结**。
> 阈值选择规则先于结果确定，证明不是看到结果后修改的。
> protocol_sha256 记录本文件冻结时的 hash，最终 artifact 必须引用。

## 0. 目的

通过**脚本化控制器动作**（不调用 VLA 策略）测量夹爪/接触/静止噪声/放置 settle 的
物理响应，为 `TerminalValidationConfig` 生成 `verified=True` 的阈值。禁止使用未来评测
候选 state3/state4，禁止用成功率结果选择阈值。

## 1. 实验原则（裁决 §一）

1. 不使用未来评测候选 state3/state4；
2. 不使用 CKG 或成功率结果选择阈值；
3. 使用确定性 MuJoCo/controller 配置（脚本动作，无 VLA，无 GPU 非确定）；
4. raw 数据 append-only；
5. 阈值选择规则在查看结果前固定（见本文件 §十一）；
6. 正常放置与高速反例分开（characterization_normal vs characterization_adversarial）；
7. 每个阈值可追溯到具体测量字段；
8. 不成功的表征 trial 不静默删除。

## 2. 环境身份（每 trial 记录）

```text
repo_commit
git_dirty
mujoco_version
controller_config（OSC_POSE）
physics_timestep
control_timestep
physics_substeps_per_control
robot_model（Panda gripper body 名集合）
object_model（合成刚体，非 state3/state4 对象）
terminal_protocol_sha256
```

## 3. 实验对象（裁决 §五）

**不**使用 state3/state4 的具体对象。使用 SAFELIBERO 中与 spatial/task1 无关的
合成刚体 + 隔离场景。三个尺寸（近最小可稳定夹持宽度 / 典型 / 近上限），两个质量档。

## 4. 六个实验

### 实验 1：空夹爪 open 响应
- 初始：完全闭合 / 中间 aperture / 接近闭合；
- 机械臂姿态：中性 / 接近桌面 / 常见运输；
- 每 substep 记录：command_time、controller_target、左右 qpos/qvel、aperture、
  aperture_velocity、mechanical_limit；
- 量：`T_open,onset`（首次 Ȧ>ε）与 `T_open,stable`（aperture 达 open-ε 且持续）。
- **H_open 用 stable 时间，不是首次运动**。

### 实验 2：夹持对象后 open 响应
- 三尺寸 × 两质量（主实验尺寸，质量补充）；
- 记录：aperture 响应、左右 finger contact、目标-支撑面 contact、目标位姿/速度、
  最后 bilateral/任意 finger contact；
- 量：`T_open,loaded`。

### 实验 3：release 后 finger contact 消失
- 量：`T_contact_gone = t_last_finger_contact - t_open_command`；
- 记录首次左右 contact gone、bilateral gone、全部 gone、是否短暂恢复/掉落/弹跳/再碰；
- 候选窗口 {3,5,10,15} substeps。

### 实验 4：静止目标速度噪声
- 条件：桌面静止 / receptacle 内 / plate 上 / 边缘轻触 / settle 后 / 长时间静止；
- 每条件 5 初始化 × 5-10s，丢弃 settle 区间，保留 raw linear/angular velocity。

### 实验 5：正常放置 settling
- 脚本化合法序列：controlled grasp → transport → open → release → settle；
- 覆盖小/中/大对象、plate/receptacle、低释放高度、轻微位置/姿态偏差；
- 候选稳定窗口 {3,5,10} control steps。

### 实验 6：高速触盘反例
- 高线速触盘 / 高角速触盘 / 短暂 goal 后弹出 / 仍夹持时触盘 / release 后滚出 /
  接触后重新被带走；
- 验收：goal 可短暂为真，但不得进入 q_acc；违规才 q_bad；其余 pending 至 goal 失效/timeout。

## 5. 样本规模（裁决 §十）

| 实验 | 条件数 | 每条件重复 |
|---|---:|---:|
| 空夹爪 open | 3 姿态 × 3 aperture | 5 |
| 载物 open | 3 尺寸 × 2 质量 | 5 |
| contact gone | 与载物共用/独立 | 5 |
| 静止速度 | ≥5 支撑条件 | 5 |
| 正常 settling | ≥3 对象 × 2 目标面 | 5 |
| 高速反例 | ≥5 机制 | 3 |

## 6. 阈值选择规则（§十一，运行前固定）

### Open 响应上界
```
H_open = ceil(1.2 * max_i T_open,stable^(i)) + 1 控制步离散裕量
```

### Contact gone
```
N_release = 候选集合 {3,5,10,15} 中 > 观测到的最大 recontact gap 的最小候选
```

### 稳定速度
```
v_max = max(v_floor, Q_0.999(||v||) + m_v)
ω_max = max(ω_floor, Q_0.999(||ω||) + m_ω)
```
- `v_floor`、`ω_floor` 本协议预固定（见 §7）；
- `m_v`、`m_ω` 为数值分辨率安全裕量（本协议预固定，见 §7）；
- **不使用** state1 的 4.38 m/s 决定阈值。

### 稳定窗口
```
在 {3,5,10} 中选择：能接受所有正常脚本放置 + 拒绝所有高速触盘反例的最小窗口，
再加一个候选档位作为安全裕量。
若不存在候选同时满足两者 → 表征 FAIL，禁止任意创造新数值。
```

### H_post
```
H_physical = H_open + H_release + H_settle + N_stable
H_post = max(H_physical, H_chunk)
```
H_chunk 覆盖当前 chunk 剩余 + 一个完整后续 chunk。

## 7. 预固定常量（本协议冻结）

- 控制频率 `control_hz = 10.0`（OSC_POSE，control_timestep=0.05s）；
- `physics_substeps_per_control = 25`；
- `physics_timestep = 0.002s`；
- 空夹爪 open 的 `ε_A = 0.002`（aperture 稳定判定容差）；
- `A_open = 0.079`（open 后 aperture 目标，来自 Panda gripper 关节 range 实测）；
- 数值速度裕量 `m_v = 0.005 m/s`，`m_ω = 0.02 rad/s`；
- 速度物理 floor `v_floor = 0.01 m/s`，`ω_floor = 0.1 rad/s`；
- contact gone 候选集合 `{3, 5, 10, 15}`；
- 稳定窗口候选集合 `{3, 5, 10}`（control steps）；
- aperture 速度阈值 `ε_Ȧ = 0.001 / substep`。

## 8. 角色分离（裁决 §十二）

```text
characterization_normal     —— 实验 1/2/3/4/5（估计上界与噪声）
characterization_adversarial —— 实验 6（验证阈值不过早接受）
```

高速反例的极端速度**不得**抬高正常稳定阈值。

## 9. artifact 结构（裁决 §十三）

```json
{
  "schema_version": "terminal_validation_characterization_v1",
  "protocol_sha256": "...",
  "environment_identity": {},
  "experiments": { "empty_open_response": {}, "loaded_open_response": {},
    "contact_gone": {}, "stationary_velocity": {}, "normal_settling": {},
    "high_speed_counterexamples": {} },
  "selection_rule": {},
  "selected_thresholds": {
    "open_response_control_steps": null,
    "release_contact_gone_substeps": null,
    "linear_velocity_max_mps": null,
    "angular_velocity_max_radps": null,
    "stable_control_steps": null,
    "post_native_control_steps": null },
  "raw_files": [],
  "raw_data_sha256": "...",
  "artifact_sha256": "..."
}
```

artifact_sha256 计算时排除自身字段，使用 canonical JSON。

## 10. PASS 标准（裁决 §十四）

1. 六类实验全完成；2. 无缺失身份字段；3. raw hash 完整；4. 正常 release 全被接受；
5. 高速反例全不被接受；6. 阈值来自预注册规则；7. 未读 state3/state4；
8. 在线离线阈值一致；9. 正式配置引用 artifact hash；
10. `require_verified()` 通过；11. 改阈值 config hash 变；12. artifact/raw 变 verified 失效。
