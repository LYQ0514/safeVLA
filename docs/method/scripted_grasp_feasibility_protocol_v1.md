# 脚本化抓取可行性门禁协议 v1（scripted_grasp_feasibility_protocol_v1）

> **状态：INVALID（2026-11-06 标记，不覆盖文件）**
>
> invalid_reason:
>   - FIXED_PREGRASP_TIMEOUT_NOT_DISTANCE_AWARE（固定 40 步 timeout 不考虑初始距离与 OSC 收敛速度）
>   - DIRECT_DIAGONAL_TRANSIT_NOT_PHASE_SEPARATED（对角线直达未分阶段，接近语义不明确）
>
> superseded_by: scripted_grasp_feasibility_protocol_v2
>
> MVP-1 证据保留：
>   - short waypoint（0.03m）：PASS，14 步
>   - long pregrasp（~0.14m）：TIMEOUT_WITH_PROGRESS（非 NAVIGATION_FAILURE），
>     最终垂直误差 ~0.03m，无接触，对象位移 0

> 日期：2026-11-06。**在运行任何对象门禁前冻结**。
> protocol_sha256 记录本文件冻结时的 hash，门禁 artifact 必须引用。
> 脚本门禁是 v2 对象选择的主要资格依据（方案 C+），几何算法仅作审计字段。

## 1. 目标与非目标

**目标**：判定「在注册的标准化脚本 + 抓取模板下，对象能否被稳定双指夹持并抬升」。

**非目标**：
- 不判定完整任务 release（那是 Commit 4 实验 2/3 的后续工作）；
- 不读取 VLA 策略结果、成功率或失败率；
- 不为单个对象特调轨迹；
- 不把「能否抓」当作「抓得容易」——门禁是布尔资格，不是难度排序。

## 2. OSC_POSE 动作语义（正式记录）

```
位置增量 Δp_t = 0.05 * a_pos_t  m（a_pos ∈ [-1,1]^3）
姿态增量 Δr_t = 0.5  * a_ori_t  rad（a_ori ∈ [-1,1]^3）
```

脚本导航动作 = 当前 EEF 误差归一化 + clip。

**脚本轨迹参数（区别于 VLA/CKG 动作限制）**：
```
max_translation_per_control_step_m = 0.01   （即 a_pos 上限 0.2）
max_rotation_per_control_step_rad = 0.05    （即 a_ori 上限 0.1）
```

## 3. shape family（引用对象选择算法 v1 §7，冻结）

```text
box_like     : alphabet_soup, bbq_sauce, chocolate_pudding, cream_cheese,
               milk, orange_juice, tomato_sauce
bowl_or_cup  : akita_black_bowl
cylindrical  : porcelain_mug, white_yellow_mug
```

## 4. 抓取模板生成（GRASP_TEMPLATE_BANK_V1）

所有模板由对象几何自动生成（PCA/caliper + 高度网格），**不人工指定抓取点**。

### 通用
- approach 方向：世界 -z（top-down）；
- grasp 中心水平位置：对象外壳（collision geom union）中心在 xy 投影；
- closing axis：水平 caliper 最小轴（第二模板用第二水平轴）。

### 模板顺序（固定，裁决 §六/§七）
1. `EXTERNAL_PARALLEL`（沿最小水平 caliper 轴闭合）；
2. `ORTHOGONAL_EXTERNAL_PARALLEL`（沿第二水平轴闭合）。

### 高度网格（固定，裁决 §五/§六/§七）
```text
box_like    : {0.50h}
bowl_or_cup : {0.40h, 0.55h, 0.70h}
cylindrical : {0.40h, 0.60h}
```
按模板顺序 → 高度网格顺序 测试；**第一个通过全部条件的模板成为 canonical**。

## 5. pregrasp 参数（裁决 §九）

```yaml
pregrasp_clearance_m: 0.10
position_tolerance_m: 0.002
orientation_tolerance_rad: 0.035   # ≈2°
max_pregrasp_steps: 40
max_descend_steps: 30
```

无法达到 pregrasp → `PREGRASP_UNREACHABLE`（不自动降低高度）。

## 6. descend 参数

- 从 pregrasp 垂直下降到 grasp 高度；
- 速度同脚本导航上限（0.01 m/step）；
- 超时 → `DESCENT_TIMEOUT`。

## 7. close 命令（裁决 §十一）

- 使用正式动作接口 `env.step(action)`，gripper 分量 = CLOSE_COMMAND（max close）；
- 不能直接写 joint target / ctrl；
- close 命令值 = `+1.0`（PandaGripper sticky，sign>0=close，已验证）。

## 8. H_close 表征依赖（裁决 §十/§二十二）

空夹爪 close 响应表征（`tools/fpck_characterize_close.py`，15 trials）：
- `max_observed_stable_steps = 13`；
- `H_close_empty = ceil(1.2*13)+1 = 17 控制步`。

脚本 close 阶段在以下任一条件结束（裁决 §十，不等待完全闭合，因为载物时会被对象阻挡）：
1. 双指接触稳定 + aperture 变化趋近零（|ΔA|<0.001/step）；
2. 达到 close timeout = `H_close_empty + ΔH_object`（ΔH_object=5，载物裕量）= 22 控制步；
3. 出现非法接触；
4. 对象脱离可抓取区域。

注：close 到机械限位后继续 close 命令会导致 QACC 数值不稳定（已观测），
故「达到 closed 后停止 close」是必要实现细节，不是对象特调。

## 9. lift 参数（裁决 §十二）

```yaml
lift_height_m: 0.05           # EEF 目标抬升
max_lift_translation_per_step_m: 0.005
lift_position_tolerance_m: 0.003
max_lift_steps: 30
h_lift_min_m: 0.04            # 对象实际抬升最低要求（1cm 跟踪裕量）
```

通过条件必须是**对象**实际抬升（z_object - z_support ≥ 0.04），不是 EEF 抬升。

## 10. hold 参数（裁决 §十三）

```yaml
hold_control_steps: 10        # 10Hz 下约 1s
stable_window_control_steps: 5
```

## 11. 稳定抓取判据（裁决 §十四，最后 N_s=5 步全部满足）

- `C_L = 1` 且 `C_R = 1`；
- `h_lift ≥ 0.04 m`；
- 目标—EEF 相对位置偏差 ≤ 2mm；
- 目标—EEF 相对旋转偏差 ≤ 3°；
- 无 support contact；
- 无 forbidden contact（robot housing/link ↔ target）；
- 无对象掉落；
- close intent 仍存在。

**不依赖**未来实验 4 的稳定速度阈值（相对位姿稳定 + 接触持续即可）。

## 12. effective aperture 定义（裁决 §十五）

在最后 5 个稳定保持步：
```text
w_effective = median(A_t)
```
记录 `aperture_min/max/median/std`。

波动判据：`max A_t - min A_t > 2mm` → `EFFECTIVE_APERTURE_UNAVAILABLE`（抓取不稳定）。

## 13. 导航过程禁止提前接触（裁决 §十六）

close 阶段之前：
- finger 提前接触 → `PREMATURE_FINGER_CONTACT`（失败，不自动继续）；
- housing/link 接触 → `PREGRASP_NON_FINGER_CONTACT`（立即失败）；
- 对象被扰动 → 记录 `PREGRASP_OBJECT_DISTURBED`（第一版用接触本身判定，不用全局位移规则）。

## 14. 通过条件（裁决 §十七，16 条）

1. pregrasp 可达；2. orientation 可达；3. 接近无非法接触；4. close intent 正确下发；
5. aperture 物理响应；6. 左右指均接触；7. 无非手指 robot-target contact；
8. 目标离开支撑面；9. 实际 lift ≥ 4cm；10. 保持 10 步；11. 最后 5 步双指持续；
12. 最后 5 步相对位姿稳定；13. 无掉落；14. 无障碍物接触；
15. effective aperture 可计算；16. 身份和 raw 日志完整。

## 15. 失败原因枚举（裁决 §十八）

```text
PREGRASP_UNREACHABLE, ORIENTATION_UNREACHABLE, PREGRASP_TIMEOUT, DESCENT_TIMEOUT,
PREMATURE_FINGER_CONTACT, PREGRASP_NON_FINGER_CONTACT, CLOSE_COMMAND_NOT_APPLIED,
NO_APERTURE_RESPONSE, NO_LEFT_FINGER_CONTACT, NO_RIGHT_FINGER_CONTACT,
NO_BILATERAL_CONTACT, NON_FINGER_ROBOT_TARGET_CONTACT, CLOSE_TIMEOUT,
OBJECT_NOT_LIFTED, SUPPORT_CONTACT_PERSISTED, OBJECT_DROPPED, RELATIVE_POSE_UNSTABLE,
BILATERAL_CONTACT_LOST, FORBIDDEN_ENVIRONMENT_CONTACT, EFFECTIVE_APERTURE_UNAVAILABLE,
IDENTITY_FAILURE, RUNTIME_FAILURE
```

多个失败 → 保留 `primary_failure` / `all_failures` / `first_failure_step`。

## 16. 身份字段（每 trial）

```text
repo_commit, git_dirty, mujoco_version, controller_config_sha256,
physics_timestep, control_timestep, physics_substeps_per_control,
robot_model_sha256, object_asset_sha256, object_class, object_body_name,
protocol_sha256, template_index, height_frac
```

## 17. raw 日志 schema

每 trial 输出 append-only jsonl，逐控制步记录：
`step / eef_pos / eef_quat / gripper_aperture / close_intent / left_finger_contact /
right_finger_contact / non_finger_robot_contact / object_pos / object_quat /
object_vel / support_contact / lift_mm / relative_pose_error / failure_flags`。

## 18. canonical hash 规则

artifact_sha256 排除自身字段，canonical JSON（key 排序）。

## 19. 明确禁止对象特调（裁决 §七/§十九）

- 模板生成只看 shape family + 几何；
- 高度网格、模板顺序固定；
- 不根据对象名称选择高度/轴；
- canonical = 第一个通过全部条件的模板，不人工指定。

## 20. 明确不读取 VLA 结果

门禁只用脚本化 env.step 轨迹，不调用 PnsgPolicy，不读任何 VLA/成功率结果。

## 21. canonical 重复要求（裁决 §二十）

canonical 模板通过后重复 3 次，要求：
- semantic outcome 一致；
- effective aperture 差异 ≤ 1mm；
- lift 时刻差异 ≤ 1 控制步；
- first bilateral contact 差异 ≤ 1 控制步。
不一致 → `SCRIPTED_GRASP_NONDETERMINISTIC`，对象不能进入正式表征集。
