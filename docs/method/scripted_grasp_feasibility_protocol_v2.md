# 脚本化抓取可行性门禁协议 v2（scripted_grasp_feasibility_protocol_v2）

> **状态：SUPERSEDED_BY_V3（2026-11-06，开发阶段被替代）**
>
> superseded_reason: 开发 MVP（cream_cheese）证明「full open → descend → close」模板
> 会侵入对象（尤其窄面对象），需引入 PRESHAPE（预闭合）状态。
> superseded_by: scripted_grasp_feasibility_protocol_v3
>
> 日期：2026-11-06。**在运行任何对象门禁前冻结**。
> supersedes: scripted_grasp_feasibility_protocol_v1（INVALID）
> protocol_sha256 记录本文件冻结时的 hash。
> 脚本门禁是 v2 对象选择的主要资格依据（方案 C+），几何算法仅作审计字段。

## 0. 相对 v1 的变更

| 变更 | 原因 |
|---|---|
| 导航分阶段（三段式） | 固定 40 步 timeout 不考虑距离；对角线直达接近语义不明确 |
| 距离相关 deadline + hard cap + stall detector | 固定步数对长距离导航不足 |
| 不增加 OSC 动作幅值 | 收敛慢是 OSC 动力学决定，增幅值导致冲击/过冲 |

## 1. 目标与非目标（同 v1）

判定「在注册标准化脚本 + 抓取模板下，对象能否被稳定双指夹持并抬升」。
不判定 release，不读 VLA 结果，不为单对象特调。

## 2. OSC_POSE 动作语义（同 v1）

```
Δp = 0.05 * a_pos m；Δr = 0.5 * a_ori rad
max_translation_per_step = 0.01 m（a_pos 上限 0.2）
max_rotation_per_step = 0.05 rad（a_ori 上限 0.1）
```

## 3. shape family（同 v1，冻结）

```text
box_like     : alphabet_soup, bbq_sauce, chocolate_pudding, cream_cheese,
               milk, orange_juice, tomato_sauce
bowl_or_cup  : akita_black_bowl
cylindrical  : porcelain_mug, white_yellow_mug
```

## 4. 抓取模板（同 v1，冻结）

- approach: 世界 -z（top-down）；
- closing axis: 水平 caliper 最小轴（第二模板第二水平轴）；
- 高度网格：box {0.50h}，bowl {0.40h, 0.55h, 0.70h}，cylinder {0.40h, 0.60h}；
- 模板顺序：EXTERNAL_PARALLEL → ORTHOGONAL_EXTERNAL_PARALLEL；
- canonical = 第一个通过全部条件的模板。

## 5. 导航状态机（v2 核心变更）

```text
INIT
→ MOVE_TO_TRANSIT_HEIGHT      （只控 z）
→ ALIGN_ORIENTATION_AT_TRANSIT （只控姿态，位置保持）
→ HORIZONTAL_TRANSIT           （只控 x,y，z 保持 transit）
→ DESCEND_TO_PREGRASP          （只控 z，到 grasp+0.10）
→ DESCEND_TO_GRASP             （慢速下降 + 提前接触门禁）
→ CLOSE
→ VERIFY_BILATERAL
→ LIFT
→ HOLD
→ PASS / FAIL
```

每阶段只控制必要自由度（裁决 §四）。

## 6. transit height（裁决 §三）

```
z_transit = max(z_eef_0, z_obj_top + h_clear)
h_clear = 0.10 m
z_transit ≤ z_max_workspace（超界 → TRANSIT_HEIGHT_UNREACHABLE，不自动降 clearance）
```

当前例子：z_eef_0=0.668 > pregrasp=0.546，故无需额外抬升，直接在当前高度对齐+平移。

## 7. 导航参数（v2 新增，裁决 §五/§九）

```yaml
navigation:
  strategy: VERTICAL_ALIGN_HORIZONTAL_VERTICAL
  transit_clearance_m: 0.10
  max_translation_per_step_m: 0.01
  max_rotation_per_step_rad: 0.05
  position_tolerance_m: 0.002
  orientation_tolerance_rad: 0.035
  effective_min_progress_m_per_step: 0.002   # 来源：cream_cheese MVP-1 开发轨迹（2-3mm/step 保守）
  deadline_settle_margin_steps: 20
  phase_hard_cap_steps: 150
  stall_window_steps: 10
  stall_min_error_reduction_m: 0.001
grasp_descent:
  max_translation_per_step_m: 0.005   # 下降更慢
```

deadline（裁决 §五）：
```
N_deadline = min(N_hard, ceil(D0 / effective_min_progress) + N_settle)
```
来源标注：`derived from cream_cheese MVP-1 development trajectory, not from characterization outcome`。

## 8. stall detector（裁决 §六）

窗口 W=10 步，`Δe_W = e_{t-W} - e_t`；若 `Δe_W < 0.001m` → `NAVIGATION_STALLED`。
进入最终容差区域（e < 2*position_tolerance）后关闭 stall 判断。

## 9. 姿态对齐（裁决 §五/§六，引用 v1 §5）

- closing axis 从运行时左右 finger pad 位置计算，不假设；
- 用旋转矩阵（data.xmat），不混用不同顺序的 quaternion；
- 目标旋转矩阵由 closing axis + approach axis + 叉乘构造，保证正交右手系（det≈+1）。

## 10. pregrasp 与 descend（同 v1 §5/§6，descend 慢速）

- pregrasp_clearance = 0.10 m；
- DESCEND_TO_GRASP 用 0.005 m/step 慢速，实时检查提前接触；
- 提前 finger contact → `PREMATURE_FINGER_CONTACT`；housing/link → `PREGRASP_NON_FINGER_CONTACT`。

## 11. close 命令与 H_close（同 v1 §7/§8）

- `env.step(action)`，gripper = +1.0（max close）；
- H_close_empty = 17（空夹爪 close 表征 15 trials，max_stable=13）；
- close 结束：双指稳定 + |ΔA|<0.001，或 timeout 22 步，或非法接触，或对象脱离。

## 12. lift / hold / 稳定判据 / effective aperture（同 v1 §9-§12）

- lift_height 0.05m，对象实际抬升 ≥0.04m，support contact 消失；
- hold 10 步，稳定窗口 5 步（双指持续 + 相对位姿 2mm/3°）；
- effective aperture = median(A_t)，波动 >2mm 失败。

## 13. 通过条件 / 失败原因（同 v1 §14/§15，新增导航失败标签）

新增：`TRANSIT_HEIGHT_UNREACHABLE`、`NAVIGATION_STALLED`、`TRANSIT_PATH_BLOCKED`。

## 14. canonical 重复要求（同 v1 §21）

canonical 模板重复 3 次，effective aperture 差异 ≤1mm、lift 时刻 ≤1 步、
first bilateral contact ≤1 步。不一致 → `SCRIPTED_GRASP_NONDETERMINISTIC`。

## 15. 身份字段 / raw 日志 / hash 规则 / 禁止对象特调 / 不读 VLA（同 v1 §16-§20）

## 16. 工作空间边界验证（裁决 §十）

每阶段检查 EEF 目标在工作空间、关节不接近 limit、横向路径不经过 distractors。
隔离场景下横向 transit 不应经过其他对象；若安全走廊不成立 → `TRANSIT_PATH_BLOCKED`，
不通过降低高度绕行（v2 无规划器）。
