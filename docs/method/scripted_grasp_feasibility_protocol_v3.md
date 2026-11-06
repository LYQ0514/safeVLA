# 脚本化抓取可行性门禁协议 v3（scripted_grasp_feasibility_protocol_v3）

> 日期：2026-11-06。**在运行任何正式候选对象前冻结**。
> supersedes: v1（INVALID）、v2（SUPERSEDED_BY_V3，开发阶段被替代）。
> protocol_sha256 记录本文件冻结时的 hash，门禁 artifact 必须引用。
> 脚本门禁是对象选择（方案 C+）的主要资格依据，几何算法仅作审计字段。

## 0. 相对 v2 的变更

| 变更 | 来源 |
|---|---|
| 引入 PRESHAPE（预闭合）状态 | cream_cheese 开发 MVP：full-open descend 侵入窄面对象 |
| 预闭合 aperture 形式化（对象宽 + 2δ） | MVP 实测对象宽 4.26cm 需预闭合到 对象宽+3cm |
| 最小轴→第二轴模板顺序 + 证据保留 | MVP：x 轴 FAIL（OBJECT_EJECTED），y 轴 PASS（48.4mm） |
| final close「碰到≠夹紧」 | MVP：瞬时双指接触不能判 PASS |
| 九项工程修复正式化 | 见 §8 |

## 1. 目标与非目标（同 v2）

判定「注册标准化脚本 + 模板下对象能否稳定双指夹持并抬升」。不判定 release、
不读 VLA、不特调对象。

## 2. OSC_POSE 动作语义（同 v2）

```
Δp = 0.05*a_pos m（上限 0.2）；Δr = 0.5*a_ori rad（上限 0.1）
max_translation = 0.01 m/step；max_rotation = 0.05 rad/step
```

## 3. shape family（同 v2，冻结）

```text
box_like: alphabet_soup, bbq_sauce, chocolate_pudding, cream_cheese, milk,
          orange_juice, tomato_sauce
bowl_or_cup: akita_black_bowl
cylindrical: porcelain_mug, white_yellow_mug
```

## 4. 抓取状态机（v3 核心：加入 PRESHAPE）

```text
INIT
→ MOVE_TO_TRANSIT_HEIGHT      （只控 z，gripper open）
→ ALIGN_ORIENTATION_AT_TRANSIT（只控姿态，gripper open）
→ HORIZONTAL_TRANSIT          （只控 x,y，gripper open）
→ DESCEND_TO_PREGRASP         （只控 z，gripper open，到 grasp+0.10）
→ PRESHAPE_AT_SAFE_HEIGHT     （预闭合到 A_preshape，对象上方无接触）
→ DESCEND_WITH_PRESHAPE       （保持 A≈A_preshape，接触即停）
→ FINAL_CLOSE                 （夹紧：双指 + aperture 稳定 + 非接触）
→ VERIFY_GRASP                （lift 前验证夹持）
→ LIFT                        （对象实际抬升 ≥4cm）
→ HOLD                        （保持 + 稳定判据）
→ PASS / FAIL
```

PRESHAPE 必须在对象上方无接触区域完成，不能在下降中临时调整开度。

## 5. 预闭合 aperture 形式化（§三）

```
w_geom(o,g)：几何预测宽度（沿 closing axis 的对象 caliper 宽度）
A_preshape = w_geom + 2*δ_clear
```

约束：
```
A_preshape < A_open（0.077）且 A_preshape > w_geom
超出 → PRESHAPE_APERTURE_UNREACHABLE
```

clearance 选择规则：
```
δ_clear = max(δ_geometry, δ_control, δ_pose)
δ_geometry = 0.005 m   # 几何宽度估计误差
δ_control  = 0.005 m   # aperture 控制误差
δ_pose     = 0.005 m   # 对象-夹爪中心定位误差
→ δ_clear = 0.005 m（每侧），A_preshape = w_geom + 0.01 m
```

**来源标注**：`derived_from_development_object`（cream_cheese MVP 用了 3cm 余量偏保守，
v3 形式化后每侧 0.5cm）。不是独立物理表征，正式候选运行前用无接触预闭合测试确认
`A_actual - w_geom > 0`。

## 6. 预闭合完成条件（§四）

闭环控制到 `|A_t - A_preshape| ≤ ε_A（0.002）` 且持续注册窗口，同时：
- 无 finger-target contact、无 housing-target contact、无对象位移；
- gripper qpos/aperture 有效；controller 命令改变物理状态；
- 未越过目标 aperture 继续闭合。
安全高度预闭合即接触 → 说明 pregrasp 高度或 tool-to-pad 几何错误。

## 7. 下降与 final close（§五/§六）

- 下降保持 `A≈A_preshape`（gripper action 维持目标 aperture，非持续 close 到零）；
  下降中单指提前接触导致对象移动 → 失败或 acquisition，不直接视为成功；
- FINAL_CLOSE 要求：deployed close intent + aperture 继续减小 + 左右指接触 +
  bilateral 持续 + 非手指无接触 + 对象未挤出 + relative pose 耦合 +
  aperture 受阻稳定（或后续 lift 验证）。**「碰到」是接触证据，不是控制证据**。

## 8. 九项工程修复（§九，正式化 + 回归要求）

| 问题 | 修复 | 回归测试 |
|---|---|---|
| closing axis = body -y | 旋转矩阵列布局用 -closing_world 作 body_y | 已知旋转矩阵下 closing axis 正确 |
| finger body ≠ 接触点 | 用 finger_joint*_tip 作参考点 | tip midpoint 与 body origin 区分 |
| 对象自接触误判 | 跳过 b1==b2 | 同对象 geom contact 不触发违规 |
| 导航重置夹爪 | nav 阶段加 gripper_mode | 阶段切换后 aperture 目标保持 |
| 碰到≠夹紧 | final close 用 aperture 稳定 | 双指瞬时接触不直接 PASS |
| stall 容差区 | 误差>2*POS_TOL 才判 stall | 容差内不被误杀 |
| 下降被对象挡 | 接触即停，非精确对齐 | 接触阻塞与到达区分 |
| 最小轴失败回退 | 第一模板 FAIL 后按顺序第二模板 | FAIL 后正确 fallback |
| 动态 deadline | ceil(D0/0.002)+20，hard cap 150 | 远距离正确 deadline |

## 9. 模板顺序与 canonical（§七/§八）

```
1. PRIMARY_MIN_CALIPER（最小水平 caliper 轴）
2. SECONDARY_ORTHOGONAL_CALIPER（第二水平轴）
```

- 不写「最小轴错误」，写「最小轴是第一几何候选，失败可能来自对象逃逸/摩擦/挤压，
  第二轴是预注册备用」；
- 第一个通过全部物理门禁的模板成为 canonical；
- 每个失败模板保留完整证据（preshape aperture、对象投影宽、final close aperture、
  接触、接触力、对象位姿、support contact、逃逸时刻、非手指接触、关键帧）。

## 10. canonical 重复要求（§十二）

canonical 模板独立运行 3 次，要求：
- semantic outcome 一致；
- effective aperture 差异 ≤1mm；
- lift 时刻 ≤1 控制步；
- first bilateral contact ≤1 控制步。
不一致 → `SCRIPTED_GRASP_NONDETERMINISTIC`，对象排除。

## 11. 开发回归清单（§十/§十二，正式候选前完成）

| 开发对象 | 要求 |
|---|---|
| cream_cheese（box） | x 轴稳定 FAIL（OBJECT_EJECTED）+ y 轴连续 3 次 PASS，aperture/lift/contact 稳定 |
| hollow container（bowl） | 至少一模板连续 3 次一致，几何宽与实际 aperture 可解释，无壁厚/外径混淆 |
| negative（不可抓） | 连续 3 次稳定 FAIL，原因一致，不因 fallback 被放行 |

三个开发对象均 DEVELOPMENT_ONLY，不进入分位选择。

## 12. 失败原因枚举（同 v2 + 新增）

```text
（v2 全部）+ OBJECT_EJECTED_DURING_FINAL_CLOSE + PRESHAPE_APERTURE_UNREACHABLE
```

## 13. 实现 hash（§十三，正式候选前冻结）

```text
protocol_v3_sha256, template_bank_sha256, controller_source_sha256,
geometry_source_sha256, contact_classifier_sha256, failure_router_sha256,
development_regression_sha256
```

正式候选运行时每个 run 引用这些 hash；代码变化 → 新 factory 版本 + 全量重跑。

## 14. 身份/raw 日志/hash 规则/禁止特调/不读 VLA（同 v2）
