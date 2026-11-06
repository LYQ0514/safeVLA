# FPCK/CKG MVP 脚本化抓取门禁 — 开发对象验证完成

> 日期：2026-11-06。DEVELOPMENT_ONLY（cream_cheese 开发对象，不进入表征选择/统计）。

## 一、里程碑结论

脚本化抓取门禁的**完整工程骨架已跑通**，cream_cheese 沿 y 轴（第二水平轴）成功完成
「导航 → 姿态对齐 → 预闭合 → 下降 → 夹紧 → 抬升 → 保持」全流程：

| 阶段 | 结果 |
|---|---|
| transit height / 姿态对齐 / 水平 / pregrasp | CONVERGED |
| 预闭合（对象宽+3cm） | 无接触 ✅ |
| 下降（接触即停） | tip 到对象中心，无接触 ✅ |
| 夹紧 | aperture=0.0421 ≈ 对象宽 0.0426（真正夹持）✅ |
| 抬升 | 对象实际抬升 **48.4mm**（>40mm 门槛）✅ |
| 保持 | 双指持续，aperture 稳定 ✅ |

## 二、核心方法发现

### 1. 方案 B（预闭合）是正确的抓取流程

- 错误顺序：降到对象中心（手指 open）→ 闭爪 → 手指侵入对象 → 滑脱；
- 正确顺序：降到对象上方 → **预闭合到 对象宽+余量** → 下降到接触 → 完全闭合夹紧。

### 2. 最小轴不一定是好抓取轴

- cream_cheese 沿最小轴 x（1.78cm 窄面）：手指闭合时对象被挤出（夹不稳）；
- 沿第二轴 y（4.26cm 宽面）：夹持稳定，抬升 48.4mm；
- 验证协议 v2「最小轴 → 第二轴」模板顺序的正确性。

## 三、修复的工程问题（9 项，均为实现 bug，非协议参数）

| # | 问题 | 修复 |
|---|---|---|
| 1 | closing axis = body 局部 -y（非 +y） | 旋转矩阵列布局用 -closing_world 作 body_y |
| 2 | finger body ≠ 接触点 | 用 finger_joint*_tip 作接触参考点 |
| 3 | 对象自接触（b1==b2）误判 non_finger | check_finger_contacts 跳过自接触 |
| 4 | 导航阶段 act[-1]=-1 重置夹爪 | nav_phase 加 gripper_mode(open/hold/close) |
| 5 | close 判定「碰到≠夹紧」 | 改为 aperture 连续稳定（真正夹持） |
| 6 | stall detector 容差区误判停滞 | 误差>2*POS_TOL 才判 stall（协议 v2 §8） |
| 7 | descend「tip 精确对齐」被对象挡住 | 改为接触即停 |
| 8 | 最小轴夹不稳 | 按协议模板顺序转第二轴 |
| 9 | 导航距离相关 deadline | v2 协议（ceil(D0/0.002)+20，hard cap 150） |

## 四、已冻结的协议（3 份）

| 协议 | sha256 | 状态 |
|---|---|---|
| terminal_validation_characterization_protocol_v1 | 02936a14... | 有效 |
| scripted_grasp_feasibility_protocol_v1 | 3b666f93... | INVALID（固定 timeout + 对角线直达） |
| scripted_grasp_feasibility_protocol_v2 | 9fa4d0ca... | 有效（分阶段导航 + 距离 deadline） |
| characterization_object_selection_algorithm_v1 | 0490faa6... | 有效（对象选择） |

## 五、下一步

MVP 骨架已跑通。剩余工作：

1. **实现完整脚本抓取门禁**（把方案 B + y 轴逻辑固化成正式门禁，含全部通过条件 + 失败原因 + canonical 重复 3 次）；
2. **对全部候选对象运行门禁**（10 个 unique object_class）；
3. **用实际稳定 aperture 生成 v2 对象 manifest**（effective_grasp_width = median aperture）；
4. 回到 Commit 4 的实验 2/3（载物 open + contact gone）；
5. 最终生成 `TerminalValidationConfig`（verified=True）。

## 六、重要提醒

cream_cheese 是**开发对象**（DEVELOPMENT_ONLY），它的成功抓取**只验证工程骨架可行**，
不进入对象分位选择，不构成「方案 C+ 已通过」的结论。正式对象门禁需在实现冻结后
对全部候选对象统一运行。
