# FPCK/CKG 见证采集工程排查报告

> 日期：2026-11-06。排查 positive control 三候选「无见证」是工程 bug 还是真实结果。

## 结论：发现 1 个工程 bug，但 Gate 5 结论不变

排查发现并修复了 1 个真实工程 bug（task_contact 漏 gripper0），但修复后
**核心结论不变**：三个候选确实是「碰撞推动假成功」，π0.5 在 long task0 上
确实不做合法抓取。

## 发现的工程 bug

### task_contact 判定漏 gripper0

- **现象**：positive control 三候选的 lifecycle 都是 RESTING，first_task_contact=None，
  看似「π0.5 从未碰 soup」；
- **根因**：`task_event_bridge.py` 的 `task_contact` 判定用
  `body.startswith("robot0")`，但 gripper 的 body 名是 `gripper0_leftfinger`、
  `gripper0_rightfinger` 等（6 个 gripper0 body），导致夹爪接触 soup 时
  task_contact 恒 False；
- **修复**：`task_contact = bool(mon["task"])`（`mon["task"]` 已定义为
  robot（robot0+gripper0 geom）↔ TASK 物体接触）。

## 修复后的验证（state2）

| 字段 | 修复前 | 修复后 |
|---|---|---|
| first_task_contact | None | (50, 50) ✅ |
| final_lifecycle | soup=RESTING | soup=TASK_CONTACT ✅ |
| first_controlled_witness | None | None（仍不成立） |
| final_e1_label | TASK_INTEGRITY_FAIL | TASK_INTEGRITY_FAIL（不变） |

## 关键诊断证据（证明 CONTROLLED 不成立是真实结果）

task_contact=True 时（step 50-60）的见证字段：

| step | lift(mm) | two_finger | grip_close | soup_disp(mm) |
|---|---|---|---|---|
| 50 | +0.00 | False | False | 0.07 |
| 55 | +0.11 | False | False | 0.35 |
| 58 | +1.09 | False | False | 5.11 |
| 60 | +4.04 | False | False | 8.57 |

**证据链**：
1. soup 抬升 ≈ 0（从未被抬起），CONTROLLED 的 lift≥10mm 条件不满足；
2. two_finger=False 全程（从未双指接触），grip 条件不满足；
3. grip_close=False 全程（gripper 从未闭合）；
4. soup_disp 0.07→8.57mm（soup 被水平推动，不是被抓取抬起）。

**结论**：π0.5 只是「碰了 soup（task_contact=True）但没有闭合 gripper、没有双指接触、
没有抬起 soup」，然后通过推动让 soup 进 basket。这是真实的「碰撞推动假成功」，
不是见证采集 bug。

## 见证字段采集的另一个改进

排查中发现 raw_events 没记见证字段（target_lifted_mm / two_finger_contact /
gripper_command_close 等），已补记到 emit，便于审计。

## 最终判定

1. **工程 bug 已修复**：task_contact 漏 gripper0（导致 lifecycle 停在 RESTING 的假象）；
2. **CONTROLLED 见证不成立是真实结果**：π0.5 确实不合法抓取（gripper 不闭合、
   不双指接触、不抬 soup）；
3. **Gate 5 结论不变**：三个候选都是碰撞推动假成功，INCONCLUSIVE；
4. **需重跑 state3/state5**（旧结果 lifecycle 是错误 RESTING，需修复后代码修正）。

## 与 STG 实验的印证

π0.5 在 long task0（soup 紧邻 mug）的「成功」大量是碰撞推动假成功，与之前
STG G2.5 几何诊断「豁免几何后 π0.5 仍抓不到 soup」一致——这是 π0.5 在该场景的
真实能力边界，不是 FPCK 实现缺陷。
