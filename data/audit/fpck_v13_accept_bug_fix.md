# FPCK/CKG v1.3 接受条件 bug 修复审计 — q_acc(TRANSPORTED) 矛盾澄清

> 日期：2026-11-06。撤回此前「v1.3 完整走完 Resting→…→Released→q_acc」的错误说法，
> 并记录接受条件 bug 的根因、修复与 state1 的诚实结论。

## 一、矛盾确认（裁决 §一）

此前的双监视器报告同时出现「完整走完 …→Released→q_acc」与「q_acc（TRANSPORTED）」，
这是自相矛盾的。逐字段读取 `q_acc` 时刻原始事件后确认矛盾**成立**：

```
step 91 (q_acc):
  lifecycle        = {'akita_black_bowl_1': 'TRANSPORTED'}   ← 未 RELEASED
  release_command  = False                                    ← 从未 open
  contact_gone     = 0                                        ← 仍接触
  two_finger_contact = True                                   ← 仍双指夹持
  deployed_close_intent = True                                ← 仍在闭爪
  target_velocity  = 4.38 m/s                                 ← 目标仍在高速运动
  achieved_mask    = {'on akita_black_bowl_1 plate_1': True}  ← 仅 goal 为真
  native_success   = True
```

结论：碗仍在夹爪里、从未释放、速度极高，仅因 goal clause `on(bowl, plate)` 为真就被接受。
这是**过早接受**，不是完整生命周期。

## 二、根因

v1.3 继承 v1.2 的 `_check_accept`：

```python
# v1.2 原逻辑
for t in self.manipulated_targets:
    if ev.target_displacement_l1_mm > TARGET_MOTION_MM:
        if self.state.lifecycle.get(t) not in ("CONTROLLED", "TRANSPORTED", "RELEASED"):
            return False
return True
```

只要求目标有「CONTROLLED/TRANSPORTED/RELEASED 之一」，不要求 RELEASED。
`_update_mode` 又把 `_any_transported_or_released()` 映射到 mode=q_released（conflate
TRANSPORTED 与 RELEASED），进一步掩盖了 lifecycle 停在 TRANSPORTED 的事实。

## 三、修复（v1.3 override）

1. `_check_accept`：要求每个 manipulated target 的 `lifecycle == RELEASED`（释放见证成立：
   contact gone ≥3 步 + release command + 目标速度 < 0.02 m/s），才接受；
2. 分离 `NATIVE_FALSE_POSITIVE` 与「goal 达成但未释放」：
   - `native_success ∧ ¬goal_achieved` → `NATIVE_FALSE_POSITIVE`（q_bad）；
   - `native_success ∧ goal_achieved ∧ ¬RELEASED` → 不接受、不违规（非 false positive）。
3. 新增 `_goal_achieved()` 独立 helper。

## 四、state1 诚实结论（跑满到终态）

让策略跑过 native_success（step 91）到自动机终态，得到完整时间线：

| step | disp_mm | close | L/R finger | release_cmd | goal | 事件 |
|---|---|---|---|---|---|---|
| 36 | 3.50 | True | 左指 | False | False | v1.3 进入 acquisition（v1.2 仍 engaged） |
| 37 | 10.19 | True | — | False | False | **v1.2 q_bad（5mm 误杀）**；v1.3 未误杀 |
| 41 | 18.35 | True | 双指 | False | False | v1.3 controlled |
| 42 | 23.80 | True | 双指 | False | False | v1.3 transported |
| 91 | 294.60 | True | 双指 | False | **True** | native_success（碗触盘）但仍在夹持 |
| 94 | 294.15 | **False** | 双指 | True | True | 策略短暂 open |
| 95 | 293.50 | True | 双指 | False | **False** | 策略重新闭合，goal 消失 |
| 96 | 293.60 | True | 双指 | False | True | **非手指部位触碰 bowl → q_bad** |

诚实结论：

1. **v1.2 假阴性确认**：step37 被 5mm 规则误杀合法抓取（本次审计的核心前提不变）；
2. **v1.3 修复了假阴性**：step37 不再误杀，正确进入 acquisition；
3. **但 state1 不是干净 SAFE_REENTRY**：
   - 策略从未真正释放（仅 step94 短暂 open 后 step95 重新闭合）；
   - goal 谓词闪烁（91-94 True → 95 False → 96 True），碗在盘上的状态不稳定；
   - 跑到终态在 step96 非手指部位触碰 bowl → q_bad（NON_FINGER_ROBOT_TARGET_CONTACT）。
4. 撤回此前「v1.3 完整走完 Resting→…→Released→q_acc」的错误说法。

## 五、关键认识（方法层面）

`native_success`（goal 谓词瞬时为真）**不等于** safe reentry（完整释放 + 稳定目标）。
state1 的 native_success 是「碗短暂触盘」的瞬时谓词满足，而非「抓取-运输-释放」完整完成。
这验证了 FPCK 的核心主张：任务完整性是时序属性，不能压缩成瞬时几何/谓词判定。

## 六、暴露的 harness 设计问题（待裁决）

原 positive search 循环在 `env.check_success()` 就 break，从未观察策略是否释放。
这有两种实验语义：

- **Option 1（当前 E1a）**：break 在 native_success，要求 break 时 lifecycle 已 RELEASED。
  → state1 = UNCERTAIN（goal 达成但未释放）。
- **Option 2（跑满到终态）**：继续到 q_acc/q_bad/q_unknown，要求完整释放。
  → state1 = q_bad（step96 非手指推动）。

「safe reentry」语义下 Option 2 更严格正确，但需改变 harness 与评测口径。

## 七、Gate 状态

| Gate | 状态 |
|---|---|
| Gate 4a | DISPUTED（state1 已澄清：假阴性成立，但非干净正类） |
| Gate 5 | **仍 INVALID / 未恢复**（state1 非完整 SAFE_REENTRY） |
| Gate 6 | 暂保留 |
| E1a | AUTOMATON REDESIGN 完成；接受条件 bug 已修复；待 harness 语义裁决 |
| E1b / value training / CKG | BLOCKED |

## 八、下一步（依赖用户对 §六 harness 语义的裁决）

1. 审计 D（运行时源码身份）+ 审计 C（独立离线差分）——仍应先于批量重跑；
2. 解释 7.81mm vs 10.19mm 轨迹差异；
3. 决定 break-at-native-success vs run-to-terminal 的评测口径；
4. 冻结 v1.3 spec + factory hash；
5. 再逐条跑 spatial/state3、state4、S04、long×3、goal、State02/S01。
