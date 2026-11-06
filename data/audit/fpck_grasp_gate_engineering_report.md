# FPCK/CKG 脚本抓取门禁 v3 工程报告 — 遇到的机器人学根因问题

> 日期：2026-11-06。协议 v3 已冻结，正式实现已提交，但发现需上报的机器人学根因问题。

## 一、当前状态

| 项 | 状态 |
|---|---|
| 协议 v3（PRESHAPE） | 已冻结（sha=e042e0ef...） |
| 正式门禁实现 | 已提交（src/fpck/scripted_grasp_gate.py） |
| cream_cheese 沿 y 轴 | 成功（48.4mm lift）✅ |
| cream_cheese 沿 x 轴 | FAIL（对象被挤出）✅（验证 fallback） |
| 正式门禁可复现性 | ⚠️ 临界非确定（见下） |

## 二、发现的机器人学根因（3 项，超出「实现 bug」范畴）

### 1. PandaGripper sticky 控制无法「保持中间 aperture」

实测：close 到 aperture=0.067 → `act[-1]=0`（hold）10 步 → aperture 继续缩到 0.052。

根因：`format_action` 的 `current_action += [-1,1]*speed*sign(action)`。act=0 → sign=0 →
current_action 不变，但 current_action 已经是 close 目标，**物理夹爪继续朝 close 目标闭合**。
只有 `act=-1`（open）才能让 current_action 往 open 累积。

→ 预闭合「保持在中间开度」在此 gripper 上**物理不可行**（act=0 不是保持，是继续 close）。

### 2. close 累积速度远快于 OSC descend 速度

close 累积 speed=0.01/step（current_action 每个 control step 变化 0.01），而 descend
5mm/step。若 descend 阶段用 act=1（强制 close），夹爪在手指降到对象前就闭合穿过。

### 3. preshape 余量临界时的「不接触」停滞

descend 到对象中心时，若 aperture 恰使 finger tip 内侧距对象表面 <1mm（如 0.6mm），
则：
- EEF 到目标（err≈0）→ 位置停止；
- gripper act=0 → aperture 稳定；
- tip 内侧刚好在对象表面外 → 永不接触 → DESCENT_TIMEOUT。

这是 sticky gripper + 慢速 descend + 固定 preshape 余量的**临界非确定性**，
取决于对象位置/姿态的亚毫米级差异。

## 三、MVP 成功的精确参数（复现关键）

```python
closing_target = [0, 1, 0]     # 沿 y 轴（第二水平轴）
obj_width = 0.0426             # y 全宽
preshape 目标 = obj_width + 0.03（close_to 用 <= 停到 0.068）
descend: act[-1]=0（hold，sticky 缓慢闭），单指接触(lf or rf)即停
close: aperture 连续稳定（真正夹紧）
→ 抬升 48.4mm
```

## 四、为什么这需要上报（而非继续迭代）

1. **「预闭合保持」在物理上不可行**：协议 v3 的 PRESHAPE 状态机假设「闭合到目标开度并保持」，
   但 sticky gripper 做不到。需要重新定义 PRESHAPE 语义（如「闭合到开度后立即进入 descend，
   由 descend 的接触反馈决定夹紧时机」）。
2. **descend 与 close 的耦合**：需要「边降边闭」的精确速度协调，或改用接触力反馈闭环，
   这是机器人学抓取控制的核心问题，不是脚本参数调优能解决。
3. **临界非确定**：preshape 余量与对象宽度的临界组合导致「不接触停滞」，
   需引入「descend 到接触」的主动反馈（而非固定 descend 目标）。

## 五、可能的解决方向（供决策）

### 方案 1：放弃「预闭合保持」，改为「preshape 后立即降+闭（单阶段）」
- preshape 到「对象宽+大余量」→ 立即 descend（act 保持 close 方向但慢）→ 接触即夹紧；
- 简化状态机，避免「保持中间开度」的物理不可行。

### 方案 2：引入接触力反馈闭环
- descend 阶段用接触力/位置反馈，到「手指接触对象」即停并夹紧；
- 更鲁棒但需要访问接触力传感器，实现更复杂。

### 方案 3：用「对象宽度的直接夹取」代替「preshape 余量」
- 不预闭合，直接 descend 到对象中心上方，然后一次 close 夹住；
- 但 cream_cheese 窄面（1.78cm）已经证明「full open descend」会侵入对象。

## 六、对后续的直接影响

脚本抓取门禁是「对象选择（方案 C+）」的主要资格依据。在上述机器人学问题解决前：
- **不能**对 10 个正式候选对象批量运行门禁；
- **不能**生成 verified=True 的 TerminalValidationConfig；
- 表征实验 2/3/4/5/6 仍被阻塞。

## 七、诚实的结论

脚本抓取门禁的**导航 + 姿态对齐骨架**已经跑通（MVP-0/1/2/1b 全 PASS），但
**抓取执行（preshape/descend/close）**遇到 sticky gripper 的物理限制，需要
机器人学层面的设计决策（方案 1/2/3），不是继续迭代实现参数能解决。
