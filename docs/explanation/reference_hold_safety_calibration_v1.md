# Reference-Hold Safety Calibration（v1）

> 日期：2026-11-01
> 状态：理论协议校准 + S0 工具修复验收标准。不启动新状态搜索。
> 前置：`intervention_timing_interface_audit.md`、`a0_goal_persistence.md`。

---

## 1. Execution Model（执行模型）

一个控制步的实际执行链（已由代码审计确认）：

```
VLA 参考 r_k (末端位姿增量, ±0.05m/±0.5rad)
  → AEGIS filter(r_k, obs)          # 每控制步一次，有状态(z_fixed,p2,计数器)
  → safe action
  → OSC set_goal (仅 j=0, achieved: g_k = ee_pos + Δ)   # 目标位姿
  → 25 substep:
       j=0..24:
         τ_{k,j} = run_controller(x_{k,j}, g_k)         # 每 substep 重算力矩
         τ_applied = clip(τ, τ_min, τ_max)              # 力矩饱和
         x_{k,j+1} = F_Δt(x_{k,j}, τ_applied)           # MuJoCo 动力学
```

**两个时间尺度**：安全判定时域 = 控制步边界一点；执行时域 = 25 substep。

## 2. Three Mathematical Quantities（三个研究量）

### 2.1 真实物理安全距离 d_phys

禁止接触的 native geom 对集合 $\mathcal F$（由安全契约决定，如 robot–obstacle 的
contype>0 geom 对）：

$$d_{\mathrm{phys}}(x) = \min_{(i,j)\in\mathcal F} \operatorname{sdist}(G_i(x), G_j(x))$$

**实现约束（MuJoCo 2.3.7，已核实）**：

- 本环境 MuJoCo **2.3.7 无 `mj_geomDistance`**（3.x 才有）。
- 因此表面距只能：
  - 接触时：`data.contact[].dist`（≤0，穿透量）；
  - 非接触时：**无原生正表面距**，`contact` 列表为空只表示「分离但未知正距离」。
- **禁止用「中心距」冒充 surface distance**（上一轮 0.16m 即此错误）。
- 若需正距离，须用保守几何近似（如 rbound 相减），并显式标记为近似 + 记录 distmax/截断。

### 2.2 几何代理保守裕量 b_geom

代理表面距离 $d_{\mathrm{proxy}}(x,z)$（z = AEGIS 代理几何状态），点态差距：

$$b_{\mathrm{geom}}(x,z) = d_{\mathrm{phys}}(x) - d_{\mathrm{proxy}}(x,z)$$

**硬约束**：

- 两距离必须**同单位、同受保护几何关系**。
- 不能把 AEGIS 归一化 CBF 值 h（无量纲，椭球规范化）直接与米单位 d_phys 相减。
- 区分「点态观测差距」（可算）与「已证明的保守裕量下界」（需几何包含关系 + 适用域）。

### 2.3 执行区间距离消耗 Γ_25

$$\Gamma_{25}(x_k, r_k) = \max_{0\le j\le25} \left[ d_{\mathrm{phys}}(x_k) - d_{\mathrm{phys}}(x_{k,j}) \right]_+$$

- $x_{k,j}$ 必须经真实 OSC + 力矩饱和 + MuJoCo 动力学。
- 区分 $\Gamma_{25}^{\mathrm{observed}}$（某次执行观测）与 $\overline{\Gamma}_{25}^{\mathrm{certified}}$
  （对规定状态/扰动/执行模型有效的可证明上界）。**只有后者可用于严格安全保证**。

## 3. Proxy-to-Physical Relationship（代理与物理的关系）

- **尚未证明** AEGIS 代理几何是真实几何的保守外包络。
- 当前只有「点态观测差距」，无「已证明的保守裕量下界」。
- 不能预设 $b_{\mathrm{geom}} \ge 0$；外包络关系未成立时它可能为负。

## 4. State Identity Contract（状态身份契约）

- **物理轨迹可重复** 与 **安全过滤器身份可重复** 是两个独立条件，分别验收。
- AEGIS filter 是有状态的：恢复须含 `z_fixed, p2, R2, Q2_diag, Q1_diag, 计数器, _last_p1`。
- **不得为记录指标而额外调用有副作用的 `filter()`**（S0 脚本 bug，见 §5）。
- 历史 $h=+0.00221$ 与重跑 $h=-0.0101$ 若来自不同 AEGIS 状态身份，标记为**不可直接比较**，
  不通过调参或反复 filter 逼近。

## 5. S0 Corrections and Acceptance（S0 修复与验收）

| 检查项 | 要求 | 上一轮问题 |
|---|---|---|
| S0-F1 AEGIS 状态身份 | 保存/恢复 filter 前完整内部状态 | restore 后 filter 又演化 z_fixed，h 漂移 |
| S0-F2 过滤调用次数 | 同一身份不因记录指标额外调 filter | 决策点+执行步各调一次，状态推进两次 |
| S0-F3 物理表面距离 | 用 native geom 距离/接触 dist，不用中心距 | 用中心距 0.16m 冒充 clearance |
| S0-F4 执行轨迹 | 记 25 substep 状态+距离+饱和后力矩 | 部分记录 |
| S0-F5 回归门禁 | 既有真值/快照恢复/缓存隔离全过 | — |

**硬条件**：若历史数据不足以恢复过滤前完整 AEGIS 状态，历史 h 与重跑 h 标记为不可比较。

## 6. 剩余执行裕量 M_exec（候选研究量，非安全证书）

在定义成立的区域：

$$M_{\mathrm{exec}}(x,r,z) = b_{\mathrm{geom}}(x,z) - \overline{\Gamma}_{25}(x,r)$$

- $M_{\mathrm{exec}} \ge 0$ **不是**完整安全证书，还需起始代理余量 + 适用几何关系 + 执行动态界共同成立。
- 它用于统一两个此前分离的问题：过度保守（任务可行性损失）与余量不足（执行风险）。
- 研究主张（待检验，非结论）：两者是否都能由「执行动态 vs 几何保护边界的不匹配」解释。

## 7. 状态

- 三量定义冻结；S0 工具待修复；不启动新状态搜索。
- 下一步：修复 S0-F1..F4 + 回归门禁 S0-F5，然后才谈预注册多状态实验。
