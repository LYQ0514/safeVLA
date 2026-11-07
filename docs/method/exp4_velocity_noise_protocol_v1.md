# 实验 4 静止速度噪声表征协议 v1

> 日期：2026-11-07。**在运行前冻结**。protocol_sha256 记录冻结 hash。
> SINGLE_OBJECT：cream_cheese。scene：CHARACTERIZATION_SCENE_V2。

## 0. 职责边界

实验 4 只估计**静止状态下的测量/仿真噪声底** `b_v, b_ω`，不决定最终
`v_max/ω_max`。最终阈值由实验 4（噪声底）+ 实验 5（正常落置稳定包络）+
实验 6（高速反例拒绝区间）联合冻结。

## 1. 固定时间采集（不循环定义稳定性）

每个条件：
- 固定 burn-in = 200 控制步（对象从摆放掉落到桌面稳定，实验 2/3 实测 ~100 步，取 2 倍裕量）；
- 固定 observation window = 100 控制步（= 2500 physics substep）；
- 全程记录，不根据观测速度提前结束。

时间长度本协议冻结，不看噪声曲线后选窗口。

## 2. 测量分辨率

- 主：physics substep（每 substep 记录 MuJoCo body cvel）；
- 辅：control step 聚合；
- 原生 cvel + 相邻 xpos/xquat 有限差分速度（一致性审计）；
- 记录：linear/angular velocity、接触对、支撑对象、penetration/contact count、
  COM 高度、control/substep/仿真时间。

## 3. 支撑条件（三个独立 condition）

```text
cream_table       —— 桌面
cream_plate       —— plate 上
cream_receptacle  —— receptacle 内
```

三者独立报告，不混成单一样本池。每个先过初始化门禁（无穿透/支撑接触语义正确/
无机器人接触/未越界/未倾覆/hash 匹配），失败 → `CHARACTERIZATION_INITIALIZATION_INVALID`。

## 4. 统计口径

线速度：`||v||_2`、`||v||_∞`；角速度：`||ω||_2`、`||ω||_∞`。
每条件输出：median、MAD、p95/p99/p99.9、max、max 时刻、连续超阈持续长度、raw trace hash。

噪声底：
```
b_v = max_c Q_0.999(||v||_2 | c)
b_ω = max_c Q_0.999(||ω||_2 | c)
```
保留绝对 max 用于异常审计。

## 5. 输出状态

```text
velocity_noise_characterization:
  verification_status: VERIFIED_WITHIN_SCOPE
  object: cream_cheese
  scene: CHARACTERIZATION_SCENE_V2
  linear_noise_floor / angular_noise_floor
  candidate_only: true
  final_terminal_thresholds: PENDING_EXPERIMENTS_5_AND_6
```

## 6. 后续联合冻结逻辑

完成实验 5/6 后，存在分离裕量才冻结：
```
max(b_v, v_settled_max) < v_max < v_unsafe_min
```
否则 `VELOCITY_THRESHOLD_NOT_IDENTIFIABLE`，不人工挑值。
