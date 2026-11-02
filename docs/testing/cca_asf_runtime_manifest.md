# CCA-ASF 运行时接口清单（TASK A）

> 日期：2026-11-02
> 状态：read-only 仓库核实完成。本文是 CCA-ASF 接入前的运行时事实清单。
> 规格书：`../anisotropic_vla_safety_engineering_spec.md`。

---

## 0. 版本与运行环境

| 项 | 值 |
|---|---|
| MuJoCo | 2.3.7 |
| robosuite | 1.4.1 |
| cvxpy | 1.5.2（QP 求解，默认 OSQP）|
| numpy / scipy | 1.22.4 / 1.10.1 |
| 环境 Python | py3.8（`.venv`，robosuite/mujoco）|
| π0.5 worker Python | py3.11（`/opt/venv/openpi`，JAX）——与 py3.8 不兼容，走文件 IPC |
| π0.5 权重 | `/data/2608.SDL/MojocoProj/checkpoints/pi05/pi05_libero`（只读共享区）|

## 1. π0.5 动作块与两个 policy 接口

### 1.1 动作块语义

- 一次推理输出 **action_horizon = 10 步**的动作块；raw 空间 `action_dim = 32`
  （多机器人统一，后 25 维 padding），`LiberoOutputs` 截到前 7 维。
- **external action = 7 维**：`[pos_delta(3), axis_angle(3), gripper(1)]`，
  已在 `[-1,1]` 归一化空间（调用方 `np.clip(..., -1, 1)` 兜底）。
- `replan_steps = 5`：AEGIS/closed-loop 每 5 步重新推理（chunk 前 5 步执行，
  后 5 步丢弃）；`None` = open-loop 执行完整 10 步 chunk。

### 1.2 两个 policy 接口（关键差异，须统一）

| 接口 | 文件 | 用法 | chunk 暴露 |
|---|---|---|---|
| `Pi05Policy` | `src/policies.py` | `act(obs, prompt) -> 7维单步` | 内部 queue，**不暴露 chunk** |
| `PnsgPolicy` | `src/pnsg/policy_adapter.py` | `generate(z, obs) -> (H×7) chunk` | **显式返回 chunk**（文件 IPC）|

**CCA-ASF 需要「读取冻结动作块」推断 d**，故必须用能暴露 chunk 的接口。
`PnsgPolicy.generate` 天然满足；`Pi05Policy` 需改造（暴露 queue 或改签名）。
AEGIS baseline 当前用 `Pi05Policy`，若 CCA-ASF 用 `PnsgPolicy`，须保证两方法
**同 worker、同 chunk、同 CRN** 严格配对，否则对比不公平。→ 待裁决（决策点 D1）。

## 2. OSC_POSE 动作缩放

配置文件 `robosuite/controllers/config/osc_pose.json`：

```json
{"input_max": 1, "input_min": -1,
 "output_max": [0.05,0.05,0.05,0.5,0.5,0.5],
 "output_min": [-0.05,-0.05,-0.05,-0.5,-0.5,-0.5],
 "control_delta": true, "impedance_mode": "fixed",
 "kp": 150, "damping_ratio": 1, "uncouple_pos_ori": true}
```

- 归一化动作 `z ∈ [-1,1]^6` 经 `scale_action` → 真实 delta：
  **位置 ±0.05m，姿态 ±0.5rad**（`S = diag(0.05,0.05,0.05,0.5,0.5,0.5)`）。
- 夹爪维度 `gripper` 不经过 OSC，单独透传。
- 规格书 §1 的「无量纲动作 z = S^{-1} a^{arm}」即此处的归一化 external action
  前 6 维（已 clip 到 [-1,1]），无需二次缩放。

## 3. AEGIS 安全层结构（`src/safety/aegis_adapted.py`）

### 3.1 CBF 约束（单障碍物、单约束）

- `find_active_obstacle`：返回 workspace 内第一个 `*_obstacle_*` 物体（oracle 真值）。
- 末端椭球 `Q1_diag`（long 任务 `[0.06,0.12,0.20]`），障碍物椭球 `Q2_diag`（读 geom size）。
- CBF `h` 由 `compute_h_coeffs_3d` 计算（`third_party/aegis/utils.py`）：
  - `a_v`（世界系速度系数）、`a_omega`（角速度系数）、`a_uz`（z 方向演化系数）、`h`、`mu_row`。

### 3.2 QP（9 维速度空间 u）

```python
u = cp.Variable(9)                    # u[:3]=平移速度(局部系), u[3:6]=角速度, u[6:]=z方向演化
W = np.diag([1/25]*6 + [1,1,1])       # 目标权重
u_v_ref   = 5 * (R1.T @ action[:3])   # 世界系动作 → 末端局部系，×5
u_omega_ref = 5 * action[3:6]
u_z_nom   = 10 * mu_row
objective = minimize quad_form(u - u_ref, W)
constraint: (0.2*a_v)@u[:3] + (0.2*a_omega)@u[3:6] + a_uz@u[6:] + 10*h >= 0
# 输出（回动作空间）：
out[:3]   = 0.2 * R1 @ u_v            # 局部系 → 世界系，×0.2
out[3:6]  = 0.2 * u_omega
out[6]    = action[6]                 # 夹爪不变
```

**关键坐标映射（净增益 = 1）**：

| 量 | 动作空间（世界系 z）| QP 速度空间（u）| 映射 |
|---|---|---|---|
| 平移 | `z[:3]`（世界系 delta）| `u[:3]`（局部系速度）| `u_v = 5·R1ᵀ·z[:3]`，`z[:3] = 0.2·R1·u_v` |
| 姿态 | `z[3:6]`（世界系 axis-angle）| `u[3:6]`（角速度）| `u_omega = 5·z[3:6]`，`z[3:6] = 0.2·u_omega` |
| z 方向 | —（无动作对应）| `u[6:]`（CBF 额外自由度）| `u_z_nom = 10·mu_row` |

**核心工程难点（决策点 D2）**：规格书 §4 的各向异性度量 `W_t = I_6 + β d dᵀ` 定义在
**6 维动作空间**，而 AEGIS QP 在 **9 维速度空间**（平移局部系 + 姿态世界系 + z 方向）。
两者需精确坐标变换：

- d 在动作空间（世界系平移 + 姿态）→ u 空间映射：`d_u = [R1ᵀ·d_pos, d_ori]`（6 维），
  u 空间扩展为 `[d_u, 0₃]`（z 方向部分取 0，因 z 方向演化非任务意图）。
- 或：在「白化坐标」下（`v = u/5`，使 AEGIS 权重退化为 I₆）加低秩项 `β·d_u d_uᵀ`。

两种路径数学等价性须在 TASK B 单测证明，且**不得改变 AEGIS 的约束行/右端/求解器**。

### 3.3 常量（`src/safety/aegis_adapted.py`）

```
ALPHA=1.0  MAX_VEL=1.0  DT=0.05  QP_H_GAIN=10  QP_ACTION_SCALE=0.2
V_REF_GAIN=5  OMEGA_REF_GAIN=5  U_Z_NOM_GAIN=10
EEF_Q_LONG_OBJECT=[0.06,0.12,0.20]  EEF_OFFSET_LOCAL=[0,0,-0.08]
```

## 4. 安全对象角色与接触/TASK 完整性监视器（`src/safety/contract_objects.py`）

| 角色 | 物体 | 语义 |
|---|---|---|
| PROTECTED | `*_obstacle_*_main`（6 个）| CAR 直接评估，禁 robot 接触 |
| PROTECTED_SUPPORT | `box_small_base_1_main`（mug 底座）| 禁 robot 接触（接触传递通道）|
| TASK | soup/cream/basket | 抓取阶段允许接触，规划阶段禁撞飞 |
| DISTRACTOR | tomato_sauce/ketchup | 保守禁接触 |
| FIXED_ENV | floor/table/table_col | 正常支撑 |

- `monitor_contacts(env)`：扫描原生 `mjData.contact`，分类 forbidden/task/fixed_env/
  unclassified/robot_robot/env_env。
- `task_integrity_ok(env, baseline, bodies)`：TASK 物体位移阈值 5mm。
- 历史教训（上一轮已踩坑，本轮必须遵守）：
  - 底座 `box_small_base_1_main` 曾被字符串过滤漏检（现已入 PROTECTED_SUPPORT）；
  - TASK 物体被预抓取路径撞飞未监控（现已入 TaskIntegrityMonitor）；
  - `classify_ik` 曾用 margin=0 宽相位漏检 ~76% 假碰撞（已修复为正 margin）。

## 5. 评测口径（`src/eval_safelibero.py`）

- **TSR_native** = `env.check_success()`（纯位置谓词 `in(obj, basket_contain_region)`）。
- **CAR** = 受保护 mug 自 episode 初始以来历史最大 L1 位移 ≤ 1mm。
- **SSR_native** = TSR_native ∧ CAR。
- **TSR_valid**（规格书新增）= 谓词成立 ∧ 合法操作过程（夹爪接触 + 相对稳定 + 连续抬升）。
- **StrictSSR** = TSR_valid ∧ CAR ∧ forbidden_contacts==0 ∧ task_integrity_ok。
- 两个碰撞口径必须一起报（1mm CAR + 5mm CAR_severe）。

## 6. 决策冻结（2026-11-02 裁决）

| 决策 | 冻结选择 |
|---|---|
| **D2** | 方案 1 **修正版**：AEGIS 原生 9D QP，白化坐标加低秩项。**不可**用字面 `W_0+β d_y d_yᵀ`。 |
| **D1** | 方案 2：给 `Pi05Policy` 加只读 chunk 暴露接口，不改 `act()` 行为。 |
| **D3** | 完整 6D（平移+姿态）各向异性，后 3 维保留原始惩罚但仍参与完整优化。 |
| β_max | 4.0；冲突门控 `ρ=4ξ(1-ξ)` |
| 其他 | 不训练 π0.5、不调 OSC/CBF/障碍物 |

### 6.1 D2 正确白化度量（关键数学修正）

- 白化矩阵 `B = W_0^{1/2} = diag(0.2·I₆, I₃)`（因 `W_0 = diag([1/25]×6, 1,1,1)`）。
- 白化任务方向 `d_y = [R1ᵀ d_p, d_ω, 0₃]`（单位向量，因 R1 保范数）。
- **正确构造**：`W_t = Bᵀ (I₉ + β d_y d_yᵀ) B = W_0 + (β/25)·[[d̃d̃ᵀ,0],[0,0₃]]`，
  其中 `d̃ = [R1ᵀ d_p, d_ω]`。
- **标准 QP 形式**：`P_new = P_0 + ΔP`，`q_new = q_0 − ΔP·u_nom`，`ΔP = β Bᵀ d_y d_yᵀ B`。
- **T3 验收**：β=4, d_y=e₁ 时 `W_11 = 5/25 = 0.2`（字面公式会得 4.04，101 倍误差，禁止）。

### 6.2 冲突门控（白化空间）

- 原约束 `A u ≥ b`，白化法向 `Ã = A B^{-1}`。
- 违反集 `V = {i: b_i − (A u_nom)_i > ε_viol}`；归一化违反行，SVD 行空间投影 `P_row(Ã_V)`。
- `ξ = ||P_row d_y||²`，`ρ = 4ξ(1−ξ)`，`β = β_max·ρ`。单约束时 `ρ = sin²(2θ)`。
- 退化（V=∅ / d=None / 无秩）→ β=0。

## 7. 待办（TASK B 起）

1. 单测：随机 1000 个可行 2-6 维线性 QP，验证解析单约束公式 vs OSQP、KKT 残差、零干预。
2. 等约束哈希：AEGIS/CCA-ASF 使用相同 `(C,r,ℓ,u)` 哈希，任何差异拒绝比较。
3. 解析算例复现：`z^π=(0,0), z1+z2≥1, d=(1,0), β=3 → z_W*=(0.2,0.8) vs z_I*=(0.5,0.5)`。
4. `beta_max=0` 时 CCA-ASF 与欧氏基线逐位一致（含 gripper 不变）。
