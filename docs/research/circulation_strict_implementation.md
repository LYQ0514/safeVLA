# Circulation Inequalities 严格实现（对应论文数学形式）

> 日期：2026-11-04
> 论文：Gonçalves, Krishnamurthy, Tzes, Khorrami, "Control Barrier Functions With
> Circulation Inequalities", IEEE T-CST 32(4): 1426-1441, 2024.（arXiv:2307.09901）
> 代码：`src/safety/circulation.py`（严格数学模块）+ `src/safety/aegis_adapted.py`（接入）
> 测试：`tests/safety/test_circulation.py`（15 项，全部通过）

---

## 1. 论文的严格数学形式

CCBF-QP（论文式 (3)）：

$$u(q)=\arg\min_\mu \|\mu-u_d(q)\|^2 \quad \text{s.t.}\quad
N(q)^T\mu\ge\alpha(D(q)),\ \ T(q)^T\mu\ge\beta(D(q)),\ \ A(q)\mu\ge b(q)$$

核心定义（论文 Assumption 1 + §III-A）：

| 符号 | 定义 |
|---|---|
| $D(q)$ | closeness 度量（$D>0\Rightarrow q\notin\mathcal C$，D 大 = 远离障碍物） |
| $N(q)=\nabla D/\|\nabla D\|$ | 法向单位向量 |
| $\Omega\in\mathbb R^{n\times n}$ | **skew-symmetric orthonormal**（$\Omega^T=-\Omega=\Omega^{-1}$），只存在于偶数维 |
| $T(q)=\Omega N(q)$ | 切向（自动 $T\perp N$，因 $N^T T=N^T\Omega N=0$） |
| $\beta(D)$ | **递减**、$\beta(0)>0$、$\beta(D)\to-\infty$（如 $\beta=a-bD$） |
| $\alpha(D)$ | $\alpha(0)=0$、$\alpha(D)<0$（如 $\alpha=-\eta D$） |

## 2. 与旧启发式实现的差异（关键修正）

旧实现（已替换）：

```python
z_h = self.z_fixed[:2]                     # 末端→障碍物方向（水平分量）
T_world = circ_sign * rotate90(z_h)        # 旋转 90°
beta = circ_gain * (circ_activate - h)
constraints.append(T_local @ u[:3] >= beta)
```

**缺陷**：法向 N 用「末端→障碍物连线方向 `z_fixed`」，但论文严格要求
**N = ∇D/‖∇D‖**。数值验证（`tools` 内联脚本）：

- ∇h_world = R1 @ a_v（`a_v` 即 ∇h 在末端局部系的表示）；
- ∇h 与 z_fixed 的夹角约 **28°**（各向异性椭球下不重合）。

用 z_fixed 当法向会引入 ~28° 的系统性切向误差。

**严格实现**：

```python
N_local = a_v / ‖a_v‖                    # ∇h 方向（局部系）
N_world = R1 @ N_local
T_world = Ω_world @ N_world              # Ω = xy 平面 skew-symmetric orthonormal
T_local = R1.T @ T_world
rhs = V_REF_GAIN * gain * (activate - h) # β 定义在速度空间，×V_REF_GAIN 映射到 QP 变量
constraints.append(T_local @ u[:3] >= rhs)
```

## 3. 维度适配（论文 §IV-C 奇数维 workaround）

末端平移是 3D（奇数维）。论文脚注 1 证明奇数维 skew-symmetric 矩阵必奇异（不可逆），
故无法定义全局 orthonormal Ω。§IV-C workaround：在 n-1=2 维子空间 orthonormal，剩余
1 维奇异。

严格实现：Ω 在 **xy 水平面**（2D）嵌入 n=2 的 Ω1/Ω2（论文式 (6)），z 维奇异（零行
零列）。对应「绕障碍物的水平切向」，适合桌面操作。此选择已文档化在代码 docstring。

## 4. 量纲说明（严格）

论文约束 $T^T\mu\ge\beta(D)$，μ 是配置速度（末端局部平移速度 $v_{local}$）。

AEGIS 的 QP 变量 $u[:3] = V_{REF\_GAIN}\cdot v_{local}$（= 5·v_local），故约束写成
$T_{local}^T u[:3] \ge V_{REF\_GAIN}\cdot\beta(h)$，与 CBF 约束的缩放
（QP_ACTION_SCALE × V_REF_GAIN = 1）一致。旧实现缺了这 ×5，β 的有效强度被放大 5 倍。

## 5. 单元测试覆盖的论文性质（15 项全过）

| 性质 | 测试 |
|---|---|
| Ω 反对称（Ω^T = -Ω） | test_omega_2d/3d_skew_symmetric |
| Ω 正交（2D 子空间 Ω^T Ω = I） | test_omega_2d_orthonormal / 3d_xy |
| Ω 奇数维奇异（det=0，§IV-C） | test_omega_3d_xy_orthonormal_z_singular |
| N 单位化、退化返回 None | test_normal_local_is_unit / degenerate |
| T ⊥ N | test_tangent_orthogonal_to_normal |
| circ_sign 控制绕向（±） | test_tangent_sign_controls_direction |
| β 递减、β(0)>0、β→-∞、β(q_g)<0 | test_beta_* |

## 6. 严格实现后的重跑结果

| 配置 | 成功 | mug 位移（mm） |
|---|---|---|
| baseline | 3/5 | [10.8, 25.6, 32.6, 0, 10.5] |
| circ_ccw | 0/5 | [0, 3.3, 6.2, 0, 0] |
| circ_cw | 0/5 | [24.5, 0, 256.9, 21.7, 0] |

**结论不变且更强**：circulation（严格实现）仍**有害**——baseline 3/5 成功（正确几何下
能完成任务），而 ccw/cw 都 0/5 成功，且 cw 出现 256.9mm 灾难性 mug 位移（init2）。

这印证了前一轮结论：circulation 针对的「平衡点卡死」是四元数 bug 假象，正确几何下
baseline 已能完成任务，circulation 是画蛇添足。严格实现修正了法向定义（∇h 而非
z_fixed）和量纲（×V_REF_GAIN）后，结论从「ccw 有害、cw 部分有效」收敛为「双向都有害」。

## 7. 诚实边界

- n=5 init 最小验证，baseline 跨进程有 GPU 非确定性（init1/init4 两次跑互换），
  但「3/5 vs 0/5」的定性结论两次一致；
- 严格实现只覆盖论文的**单障碍物 circulation**；多障碍物（extra_obstacles）与
  §IV-A 的 softmin 多约束聚合未实现（AEGIS 场景是单 active obstacle，非必需）；
- 维度 workaround（xy 平面 2D）是工程选择，论文允许（§IV-C），但非唯一合法选择。
