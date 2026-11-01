# H1-W 坐标定义与执行协议（v1）

> 日期：2026-11-01
> 状态：H1-W 最小判别实验的坐标定义（待执行）；H5 为新增候选机制，不改 H1–H4 判决标准。
> 前置：`docs/explanation/convex_safety_mode_mismatch_hypothesis_v0.md`

---

## 0. 决定（冻结）

- **物理见证空间**：固定夹爪的 6 维外部可执行连续动作 $a_c = (a_v, a_\omega) \in \mathbb R^6$。
- **代理约束空间**：AEGIS 9 维 QP 可行域，经固定状态线性投影到 6 维动作空间。
- **物理分支直接执行候选动作，不经过 AEGIS 重新过滤**（否则比较的就不再是原始三动作）。
- 现有数据范围：**仅 task0 可恢复快照**（`aegis_long_design/ep00-05/decision_*.npz`）。
  task3 缺少快照，暂不批准新增整条 rollout 采集。

---

## 1. 三种空间（严格区分）

AEGIS 执行链（已核对 `src/safety/aegis_adapted.py`）：

$$
u = \begin{bmatrix} u_v \\ u_\omega \\ u_z \end{bmatrix} \in \mathbb R^9,
\qquad
a = \begin{bmatrix} 0.2\,R_1 u_v \\ 0.2\,u_\omega \\ g_0 \end{bmatrix} \in \mathbb R^7
$$

| 空间 | 维度 | 作用 |
|---|---|---|
| QP 决策空间 $u$ | 9 | AEGIS 约束与优化 |
| 可执行连续动作 $a_c$ | 6 | 真实动作的非凸性见证 |
| 外部完整动作 $a=(a_c, g_0)$ | 7 | 实际下发环境 |

夹爪固定后，研究对象是 6 维连续动作空间。

---

## 2. 线性投影与保凸性（证明）

固定状态 $x$ 下 $R_1$ 为常量。定义线性映射

$$
a_c = P_x u,
\qquad
P_x = \begin{bmatrix} 0.2 R_1 & 0 & 0 \\ 0 & 0.2 I_3 & 0 \end{bmatrix} \in \mathbb R^{6\times 9}
$$

设 QP 可行域 $\mathcal U_{\mathrm{QP}}(x) = \{ u : A(x)u + b(x) \ge 0 \}$（其余硬约束均凸时它是凸集）。

**命题（线性投影保凸）**：$\widehat{\mathcal A}_{\mathrm{QP}}(x) = P_x \mathcal U_{\mathrm{QP}}(x)$ 是凸集。

**证明**：任取 $a_1 = P_x u_1$、$a_2 = P_x u_2$（$u_1, u_2 \in \mathcal U_{\mathrm{QP}}$），对任意
$\lambda \in [0,1]$：

$$
\lambda a_1 + (1-\lambda) a_2 = P_x[\lambda u_1 + (1-\lambda) u_2]
$$

由 $\mathcal U_{\mathrm{QP}}$ 凸性，$\lambda u_1 + (1-\lambda) u_2 \in \mathcal U_{\mathrm{QP}}$，
故其投影也在 $\widehat{\mathcal A}_{\mathrm{QP}}$。∎

**关键区分（必须写死）**：

- 命题对**可行域投影** $\widehat{\mathcal A}_{\mathrm{QP}}$ 成立（凸）；
- 命题**不适用**于 `filter(action, obs)` 这个「输入动作 → QP 最优解」的**算子**。
  该算子可以非线性，其**输出集合**不能未经证明等同于整个投影可行域。

---

## 3. H1-W 的三动作非凸性见证（物理空间）

在固定夹爪、同一完整初始状态 $x$、同一执行协议与时域 $H$ 下，构造

$$
a_L, a_R \in \mathbb R^6, \qquad a_M = \frac{a_L + a_R}{2}
$$

用 MuJoCo 原生动力学判断（不经过 AEGIS 过滤）：

$$
\Phi_c(x, a_L; H) = 1,\quad \Phi_c(x, a_R; H) = 1,\quad \Phi_c(x, a_M; H) = 0
$$

其中 $\Phi_c = 1$ 表示满足契约 $c \in \{\mathrm{disp}, \mathrm{contact}\}$（分别计算）。

**判定**：找到该见证 ⟹ 在所研究动作域与时域内，真实物理安全动作集非凸。
找不到 ⟹ 仅说明预设搜索域内未发现，不证明物理安全集凸。

### 3.1 代理约束侧的析取判定

对同一 $a_L, a_R, a_M$，映射回 QP 物理控制分量：

$$
u_v = 5 R_1^\top a_v, \qquad u_\omega = 5 a_\omega
$$

$u_z$ 用存在性可行性判定：

$$
\exists u_z : \begin{bmatrix} 5R_1^\top a_v \\ 5 a_\omega \\ u_z \end{bmatrix} \in \mathcal U_{\mathrm{QP}}(x)
$$

由命题 1（v0 文档），凸代理必然满足析取：

$$
a_L \notin \widehat{\mathcal A}_{\mathrm{QP}} \;\lor\; a_R \notin \widehat{\mathcal A}_{\mathrm{QP}} \;\lor\; a_M \in \widehat{\mathcal A}_{\mathrm{QP}}
$$

即：若物理安全标签成立，凸代理要么排除至少一个安全端点，要么允许不安全中点。

---

## 4. 三个必须分别记录的对象（不能混为一谈）

| 对象 | 定义 | 记录方式 |
|---|---|---|
| **真实物理安全集** $\mathcal A_{\mathrm{phys}}^{c,H}$ | 直接执行候选，MuJoCo 判定 | 物理安全标签（双契约）|
| **QP 投影可行域** $\widehat{\mathcal A}_{\mathrm{QP}}$ | 9 维可行域投影到 6 维 | $u_z$ 存在性判定（见 §3.1）|
| **实际过滤器输出** | `filter(a, obs)` 返回的动作 | 逐候选记录输出 + info |

> 「存在某个可行 $u_z$」**不等同于**「真实过滤器一定会输出该动作」。二者分别记录。

---

## 5. 执行协议（冻结）

1. 固定完整状态、真实 obs、时域 $H$、夹爪 $g_0$、后续控制规则与随机性。
2. 优先检查 $H=1$ 与 $H=5$ 两个控制步，**两个时域结论不得混合**。
3. **物理分支直接执行 $a_L, a_R, a_M$，不再经 AEGIS 过滤**。
4. 若 $H > 1$，必须预先固定候选动作之后的续行控制规则，避免把不同 $u_z$ 的
   控制器状态演化混入同一物理动作定义。
5. 位移契约的参考初始物体位姿**不能随分支恢复而重新设置基准**（保留官方口径）。

---

## 6. H5：辅助几何变量 vs 物理执行的安全实现差异（新增候选机制）

单条 CBF 约束写为（$a_v, a_\omega, a_z$ 是约束系数，不是动作）：

$$
a_v^\top u_v + a_\omega^\top u_\omega + a_z^\top u_z + 10h \ge 0
$$

分解为

$$
m_{\mathrm{physical}} = a_v^\top u_v + a_\omega^\top u_\omega + 10h,
\qquad
m_{\mathrm{aux}} = a_z^\top u_z
$$

**候选机制 H5**：是否存在

$$
m_{\mathrm{physical}} < 0, \qquad m_{\mathrm{full}} = m_{\mathrm{physical}} + m_{\mathrm{aux}} \ge 0
$$

即 QP 安全约束**依靠辅助几何变量 $u_z$ 的贡献才满足**，而真实机器人控制分量
（$u_v, u_\omega$）未获得充分安全改善。

**性质**：

- $u_z$ 控制椭球朝向演化（`dz = (I - z z^\top) u_z`，`z_fixed` 归一化更新）。
- 若 $u_z$ 的安全语义正确且动态被完整纳入安全证明，它可以是合法自由度。
- 若辅助几何调整与真实 robot–obstacle 安全之间缺少保证，则可能出现「扩展几何
  状态空间中约束满足，但真实物理执行未获相应安全改善」。

**第一个决定性证据**：QP 约束贡献分解（$m_{physical}$ vs $m_{aux}$）与对应候选的
物理安全结果对齐。

> H5 是新增候选机制，**不自动改变 H1–H4 判决标准**，也不在未核实 $u_z$ 对椭球动态、
> 安全函数与后续控制器状态的完整影响前，认定 AEGIS 存在缺陷。

---

## 7. 研究优先级（两个可独立验证的方向）

| 假说 | 核心问题 | 第一个决定性证据 |
|---|---|---|
| **H1**：物理安全集与凸 QP 代理的模式覆盖冲突 | 安全动作非凸结构是否造成错误排除/放行 | 同状态安全端点 + 不安全中点 |
| **H5**：辅助几何变量与物理执行的安全实现差异 | 是否靠 $u_z$ 满足约束却未产生充分物理保护 | QP 约束贡献分解 + 物理安全结果对齐 |

两假说可能解释不同失败行为，也可能最终关联。二者都需先完成 H1-W 坐标定义校验后，
在 task0 可恢复快照上做第一轮验证。

---

## 8. 状态

- **坐标定义冻结**，待 task0 快照上执行第一轮 H1-W 验证。
- 不新增整条 rollout 采集；task3 缺快照，暂不启动。
- 本实验是独立新假说的最小实验，不重开旧 PNSG / AEGIS 上游 / 源空间 steering 研究线。
