# CCA-ASF v1：四元数修复后严格实现、审计与评测规范

**文档版本**：1.0（2026-11-04）  
**状态**：POST-FIX REVALIDATION / 算法冻结，允许审计与重验，禁止按结果调参  
**代码基线**：四元数修复点 `b2226d9`；真正运行时必须记录 `HEAD`、工作区 dirty 状态、依赖版本及模型/场景哈希。  
**算法对象**：Frozen π0.5 + AEGIS 9D CBF-QP + CCA-ASF 各向异性目标；不包含在线 MPC、RRT、HVNP、策略再训练。  
**证据身份**：本文以项目历次提交报告/用户提供的技术事实为依据，**不是对当前仓库源码的逐行审计**。凡标记 `[SOURCE-CHECK]` 的项目，必须由实际代码、实测和可追溯记录核实；未核实禁止开展正式性能结论。

---

## 0. 最高优先级的不可违反约束（Contract）

1. **唯一算法差异**：CCA-ASF 仅在原 AEGIS QP 的 **9D 目标二次型**上添加白化后的 rank-1 正半定项，保持原问题的全部约束、变量、额外代价、求解器及执行映射不变。`euclidean` 分支必须复用原生 AEGIS 计算路径。
2. **不可更改安全契约**：不得改变 CBF 几何、障碍物、椭球尺寸、margin、阈值、OSC、夹爪、控制子步数、工作空间界限、回退策略、QP solver 参数。
3. **6D≠9D**：VLA 动作方向来自外部 6D（位置+轴角），但 QP 原生 9D，后 3 维继续参与原优化；不能重写成 6D QP，也不能将后 3 维冻结。
4. **白化坐标必须一致**：不得直接给原生 `u` 加 `β d_y d_yᵀ`；必须是 `ΔW=β(Bᵀd_y)(Bᵀd_y)ᵀ`。
5. **当前真实几何优先**：四元数及椭球中心 `p1` 必须在修复后独立核验。曾经约 16cm 的 `p1` 偏差足以使全部物理安全比较失效。
6. **零权重回归**：在完全相同 QP 输入下，`cca_asf(beta_max=0)` 与 `euclidean` 的优化问题和最终输出必须一致。
7. **严格配对**：相同 state/stream/初始 obs/初始动作 chunk；发生状态分叉后各自闭环重规划，禁止强制共用后续 chunk。
8. **指标分离**：Native TSR、Native CAR、Native SSR、Valid TSR、Forbidden Contacts、Valid StrictSSR 各自定义；严禁把 `success` 与 `CAR` 混用。
9. **算法冻结**：`beta_max=4`。`eta`、`intent_len`、权重窗、违反阈值、SVD 秩阈值等必须从真实源代码导出并记录；在当前 R1 中不允许根据结果更改。
10. **遇到不一致即停止**：非有限几何、非法坐标旋转、约束符号不明、首 chunk 不同、QP 约束不匹配或 CAR 口径不清，均禁止继续生成主结果表。

## 1. 研究目标及非目标

在冻结 π0.5 的名义动作 `z_t` 与原 AEGIS 安全可行域不变的前提下，根据**当前未过滤的 VLA 动作块**提取短时任务方向，在已经触发安全修正的 QP 中改变“修正分配方向”，以尽量减少动作沿代理任务方向的偏移。该方法**没有**建立真实语义任务意图、未来恢复负担或长期安全的保证。

- **目标**：相同状态/名义动作下，比较 Euclidean AEGIS 与 CCA-ASF 的 QP 修正、真实闭环任务收益、物理安全和计算成本。
- **非目标**：不替换安全约束、不引入新几何代理、不训练额外网络、不改 π0.5、不预先搜寻多条候选轨迹、不利用真值物体位姿进行额外在线规划。

## 2. 冻结的软件与动作接口

### 2.1 代码入口（必须在当前 HEAD 重新确认）

| 模块 | 预期职责 | 不允许的行为 |
|---|---|---|
| `src/safety/cca_asf/intention.py` | 未过滤 chunk → 6D 单位任务方向 | 向 worker 重新发起推理、读取已过滤动作、跨 chunk 用旧数据 |
| `src/safety/cca_asf/metric.py` | 坐标映射、约束冲突、β 与 `ΔW` | 修改 AEGIS 约束或物理状态 |
| `src/safety/aegis_adapted.py` | 维持原 QP，仅替换目标矩阵 | 自行重建约束、减去辅助变量、擅改 fallback |
| `src/policies.py` | `Pi05Policy` 只读 chunk/metadata | 改写 `act()` 的消费顺序、隐式推理 |
| `src/pnsg/policy_adapter.py` | 如采用 `PnsgPolicy`，提供冻结策略的确定性查询 | 混合两套不同动作约定而未核验 |
| `contract_objects.py`（实际路径待查） | Scene roles / Native CAR / forbidden 统计 | 将严格接触与官方 CAR 合并为一个成功标志 |
| `tools/cca_asf_requick.py` | Post-fix 重验入口 | 在对照分支中使用不同场景、超参或成功定义 |

**[SOURCE-CHECK]** 当前真实入口名称、调用签名、`filter_mode` 分支位置、`replan_steps`、实际外部动作消费逻辑、`eta/intent_len` 值。`filter_mode={euclidean,cca_asf}` 是已报告的**安全层**接口；`vanilla` 如存在，应在实验 runner 中绕过 QP，不能擅自声称它是安全层已有模式。

### 2.2 外部动作与执行语义

未过滤策略动作：

\[
z_t=(\Delta p_t^w\in\mathbb R^3,\; \Delta\theta_t\in\mathbb R^3,\;g_t\in\mathbb R),\quad z_t\in\mathbb R^7.
\]

- `z[0:3]`：**世界坐标系**平移增量的策略归一化控制量；OSC 平移尺度 `0.05`。
- `z[3:6]`：axis-angle 姿态增量的策略归一化控制量；OSC 旋转尺度 `0.5`；**旋转增量的参考系及组合次序须从现有 OSC 源码实证**，不能仅凭 axis-angle 名称假定。
- `z[6]`：gripper；不参与方向提取、不参与 CCA 的低秩项；保留原控制通路和夹持符号。
- 曾核实 AEGIS `u[0:3]≈5R_1^T z[0:3]`、`u[3:6]≈5 z[3:6]`，`u[6:9]` 为原 AEGIS 的三个辅助控制/演化维度；这些公式属于项目既有 QP 代理映射，**不允许从 0.05、0.5 或 25 个子步数重新“推导”一个比例覆盖原实现**。逐项比对真值。
- 历史报告为 `replan_steps=5`、每 outer step 25 个 MuJoCo 子步；以运行 manifest 的当前实际值为准并核对。

### 2.3 QP 变量含义

\[
u\in\mathbb R^9,\quad u=(v^l_x,v^l_y,v^l_z,\omega_1,\omega_2,\omega_3,a_1,a_2,a_3).
\]

最后三项用 `a_i` 代表**原有辅助维度**，不是随意添加的三个独立 slack，也不能假定其具体物理含义。应在 runtime manifest 中填写真实变量名、上下界、约束耦合关系。不可将 `u[6:9]` 设 0、固定为 `u_ref` 或从 QP 消去。

## 3. 数学定义：唯一允许的目标修改

### 3.1 原始基线

按既有 AEGIS 实现，主目标为

\[
J_0(u)=(u-u_{\rm ref})^\top W_0(u-u_{\rm ref})+J_{\rm other}(u),
\]

其中，在已报告的固定 9D 版本中：

\[
W_0=\operatorname{diag}(\underbrace{1/25,\ldots,1/25}_{6},\underbrace{1,1,1}_{3}),\quad
B=W_0^{1/2}=\operatorname{diag}(0.2I_6,I_3).
\]

**[SOURCE-CHECK]** 真实基线 `W_0` 是否确为上述矩阵；`J_other` 是否存在；`u_ref`（QP 目标中心）和 `u_nom`（用于约束冲突判别的未过滤名义输入）是否同一 9D 向量。即便二者数值相同，也必须分别命名以防后续误用。

安全约束写成统一符号：

\[
\mathcal U_t=\{u:A_tu\ge b_t,\; \underline u_t\le u\le\overline u_t,\; C_{eq,t}u=e_t\},
\]

其中 `A,b` 只是统一数学表述。实际 CVXPY 可能有其他约束、非 `>=` 符号，必须逐行导出并转换、标记原约束来源；**绝对不能为了冲突计算而更改送给求解器的约束**。

### 3.2 从冻结动作块提取任务方向

设当前 `act()` 关联的**未过滤** chunk 为 `Z=[z_0,...,z_{K-1}]`，当前消费索引 `k_t`。定义：

\[
v_z=\sum_{j=0}^{L_t-1}\omega_j z_{k_t+j,0:6},\quad
L_t=\min(\mathrm{intent\_len},K-k_t),\quad
d_z={v_z\over \|v_z\|_2}.
\]

以上是接口与归一化的数学模板；`ω_j` 的实际定义以及 `eta` 如何参与权重计算须以当前 `intention.py` 为准，写入配置，**不能把未知 `eta` 擅自解释为某种指数折扣**。如实际实现的聚合方式不同，必须先审查是否仍符合“当前/未来未过滤 chunk 的单一方向代理”并将真实公式固化于 manifest，而不是默默采用此模板。

必要约束：
- 索引 `k_t` 是本次名义动作实际对应的索引，不能使用已前进的下一个索引。
- 不包含上一个 chunk、未来尚未生成的 chunk、过滤后动作或 gripper。
- 输入、权重、输出必须 finite；方向范数过小则标记 `DEGENERATE_DIRECTION`，使 `β=0` 并退化到原生 Euclidean AEGIS；记录原因。
- 任何需要额外调用 worker 的方向获取实现均拒绝。

### 3.3 6D 外部方向 → 9D 白化坐标

在已核实的坐标接口下：

\[
J_t=\begin{bmatrix}5R_1^\top&0\\0&5I_3\\0_{3\times3}&0_{3\times3}\end{bmatrix}\in\mathbb R^{9\times6},
\quad
d_y={BJ_td_z\over\|BJ_td_z\|_2}=
\begin{bmatrix}R_1^\top d_{z,p}\\d_{z,\omega}\\0_3\end{bmatrix}.
\]

要求 `R1` 来自**当前** AEGIS 使用的同一参考系与状态，满足 `R1ᵀR1≈I`、`det(R1)≈+1`。当旋转 3D 项的实际坐标语义不符合上述块对角 `J` 时，必须暂停、核实和改写映射推导，不能继续把该公式当作真值。

`||d_y||=1`（数值容差内），`d_y[6:9]=0`。**不要对 6D 平移和旋转再乘 OSC 的 0.05/0.5 比例，导致重复缩放**；实际 `J` 的变换从原 AEGIS 接口核对。

### 3.4 当前约束冲突与门控

用原生名义 9D 向量 `u_nom` 对**预先冻结的参与门控约束行**评价违反量：

\[
r_i=b_i-a_i^\top u_{\rm nom},\qquad
I_{\rm viol}=\{i:r_i>\varepsilon_{\rm viol}\}.
\]

对选中约束进行白化：

\[
\widetilde A=A_{\rm viol}B^{-1},\quad
V_r=\operatorname{RowBasis}(\widetilde A),\quad
\xi=\|V_r^\top d_y\|_2^2\in[0,1].
\]

`V_r∈R^{9×r}` 是 SVD 得到的正交**行空间**基。若使用 `numpy.linalg.svd(Awhite)`，其右奇异向量的前 `r` 行转置得到 `V_r`；不能误用左奇异向量。若没有违反行或秩为零，设 `ξ=0`，并完整记录原因。

\[
\rho=4\xi(1-\xi),\quad \beta=\beta_{\max}\rho,\quad\beta_{\max}=4.
\]

这是**部分重合**门控，`ξ=0` 和 `ξ=1` 均得到 `β=0`，不是“越冲突越大”的单调函数。若多个约束法向覆盖任务方向，`ξ≈1` 时可能完全不激活。记录选中行数、数值秩、SVD 阈值、`ξ/ρ/β`。是否把 bounds / equality 行纳入门控**必须与既有 `metric.py` 一致**；扩充行集合会改变方法，R1 禁止这样做。

### 3.5 白化 rank-1 目标

\[
h=B^\top d_y,\qquad
\Delta W=\beta hh^\top,\qquad
W_t=W_0+\Delta W=B^\top(I_9+\beta d_yd_y^\top)B.
\]

保持原完整目标：

\[
\boxed{J_{\rm CCA}(u)=(u-u_{\rm ref})^\top(W_0+\Delta W)(u-u_{\rm ref})+J_{\rm other}(u).}
\]

CVXPY 的 `cp.quad_form(x,P)` 是 `xᵀPx`，**没有 `1/2`**。已报告原 AEGIS 使用 `cp.quad_form(u-u_ref,W0)`，因此 CCA 应使用 `cp.quad_form(u-u_ref,W0+ΔW)`，不得额外乘、除 2。若项目改为标准 `1/2 uᵀPu+qᵀu`，纯中心化主项对应 `P=2(W0+ΔW)`、`q=-2(W0+ΔW)u_ref`（常数可略去），另有成本项另算；**必须同时更新 P 与 q**。

单位测试黄金值：`β=4,d_y=e1` 时 `W_t[0,0]=0.04+4×0.2²=0.20`；错误原生 `W0+β d_y d_yᵀ` 会得到 `4.04`。白化坐标的特征值为 `1`（8 个）和 `1+β`（1 个），目标正定；`ΔW[6:9,:]=ΔW[:,6:9]=0`，但**最终优化的辅助 3D 仍可能由于约束耦合发生变化**，不应断言输出相同。

若 `β=0`，必须 `ΔW=0`，得到原基线完全相同的 CVXPY 问题，而非另写“近似等价”的基线 QP。

### 3.6 理论保证边界

严格保证的仅是：`W_t≻0`（当原 W0 正定、β≥0）、**QP 可行集不变**、β=0 的目标等价。若基础目标恰为白化欧氏修正的平方范数、可行集固定、最优解精确，保护方向的惩罚单调性成立；这不证明下游任务成功、真实碰撞避免、CAR 或 StrictSSR。QP 约束可行也不等于 MuJoCo 离散物理安全。

## 4. 四元数与 CBF 几何：强制先行审计

### 4.1 四元数的唯一准则

- MuJoCo 的 `quat` 数据是 scalar-first `[w,x,y,z]`；第三方旋转库的输入顺序必须逐一核对（例如 SciPy `Rotation.from_quat` 一般接收 `[x,y,z,w]`）。
- 优先使用同一个 `mjData` 当前时刻的 `xmat/xpos` 构建独立参考，避免把两个相同 bug 的旋转转换函数互相当 oracle。
- 四元数 `q` 与 `-q` 应产生相同旋转；必须归一化并审查非单位数值，旋转顺序、左右乘、body/world 与 local/world 坐标不能混用。
- 检验 identity、绕 X/Y/Z 的非零旋转、复合旋转、随机合法姿态；特别是 `p1` 局部偏移非零时的测试。

### 4.2 `p1` 独立校验

文档中 `p1` 指修复前出现约 16cm 错位的 AEGIS 椭球/约束参考中心。其具体公式和依附 body/site 必须从源代码找到：

\[
p_{1,\rm ref}^w=p_{\rm anchor}^w+R_{\rm anchor}^w\,r_{\rm local}\quad\text{（若源码确实使用固定局部偏移）}.
\]

以 MuJoCo 的 body/site world pose 独立计算 `p1_ref`，测试 `||p1_impl-p1_ref||` 达到项目预注册几何容差（建议数值几何内核级 `1e-6 m`，具体按浮点和源实现确认）。在 S01/S02/S04/S09 的实际轨迹上抽样验证，不只测单位旋转。记录 `p1`、`R1`、quaternion、anchor geom/body ID、offset、同一时刻 `qpos` hash。**发生任何厘米级错位直接 BLOCK**。

### 4.3 模型几何覆盖≠物理安全

确认 AEGIS CBF 具体覆盖哪些 link/obstacle，并作列表；原始方法未覆盖的 `gripper→box_small_base→mug` 间接碰撞链不可被错误描述为“QP 约束已经认证安全”。保持原 CBF 集合不变用于公平对照，新增严格物理监控**只负责评价**，不进入在线控制。`margin` 限幅查询无 contact 时可能只是下界，不能宣称精确间隙；旧版 MuJoCo 2.3.7 不应擅自调用 `mj_geomDistance`。

## 5. 策略调用、Chunk 时序与配对 CRN

1. 确定本轮唯一策略入口：`PnsgPolicy` 的确定性 CRN 版本**或** `Pi05Policy` 只读接口；不得一种算法走一个 wrapper。比较策略版本、checkpoint、图像预处理、动作归一化/反归一化、输出剪裁、episode reset。
2. 每个 `(task,state_id,stream)` 的实验条件完全一致；必须记录 `seed` 构造、worker 状态、query index、`chunk_sha256`、`buffer_index`、`nominal_action_sha256`。
3. `get_current_chunk()` 返回防御性拷贝，不额外推理、不能修改 `_chunk`、不能消费动作或推进索引。
4. 在 `act()` 消费前后明确锁定“本次动作对应的 chunk 下标”；任何 off-by-one 为 BLOCK。
5. 采用独立推理时，首个 chunk 逐项相同才可称为严格配对；否则用单次生成且**正确同步 query index/RNG/cache 状态**的首 chunk replay。不得仅粗暴设置 `_chunk`。
6. 首次 chunk 共享后，两条控制路径在不同真实观测下各自重新规划；不重放后续名义动作以制造虚假相等。
7. 同一物理/QP 快照的 β=0 对照可测试算法等价；两条已分叉的 episode 不要求逐步控制一致，也不能作为 β=0 等价性的主要证据。
8. 完整的物理状态、控制器状态、filter state、policy state、随机状态及必要 warm-start 一起恢复，才能声称单状态 fork；不完整 `sim.get_state()` 不足以证明反事实状态完全相同。严格测试优先从相同初始条件重跑。

## 6. 运行时计算顺序（不可倒置）

1. **reset**：从共同初始状态构建 env/policy/AEGIS；清零方向缓存、求解器状态按原方法 reset、计数器和指标历史。
2. **observe**：获取当前 `obs_t`、`mjData` 及真实 `R1,p1`；记录状态身份。
3. **policy**：用同一冻结 worker 得到未过滤 `z_nom[7]`；只读抓取本次 chunk + 精确索引。
4. **intent**：从未过滤 chunk 提取 `d_z[6]`；退化时显式标记。
5. **baseline QP**：原 AEGIS 从当前状态构造 `u_nom[9]`、`u_ref[9]`、`W0[9,9]`、全部原约束、bounds、其他项。
6. **CCA metric**：使用当前 `R1` 将 `d_z→d_y`；从冻结的门控约束提取已违反行，计算 `ξ,ρ,β,ΔW`。
7. **solve**：`euclidean` 保留原函数路径；`cca_asf` 仅将主目标 `W0` 换为 `W0+ΔW`。原 `cp.OSQP`、求解精度、warm start、fallback 保持一致。
8. **map-back**：原 AEGIS 将 9D 最优解映射成外部 6D 修正，拼回原 `gripper`；不能重新缩放、二次旋转或覆盖抓取控制。
9. **execute**：原 OSC 和 MuJoCo physics substeps；逐 substep 捕获 contact / mug 位移历史。
10. **log**：记录求解状态、约束残差、名义/执行动作、CBF 激活、方向及全部任务/安全指标。

**无效方向**可以退化为 Euclidean 并标记；**无效几何、CBF 矩阵非有限、输入错维、solver 非法状态**必须依原项目错误处理策略中断或显式报告，严禁悄悄输出未经评估的名义动作并标记为“安全”。

### 6.1 仅供核对的核心伪代码（不是替换真实源码的补丁）

```python
# Preconditions: metric.py implements mapping/gate identically;
# all arrays in native AEGIS 9D coordinates, original cost is centered.

def augment_original_qp(W0, u_ref, u_nom, d_z, R1,
                        A_gate, b_gate, beta_max, eta, intent_len):
    assert W0.shape == (9, 9)
    assert u_ref.shape == u_nom.shape == (9,)
    assert d_z.shape == (6,) and R1.shape == (3, 3)
    assert np.isfinite(W0).all() and np.isfinite(R1).all()
    # Call reviewed metric.py. It owns normalization, violation
    # tolerance, rowspace SVD threshold and degenerate fallback.
    d_y = map_dz_to_dy(d_z, R1)
    beta, gate_meta = compute_conflict_beta(
        d_y, u_nom, A_gate, b_gate, beta_max=beta_max
    )
    Bdiag = np.array([0.2] * 6 + [1.0] * 3)
    h = Bdiag * d_y
    delta_W = beta * np.outer(h, h)
    W_new = W0 + delta_W
    assert np.allclose(W_new, W_new.T)
    assert np.linalg.eigvalsh(W_new).min() > 0
    return W_new, gate_meta | {"delta_W": delta_W}

# In original filter():
# original_constraints = build_original_constraints(...)
# original_other_cost = build_original_other_cost(...)
# W_objective = W0 if mode == "euclidean" else W_new
# objective = cp.quad_form(u - u_ref, W_objective) + original_other_cost
# problem = cp.Problem(cp.Minimize(objective), original_constraints)
# problem.solve(solver=cp.OSQP, ... ORIGINAL solver arguments ...)
# output = ORIGINAL map_back / fallback / gripper path
```

实际函数签名、返回值顺序要按源码适配，**禁止照抄示例后未经代码核验直接运行**。尤其不可把非 finite `d_z` 的 fallback 和非 finite `R1/p1` 的致命错误当成同一类异常。

## 7. 安全、任务和接触统计的不可混淆契约

| 名称 | 评价对象 | 定义与审计条件 |
|---|---|---|
| `TSR_native` | benchmark 任务谓词 | 读取原官方任务成功函数；如 `success=True` 应标明这是何种 success |
| `CAR_native` | 官方 active obstacle | 从实际 `find_active_obstacle` 源码确认对象及历史 Dmax 阈值；本场景已报告主对象为 mug，不得假定 ketchup 也属于同一官方 CAR |
| `SSR_native` | 任务+官方安全 | `TSR_native AND CAR_native` |
| `TSR_valid` | 合法操作行为 | 通过已有 TASK integrity：合法接近、抓取、运输、放置；不能以碰撞导致物体进区当作成功 |
| `ForbiddenContacts_strict` | 额外全身禁止接触 | 按冻结 role 映射，统计 gripper–ketchup、robot–mug 等；标注允许的任务合法接触例外 |
| `StrictSSR_valid` | 严格联合成功 | `TSR_valid AND CAR_native AND (strict forbidden events == 0)` |
| `Dmax_mug_mm` | mug 的历史最大位移 | 从相同初始姿态定义，单位 mm；不得与最大接触深度、终点位移混淆 |

需从评测源码核验官方 CAR 具体对象、阈值和测量参考。此前报告出现 `Dmax=30.5mm` 但 `success=True`，强烈提示该 `success` 不是 CAR 或 SSR。**禁止在字段含义不明时将其解释为安全成功**。

- 真实场景中的 ketchup（约 `[-0.212,-0.132,0.509]m`）在 workspace 内，属于严格接触评估对象；远离 workspace 的 tomato_sauce（约 `[-10.15,-0.25,0.038]m`）不是实际近场障碍，不能因对象名出现在 XML 就视为已摆放。
- `box_small_base` 可通过接触链间接推动 mug；不能只监控 robot–mug 直接碰撞。
- 逐 physics substep 记录 `mjData.contact` 的 geom IDs、body/role、`dist`、`efc_address`；`efc_address` 合格时可用 `mj_contactForce` 取力。**持续接触多个 substep 不等于相同数量的独立碰撞事件**，需同时记录 substep-contact count 与连续区间 episode count。参见 MuJoCo 2.3.7 Simulation 文档。
- `mujoco mj_geomDistance` 在现有 2.3.7 项目环境中不可依赖；截断 margin 下的非接触结果是距离界而不是无限精度真间隙。

## 8. 配对实验与统计：修复后 CCA-R1

### 8.1 固定配置

- commit：`b2226d9` 后的明确 HEAD；记录 dirty status、Python/NumPy/MuJoCo 2.3.7/robosuite 1.4.1/CVXPY 1.5.2/OSQP、GPU、模型 checkpoint sha、场景 XML/BDDL 哈希。
- 策略：修复重验报告使用确定性 `PnsgPolicy` + CRN seed；正式对照必须固定同一路径，不能在不同方法中切换 wrapper。
- 方法：`euclidean` vs `cca_asf(beta_max=4)`；其他方向提取和求解配置全冻结。
- 小规模 Pilot：`safelibero_long` Long-I task0，`S01/S02/S04/S09 × 4 streams × 2 methods = 32 episodes`。这是**探索性样本**，不作为统计显著性或非劣效证明。
- 单对必须同 init、同模型、同初始 chunk/CRN 生成规则；每方法按自己的真实后续状态闭环重规划。运行顺序预注册或成对平衡，避免资源/热启动污染。

### 8.2 性能主指标和辅助指标

**主指标**：`SSR_native`、`StrictSSR_valid`，分别报告 success 2×2 配对翻转表；同时报告 `TSR_native`、`CAR_native`、`TSR_valid`，避免误把任务成功当安全成功。

**辅助指标**：mug Dmax 每对差值及尾部最坏例；严格接触区间次数与持续时长；实际 intervention rate（需归一化为合格 filter calls）；累计控制修正 `Σ ||B(u_safe-u_nom)||²`；`D_parallel=|d_yᵀ B(u_safe-u_nom)|`、`D_perp=||(I-d_y d_yᵀ)B(u_safe-u_nom)||`；β 正值率、`ΔW` 与 QP 解变化率；solver elapsed mean/P95、fallback 与 infeasible 数；gripper 与任务阶段。

**干预的三个口径不可合并**：实际执行动作变化 `||u_safe-u_nom||>ε`、CBF 约束激活、QP 求解调用。阈值 `ε` 从实现中冻结并记录。比较干预次数时记录每条 episode 执行长度与 eligible calls。

Post-fix 4-stream 简单重验已有数据：Euclidean Dmax `[25.6,30.5,8.8,16.0]mm`，CCA `[13.1,16.3,13.3,26.1]mm`；配对各有 2 个 stream CCA 较小/较大。介入计数 Euclidean `[432,171,136,152]`，CCA `[148,130,112,141]`。**仅为启动重验的信号**；既不支持“安全不劣”，也不能直接推出“少干预导致更安全”。若成功标识矛盾必须先修指标解释。

### 8.3 版本依赖

- 修复前 CCA E0 的 46/46 QP 方向保护属于历史数学机制证据；与错误 `p1` 相关的违反行、最优解和闭环结果不能充当 post-fix 的真实 QP 数据。修复后重新抽真实 QP 快照做同输入对照。
- 先前 `CCA 27–30mm vs Euclidean 0` 的灾难性比较**已撤销为修复前受污染证据**。但不能仅凭“修复四元数 + 切换为确定性 PnsgPolicy”的新旧比较证明错误完全由四元数 bug 单独造成。
- 依赖旧 AEGIS 的 M3-P1a/P1a-R 结果应标记 `PRE_FIX_EVIDENCE` 直至运行 SHA 审查；独立 2D M3-P0 不受此 bug 影响。

## 9. 分层验收测试矩阵（按次序阻断）

| Gate | 必验内容 | PASS 标准 / BLOCK 原因 |
|---|---|---|
| G0 源码身份 | baseline SHA、四元数修复 HEAD、模型/场景哈希、版本、solver 参数、未跟踪修改 | 缺失任何核心运行身份 → BLOCK |
| G1 动作语义 | 7D/6D/9D shape、world/local、axis-angle、gripper、`u↔z` 比例、25 子步 | 任何坐标或尺度未核实 → BLOCK |
| G2 几何 | `p1` 对独立 MuJoCo 世界系 oracle；非平凡旋转；`R1` 正交 | 绝对世界系几何偏差超过预注册数值容差 → BLOCK |
| G3 数学 | `β=0`、`W11=.20`、SPD、rank-1、aux block、中心化 CVXPY 目标、零/NaN 方向 | 任一失败 → BLOCK |
| G4 门控 | `Awhite=Aviol B^-1`、正确行空间、固定参与行、`ξ∈[0,1]`、零/满重合关闭 | 约束符号、违反阈值或参与行不明 → BLOCK |
| G5 QP 等价 | 同一 9D QP 快照：原 Euclidean vs CCA β=0，约束和输出一致 | 逐位可复现优先；至少 max-abs 误差 ≤ 1e-8 或预注册 solver tolerance |
| G6 真正激活 | β>0、`ΔW≠0`、同输入控制差异，核实 unchanged constraints | 只有 β>0 但无任何控制影响 → 不能声称机制作用 |
| G7 Policy/CRN | 首 chunk SHA、chunk[index]=act nominal、query index、reset、replan 节奏一致 | 首个 chunk 对不上或双推理 → BLOCK |
| G8 物理审计 | 按 substep 记录碰撞及间接传递；CAR 历史 Dmax；p1 跟踪 | 监控不全、角色不明 → 禁止安全结论 |
| G9 指标审计 | `TSR_native/CAR_native/SSR/Valid StrictSSR`、阈值和对象一致 | `success` 含糊、违反 CAR 却标 SSR 成功 → BLOCK |
| G10 Pilot | 16 对预注册对照、逐对差值、成功翻转、失败审计、性能与时延 | 只报均值、只报最优案例或基线不同版本 → BLOCK |

### 9.1 建议的关键自动化测试

- `test_quat_identity_xyz_mixed_and_sign_equivalence`：独立 MuJoCo world-transform oracle；覆盖 `q` 与 `-q`。
- `test_p1_world_anchor_under_rotation`：本地非零偏移，各类旋转；并在真实 S01 数据上核对。
- `test_local_velocity_frame`：`R1` 的转置方向和外部世界系平移映射。
- `test_axis_angle_frame_no_double_scaling`：原 OSC 单测比对原 AEGIS 反映射。
- `test_whitened_rank_one_intensity`：`β=4,dy=e1,W11=.20`。
- `test_conflict_rowspace_rotation_invariance`：对约束行的等价缩放和重复行，数值子空间不应不合理变化；需正系数缩放以保持不等式方向。
- `test_conflict_gate_endpoints_and_partial`：`ξ=0,1→β=0`，`ξ=.5→β=β_max`。
- `test_objective_center_and_factor`：CVXPY `quad_form` 对常量、线性项展开一致。
- `test_aux_dimensions_remain_optimized`：rank-1 项不直接惩罚 aux，但不能强制 aux 不变。
- `test_constraint_matrix_and_bounds_identity`：两方法原始约束对象/数值哈希、违反残差一致。
- `test_euclidean_legacy_regression`：新版 `euclidean` 与修复后未改动 AEGIS 路径相同。
- `test_beta_zero_complete_filter_equivalence`：同 QP 输入含 gripper 和 fallback，误差合格。
- `test_real_chunk_readonly_no_extra_inference`：worker query count 不变，索引、reset 无副作用。
- `test_nan_direction_fallback_not_nan_geometry`：退化方向可退化；非 finite 几何必须阻断。
- `test_metric_contract_tsr_vs_car`：注入 `TSR=True, Dmax` 超阈值，SSR 必须 False。
- `test_indirect_base_to_mug_contact_log`：至少能保留 gripper–base 以及 mug Dmax 变化时间线。

## 10. 必须落地的日志数据契约

每次实验：`run_manifest.json` 包含 HEAD/dirty、commit parents、Python/NumPy/CVXPY/OSQP/MuJoCo/robosuite、GPU、模型/场景 SHA、`beta_max/eta/intent_len`、SVD 与违反阈值、`replan_steps`、OSC 与物理子步数、CAR 规则、policy wrapper、CRN 规则、timestamp。

每个 outer step 至少记录：

```json
{
  "task": "safelibero_long_L0", "state_id": 1, "stream": 3,
  "mode": "cca_asf", "outer_step": 0,
  "query_index": 0, "chunk_index": 0, "chunk_hash": "...",
  "u_nom_9": [0,0,0,0,0,0,0,0,0],
  "u_ref_9": [0,0,0,0,0,0,0,0,0],
  "u_safe_9": [0,0,0,0,0,0,0,0,0],
  "d_z_6": [0,0,0,0,0,0], "d_y_9": [0,0,0,0,0,0,0,0,0],
  "xi": 0.0, "rho": 0.0, "beta": 0.0,
  "n_violated_gate_rows": 0, "rowspace_rank": 0,
  "qp_status": "optimal", "qp_residual_min": 0.0,
  "solver_ms": 0.0, "fallback_reason": null,
  "p1_world": [0,0,0], "R1_world_row_major": [1,0,0,0,1,0,0,0,1],
  "nominal_action_7": [0,0,0,0,0,0,0],
  "executed_action_7": [0,0,0,0,0,0,0],
  "mug_dmax_mm": 0.0,
  "tsr_native": false, "car_native": true,
  "forbidden_contact_substeps": 0,
  "forbidden_contact_episodes": 0
}
```

上述数值均为**schema 占位符，不是真实实验记录**。若日志巨大，可将大矩阵按预注册抽样或危险事件保存为 `.npz`，但必须记录散列、关联 step、采样规则以及完整关键事故窗口。不可只保存有利于方法的步骤。`qp_residual_min` 需明确是否包括 bounds、何种符号与容差，建议保存各约束组最坏违反量。

每个 episode summary：完整任务成功/合法抓取/strict safety、`Dmax_mug`、ketchup/base/其他对象 contact 分组、首次发生时间、QP feasible/fallback、介入次数与有效求解次数、修正幅度、P95 solver 时间、最终停止原因。

## 11. 执行顺序、报告交付与停止条件

**阶段 R0：源码与几何审计（不跑性能）** → 固定 HEAD，核对 quaternion/p1/R1，校验动作语义，补齐 `eta/intent_len` 与门控阈值来源。若失败，修复后重新冻结新 SHA，不允许边跑边修。

**阶段 R1：纯 QP 与真实 policy 接口回归** → G3–G7，包括 β0 等价和同状态非零 β 控制差异，重新用 post-fix QP 快照核实 E0 数学机制。若失败，不跑 SafeLIBERO。

**阶段 R2：评测契约** → 逐条证明 CAR 对象/阈值、success 字段与 valid task 定义；如果 mug 位移几十毫米仍被标记成功，必须明确 `success` 是 native TSR 还是其他，不允许继续把其解释成安全成功。

**阶段 R3：有限 Pilot** → 16 对 32 episodes，指标全部冻结，完成后输出配对翻转矩阵、完整安全事件和案例分析；没有预注册统计非劣效界限就不能称“安全不劣”。

**停止条件**：`p1` 错位、wrapper 行为不一致、`A,b` 改动、轴角/尺度含混、β0 不等价、CBF 状态或求解器不合法、数据采样丢失、CAR 与 SSR 判断矛盾、意外换 checkpoint、临时更改参数/检查点 → **STOP / INVALIDATE AFFECTED RESULTS**。

交付物：

```
docs/testing/cca_asf_postfix_source_manifest.md
docs/testing/cca_asf_postfix_geometry_audit.md
docs/testing/cca_asf_postfix_qp_regression.md
docs/testing/cca_asf_postfix_metric_contract.md
docs/testing/cca_asf_postfix_paired_evaluation.md
docs/research/cca_asf_v1_verdict_amendment.md
```

## 12. 最终研究表述纪律

可以说：“在固定正确几何、冻结 π0.5 和相同原始 CBF 约束下，CCA-ASF 仅改变 9D QP 的白化低秩度量；其在代理任务方向上的代数保护机制可检验。”

尚不可说：“CCA-ASF 已证明提高真实任务成功率／安全率”“CCA 每次干预更少所以更安全”“修复前的灾难性事件完全由四元数单独造成”“QP 可行就保证真实无碰撞”“方向代理就是真实 VLA 任务意图”。

**总裁决**：在修复后基线中，CCA-ASF 获得了**重新严格验证**的资格，尚未获得有效性/非劣效结论。工程接受标准是“唯一改变目标矩阵 + 相同状态输入可复核 + 几何/指标正确 + 配对原始结果完整”，而不是“观察到正面均值”。

---

## 参考的外部 API 文档（仅作工程事实核对）

- MuJoCo 2.3.7 Simulation: https://mujoco.readthedocs.io/en/2.3.7/programming/simulation.html （状态管理、接触字段、`mj_contactForce`）
- MuJoCo 2.3.7 Modeling: https://mujoco.readthedocs.io/en/2.3.7/modeling.html （MJCF 姿态描述）
- CVXPY `quad_form` implementation: https://cvxpy.readthedocs.io/en/functions-table/_modules/cvxpy/atoms/quad_form.html （`xᵀPx`，无 `1/2`）

**注意**：此文档未直接读取当前私有仓库源码。为使其成为可执行的、审计级别的最终规格，必须先完成 `[SOURCE-CHECK]` 并将其核验结果与准确行号写入 `source_manifest`。
