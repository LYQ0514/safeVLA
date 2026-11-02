# 面向冻结 π0.5 的约束冲突感知各向异性安全过滤

**工程方法书 v1.0 · 2026-11-02**  
**英文工作名：Constraint-Conflict-Aware Anisotropic Safety Filtering (CCA-ASF)**  
**目标：在 MuJoCo / robosuite / SafeLIBERO 中，以最小改动验证“任务意图定向保护”是否优于同约束下的欧氏 QP。**

> **研究性质与证据边界**：这是可实施且可证伪的研究原型，而非已经验证有效或获得安全认证的算法。各向异性 QP/任务一致性 CBF 已有先行工作；候选创新点是“冻结 VLA 动作块推断任务方向 + 安全约束冲突感知的有界方向度量 + 等约束成对验证”。尚不能宣称 T-RO 创新充分或真实机器人形式化安全。

## 0. 交付目标、严格边界与先验冻结

### 目标
在不训练/微调 π0.5、不修改 OSC_POSE、任务文本、AEGIS 安全约束、障碍物模型、episode 起点和 CRN 的前提下，**仅替换 QP 目标函数的二次项**。衡量相同安全约束下安全修正是否更能保留 VLA 计划运动方向，并改善真实任务成功与合法操作成功。

### 不做
- 不引入 RRT、IK、外部路径规划、VLA 接管、恢复预测器、因果漂移图、H 步物理 Oracle 在线推演。
- 不为某个状态调节 CBF/障碍物半径、margin、碰撞规则；不凭借重新标定 safety contract 使新方法获胜。
- 不以缺失/不可信的 `exact_gap`、宽相位接触列表代替真实安全事实。
- 不将“不满足当前 QP 约束时停止”称作保证物理安全，也不把原生 `check_success=True` 等同于合法完成。

### 既有实验证据（提供给实验智能体，**须复核当地仓库**）
冻结 π0.5；`replan_steps=5`；SafeLIBERO long；MuJoCo 2.3.7；robosuite `OSC_POSE`；每个 policy action 由若干 OSC / MuJoCo substeps 执行（过去记录为 25，必须读取运行时配置，不允许硬编码）。历史项目曾出现：底座 body 漏检、TASK 物体被误撞、几何 margin 截断、`state/stream` 硬编码、MuJoCo fork 不精确。已有 `src/safety/contract_objects.py` 的角色体系及 TASK 完整性监控，须从真实仓库核实，不得重新发明一套有冲突的判据。

## 1. 动作空间与符号（严禁把 OSC delta 等同于关节速度）

- 机器人状态：$x=(q,\dot q, o, b)$，其中 $q\in\mathbb R^7$ 为 Panda 机械臂关节，$b$ 为控制器内部状态，$o$ 含真实感知/任务信息。
- 冻结 VLA 生成动作块 $A_t^{\pi}=[a_{t,0}^{\pi},\ldots,a_{t,K-1}^{\pi}]$，每步 $a=[a^{\rm arm}_{1:6},a^{\rm grip}]$。先在真实仓库读取动作语义；默认臂部六维为 OSC 位移增量三维 + 世界系 axis-angle 姿态增量三维，夹爪维度单独保留。
- 令 $a_t^{\rm arm}\in\mathbb R^6$ 为**传给 OSC 控制器之前**、但已明确规范化语义的动作；$S=\mathrm{diag}(s_1,\ldots,s_6)$ 表示动作尺度（正值），定义无量纲动作 $z=S^{-1}a^{\rm arm}$。若模型原本已经输出 [-1,1] action，令 $z$ 为该坐标，$S$ 只是从归一化坐标到真实 OSC delta 的已配置尺度。不可二次缩放。
- $z_t^{\pi}$ 为当前名义六维向量；$[\ell,u]$ 为原有动作盒约束；$C_tz\ge r_t$ 为**原版 AEGIS 不改变行、右端及有效安全裕度**的仿射约束。若原始代码约束为 $A a\le b$，必须做精确坐标/符号变换，绝不可直接复制行列。
- 当前可行集 $\mathcal F_t=\{z:\, C_tz\ge r_t,\ell\le z\le u\}$；若基线含其他线性约束也原样纳入。数值求解器、容差、最大迭代数和 warm-start 规则在 baseline/ours 中一致。
- 下文的“安全”在数学证明中指满足**给定的 QP 安全仿射约束**；实际不碰撞、CAR 和 TASK 完整性属于独立的物理审计，不得混淆。

**3D / 6D 兼容门禁**：官方 AEGIS 在 SafeLIBERO 中报告过 *translational-only* 与 *full-action-space* 两类配置。若本地安全层原本只约束平移三个量，**第一阶段只在同样 3D 子空间对比**，把所有 $I_6,W_{6\times6},z\in\mathbb R^6$ 相应替换为 $I_3,W_{3\times3},z\in\mathbb R^3$，不受安全过滤的姿态三维直接透传；不得凭空为其余三维添加没有物理意义的零 CBF 行。若原生 6D QP 存在并经过等约束接口审计，再开展完整 6D 对比。本文公式使用 $n=6$ 的主版本，单元测试应支持 $n\in\{3,6\}$。

## 2. 核心假说

**H1（受限方向保留）**：对同一项激活安全约束，当任务方向 $d$ 与约束法向既不正交也不平行，低秩各向异性度量能严格减小该任务方向上相对名义动作的偏移；当平行时无法获得方向再分配优势。

**H2（条件性任务价值）**：在 SafeLIBERO 配对试验中，在**相同 CBF 可行集**与安全评测下，H1 的方向偏移改善能否转化为更好的合法抓取/任务完成率？这完全是经验假说，不由 H1 的数学结论保证。

**H3（冲突门控的必要性）**：根据当前违反/趋近约束与任务方向的关系激活各向异性，比恒定方向权重具有更好的稳定性/减少不必要的重分配。须用消融验证，而非预设为真。

## 3. 冻结 VLA 动作块推断任务意图

### 3.1 不使用训练标签的首版任务方向

在一次真实 VLA 重规划时，读取当前冻结动作块中前 $L\le K$ 条臂部归一化增量 $z_{t,k}^{\pi}$。定义

$$
\bar d_t=\sum_{k=0}^{L-1}\omega_k z_{t,k}^{\pi},\qquad
\omega_k=\frac{\eta^k}{\sum_{j=0}^{L-1}\eta^j},\quad 0<\eta\le 1.
$$

$$
d_t=\begin{cases}\bar d_t/\|\bar d_t\|_2,&\|\bar d_t\|_2>\epsilon_d,\\ z_{t,0}^{\pi}/\|z_{t,0}^{\pi}\|_2,&\|\bar d_t\|_2\le\epsilon_d\ \text{且}\ \|z_{t,0}^{\pi}\|>\epsilon_d,\\ \varnothing,&\text{否则.}\end{cases}
$$

- **同一 action chunk 内**可以每步使用将要执行的剩余名义动作更新 $d$，但必须明确缓冲索引；不能额外调用 VLA 或偷看未来真实状态。
- $d$ 是**短时名义运动方向代理**，不是语义上正确的抓取方向，更不是 VLA 隐层因果任务意图。若名义计划本身碰撞，保护它的某些分量也可能伤害任务。该局限需在论文与负结果中明确。
- 建议首轮固定 $L=5$（若 chunk 少于 5，则取真实可用长度）、$\eta=0.8$、$\epsilon_d=10^{-6}$。这些是**待在验证集冻结的初始建议值**，不是先验最优值。若无有效 $d$，严格退化为各向同性 QP。
- 初版不使用 privileged soup/basket 位置，不加入任务阶段分类，从而避免重新引入不稳定物体语义管线。未来可单独测试由任务目标构造 $d$ 的上界版本，但不能混入主试验。

### 3.2 强制实现约束

- 用真正发送给 OSC 的相同坐标系生成 $d$。归一化动作的位移与姿态不同物理量由既有 $S$ 统一处理，不能在比较方法之间变更尺度。
- 当前六维指令可以是 delta target，而非严格 Cartesian velocity；$d$ 的数学角色是目标函数内的**动作变化方向**，不是安全动力学模型中的真实 $\dot q$。
- 夹爪指令原样透传；如果有安全逻辑会修改夹爪，那是另一个干预变量，必须在所有方法中一致。

## 4. 冲突感知各向异性度量

### 4.1 判断名义动作与 CBF 是否冲突

定义违反集合 $\mathcal V_t=\{i: r_{t,i}-(C_tz_t^{\pi})_i>\epsilon_{\mathrm{viol}}\}$。当名义动作已可行，**不调用优化或优化必返回名义动作**。

对 $i\in\mathcal V_t$ 取第 $i$ 行 $c_i^\top$，将各行按 $\|c_i\|_2$ 归一化后构造 $C_{\mathcal V}$。通过 SVD（不使用不稳定显式逆）计算其行空间的正交投影矩阵 $P_{\mathcal V}$。令

$$
\xi_t=\|P_{\mathcal V}d_t\|_2^2\in[0,1],\qquad
\rho_t=4\xi_t(1-\xi_t)\in[0,1].
$$

$\xi=0$ 表示方向与当前约束行空间正交，$\xi=1$ 表示完全落入行空间；二者都不一定存在有益的方向重新分配，故门控抑制极端情况。若 $\mathcal V=\varnothing$、$d=\varnothing$、SVD 无有效秩或数据无效，令 $\rho=0$。

**说明**：这一门控在单约束情形具有清楚的几何意义；多约束下它只是可测试的设计假设，不能宣称全局最优。若加了盒约束，真实激活集可能变化，不能用仅由违反行计算的 $\rho$ 推出多约束保证。

### 4.2 构造 SPD 度量与最终 QP

$$
\beta_t=\beta_{\max}\rho_t,\qquad
W_t=I_6+\beta_t d_t d_t^\top.
$$

若 $d=\varnothing$，令 $W_t=I_6$。取首版预注册 $\beta_{\max}=4$（可在单独验证集检查敏感性，绝不在测试样本调节）。

$$
\boxed{\begin{aligned}
z_t^*&=\arg\min_{z\in\mathbb R^6}\frac12(z-z_t^{\pi})^\top W_t(z-z_t^{\pi})\\
\text{s.t. }&C_tz\ge r_t,\quad \ell\le z\le u.
\end{aligned}}
$$

发送给控制器的动作：$a_t^{\rm arm,*}=Sz_t^*$，夹爪动作维持原值。

用于严格匹配的欧氏基线：$W=I$；不改变任何其他项。若 AEGIS 原 QP 还有正则化/软约束/方向权重，须先拆分并记录：可在原有正定矩阵 $W_0$ 上通过白化坐标添加低秩项，但**不能在不明确等价性的前提下说基线是 $I$**。推荐首轮在完全相同归一化坐标统一重写为 $W=I$ 并验证数值等价，再做实验。

### 4.3 QP 数值形式（OSQP）

令目标 $\frac12 z^\top P z+q^\top z$，那么

$$P=W_t,\quad q=-W_t z_t^{\pi}.$$

常量项 $\tfrac12(z_t^{\pi})^\top W_t z_t^{\pi}$ 可忽略。为了 OSQP 使用标准双侧边界，堆叠 $G=[C_t;I_6]$、$l=[r_t;\ell]$、$u=[+\infty;u]$。

QP 不得依赖含惩罚松弛的伪“硬安全”：若历史 AEGIS 必须使用 slack，保留相同 slack 惩罚并在报告中明示 `SLACK_USED`；不能从 slack QP 可解推导硬安全成立。

## 5. 可以严格证明的数学性质

### 定理 1：严格凸性、唯一解以及零干预

**条件**：$d$ 有定义时 $\|d\|_2=1$，$\beta_t\ge0$；可行集 $\mathcal F_t$ 非空、闭，盒约束有界。

由 $W=I+\beta dd^\top$ 可知：对任意 $x\ne0$，
$$x^\top Wx=\|x\|_2^2+\beta(d^\top x)^2>0.$$
所以 $W\succ0$，且 $\lambda(W)=\{1,\ldots,1,1+\beta\}$、条件数 $\kappa_2(W)\le1+\beta_{\max}$。目标严格凸；在非空紧可行集上存在唯一解。

若 $z_t^{\pi}\in\mathcal F_t$，则取 $z=z_t^{\pi}$ 使目标为零，故 $z_t^*=z_t^{\pi}$。**非风险名义动作严格不受影响**（数值实现可直接 fast path）。

### 定理 2：各向异性不改变硬约束可行性

各向同性与各向异性使用同一 $\mathcal F_t$，所以
$$\mathcal F_t^{\mathrm{iso}}=\mathcal F_t^{\mathrm{aniso}}.$$
只要求解正确，若硬约束有解，两者输出都满足相同 QP 硬约束；若无解，改变 $W$ 不能使其有解。**这是 QP 代数事实，不等于 MuJoCo 每个 substep 不碰撞的物理保证。**

### 定理 3：单个激活线性约束时的任务方向偏移减小

忽略不激活的盒约束，设一个安全约束 $c^\top z\ge r$，名义动作为违反状态，$\delta=r-c^\top z^{\pi}>0$。记 $\theta$ 为单位任务方向 $d$ 和 $c$ 的夹角，$W=I+\beta dd^\top$，$\gamma=\beta/(1+\beta)$。

KKT 得：
$$\Delta z_W=z_W^*-z^{\pi}=\frac{\delta W^{-1}c}{c^\top W^{-1}c},$$
$$W^{-1}=I-\gamma dd^\top.$$

因此
$$
\left|d^\top\Delta z_W\right|=
\frac{\delta(1-\gamma)|d^\top c|}{\|c\|_2^2-\gamma(d^\top c)^2}.
$$

等权投影为 $\Delta z_I=\delta c/\|c\|^2$。当 $d^\top c\ne0$，二者任务方向偏移绝对值之比为

$$
\boxed{\frac{|d^\top\Delta z_W|}{|d^\top\Delta z_I|}=
\frac{1-\gamma}{1-\gamma\cos^2\theta}\le1.}
$$

- 当 $0<|\cos\theta|<1$ 且 $\beta>0$：严格小于 1。
- 当 $|\cos\theta|=1$：没有可以重新分配的方向，比例为 1。
- 当 $\cos\theta=0$：两种修正原本对 $d$ 的偏移均为 0（比值 $0/0$ 不定义），无法声称改进。

**定理只保证相对名义动作沿 $d$ 的偏移更小，不保证几何路径更安全、更容易抓取、未来任务回报更高。** 盒约束、多活跃 CBF、非线性 OSC 动力学时不适用该严格比例，需要实证检查。

### 定理 4：任务方向强保护的极限

设 $\mathcal F$ 为非空紧凸集，$z_\beta$ 最小化 $\|z-z^\pi\|^2+\beta[d^\top(z-z^\pi)]^2$。令 $\beta\to\infty$，则任意收敛子序列的极限先最小化 $|d^\top(z-z^\pi)|^2$，再在同一最小值水平集内最小化欧氏扰动。证明：将最优性不等式与任意在任务误差上最优的 $z_0$ 比较，得 $\beta(e(z_\beta)^2-e(z_0)^2)\le\|z_0-z^\pi\|^2$，故 $e(z_\beta)^2\to e(z_0)^2$；极限中的二级最小范数由紧性与最优性比较得到。

**工程含义**：大 $\beta$ 不能创造可行任务方向；可能显著增加其他维的动作扰动和接触风险，因此必须有 $\beta_{\max}$ 与数值稳定性测试。

### CBF 前向不变性：仅作为条件性命题

对于实际连续控制仿射系统 $\dot x=f(x)+g(x)v$，若真正实现了 $\dot h(x,v)\ge-\alpha h(x)$，初始 $h(x_0)\ge0$，解存在且满足常规光滑/正则性条件，则比较原理给出 $h(x(t))\ge e^{-\alpha t}h(x_0)\ge0$。该性质由可行集成立，与 $W$ 无关。

**重要**：SafeLIBERO 的真实操作是归一化 OSC **目标增量**、25 左右物理 substeps、控制器动态和有限观测；不能直接将上述连续速度 CBF 定理套到本系统上。实际必须报告逐 substep 物理接触和 CAR，且区分 `CBF_QP_feasible` 与 `physical_safe`。若使用线性化下一步 barrier，需要明确模型误差/截断界，不能声称无条件安全。

## 6. 最小数值示例（单约束解析验收）

设 $z^{\pi}=(0,0)$，约束 $z_1+z_2\ge1$，任务方向 $d=(1,0)$，$\beta=3$。则
$$z_I^*=(0.5,0.5),\qquad W=\mathrm{diag}(4,1),\qquad z_W^*=(0.2,0.8).$$

两者同样满足 $z_1+z_2=1$，但沿任务维 $z_1$ 的偏移由 $0.5$ 降至 $0.2$。若改成 $d=(1,1)/\sqrt2$，两者应一致；若 $d=(1,-1)/\sqrt2$，两者沿 $d$ 的偏移都为零。此例**仅证明动作扰动再分配能力**，没有模拟任务进度。

**务必单测**：随机生成 1000 个可行的 2–6 维线性 QP、对比 OSQP 与解析单约束公式、对 KKT 残差和零干预条件做断言。

## 7. 运行时主要伪代码

```python
# Function contracts are logical interfaces; do not assume these exact names exist.
def cc_asf_filter(state, frozen_action_chunk, buffer_index,
                  aegis_constraint_adapter, action_adapter, cfg):
    # A. Interface and immutable inputs
    nominal_full = frozen_action_chunk[buffer_index].copy()
    z_nom, grip = action_adapter.to_normalized_osc6_plus_grip(nominal_full)
    assert z_nom.shape == (6,)

    # B. SAME safety set as matched isotropic QP
    C, r, lower, upper, safety_meta = aegis_constraint_adapter.build(state)
    assert constraints_verified(C, r, lower, upper, safety_meta)

    # C. Nominal preservation
    if hard_feasible(z_nom, C, r, lower, upper, cfg.qp_tol):
        return nominal_full, {"status": "NOMINAL", "beta": 0.0}

    # D. Intention inferred only from available frozen chunk
    remaining = frozen_action_chunk[buffer_index:
                      min(buffer_index + cfg.intent_len,
                          len(frozen_action_chunk))]
    d = chunk_direction(remaining, action_adapter,
                        eta=cfg.eta, eps=cfg.eps_dir)

    # E. Derive constraint-conflict gating
    if d is None:
        beta, W = 0.0, eye(6)
    else:
        v = r - C @ z_nom
        Cv = normalize_rows(C[v > cfg.violation_tol])
        P = row_space_projector_svd(Cv, rank_tol=cfg.rank_tol)
        xi = clip(float(d @ P @ d), 0., 1.) if P is not None else 0.
        beta = cfg.beta_max * 4. * xi * (1. - xi)
        W = eye(6) + beta * outer(d, d)

    # F. All methods must use exact same constraints / solver settings
    ans = qp_solver.solve(P=W, q=-W @ z_nom,
                          C=C, r=r, lower=lower, upper=upper,
                          warm_start=cfg.matched_warm_start)
    if not ans.is_optimal or not finite(ans.z):
        return NONE, {"status": "NO_VERIFIED_ACTION", "reason": ans.status}
    if not hard_feasible(ans.z, C, r, lower, upper, cfg.qp_tol):
        return NONE, {"status": "QP_RESIDUAL_FAILURE"}

    return action_adapter.from_normalized_osc6_plus_grip(ans.z, grip), {
        "status": "FILTERED", "beta": beta,
        "task_direction": None if d is None else d.copy(),
        "qp_residual": ans.primal_residual,
        "norm_deviation": norm(ans.z - z_nom),
        "directional_deviation": 0.0 if d is None else abs(d @ (ans.z-z_nom)),
        "safety_meta": safety_meta,
    }
```

**NO_VERIFIED_ACTION 处理**：不可悄悄返回未认证的名义动作、上一动作或 Hold。应按照现有 benchmark 的统一安全失败协议终止当前 episode，并单独计为 `ABSTAIN/INFEASIBLE`；如果选用预先定义的缓停/停止控制，必须在物理仿真中验证，且对所有对照组采用相同策略。

## 8. MuJoCo / SafeLIBERO 实施接口

### 8.1 只在安全 QP 目标函数中插入

建议封装 `ActionAdapter`、`ConstraintAdapter`、`IntentExtractor`、`MetricBuilder`、`QPSolver` 与 `SafetyAudit`。首先定位仓库内 AEGIS CBF-QP 构造、π0.5 policy server、OSC command scaling、LIBERO rollout runner。**不要假定具体源码函数名与参数**，从仓库打印配置并保存 `runtime_manifest.json`。

| 组件 | 输入 | 输出 | 不可改的部分 |
|---|---|---|---|
| PolicyAdapter | 冻结 π0.5 chunk | 真实待执行 action | 模型权重、推理参数、CRN |
| ActionAdapter | action + controller config | $z_{nom}$, grip, $S$ | `control_delta`, output scaling, axis-angle 坐标约定 |
| ConstraintAdapter | sim state + AEGIS state | $(C,r,\ell,u)$ | 原障碍物、CBF $h/\alpha/\delta$、约束数与安全边界 |
| MetricBuilder | chunk + violated rows | SPD $W$ | 不读取 privileged future rollout |
| QP | $W,z_{nom},C,r$ | $z^*$ | solver / hard constraints / tolerances |
| Audit | real MuJoCo 25-ish substeps | contacts, CAR, task integrity | 只读、独立评测 |

### 8.2 MuJoCo 细节

- `mujoco.mj_jacSite(model,data,jacp,jacr,siteid)` 和 `mj_jac` 等可用于可选分析任务/碰撞点的灵敏度；`jac` shape `(3, model.nv)`，索引取真实 Panda arm DoF，而不是假定 robot qvel 位于前七列。
- `data.qfrc_bias` 是动力学 bias force，`mj_fullM` 可展开质量矩阵。但**本方案不需要修改底层 OSC 控制力矩**，避免引入不必要的动力学控制器重写。
- 每次记录 `sim.data.contact` 时，务必识别 `geom1/geom2`、body 角色与真实接触活跃状态；单一 `contact.dist` 不是任意远距离 geom 对的精确距离。2.3.7 的 margin 查询若只有半径下界，记录 `clearance_lower_bound`，不要伪装成 `exact_gap`。
- 子步物理监控用 `mujoco` 原生 physics copy；不要将 query model 的放大 margin 写回实际 physics model。
- 若涉及 solver 快照、warm start 或动作缓存，按已验证的**确定性完整重跑**机制配对；`get_state()` 快照不一定恢复 OSC 内部状态与约束求解器缓存。

### 8.3 CONTRACT 与 TASK 完整性

- 原生 CAR：从物理真实位置计算受保护 mug 自 episode 初始时刻以来的历史最大 L1 位移，阈值 1mm；不得用瞬时位移或任意带 margin 的几何间隙替代。
- 原生 TSR：`check_success`。合法 TSR：原生谓词 + 合法抓取/运输/释放过程（抓取需接触、相对稳定、连续抬升等证据，不能仅看 gripper 开合数值）。
- `StrictSSR=TSR_valid AND CAR AND forbidden_contacts==0 AND task_integrity_ok`。额外同时报告 benchmark `SSR_native=TSR_native AND CAR`。
- 受保护底座属于禁止机器人接触对象；TASK（soup/cream）允许抓取阶段合法接触，**不允许预抓取规划/过滤时非预期撞飞**。
- 必须把 `contact_registry_version / physics_model_hash / task_integrity_version` 写进每条实验记录；若既有审计回归失败，暂停实验，不允许自行增大/减小障碍半径补救。

## 9. 对照、消融与实验设计

### 9.1 方法对照（均使用同一物理起点/动作流）

**主比较**：
1. Frozen π0.5（无过滤，反映任务上限与物理风险）。
2. AEGIS / Euclidean normalized QP（$W=I$）。
3. **CCA-ASF**（$W=I+\beta_tdd^\top$，$\beta_t$ 冲突门控）。

**必要消融**：
4. Constant-Aniso（$W=I+\beta_{max}dd^\top$，无冲突门控）。
5. Random-Direction（在同样 $\beta_t$ 下使用固定 seed 随机单位方向，检验 VLA 意图有无价值；随机方向与 safety constraint 不应泄露目标）。
6. Per-axis-Diagonal（仅使用 $\operatorname{diag}(W)$），检验交叉项的价值。
7. Current-Action-only（$d=z_{nom}/\|z_{nom}\|$），检验 chunk 信息价值。

**相关文献对照**：OSCBF（Morton & Pavone, IROS 2025）是任务一致性 CBF 的重要同类方法。若该基线不能在当前 OSC 接口与同等观测/控制信息下公平实现，必须在论文中说明限制；不能只与 AEGIS 比较就自称首创任务保持。其他对照如 VLPSA/WBAG 需先确认代码、设置与 SafeLIBERO 任务一致。

### 9.2 测试阶段（防止重新陷入工程死角）

**Phase 0：纯数值单元测试（无 MuJoCo）**
- 解析算例、唯一性、等可行集、零干预、QP 容差；≥1000 随机可行 QP；确认相同约束、不同目标函数。
- 当 $\beta=0$ 时，ours 与欧氏基线动作在 `1e-6`（或实际 solver 配置可支持的）精度内相同。

**Phase 1：小规模 MuJoCo smoke test**
- 选择预注册 1 个适合安全动作过滤的任务、2–3 个 state、2 个 CRN stream；共 3 方法 × 约 4–6 配对单元。先只验证 wrapper 输入输出、归一化、solver status、零干预、物理记录与 TASK 完整性。不得使用旧的安全被碰撞清单所推翻的“安全路径可行”标签。
- 不允许因首次失败而更换状态、调 $\alpha$、margin 或任务目标；技术修复必须对所有方法一致。

**Phase 2：SafeLIBERO paired eval**
- 优先使用**原生官方 SafeLIBERO** 任务/难度设置，包含 long 作为困难分层，避免仅用曾反复调试的 task0/state02。
- 冻结名单：建议至少 4 个任务 × 每任务 10 个 distinct init states × 2 个真正不同的 CRN streams = 80 个配对单元，每个单元运行 Vanilla、Euclidean、CCA-ASF（240 episodes）；若资源不足可先固定 20 配对单元作 pilot，**不能当成投稿级泛化结论**。
- 三方法在相同 worker、相同父初始状态、相同初始模型参数及 `(task,state,stream,query-index)` CRN 映射下比较，核验相同 stream 重跑一致与不同 stream 噪声哈希不一致。不要把随机重复算独立初始布局。
- **所有参数只在独立 validation states 上确定，test 一次冻结**。主测试不因结果失败改 $\beta$、任务列表或 obstacle set。

### 9.3 指标

**主要**：`TSR_native`, `TSR_valid`, `CAR`, `SSR_native`, `StrictSSR`；注明按 episode 统计，零除保护和提前停止皆纳入分母。

**机制**：
- `risk_step_count`, `filter_step_count`, `abstain_count`, QP infeasible/solver fail；
- `directional_deviation = |d^T(z^*-z_nom)|`；
- `orthogonal_deviation = ||(I-dd^T)(z^*-z_nom)||`；
- `total_deviation`, `Δorientation`, `Δposition`, `clip_fraction`, `constraint_residual`；
- `nominal_violations`, `mug Dmax`, `robot-base contacts`, `task_object_integrity`, `physical contact first step`；
- `policy progress`: 初始/最终目标距离、合法抓取时间、合法放置时间、任务完成步数；
- QP mean/p95 latency **单独计时**，不要混入 π0.5 GPU 推理；episode total wall time 另报。

**统计**：按 state 单元成对对比；展示配对成功/失败四格表和 paired bootstrap CI，区分初始状态独立性与随机流的簇内相关性，建议以状态作聚类自助法单位。不能将 2 stream × 1 state 当成两个独立环境。主指标 pre-register `StrictSSR`，若因评测定义受限，则单独披露 Native SSR。

## 10. 工程验收、回归测试与退出标准

### 接入验收（必须全过）
1. `beta_max=0` 与 Euclidean QP 逐步动作数值一致（包含 gripper 完全不变）。
2. AEGIS/ours 使用相同 $(C,r,\ell,u)$ 哈希；任何差异直接拒绝比较。
3. 单约束解析示例可复现 $(0.2,0.8)$ vs $(0.5,0.5)$。
4. 约束不可行时标记 `NO_VERIFIED_ACTION`，不执行不可认证动作；所选应急方案两组一致。
5. 运行时记录每个 OSC action 的 normalized/physical 值与 applied clipping；避免双重尺度变换。
6. `PROTECTED/SUPPORT/TASK` 角色及 `monitor_contacts` 回归通过；尤其 Link5-base 与 pregrasp soup 非法碰撞不得漏检。
7. 原生 CAR 由真实物体全程最大历史位移计算；正确地区分 native/valid TSR。
8. 预注册 action-pair 单元用确定性完整重跑而非不完整 fork；CRN 绑定 `query-index` 不因方法调用次数而漂移。
9. 所有方法真正执行相同的 `replan_steps` 和 controller policy/physics 频率。

### 研究止损门禁
- Phase 0 未通过：停止 MuJoCo 接入。
- Phase 1 出现基础设施与角色清单失配：停止结果解释，修复统一接口，不推进任务调参。
- Phase 2 对 `StrictSSR`、`TSR_valid` 没有稳定改善，只有 `directional_deviation` 下降：判为“代数方向保护成立、任务有效性证据不足”，不继续扩增系统。
- 若仅在手工调权重或少数训练状态提升，不宣称泛化。

## 11. 给实验智能体的直接任务分解

```text
TASK A [read-only]: inspect repository, pin MuJoCo/robosuite versions, identify
  policy chunk buffer, OSC action scaling & gripper semantics, AEGIS objective,
  safety constraint assembly, contact/task-integrity monitors. Write manifest.
TASK B [pure math]: implement intent / SVD conflict gate / W builder / QP,
  and analytic/unit tests. No MuJoCo until tests pass.
TASK C [adapter]: add config flag filter_mode={vanilla,euclidean,cca_asf}
  and beta_max, eta, intent_len; do NOT touch AEGIS safety constraints.
TASK D [audit]: verify unit scale, C/r and bounds hash, QP feasibility,
  contact registry regression, strict / native outcome definitions.
TASK E [pilot]: 2-3 preregistered states, two streams, matched within worker,
  output per-step logs and method-level table.
TASK F [gate]: compare against stop rules. Only if valid proceed to paired
  SafeLIBERO evaluation, then ablations.
```

**建议文件路径**（若仓库已有同名类则按实际工程适配）：

```
src/safety/cca_asf/intention.py
src/safety/cca_asf/metric.py
src/safety/cca_asf/qp_adapter.py
src/safety/cca_asf/runner_hook.py
tests/safety/test_cca_asf_math.py
tests/safety/test_cca_asf_constraints.py
experiments/cca_asf/run_paired_safelibero.py
experiments/cca_asf/config.yaml
docs/testing/cca_asf_runtime_manifest.md
docs/testing/cca_asf_validation_report.md
```

### 配置建议（仅起始验证值，必须经独立 val 冻结）

```yaml
method: cca_asf
policy: frozen_pi05
controller: OSC_POSE
replan_steps: 5   # verify at runtime
intent_len: 5
eta: 0.8
beta_max: 4.0
rank_tol: 1.0e-8
violation_tol: 1.0e-7
eps_dir: 1.0e-6
qp_tol: 1.0e-5
fallback: terminate_unverified  # identical to baseline protocol
no_train: true
no_geometry_retune: true
```

## 12. 论文科学边界与未来扩展（首轮不实现）

- **不可宣称**“各向同性是所有现有 VLA 安全方法的通病”，OSCBF 已针对任务一致性进行研究。首轮方法本质仍是结构化低秩加权 QP，创新性要由**从冻结动作块自动构造权重 + 冲突门控 + 严格公平的任务收益**共同支持；若实测与 task-consistent OSCBF 相当，也应诚实承认。
- **不可宣称**模型有无限时域/真实机器人安全证明。CBF 推导需理想精确控制仿射假设；SafeLIBERO 的 OSC 和几何感知不能默认满足。
- **不能由单步解析证明推断全 episode SSR 提升**。任务成功仍需真实实验。
- 可作为第二阶段的研究问题：动作块依赖的**有界状态相关度量** $W(s_t,A_t^\pi)$ 是否优于静态任务层级度量。只有第一阶段显示可重复收益后，才考虑短视性和未来干预成本。

## 参考与实施依据

1. Hu 等，*VLSA: Vision-Language-Action Models with Plug-and-Play Safety Constraint Layer*，AEGIS / SafeLIBERO 项目页面：https://vlsa-aegis.github.io/
2. Morton & Pavone，*Safe, Task-Consistent Manipulation with Operational Space Control Barrier Functions*，IROS 2025：https://stanfordasl.github.io/oscbf/
3. robosuite 官方控制器文档（核实 OSC_POSE delta / scaling / orientation）：https://robosuite.ai/docs/modules/controllers
4. MuJoCo 官方 2.x API reference（核实 mj_jacSite 等）：https://mujoco.readthedocs.io/en/2.3.2/APIreference.html
5. MuJoCo 官方计算文档（接触 / margin / gap 语义；应用于当前版本需核实）：https://mujoco.readthedocs.io/en/latest/computation/

**最终交付判据**：实验智能体应先交《运行时接口清单 + 单元测试通过记录 + 对照约束哈希一致性》，再提供 SafeLIBERO 匹配实验，而不是先发一张 SSR 改善表。
