# AEGIS-STG：冻结 VLA 的时空联合安全控制探索
## 实验侧智能体工程任务书 v1.0 · 2026-11-05

> **文件用途**：可直接交付实验侧智能体执行的分阶段任务书。方案为待证伪研究原型，并非已验证算法或正式安全证书。先验证问题、再实施控制律、最后做完整实验；任何门禁失败都必须停止并报告，不允许为了得到正结果绕过门禁。

## 0. 核心指令与完成定义

**研究问题**：在冻结 π0.5 + 原 AEGIS CBF-QP 的条件下，是否可以借鉴传统机器人控制的环流场（circulation field）与动态参考管理（reference governor），将空间绕障方向、跨周期控制记忆及局部动态安全余量耦合进**同一个实时闭环控制律**，在不做在线多候选搜索、不额外调用 π0.5 的前提下，提高合法安全任务成功率并减少停滞与时间损失？

**临时研究代号**：STG（Spatiotemporal Task-Guided Governor）；不要把它写作原创的理论名称或论文最终命名。**不允许预先宣称** STG 解决“真正的各向异性与短视性”。

**成功交付**：可重放的代码提交、完整审计证据、等约束可运行控制器、至少一个通过门禁的配对 pilot、时间/任务/物理安全指标、消融、失败案例与是否继续扩大验证的研究结论。

**不可修改**：冻结 π0.5 checkpoint、图像/语言/归一化预处理、名义动作及 chunk 消费逻辑、AEGIS CBF 几何与 margin/阈值、原始 QP 约束和变量、OSC 控制器、夹爪、动作执行子步数、任务与安全评测契约、各组随机性协议。**本研究只允许在原 QP 目标中增加软引导项、在其外部增加小状态的参考管理器**。若发现原实现必需修复，先单独修复并在全部对照组统一使用，形成新的冻结基线。

**硬在线预算**：每个现有控制周期最多 1 次正常 VLA 获取（按原 `replan_steps` 使用缓存，不增频）、最多 1 次安全 QP；不引入在线 RRT/IK/MPC、多分支物理仿真、N 候选 QP/动作认证、额外 VLA worker 查询、外部学习预测器。仅记录本次小规模线性代数开销。候选比较可**离线只读诊断**，不得进入部署算法。

**当前项目已知约束（需从运行仓库复核）**：SafeLIBERO、MuJoCo 2.3.7、robosuite 1.4.1；π0.5 的外部动作 `z∈R^7`（世界系平移增量3 + axis-angle旋转增量3 + gripper1）；AEGIS 原生 `u∈R^9` QP（6D 主运动变量 + 3D 原有辅助变量）；`replan_steps=5` 及每 outer step 约25个模拟子步仅为历史记录，不可硬编码。历史几何四元数问题及 AEGIS Oracle perception 未闭环等价问题仍需以最新代码核实。当前无权直接认定“原版 AEGIS 已严格复现”。

## 1. 可证伪假说及机制界定

- **H0（基线能力）**：在给定严格评测契约下，原 AEGIS 的失败是否真的表现为“约束活跃+任务进展停滞/反复纠偏”，而不是单纯抓取失败、碰撞几何遗漏、OSC 调用错误或 VLA 原本无能力？只有相应机制分层成立，才进入 H1/H2。
- **H1（局部环流）**：在同样 CBF 可行集内，连续切向任务引导能降低局部停滞，但未必提高合法任务成功。
- **H2（参考记忆）**：在同样约束和几何下，跨周期参考状态与方向迟滞是否降低修正方向振荡、无效回撤，且不增加物理碰撞？
- **H3（时空耦合）**：环流方向与动态参考更新联合使用，是否相较各自单独使用提供额外增益？使用 2×2 因子设计验证交互作用，不能只比较联合版本与基线。
- **H4（实时性）**：增量在线成本足够低，不导致任务墙钟时长或单步 p95 控制耗时显著恶化。速度与严格安全结果均为必须报告的硬指标。

**重要区分**：经典 ERG 依赖稳定且可建模的受控闭环；本项目首轮的 `governor` 只是 ERG 启发的 **动态参考管理原型**。如果没有有效的动态安全裕度与稳定性证明，必须称作 `dynamic reference heuristic`，不可声称继承 ERG 的安全保证。

## 2. 基础数学对象与算法边界

### 2.1 原生 AEGIS（原样保留）

记原生控制变量为 `u∈R^9`，原 AEGIS 优化为

\[
\min_{u} J_0(u;z_t,x_t),\quad
\text{s.t.}\ u\in\mathcal U_{\rm AEGIS}(x_t).
\]

其中 `J0` 是真实源码中的**完整**目标，包括 9D 中原有权重、辅助维度成本与其它项；`U_AEGIS` 包含原生 CBF 线性约束、bounds 和已有等式/松弛结构。曾记录 `W0=diag(1/25×6,1×3)`，但**必须读取源码确认**。原样调用全部现有约束/执行映射/solver/fallback；不能重写成 3D/6D 替代 QP。`u_ref` 与 `u_nom` 分别记录，不假定相同。

定义 `P_t∈R^{3×9}` 为从 9D QP 决策变量的**增量**到世界坐标系、统一量纲的外部平移增量的局部线性映射。它必须来自 AEGIS 原有 `u→OSC动作` 映射的精确解析形式。令 `z_{p,ref,t}∈R³` 为 `u=u_ref,t` 时由原 map-back 获得的外部平移指令，则在该仿射区间：`z_{p,out}(u)=z_{p,ref,t}+P_t(u-u_ref,t)`。若映射非仿射或含裁剪，应只在已核实有效区域使用 Jacobian，记录近似误差，并阻止跨越裁剪/非光滑区。禁止直接用 `u[:3]` 冒充世界平移、或把 7D/9D 维度错误等同。

### 2.2 统一方案：有记忆的空间参考场 + 单次 QP

在接近约束或名义 QP 有风险时，根据**基线本就可访问的**几何代理，提取最相关障碍面的世界系法向 `n_t` 和当前未过滤运动意图 `d_t`（来源于真实 chunk/当前动作，不多查询策略）：

\[
\bar t_t=(I-n_tn_t^\top)d_t,\qquad
\hat t_t=\bar t_t/\|\bar t_t\| \quad (\|\bar t_t\|>\epsilon).
\]

法向必须来自已经通过几何/坐标审计的 AEGIS 代理；若约束法向仅存在于 9D 控制空间，须通过实际 `P_t` 的任务空间推导验证，不得直接截取维度当作物理法向。多个活动障碍按**预先定义的最小代理裕度+迟滞**选定，不做在线全排列候选或轨迹评分。

在 `\bar t` 退化时，采用可重现的固定备用基轴与 `n` 生成一个切向方向，但必须显式标记 `DEGENERATE_TANGENT` 并保留防止方向抖动的记忆；不允许在线尝试左右两边并仿真选最优。为避免逆向绕行，记录 `d_nom·t`、连续两次切向内积，以及实际目标进展。

引入轻量参考状态 `r_t∈R^3`（世界系**虚拟末端参考位置**），与内部绕行侧 `s_t`、风险进入时刻、前一步指导方向构成小状态 `m_t`。在首次进入风险区时使用当前真实末端位置初始化 `r_t`。局部愿望参考 `r_des` 从当前可用名义位移或**已缓存、尚未消费的** chunk 计算并进行物理范围裁剪；不得读取未来真值轨迹、任务 oracle 目标或额外重规划。

离散动态参考管理模板：

\[
 r_{t+1}=r_t+\Delta t\,\alpha_t\,\rho(x_t,r_t,r_{des,t},\hat t_t,m_t),
\]

其中 `ρ` 由目标吸引与沿障碍切向环流组成；`α_t` 根据**有单位的**代理间隙、速度与保守动力学量控制参考变化率，并有界限与迟滞。若只是几何间隙减制动距离的代理，则只能称为 `braking-aware proxy`，不能称 certified DSM；若当前 AEGIS 的 `h` 无米单位，不得直接代入 `v²/(2a)`。优先先做 `α=1` 的固定速率版本以验证记忆机制，再加入动态裕度，避免一开始把复杂性和减速混在一起。

引导目标 `g_t∈R^3` 是 `r_{t+1}-p_ee,t` 经**现有 action normalization/scale**转换的世界平移命令。`g_t` 与 `P_t u` 必须具有相同坐标、单位和离散控制周期含义。设计软约束成本：

\[
\boxed{
 \min_{u\in\mathcal U_{\rm AEGIS}(x_t)}
 J_0(u)+\lambda_t\|z_{p,ref,t}+P_t(u-u_{ref,t})-g_t\|_Q^2
 }
\]

> **公式是接口模板**：`z_{p,ref,t}`、`P_t(u-u_ref)` 和 `g_t` 要处于同一坐标系、单位、尺度与离散控制周期。若真实 action map 存在非线性，必须以已验证的 `z_{p,out}(u)` 为准，并给出局部凸二次近似成立的范围；避免双计 `u_ref` 或漏掉 offset。实验侧必须给出准确推导，不能照抄模板而忽略真实源码。

将原生 9D QP 的全部硬约束、辅助变量和 `J_other` 保持不变；引导只改变目标。选择 `λ_t≥0`，目标新增项为 PSD 二次型，不额外新增一个在线 QP。无风险/方向无效/诊断不通过时 `λ_t=0` 且回归原生 AEGIS；如果风险进入期间名义动作原本 QP 可行，允许轻微引导但必须报告 `NOMINAL_MODIFIED_WHILE_FEASIBLE` 事件，不能隐瞒。软项不提供新的物理安全证书，原安全证明的条件仍需独立审计。

**首轮不要实现完整 RMPflow 或 CCM**；它们是机制解释与后续上界，不是 P1 的预设交付目标。

## 3. 四组公平消融（2×2 因子设计）

- `B0 = aegis_native`：当前经过审计的 **AEGIS-QP (Oracle Perception)**，原 QP，参考管理关闭，环流关闭。
- `B1 = circulation_only`：仅当前瞬时切向引导，风险退出后关闭，无时间记忆；相同 `λ` 和 QP 接口。
- `B2 = governor_only`：动态参考/迟滞与惯性/方向平滑，但不添加环流场；无几何辅助绕障方向。
- `B3 = joint_stg`：同一参考演化同时采用环流方向与跨周期参考管理；不是运行两个 QP 的串联。

作为外部定位参考另报告 `N0 = vanilla_pi05` 的既有结果（如果协议严格可比），但绝不把它当作等约束对照。`CCA-ASF` 已有独立归档和历史任务完整性负结果，不得把旧实现改头换面合入 B3；若重用接口，只能重用经过审计的 read-only adapter。`Route-Guided CBF-QP Repair` 需列为相关工作并在论文阶段单独考虑等条件复现，不强制在 P1 中实现。

**事先登记四组共有参数**：风险阈值（进入/退出，退出阈值必须更宽），`λ`、引导最大幅度、参考变化上限、切向退化阈值、方向锁定最长时长、重置规则、QP 超时与回退；允许在开发集选择有限网格，测试集冻结。若方向锁定导致反复停滞，作为失败表型记录，不能现场加入未登记的搜索或第三控制器。

## 4. G0：源码真实性、物理几何与评测契约审计（硬门禁）

**任务**

1. 只读确定仓库 HEAD、dirty、Python/包/CUDA/MuJoCo/robosuite/cvxpy/OSQP 版本、模型权重 SHA256、SafeLIBERO 场景/任务文件 hash；定位 `Pi05Policy`、AEGIS-QP 构造、执行 mapping、OSC_POSE、物理 contact recorder、现有 CAR/TSR/StrictSSR。
2. 打印一次完整 7D 名义 action→9D `u_nom/u_ref`→9D 解→7D action→OSC target→每 substep 物理状态的链路。明确世界/局部坐标、四元数 `[w,x,y,z]`/第三方库顺序、单位/比例、gripper 正负、动作 buffer 下标和 `replan_steps`。
3. 核验 `p1` 椭球中心与 `R1`：以 `mjData.xpos/xmat` 为独立真值重建几何中心，在多个非单位姿态上验证；旧数据中的 `16cm` 级偏差不可再现。查清所有 CBF 覆盖/未覆盖 link、障碍角色与 box_small_base→mug 间接位移链。
4. `AEGIS-Oracle` 与原论文视觉感知 AEGIS 严格分开命名，未匹配感知链路不得称为 full reproduction。核对 `replan_steps=5` 是否对 nominal 与 filter 一致；若不一致先统一后重新跑基线。
5. 核验 `TSR_native=check_success`、`CAR_native` 的确切对象/位移方式，原生 `SSR_native=TSR_native∧CAR_native`；合法操作 `TSR_valid` 需合法抓取/运输/释放证据；`StrictSSR_valid=TSR_valid∧CAR_native∧no_forbidden_contacts∧task_integrity_ok`。历史 protected mug 的最大 L1 位移 1mm 是先前使用口径，必须和本轮真实评测代码比对。逐物理子步保存位置/接触，不能使用缓存位移、只看末态或提前停止历史最大值。
6. 锁定所有实验方法相同的 solver、约束构造、QP slack、执行映射、fallback。零引导（`λ=0`）必须在固定输入上数值复现原生 AEGIS（推荐输出 L∞≤1e-6，若基线数值精度不支持则记录更宽容差的独立依据）。

**G0 PASS**：`runtime_manifest.json`、`reproduction_boundary.md`、`geometry_audit.csv`、`safety_contract.md`、`action_mapping_test.json`、`baseline_equivalence.json` 均存在且可核查；物理几何、策略时序、评测身份无未解决的致命冲突。**任一 BLOCK 则不得跑性能结论。**

## 5. G1：纯数学与接口实验（不消耗 GPU rollout）

- 校验 `P_t`：有限差分检验任务空间输出与理论 Jacobian/仿射映射一致（包括旋转姿态和辅助变量耦合），不得忽略 9D 的后三维。
- 测试所有控制变体 `λ=0` 恢复 B0；QP 变量 9D、全部原始 CBF 行/边界/辅助变量不变；gripper 与 nominal chunk 查询次数不变。
- 在 2D toy 单障碍及 3D 偏置障碍测切向性 `|nᵀt|≤1e-6`，方向连续性、旋转等变性和对侧选择确定性；对平行、反向、零动作和多障碍退化分别测试。
- Governor 检验 reset 不串 episode、风险离开恢复 B0、chunk 交界不会误清空物理参考、低速/高速接近在代理裕度下有不同响应、不会零间隙除零/非有限。
- 对原约束有解/无解两种情形测试，不得把 `QP infeasible` 靠新增软项“修好”；所有硬约束残差遵守同一 solver 容差。不得在新增模块异常时静默返回未认证动作。
- 对 `β/λ=0`、`λ>0` 下 `u∈R^9` 最优解、warm start、QP solve count 做自动断言。
- CPU 本地 1000 次以上固定问题微基准和 3 类病态样本；记录 governor/geometry/QP 单独 p50/p95，保证四组使用一致的测时方法。

**G1 PASS**：所有数值/接口测试通过；仅产生固定维度向量计算 + 原有一次 QP；没有在线候选池和二次策略查询。

## 6. G2：机制鉴别——先证明失败可被控制机制影响

在**仅使用 B0 结果**的情况下，预先冻结病例选择表，覆盖：

- S1：CBF 长时间活跃、末端进展停滞，原 π0.5 具有任务进展线索（疑似不良局部平衡/方向问题）。
- S2：CBF 修正方向交替/返回、目标或物体状态变化异常（疑似跨周期参考不一致）。
- S3：冻结 π0.5 本身抓取失败或受保护物体已被初始惯性推走（控制器无法挽回，必要负对照）。
- S4：已存在任务完整性违规/间接碰撞（审计限制而非绕障改进）。

优先使用此前 G1a/W2 中的稳定翻转案例，但必须在修复后原生 B0 里**重新确认**，禁止旧缓存结果直接作为主证据。若方便纳入 long-I task0/1 和 long-II 中其他任务，按已冻结规则覆盖，不能只挑 B3 有利的样本。以 `state_id / stream / phase / selection_reason / baseline_hash` 记录选择。

对每个冻结 episode，配对运行 B0–B3。每步记录：几何 proxy 清距、CBF 约束活动数/残差、真实末端位置和速度、名义 7D、修正 9D、参考 `r_t`、方向 `n,t`、过滤方向翻转、进展 proxy、真实物体/接触状态、VLA query index、新 chunk hash、求解状态、所有时间拆分。

**机制结论必须有归因等级**：
- `CONTROL_DEADLOCK_SUPPORTED`：存在长时间连续 CBF 活跃、任务进展低、且至少一种变体在同初始态下突破停滞；
- `VLA_LIMITATION`：各组均无有效任务推进，且 nominal 本身无合理动作线索；
- `CONTACT_OR_GEOMETRY`：物理碰撞、惯性/geometry gap 主导；
- `INCONCLUSIVE`：不足以区分。**禁止把所有失败硬归因于 QP 的短视性。**

## 7. G3：小样本配对 Pilot（正式开发集，不与最终测试混用）

建议首批：`2 tasks × 4 initial states × 2 frozen CRN streams × 4 methods = 64 episodes`。如先验任务覆盖缺乏上述 S1/S2，可用**额外的固定机制集**，但不得声称统计显著或代表整套 SafeLIBERO。

只有 G0/G1 全 PASS 才可以执行。以相同初始 scene、RNG、完整 controller/filter/policy 状态进行配对；首个名义 chunk 必须完全一致；之后状态分叉，各分支按照各自观察**独立**闭环重规划，绝不强迫后续共享 chunk。`get_state()` 无法完整 fork 时从相同初始条件确定性重跑，不宣称精确 fork。每个方法必须同一 episode 步数上限、同一强制停止条件。

**预注册参数**：只在开发集上试至多 3 档 `λ`、至多 2 档风险迟滞宽度；其它参数固定。总开发预算在报告中明确，一旦发现严重不可逆违例即停；不得针对具体 case 逐例调参。最优配置需同时满足严格安全/成本约束，不得只最大化原生 TSR。

**Pilot 继续门槛（工程性，非统计显著门槛）**：没有新的硬安全/任务完整性违规；B3 相对 B0 至少出现可解释的停滞解除或合法操作改善；和 B1/B2 相比存在可归因优势迹象；p95 新增控制开销 ≤ B0 过滤子系统 p95 的 20%（若基线非常快，应另报告绝对毫秒，不能只用百分比掩盖问题）；额外 VLA 调用数 = 0；QP 调用数/outer step ≤ 1。若 B3 成功提升但 task wall clock 变慢，判 `SPEED_TRADEOFF` 并解释，不自动晋级。

**不满足继续条件的输出**：负结果及主因分析；若仅 B1 有效，主线应退回已有 circulation，不要硬造联合创新。

## 8. G4：独立、锁参的扩展测试

建议冻结后：`4 tasks × 16 initial states × 3 CRN streams × 4 methods = 768 episodes`（具体 4 个任务以现有 SafeLIBERO 一致跑通的 task IDs 预注册；若资源不足，先执行分层 4×8×2×4=256 pilot-extended，仅作探索）。最终争取覆盖更多 suite/level；原论文是 4 suites×2 levels×4 tasks×50 episodes=1600 **每方法**，本阶段小集不可叫做完整官方复现。

独立随机种子、初始状态和冻结选择表，不与 G2/G3 混用。四组按 `(task,init,stream)` 成组排列并随机交错运行，控制硬件负载、policy worker 缓存和热启动差异；按 worker 做一致性 spot-check。提交逐 episode 数据，不准仅提交聚合均值。

### 主要指标（全部要有分母与失败类型）

1. `TSR_native, CAR_native, SSR_native`（基准口径）；`TSR_valid, StrictSSR_valid`（严格口径）；`forbidden_contact_episode_rate`, `task_integrity_violation_rate`, `Dmax_mug_mm`。
2. `episode_steps`, `task_wall_clock_seconds`（包含推理、QP、仿真；成功与失败分层）、`policy_inference_ms`, `governor_ms`, `qp_ms`, `total_control_ms` 的 p50/p95/p99；每条真实 VLA 新推理调用次数、每步 QP 次数。
3. 时空机制：参考方向翻转次数、CBF 激活时长/比率、累计动作修正范数、任务阶段转换时间、末端往复/回撤距离、停滞段数与时长、碰撞发生时刻与归属物体。
4. 若终局 timeouts 多，报告步数上限截尾的 `restricted mean steps` 或 capped wall-time；不允许只在成功样本上比较均值从而产生幸存者偏差。

### 统计分析

对每个 `(task,init,stream)` 计算 B3-B0、B3-B1、B3-B2 的配对差异；0/1 结果用配对翻转矩阵及 exact McNemar（样本小则优先描述效应和置信区间），连续结果提供配对中位差和 bootstrap 95% CI。**bootstrap 以 initial state 为簇**，保留同一 state 的 streams/方法关联；不得将每个物理 substep 当独立样本。2×2 交互作用预注册为

\[
I=(Y_{B3}-Y_{B2})-(Y_{B1}-Y_{B0}).
\]

对成功率 `Y` 越大越好，对耗时/违规率 `Y` 越小越好，解释要注意符号及尺度。仅凭交互作用非零并不能证明某种新的安全定理。

## 9. 计算开销控制（不可妥协）

- **首要限制**：VLA 额外 query=0；在线多候选=0；额外 QP=0；无未来物理 rollout。
- 单次 QP 的约束和变量维度与 B0 一致；仅目标矩阵/线性项随参考变化。全部解析运算应为固定维度矩阵和状态更新。不得将 CVXPY 每步重建与 warm-start 差异误当算法本身开销；基线与新方法同一求解路径做公平比较。
- 默认风险关闭时 `lambda=0` 走 B0 原生路径；计数风险开启频率，以揭示在线平均成本是否受触发率影响。
- 计算成本硬门槛建议先预注册 `p95(total filter+governor) ≤ 1.2× p95(B0 filter)`，同时报告毫秒绝对增量；如果原基线小于 1ms 则该百分比可能不稳定，须单独审计，不可随意放宽阈值以过门。
- **物理执行效率独立**：不仅测计算延迟，还测 action step 数、成功/失败共同截止时间内的实际任务进展和 wall-time。仅让机器人更慢、更保守来减少碰撞不视为成功。

## 10. 必须支持的日志字段和结果目录

推荐输出（可按真实仓库目录调整，但字段语义必须一致）：

```text
experiments/stg_v1/
  audit/runtime_manifest.json
  audit/reproduction_boundary.md
  audit/geometry_audit.csv
  audit/action_mapping_test.json
  audit/safety_contract.md
  audit/baseline_equivalence.json
  config/frozen_dev.yaml
  config/frozen_test.yaml
  src/                         # 新模块；原安全层尽量最小 patch
  tests/test_mapping.py
  tests/test_tangent.py
  tests/test_governor_reset.py
  tests/test_qp_equivalence.py
  tests/test_safety_contract.py
  outputs/g2_mechanisms.csv
  outputs/g3_episodes.jsonl
  outputs/g4_episodes.jsonl
  outputs/g4_steps.parquet
  outputs/pairwise_results.csv
  reports/G0_AUDIT.md
  reports/G1_NUMERICS.md
  reports/G2_MECHANISM.md
  reports/G3_PILOT.md
  reports/G4_FINAL.md
  reports/NEGATIVE_FINDINGS.md
  reports/REPRO_COMMANDS.md
```

逐 episode 必备：`run_id,git_sha,model_sha,scene_sha,task_id,level,initial_state_id,crn_stream,method,seed,first_chunk_sha256,success_native,car_native,ssr_native,tsr_valid,strict_ssr_valid,task_integrity_ok,forbidden_contact_count,dmax_mug_mm,termination_reason,episode_steps,task_wall_s,total_vla_queries,total_qp_solves,qp_fail_count,filter_p95_ms,governor_p95_ms`。

逐控制步必备：`step,chunk_index,nominal_action_7d,qp_u_nom_9d,qp_u_ref_9d,qp_u_out_9d,executed_action_7d,reference_r_world,tangent_world,obstacle_proxy_id,risk_mode,lambda,cbf_min_residual,qp_status,qp_time_ms,governor_time_ms,ee_pos_world,ee_vel_world,contact_events,protected_object_true_pos,task_phase_evidence`。禁止将 `task_phase_evidence` 的 privileged 真值反馈给在线算法。

## 11. Stop / Go / Pivot 判决表

| Gate | GO | STOP | PIVOT |
|---|---|---|---|
| G0 | baseline/坐标/几何/评测完整一致 | 几何、策略 chunk、CAR/SSR、QP 不可信 | 先修全方法共同基线 |
| G1 | 单次 QP、9D 保真、重置与等价通过 | 映射、维度、solver、方向退化不稳定 | 仅修控制接口，不跑主结果 |
| G2 | 有明确 B0 停滞/振荡机制且可被至少一组缓解 | 失败都来自策略/碰撞漏检 | 转几何覆盖或 VLA 能力研究 |
| G3 | 同约束安全不恶化，有可复现合法进展且成本合格 | 物理违规增加、在线变慢严重、无法复现 | 若 B1 单独足够则不主张联合贡献 |
| G4 | 独立测试有效、稳定、成本成立 | 置信区间/任务扩展不支持 | 发布负结果并回到问题诊断 |

**严格禁止**：对 G4 test 调参；改 CBF 障碍代理/阈值使 B3 变好；把自动抓取撞飞当成功；只报告完成任务的耗时；从单个成功视频得出提升；把原始 AEGIS Oracle 结果叫完整视觉方法复现；把 proxy margin 当作已认证 barrier；把小样本初步优势称为统计结论。

## 12. 给实验侧智能体的工作与汇报协议

1. **立即只读审计**：先提交 G0 证据清单、真实命令入口和 baseline 健康检查；不得在未过门禁时直接写方法代码。
2. **G0 PASS 后创建独立分支**：最小侵入实现 B1/B2/B3；必须先完成 G1 的零引导等价与数学单测，才允许 GPU rollout。
3. **每完成一个 gate**，交付 `PASS/BLOCK/INCONCLUSIVE`、Git SHA、配置文件、原始数据路径、可运行命令、关键日志摘要与下一步是否值得执行的判断。若中断，不可自作主张扩大试验覆盖或重定义指标。
4. **阶段失败时同样交付**：错误日志、可重放最小状态、失败归类、不能做出的结论；优先证明失败原因，而不是调整结果。
5. **结果图**：至少绘制严格成功率配对翻转图、阶段完成分布、干预/进展时序图、四组计算延迟分布、耗时–安全二维 Pareto 图。
6. **论文级总结**：将“经典控制方法可解释的问题”“冻 VLA 独有的续接行为”“超出现有环流/ERG 的实证增益”分开陈述；RMPflow/CCM 只做后续方向，不用未经证明的收缩/可行性保证包装本方法。

## 13. 参考理论与新近相关工作（非本项目的已验证结论）

- AEGIS / VLSA: https://github.com/THU-RCSCT/vlsa-aegis ；benchmark 范围及原始实现以该项目和其论文为准。
- RMPflow (Cheng et al., T-ASE 2021): https://doi.org/10.1109/TASE.2021.3053422
- Circulation CBF (Gonçalves et al., IEEE TCST 2024): https://doi.org/10.1109/TCST.2024.3372802
- ERG with Dynamic Safety Margin and Navigation Field (Automatica 2024): https://doi.org/10.1016/j.automatica.2024.111696
- Route-Guided CBF-QP Repair (2026 preprint / workshop): https://arxiv.org/abs/2611.01713 。它已将路径引导引入 AEGIS 类 QP；本研究不能把“有方向的绕障”本身称为原创贡献。

---

**最终决策原则：先证明机制与在线可部署性，再决定是否形成论文方法。** 任何统计和新颖性结论只能根据 G4 的独立锁参数据得出。
