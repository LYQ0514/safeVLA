# CHANGELOG

## [Unreleased]

### 2026-11-01 — step330 反事实动力学归因（历史惯性，终止案例）

- **新增**：`docs/testing/step330_counterfactual_attribution.md`、
  `tools/attribution_counterfactual.py`。
- 反事实实验：a_M vs 零动作 a_0，mug 终点位置差异 Δ_causal = 0.000mm。
- mug 位移 4.58mm 是历史碰撞残余动力学（初速 0.66m/s 水平滑动 + 与 box_base 推挤），
  与当前动作无关。step330 双契约差异例证已证伪。
- 理论认识：基于历史最大位移的 CAR 单调不减，历史违规不可撤销——当前控制器只能
  预防未来违规，不能撤销已发生违规（非算法贡献，纳入方法论）。
- 三次解释转变完整记录；不启动 task3、不恢复 H1。
### 2026-11-01 — H1-W 第一轮（负结果 + 更正：375mm 是读取伪影）

- **新增**：`docs/testing/h1w_round1_results.md`、`tools/h1w_witness.py`、
  `docs/explanation/h1w_coordinate_protocol_v1.md`。
- H1 非凸见证：task0 预设范围未发现（Level 1 未达成），5 个不安全状态均为幅度型。
- **更正**：上一版「cream_cheese 撞飞 375mm / 契约保护盲区」是观测缓存读取伪影，
  已撤回。用 sim.data.xpos 重测：cream_cheese 位移 0.000mm；mug 4.58mm、box_base 1.74mm
  是真实的。
- 时间身份已核实：D_before=10.62mm（早已违规），mug 4.58mm 是 a_M 新增，非伪影。
- 不批准 task3 新采集。
### 2026-11-01 — AEGIS 任务类型失败表型分析（描述性，非因果）

- **新增**：`docs/explanation/aegis_task_failure_modes.md`。
- **核心发现**：long-I vs long-II 任务失败率仅差 1pp（86.5% vs 87.5%），但失败任务
  中碰撞比例 P(C|F) 差 23.6pp（35.3% vs 58.9%）。相似失败率、不同安全构成。
- **三失败表型**：无碰撞无成功 / 早期碰撞 / 中后期碰撞（表型 ≠ 机制）。
- 修正两处过强表述：550/550 超时 ≠ 持续有效尝试；后期碰撞多 ≠ 时间退化（需按
  暴露时间/风险中 episode 数归一化，区分首次与重复碰撞）。
- 概念性成果：Failure-Conditioned Safety Analysis（诊断工具，非算法名）。
- 科学状态 INCONCLUSIVE + 执行状态 PAUSED 维持不变，不触发新采集。

### 2026-11-01 — 干预后可恢复性因果审计（P0-A，INCONCLUSIVE + PAUSED）

- **新增**：协议 `docs/reference/intervention_recoverability_audit_protocol.md` v1
  （四类失败机制 / G1-G3 门槛 / 轨迹级 vs 局部状态分支 / GO-INCONCLUSIVE-STOP 判决 /
  科学结论与执行状态分离 / 条件选择注意事项）。
- **新增**：审计 `docs/testing/p0a_evidence_audit.md`（6 条 long 轨迹逐条证据）。
- **判决**：科学结论 INCONCLUSIVE，执行状态 PAUSED。观测缺口（前缀 4 步干预未记录）
  + 「未观察到过滤」+ 无法识别 Policy Deficiency 与 Induced Degradation 相对贡献。
- 证据边界：不支持「即时过滤直接造成所有 long 失败」；「历史干预导致退化」未识别，
  不肯定也不否定。

### 2026-11-01 — Source→AEGIS 串接实现（归档，停止规模扩展）

- **新增**：`src/safety/source_aegis.py`（源空间 steering 提案 + AEGIS 过滤）、
  `tools/run_eval_source_aegis.py`。
- **冒烟**：spatial task0 5 episode，TSR 20% / CAR 80% / SSR 20%（小样本）；
  单案例 ep35 基线碰撞 12.15mm → 组合后 0.002mm。
- **结论**：触发机制风险辨别能力未量化、计算开销高（5 条 37min，外推 400 条约 50h）、
  源空间相对同条件动作修正的增量价值未建立 → 停止规模扩展。
- 归档：`docs/explanation/source_aegis_concat_archive.md`。
- 执行边界：本实现无 C_2 认证，是「AEGIS 过滤后执行」，不能继承「认证后执行」表述。

### 2026-11-01 — AEGIS 上游 Source 改造可行性探索（收口，暂停）

- **新增**：最小采集器 `tools/aegis_long_collect.py`（显式源噪声，6 episode long task0）、
  smoke `tools/aegis_long_smoke.py`、时间对齐分析 `tools/aegis_long_align.py`。
- **AEGIS 状态保存/恢复**：`src/safety/aegis_adapted.py` 新增 `save_state`/`restore_state`
  （z_fixed/p2/R2/Q2/Q1/计数器等有状态字段）。
- **结果**：6/6 soup 未完成；「CBF 阻碍推进」动机未被支持，主因是策略提案不足
  （5/6）与抓取失败（EP2）。冻结的 Source 生成器与失败间缺少作用机制。
- **决定**：暂停 AEGIS 上游 Source 候选续轨实验。分析见
  `docs/explanation/aegis_long_alignment_analysis.md`。

### 2026-11-01 — PNSG 固定预算设计批次（12 状态）与方法主张收缩

- **新增**：12 状态固定批次（deployment obs 协议）结果
  `docs/testing/pnsg_batch12_result.md`、编排器 `tools/pnsg_run_batch.py`。
- **结果**：状态级四分类 — 两组都能修复 5 / 仅 Source 2 / 仅 Action 1 / 都不能 4。
  Source 修复 7/12，Action 6/12；共同成功的 5 状态中 Action 的 C_act 更省 4 个。
- **结论**：未建立源空间 steering 相对动作空间修正的额外价值。
  **停止扩大当前 steering 方案**；保留局部修复能力与候选族间差异的机制观察。
- 设计文档 `docs/设计.md` 第 21 节固化方法主张收缩。

### 2026-11-01 — PNSG 代价匹配诊断与最终收口

- **新增**：代价匹配诊断 `docs/testing/pnsg_cost_match.md`、脚本
  `tools/pnsg_cost_match.py`。两个 source-only 状态均可由原 Action 方向在匹配
  部署动作代价后修复（5/5）→ 幅度覆盖不足解释独占案例，无须诉诸方向优势。
- **新增**：最终研究报告 `docs/testing/pnsg_steering_final_report.md`（实验线收口，
  固定最终判断）。
- **最终措辞**：在当前 π0.5、任务切片、代理约束、两步认证和候选生成规则下，实验
  未建立源空间 steering 相对动作空间修正的额外价值，因此停止扩大当前方案。

### 2026-10-31 — PNSG obs 协议冻结与三状态身份统一

- **新增**：obs 协议 `docs/reference/pnsg_obs_protocol.md` v1
  （deployment 主实验 / synchronous 诊断）。
- **修正**：机器人 eef sensor 读 site_xpos[grip_site] 非 body xpos；复现应加载保存
  obs 而非从 snapshot 重建。
- **复核**：deployment 身份下 state0/state1 均为「两组都成功」，state3「两组都失败」；
  撤回「已观察到 Source 独有修复优势」。
- 契约：`collision_event_contract_v2.1`、CRN `step5_v1`。
