# CHANGELOG

## [Unreleased]

### 2026-11-01 — 干预后可恢复性因果审计（P0-A，INCONCLUSIVE）

- **新增**：协议 `docs/reference/intervention_recoverability_audit_protocol.md` v1
  （四类失败机制 / G1-G3 门槛 / 轨迹级 vs 局部状态分支 / GO-INCONCLUSIVE-STOP 判决）。
- **新增**：审计 `docs/testing/p0a_evidence_audit.md`（6 条 long 轨迹逐条证据）。
- **判决 INCONCLUSIVE**：观测缺口（前缀 4 步干预未记录）+ 反向证据（5/6 策略在
  cream 后不转向 soup），不足以支撑继续投入。原则上暂停。
- 低成本补全方案：采集器记录执行前缀全程 5 步干预（仅记录，不自动启动）。

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
