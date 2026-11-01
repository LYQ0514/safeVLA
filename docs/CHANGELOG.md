# CHANGELOG

## [Unreleased]

### 2026-11-01 — PNSG 固定预算设计批次（12 状态）与方法主张收缩

- **新增**：12 状态固定批次（deployment obs 协议）结果
  `docs/testing/pnsg_batch12_result.md`、编排器 `tools/pnsg_run_batch.py`。
- **结果**：状态级四分类 — 两组都能修复 5 / 仅 Source 2 / 仅 Action 1 / 都不能 4。
  Source 修复 7/12，Action 6/12；共同成功的 5 状态中 Action 的 C_act 更省 4 个。
- **结论**：未建立源空间 steering 相对动作空间修正的额外价值。
  **停止扩大当前 steering 方案**；保留局部修复能力与候选族间差异的机制观察。
- 设计文档 `docs/设计.md` 第 21 节固化方法主张收缩。

### 2026-10-31 — PNSG obs 协议冻结与三状态身份统一

- **新增**：obs 协议 `docs/reference/pnsg_obs_protocol.md` v1
  （deployment 主实验 / synchronous 诊断）。
- **修正**：机器人 eef sensor 读 site_xpos[grip_site] 非 body xpos；复现应加载保存
  obs 而非从 snapshot 重建。
- **复核**：deployment 身份下 state0/state1 均为「两组都成功」，state3「两组都失败」；
  撤回「已观察到 Source 独有修复优势」。
- 契约：`collision_event_contract_v2.1`、CRN `step5_v1`。
