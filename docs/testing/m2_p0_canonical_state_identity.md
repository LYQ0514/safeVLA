# M2-P0 G0：Canonical State Identity 验证

> 日期：2026-11-02
> 状态：G0 门禁，正面结果（NVS 状态确定性可复现）。
> 前置：`../testing/m1_t0_task_integrity_audit.md`（M1-T0 收束）。

---

## 0. 结论

> `reconstruct_nvs`（同版本 + 修正后的 `select_action`）重建的 NVS 状态是
> **确定性可复现**的：S01/S04/S09 各 5 次重建的 NVS 步数与 q_start 哈希完全一致。
> 此前观察到的「NVS 步数波动」是**跨脚本版本差异**（`select_action` 参数版本
> 不同），不是运行时非确定。

因此 canonical parent state 可直接用 `reconstruct_nvs` 确定性重建，无需额外的
「固定绝对决策步」机制。但仍需在正式实验前对每个状态做一次「5 次重建一致性」
门禁，作为状态身份验证。

## 1. NVS 稳定性诊断

| 状态 | NVS 步数（5 次）| q_start 哈希（5 次）| soup 位置 |
|---|---|---|---|
| S01/st3 | 30, 30, 30, 30, 30 | 全一致 | 一致 |
| S04/st3 | 151, 151, 151, 151, 151 | 全一致 | 一致 |
| S09/st3 | 11, 11, 11, 11, 11 | 全一致 | 一致 |
| S02/st0 | **NVS=None（任务成功@267）** | 全一致 | 一致 |

## 1.5 State02 的特殊性（重要）

State02 在 `reconstruct_nvs` 里**从不触发 NVS**——HVNP 自己就能安全完成任务
（成功@267，与 P1b 的 275 一致，跨 run 微小差异）。State02 是「容易案例」，
**不属于 M2 的「困难状态需要任务级绕行」范畴**。

因此 State02 作为 M2-P0 的「接管候选」意义有限：它不需要外部规划给预抓取状态。
M2 的核心对象是 S01/S04/S09（HVNP 无法处理的困难状态）。State02 仍可作为
「HVNP 自主成功的正例」保留，但不参与「接管状态区分」实验。

## 2. 根因澄清（重要）

之前 P4-P0e-r 与 P4-P1 报告中出现的「S04 NVS@86/151/419」波动，根因是：
- `reconstruct_nvs` 与 `select_action` 在多个脚本中被复制，且**参数版本不一致**
  （有的用 12 参数 `select_action`，有的用 14 参数修正版）；
- 修正为统一 14 参数版（显式传 state_id/stream）后，NVS 确定性稳定。

这是「代码版本管理」问题，不是「物理仿真非确定」问题。

## 3. G0 通过标准与门禁

| 门禁 | 通过条件 | 结果 |
|---|---|---|
| canonical parent state 确定性 | 同状态 5 次重建 NVS 步数 + q_start 哈希一致 | ✅ S01/S04/S09 |
| stream 噪声哈希验证 | 不同 stream 噪声不同（已在 M1-P0e 验证）| ✅ |
| P-V/P-H 配对状态一致 | 同一候选两分支用同一 q_start | 待 G2 验证 |

## 4. 对 M2-P0 的含义

1. `reconstruct_nvs` 可作为 canonical parent state 的确定性重建机制；
2. 每个状态在正式实验前，做一次 5 次重建一致性门禁（记录 q_start 哈希 + NVS 步数）；
3. 配对实验的「共同规划起点」= 该状态的 canonical q_start（确定性）。

## 5. 工程产物

- `tools/m2_p0_g0_identity_diag.py`：NVS 波动诊断。
