# CCA-ASF 研究线索引

> 日期：2026-11-02
> 状态：v1 归档。本文件是研究线单一入口，按时间顺序索引全部文档与关键证据。

---

## 0. 一句话结论

各向异性安全过滤（CCA-ASF v1）证明了「局部方向重分配」的数学机制与工程实现，
但在真实闭环中暴露出反例：减少任务方向代理偏移**不能**保证闭环物理安全
（S01/st3 mug 27-30mm 位移 vs euclidean 0mm）。已于 `66c9864` 归档。

## 1. 文档索引（按阶段）

| 阶段 | 文档 | 关键结论 |
|---|---|---|
| 规格书 | `docs/anisotropic_vla_safety_engineering_spec.md` | 方法定义 + 6D/9D 白化修正 |
| TASK A | `docs/testing/cca_asf_runtime_manifest.md` | 运行时核实 + D1/D2/D3 决策冻结 |
| TASK B | `src/safety/cca_asf/`（metric/intention）| 白化度量 + 任务方向提取 + 17 单测 |
| TASK C | `docs/testing/cca_asf_task_c_integration_report.md` | 真实 AEGIS QP 集成 + filter_mode |
| TASK D | `docs/testing/cca_asf_task_d_runtime_smoke_test.md` | 端到端 5 门禁（β=0 逐位等价）|
| E0 | `docs/testing/cca_asf_task_e0_qp_effect_audit.md` | 46/46 方向重分配 + ΔD_⊥>0 trade-off |
| A0 | `docs/testing/cca_asf_e1_metric_contract.md` | 四类指标口径冻结 |
| B | `docs/testing/cca_asf_e1_catastrophic_car_audit.md` | 27.5mm mug 位移归因 |
| B1 | `docs/testing/cca_asf_e1_trajectory_divergence_closure.md` | 三问题证据补全 |
| 判决 | `docs/research/cca_asf_v1_final_verdict.md` | 四项结论 + 归档范围 |

## 2. 关键证据锚点（后续立题可快速定位）

1. **T3 白化强度**：`β=4, d_y=e₁` 时 `W_11 = 0.2`（非字面公式的 4.04）。
2. **D1 β=0 逐位等价**：单步 filter 对比最大差 0.0（80 步）。
3. **E0 trade-off**：ΔD_∥=−0.050 / ΔD_⊥=+0.070 / ΔD_total=+0.019（46/46）。
4. **B 稳定复现**：cca_asf Dmax=27.497mm（2 次逐位一致）vs euclidean 0mm。
5. **B1 时间线**：step0 分叉 → step219 首次位移 → step221-222 突发 >10mm。
6. **碰撞步 β=0**：名义动作可行，撞飞是历史干预改变轨迹的后果，非单步直接碰撞。

## 3. 核心反例（留给论文）

$$\boxed{D_\parallel^{\mathrm{CCA}} < D_\parallel^{\mathrm{AEGIS}} \;\not\Rightarrow\; \mathrm{CAR}_{\mathrm{CCA}} \ge \mathrm{CAR}_{\mathrm{AEGIS}}}$$

局部优化目标（代理任务方向偏移）与闭环安全性能不一致。

## 4. 两个待立题方向（不立即开展工程）

1. **任务语义感知的各向异性**：识别哪些控制自由度真正影响任务完成，而非动作块方向代理。
2. **长时域安全干预**：评价当前安全修正对未来策略状态/安全风险/任务进度的影响。

## 5. 资产保留

- `src/safety/cca_asf/`：数学模块（保留）；
- `AegisSafetyLayer.filter_mode`：euclidean/cca_asf 两模式（保留，euclidean 仍是原始 AEGIS）；
- `tests/safety/test_cca_asf_*.py`：26 个测试（保留）；
- 终止：E1 正式评测、E2、β/方向调参。
