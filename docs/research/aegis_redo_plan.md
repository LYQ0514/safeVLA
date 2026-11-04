# AEGIS 复现修复后的重做计划

> 日期：2026-11-04。触发：发现四元数约定 bug（`aegis_reproduction_audit.md`）。
> 本文列出修复后必须重做的全部内容，按依赖顺序排列，作为「重做」的执行清单。

---

## 0. 背景一句话

AEGIS 复现的 `_quat_wxyz_to_rot` 把 `robot0_eef_quat`（xyzw）误当 wxyz，导致末端椭球
中心 p1 偏移 16cm。修复后 long task0 TSR 从 0% → 45%，碰撞率 24% → 55%。**所有基于
bug 时代 AEGIS 的定量结论作废。**

## 1. 已修复（✅）

- [x] `_quat_wxyz_to_rot` 改为 xyzw 直通（`R.from_quat`）；
- [x] 验证 R1/p1 正确；
- [x] long task0 20-episode 快速验证（TSR 45% vs bug 0%）；
- [x] 4 份结论文档标记作废（`aegis_metric_attribution.md`、`circulation_cbf_aegis.md`、
  `grasp_aware_ellipsoid_validation.md`、`grasp_aware_gate_result.md`、
  `grasped_sep_result.md`）；
- [x] CHANGELOG + 归档索引更新。

## 2. 进行中（⏳ 后台）

- [ ] **全量 AEGIS baseline**：`data/eval/aegis_1024_fixed/`（4 suite × 2 level × 4 task
  × 50 episode = 1600），与 bug 时代 `aegis_1024/` 同参数（1024 分辨率、replan5、
  num_steps_wait=10）。预计 13+ 小时，支持 `--resume` 断点续跑。

## 3. 待做（依赖全量结果）

### 3.1 重做「AEGIS 指标归因」（优先级最高）

- 对比修复后 vs vanilla 的 TSR / CAR / SSR / ETS，分场景、分 task；
- **重新回答**：AEGIS 的真实缺陷是什么？
  - 修复后碰撞率 55%（long task0），可能不再是「任务-安全空间冲突」（bug 假象），
    而是「间接传递链」（碰底座带 mug）或「全身覆盖缺口」（末端 vs wrist/forearm）；
  - 修复后 TSR 45%（不再 0%），说明「任务-安全冲突」被 bug 严重夸大。

### 3.2 重新评估支撑感知线的价值

- 支撑感知（`contact_propagation_aware_direction.md`）的因果证据（离线 replay：
  S05 344→6mm、S06 12.6→0mm）是否**在修复后仍成立**？
  - 注意：这些 replay 也用了 `aegis_adapted.py`，p1 也是错的——但其「传递链」机制
    结论（碰底座带 mug）可能仍成立（是物理事实，不依赖 p1），需重验定量。

### 3.3 重做 circulation / grasp-aware / 分离约束

- 这三个方向的「问题」（末端无法接近 soup、禁区 19.3cm、7.4cm 瓶颈）是 bug 假象；
- 修复后 AEGIS 已经能到 soup（0.6cm），这些方向的「必要性」可能消失；
- 但它们的「实现代码」仍保留（`use_circulation`/`grasp_aware`/`grasped_object_body`
  参数），修复后若有新问题可复用。

## 4. 诚实声明

- 这是一次重大自我纠正：我之前基于 AEGIS 的多轮「数据驱动归因」建立在错误复现上；
- 修复后，真实图景可能完全不同：AEGIS 的任务完成能力被严重低估，而其真实缺陷
  （碰撞率 55%）指向「间接传递链 / 全身覆盖」等「支撑感知线」真正研究的问题；
- 在修复后全量 baseline 完成前，不再对 AEGIS 下任何定量结论。
