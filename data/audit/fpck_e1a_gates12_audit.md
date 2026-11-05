# FPCK/CKG E1a 采集器基础设施 + 两个门禁审计

> 日期：2026-11-05。判定：门禁 1 PASS、门禁 2 PASS，可进入 witness 重发现。

## 采集器（src/fpck/collector.py）

- **透明包装**：只记录，不改动作、不增推理、不推进随机流、不额外 env.step；
- **确定性机制**：PnsgPolicy + CRN（nominal_seed + sample_z），query_idx 显式决定 chunk，
  恢复时只需恢复 query_idx/buffer_index（policy cursor），不依赖内部 _chunk 缓存；
- **AnchorIdentity**：完整身份（repo commit / checkpoint hash / suite / level / task /
  state / stream / prompt hash / manifest hash / contract hash / spec version /
  replan_steps / full_state hash / chunk hash / buffer index）；
- **append-only**：原始事件日志只追加，汇总另写。

## 门禁 1：透明采集等价（PASS）

- 配置：long/I/task0, init 0-4, replan=5；
- 结果：**5/5 条 act_err=0.00**，success/steps/CAR/chunk hash/buffer index/推理次数全一致。

**过程中发现并修复的实现问题**：初版关 collector 用 `q = step//REPLAN` 每个 step 都
重新 generate（浪费推理），而 collector 用缓存节奏（每 REPLAN 步 generate 一次），导致
推理次数不一致。修正为两边用相同缓存节奏后，全项一致（act_err=0）。

## 门禁 2：snapshot round-trip（PASS）

- 3 个时刻（step 10/50/150）执行：save full state + policy cursor + chunk → 执行 →
  恢复 → 重新执行；
- 结果：**3/3 动作/关键真值（eef/obstacle/goal truth/native success）/contact 集合全一致**。

关键：恢复包括 policy state（query_idx/buffer_index）+ chunk payload，不只恢复 MuJoCo。
这满足裁决「只恢复 MuJoCo state、不恢复 π0.5 chunk/cursor 和随机状态，不算通过」的要求。

## 结论

采集器基础设施可靠，两个门禁通过，可以进入 witness 重发现（State02 / S01 / S04 各
最多 3 次恢复运行）。重型 RRT 运行的前置条件已满足。
