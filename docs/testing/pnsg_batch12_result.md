# PNSG 12 状态固定批次（deployment 身份）结果

> 日期：2026-11-01（批次运行完成）
> 协议：`docs/reference/pnsg_obs_protocol.md` v1（deployment）
> 数据根：`/data/safeVLA/data/audit/`
> 主比较单位：**状态**（非候选 / 非 β）

## 摘要结论（准确表述）

> 在 deployment 身份下，按冻结顺序扫描 30 个 episode、纳入 12 个独立危险状态，
> 状态级四分类为：**两组都能修复 5、仅 Source 2、仅 Action 1、都不能 4**。
> 各候选认证结果在五次跨进程重复中均未翻转。
> 在 5 个「两组都能修复」的状态里，最小部署动作代价 C_act 的配对比较中
> **Action 更省 4 次、Source 更省 1 次**。源空间修正未展现出系统性的相对优势。

---

## 1. 冻结批次规则（回顾）

- episode 4 起，扫描上限 30 个 episode（ep4..ep33）。
- 每个 episode 最多纳入一个状态：取首个通过两层 5/5 nominal 稳定性门的 query。
- deployment obs 协议、完整 snapshot、冻结九候选、五进程评估。
- 达到 12 个合格状态或耗尽 30 episode 即停止。
- 不根据 Source/Action 结果决定纳入/跳过/提前结束；筛选失败全部保留。

## 2. 状态级四分类（主比较单位）

| 结果 | 状态数 | 占比 |
|---|---|---|
| 两组都能修复 | 5 | 41.7% |
| 仅 Source 能修复 | 2 | 16.7% |
| 仅 Action 能修复 | 1 | 8.3% |
| 两组都不能修复 | 4 | 33.3% |
| **合计** | **12** | 100% |

**没有任何一方占优**：Source 独占 2、Action 独占 1，双方都有「独占修复」的案例，
也都有「无法修复」的案例。

## 3. 状态明细

| episode/query | 障碍物 | native 穿透 | 分类 | Source 最小通过 | Action 最小通过 |
|---|---|---|---|---|---|
| ep04/q14 | red_coffee_mug | −4.13 mm | neither | — | — |
| ep06/q28 | moka_pot | −0.75 mm | source_only | β=0.25 (C_act=1.440) | — |
| ep07/q18 | white_storage_box | −1.28 mm | source_only | β=1.0 (C_act=0.758) | — |
| ep09/q27 | wine_bottle | −0.08 mm | neither | — | — |
| ep10/q15 | red_coffee_mug | −2.27 mm | action_only | — | β=0.75 (C_act=2.235) |
| ep12/q19 | moka_pot | −0.22 mm | both | β=0.5 (0.329) | β=0.25 (0.079) |
| ep15/q10 | wine_bottle | −0.01 mm | both | β=0.75 (0.737) | β=0.5 (0.526) |
| ep16/q16 | red_coffee_mug | −0.28 mm | neither | — | — |
| ep17/q19 | yellow_book | −0.23 mm | both | β=1.0 (0.940) | β=0.75 (0.500) |
| ep18/q21 | moka_pot | −0.24 mm | neither | — | — |
| ep19/q26 | white_storage_box | −0.72 mm | both | β=0.25 (1.712) | β=1.0 (0.357) |
| ep20/q19 | milk_obstacle | −0.20 mm | both | β=0.25 (0.101) | β=0.25 (0.157) |

## 4. 「两组都成功」状态的 C_act 配对（最小幅度通过候选）

| 状态 | Source C_act | Action C_act | 更省 |
|---|---|---|---|
| ep12 | 0.329 | 0.079 | **Action** |
| ep15 | 0.737 | 0.526 | **Action** |
| ep17 | 0.940 | 0.500 | **Action** |
| ep19 | 1.712 | 0.357 | **Action** |
| ep20 | 0.101 | 0.157 | Source |

Action 更省 4/5，Source 更省 1/5。延续了开发案例（state0/state1）里
「Action 部署代价更低」的趋势，但非绝对（ep20 例外）。

## 5. 独占修复状态观察

- 2 个 `source_only`：ep06（穿透 −0.75mm，β=0.25 即修）、ep07（−1.28mm，需 β=1.0 预算边界）。
- 1 个 `action_only`：ep10（−2.27mm，需 β=0.75）。
- 数量少，且 source_only 的穿透（−0.75/−1.28）与 action_only（−2.27）不构成清晰分界，
  **不足以声称「Source 修浅碰撞、Action 修深碰撞」或反之**。

## 6. 连续进度（2 步前缀）

所有 12 状态 lift 与 target–goal 均零变化（两步内无抓取、无移动）；
EEF–target 距离变化随状态而异（接近幅度不同）。**两步前缀内无任何候选完成抓取**，
进度维度无候选族差异（受 EXECUTE_STEPS=2 限制，本批次不回答更长 horizon 的进度问题）。

## 7. 策略调用与耗时

- Source 组每个 β 需 1 次额外策略前向（4 次额外前向）+ Jacobian 求解 + Δz* 求解；
  Action 组复用 nominal 基准，无额外前向，仅 Δa* 求解。
- 每候选两步认证 ≈ 4.5 s（不含前向）。
- Source 的总计算成本系统性高于 Action（策略前向 + Jacobian）。

## 8. 数据文件

| 内容 | 路径 |
|---|---|
| 批次汇总 | `data/audit/batch_12state_summary.json` |
| 各状态入选 | `data/audit/stable_dangerous_state_batch_{ep}/`（含 obs.npz + 身份） |
| 各状态候选评估 | `data/audit/candidate_repro_batch_s{ep}.json` + `_rep{0..4}.json` |
| 各状态连续进度 | `data/audit/progress_batch_s{ep}.json` |
| 筛选失败 | 汇总内 `failures` 字段（ep5/8/11/13/14 无合格状态） |
| 运行日志 | `data/audit/batch_12state.log` |

## 9. 冻结结论与方法主张

1. **无系统性 Source 优势**：状态级 `source_only=2` vs `action_only=1` 不构成显著差异；
   `both` 状态的 C_act 配对中 Action 更省 4/5。
2. **保留的客观事实**：Source steering 能在 Action 失败的状态上产生通过认证的修复
   （2 个 source_only 状态），Action 亦然（1 个 action_only 状态）——两种方法互补，
   无一方全面占优。
3. **方法主张应收缩**：当前 steering 方案未在安全（状态级修复率）、进度、成本任一维度
   展现出优于 Action 的系统性证据；且 Source 的计算成本（策略前向 + Jacobian）更高。
4. **建议停止扩大当前 steering 方案**，除非有新的设计动机（如更长的 horizon、
   不同的约束构造、或代价匹配后重新评估）。不追加预算寻找有利案例。

## 10. 已知局限（如实记录）

- rollout 跨进程发散：同一 episode 不同运行可能选到不同 query（开发期已确认），
  批次内自洽（筛选+评估同一次 rollout），但「状态」的定义依赖 rollout 实现。
- 两步认证窗口（EXECUTE_STEPS=2）：只回答两步内的安全与进度，不覆盖完整任务。
- 12 状态仍是有界小样本，四分类比例不承担统计显著性。
- C_act 是部署 pose 相对 nominal 的 L2，未含旋转的物理等价性；加权版见原始 JSON。
