# PNSG deployment 身份三状态汇总

> 日期：2026-10-31
> 协议：`docs/reference/pnsg_obs_protocol.md` v1（deployment = 主实验身份）
> 数据根：`/data/safeVLA/data/audit/`
> 状态数：3（冻结批次内全部案例，非挑选）

## 摘要结论（准确表述）

> 在 deployment（真实部署输入）身份下，已采集的 3 个独立危险状态中：
> 2 个「两组都成功」、1 个「两组都失败」，**没有「仅一组成功」的案例**。
> 所有候选的认证结果在五次跨进程重复中均未翻转。
> 两个「都成功」状态里，通过候选的最小部署动作代价 C_act 均为 **Action 更小**。
> 当前证据只支持「候选族内，Action 更省动作修正」，**不支持源空间修正的相对优势**。

---

## 1. 三个状态的身份与结果

| 状态 | 障碍物 | native 穿透 | 传感滞后 | Source | Action | 结论 |
|---|---|---|---|---|---|---|
| state0/query20 | moka_pot | −0.17 mm | 0.468 mm | 成功(β=0.25) | 成功(β=0.25/0.5/1.0) | 两组都成功 |
| state1/query19 | storage_box | −1.00 mm | 0.161 mm | 成功(β=0.75/1.0) | 成功(β=1.0) | 两组都成功 |
| state3/query10 | wine_bottle | −1.87 mm | 1.106 mm | 全失败 | 全失败 | 两组都失败 |

- state0/state1 的 deployment 复核见 `docs/testing/pnsg_deployment_recheck.md`。
- state3 是 deployment 新采集（旧 synchronous 记录 snapshot 不同，见 §5）。
- 三个状态均为「两层 5/5 复现门」合格的危险状态（nominal 5/5 危险 + zero-action 稳定）。

## 2. 认证通过率（各候选族 4 个 β，通过 = 5/5 复现通过）

| 状态 | Source 通过 | Action 通过 |
|---|---|---|
| state0 | 1/4（β=0.25） | 3/4（β=0.25/0.5/1.0） |
| state1 | 2/4（β=0.75/1.0） | 1/4（β=1.0） |
| state3 | 0/4 | 0/4 |

合计：Source 3/12，Action 4/12。**无候选族占优**。

## 3. 通过候选的最小部署动作代价 C_act

| 状态 | Source 最小成功 | Action 最小成功 | 更省 |
|---|---|---|---|
| state0 | β=0.25，C_act=0.178 | β=0.25，C_act=0.078 | **Action** |
| state1 | β=0.75，C_act=1.699 | β=1.0，C_act=0.715 | **Action** |
| state3 | 无 | 无 | — |

两个「都成功」状态里，**Action 的最小成功 C_act 均小于 Source**。
Source 的 C_act 来自非线性策略前向（不可直接预算），Action 的 C_act 来自直接
动作修正（线性 Δa*）；二者同一量纲（部署 pose 相对 nominal 的 L2），可比较。

## 4. 执行后任务进度（2 步前缀）

| 状态 | EEF–target Δ 范围 | lift Δ | target–goal Δ |
|---|---|---|---|
| state0 | −2.9 ~ −4.8 mm | 0 | 0 |
| state1 | −3.7 ~ −7.4 mm | 0 | 0 |
| state3 | −12.5 ~ −14.4 mm | 0 | 0 |

三个状态 lift 与 target–goal 均零变化（未抓取、未移动），仅 EEF 靠近目标
（幅度随状态而异）。**两步前缀内无一完成抓取**，进度上无候选族差异。

## 5. 策略调用与总耗时

- 每候选两步认证（恢复+执行+认证，不含策略前向）≈ 4.5 s。
- 策略调用：Source 候选每个 β 需 1 次额外策略前向（z0+αΔz），Action 候选
  只复用 nominal 基准，无额外前向。故 Source 组策略调用次数多于 Action 组
  （4 vs 0 次额外前向），但每候选认证耗时相近。
- 总计算时间：单状态 5 进程 × 9 候选 ≈ 45 次认证 + 冻结求解 + 策略前向。

## 6. state3 的 deployment 与 synchronous 关系

- 旧 synchronous 记录（`stable_dangerous_state3/`）：snapshot `e6b31c76…`，
  native −1.745 mm，结论「两组都失败」。
- deployment 新采集（`stable_dangerous_state3_deploy/`）：snapshot `195bfdb3…`，
  native −1.866 mm，结论「两组都失败」。
- **两个 snapshot 不同**（max diff 0.002，rollout 跨进程发散），故不能写成
  「同状态的协议对照」。deployment 新采集是独立记录，旧 synchronous 记录保留。
- 定性结论一致（两组都失败），但穿透深度不同（−1.87 vs −1.75）。

## 7. 数据文件

| 内容 | 路径 |
|---|---|
| state0 deployment | `data/audit/candidate_repro_state0_q20_deploy.json` + `_rep{0..4}.json` |
| state1 deployment | `data/audit/candidate_repro_state1_q19_deploy.json` + `_rep{0..4}.json` |
| state3 deployment | `data/audit/candidate_repro_state3_q10_deploy.json` + `_rep{0..4}.json` |
| 入选状态 | `data/audit/stable_dangerous_state{0,1,3}_deploy/` |
| obs 协议 | `docs/reference/pnsg_obs_protocol.md` |
| 复核记录 | `docs/testing/pnsg_deployment_recheck.md` |

## 8. 冻结结论与方法主张

1. deployment 身份下 3 状态结果：两组都成功 ×2、两组都失败 ×1、**无「仅 Source 成功」**。
2. 通过候选的最小 C_act 均为 Action 更小 → 当前候选族内 Action 更省动作修正。
3. **源空间修正的相对优势未被观察到。** 撤回「已观察到 Source 独有修复优势」。
4. 保留的客观事实：**Source steering 能产生通过两步非线性认证的修复**（state0/state1
   的 source 候选均通过），但不优于 Action，且部署动作代价更高。
5. 后续扩状态的目的应是**估计两组在独立危险状态上的表现**，不以「找到仅 Source 成功」
   为停止条件。若后续安全、进度、成本持续无 Source 优势，应收缩方法主张，
   而非继续改对照寻找优势。
