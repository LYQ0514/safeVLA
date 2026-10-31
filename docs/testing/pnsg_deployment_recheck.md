# PNSG deployment 身份复核结果（state0 / state1）

> 日期：2026-10-31
> 协议：`docs/reference/pnsg_obs_protocol.md` v1（deployment = 主实验身份）
> 数据根：`/data/safeVLA/data/audit/`
> 性质：复核，取代旧 synchronous 身份记录的结论地位

## 摘要结论（准确表述）

> 在 deployment（真实部署输入）身份下复核 state0/query20 与 state1/query19：
> **两个状态的 Source 与 Action 冻结候选族均包含通过两步非线性认证的候选**，
> 各候选认证结果在五次跨进程重复中均未翻转。源空间修正的相对优势**尚未观察到**。
> 之前 state1「仅 Source 成功」的结论是 synchronous（force 刷新）身份的产物，
> 不代表部署行为，予以修正。

---

## 1. 为什么复核

首轮候选评估存在 obs 身份缺陷：复现脚本从 snapshot 重建 obs（force 刷新），
而非加载筛选时策略实际消费的 obs。修正后冻结两种身份：

- **deployment（主实验）**：obs = `env.step()` 返回值（π0.5 真实部署输入）。
- **synchronous（诊断）**：obs = force 刷新（消除采样相位滞后）。

此前 state0/state1 的「结论」分别在混乱身份 / synchronous 身份下得到，需按
deployment 重新评估。

## 2. 身份记录

| 状态 | obs_protocol | 传感滞后（eef） | native 穿透 |
|---|---|---|---|
| state0/query20 | deployment | 0.468 mm | −0.17 mm |
| state1/query19 | deployment | 0.161 mm | −1.00 mm |

> 传感滞后 = `env.step` 返回 obs 的 `robot0_eef_pos`（site_xpos[grip_site]）vs
> snapshot 最新 site 位置，读同一 site。本次 rollout 为 sub-mm 级。

## 3. state0/query20（deployment）

| 候选 | 通过 | native (mm) | C_act | C_act(加权) | C_src |
|---|---|---|---|---|---|
| nominal | 0/5 | −0.17 | — | — | — |
| **source β=0.25** | **5/5** | 无接触 | 0.178 | 0.0019 | 4.5 |
| source β=0.5 | 0/5 | −0.05 | 0.369 | 0.0033 | 18.0 |
| source β=0.75 | 0/5 | −0.01 | 0.595 | 0.0053 | 40.5 |
| source β=1.0 | 0/5 | −0.13 | 0.893 | 0.0086 | 72.0 |
| **action β=0.25** | **5/5** | 无接触 | 0.078 | 0.0043 | — |
| **action β=0.5** | **5/5** | 无接触 | 0.155 | 0.0086 | — |
| action β=0.75 | 0/5 | −0.08 | 0.233 | 0.0129 | — |
| **action β=1.0** | **5/5** | 无接触 | 0.310 | 0.0172 | — |

**两组都成功**，非单调。Source 最小成功 β=0.25（C_act=0.178），
Action 最小成功 β=0.25（C_act=0.078）。

## 4. state1/query19（deployment）

| 候选 | 通过 | native (mm) | C_act | C_act(加权) | C_src |
|---|---|---|---|---|---|
| nominal | 0/5 | −1.00 | — | — | — |
| source β=0.25 | 0/5 | −0.77 | 0.345 | 0.0064 | 4.5 |
| source β=0.5 | 0/5 | −0.25 | 0.878 | 0.0173 | 18.0 |
| **source β=0.75** | **5/5** | 无接触 | 1.699 | 0.0344 | 40.5 |
| **source β=1.0** | **5/5** | 无接触 | 2.239 | 0.0481 | 72.0 |
| action β=0.25 | 0/5 | −0.67 | 0.179 | 0.0024 | — |
| action β=0.5 | 0/5 | −0.16 | 0.358 | 0.0047 | — |
| action β=0.75 | 0/5 | −0.85 | 0.536 | 0.0071 | — |
| **action β=1.0** | **5/5** | 无接触 | 0.715 | 0.0094 | — |

**两组都成功**，非单调。Source 最小成功 β=0.75（C_act=1.699，C_src=40.5，
预算率 0.75），Action 唯一成功 β=1.0（C_act=0.715）。

## 5. 关键发现：身份敏感性 + rollout 发散

1. **state1 的结论随 obs 协议翻转**：
   - synchronous（旧）：source β=1.0 成功，action **全失败** → 误判「仅 Source 成功」。
   - deployment（复核）：source β=0.75/1.0 成功，action β=1.0 **也成功** → 两组都成功。
2. **snapshot 也不同**：两次筛选的 snapshot_sha256 不同（`51fdd9fd…` vs `2503049f…`），
   即 rollout 跨进程发散导致物理状态不同。因此「仅 Source → 两组都成功」的翻转
   **不能唯一归于 obs 协议**，obs 身份与物理发散混杂，本轮未做消融。
3. **部署动作代价**：两个状态里 Action 成功候选的 C_act 均小于 Source 成功候选
   （state0：0.078 < 0.178；state1：0.715 < 1.699）。若只看「最小部署代价的修复」，
   Action 反而更省。

## 6. 代价匹配是否进行

**不进行。** 代价匹配的前提是「state1 在 deployment 身份下仍为仅 Source 成功」，
但复核结果是两组都成功，前提不成立。源空间相对优势在 deployment 身份下
**尚未被观察到**，无可匹配的「Source 独有成功」案例。

## 7. 数据文件

| 内容 | 路径 |
|---|---|
| state0 复核 | `data/audit/candidate_repro_state0_q20_deploy.json` + `_rep{0..4}.json` |
| state1 复核 | `data/audit/candidate_repro_state1_q19_deploy.json` + `_rep{0..4}.json` |
| 入选状态 | `data/audit/stable_dangerous_state{0,1}_deploy/`（含 obs.npz + 身份） |
| obs 协议 | `docs/reference/pnsg_obs_protocol.md` |

## 8. 冻结结论

1. deployment 身份下，state0 与 state1 均为「两组都成功」，无候选翻转。
2. 之前「仅 Source 成功」是 synchronous 身份的产物，已修正。
3. 源空间相对优势尚未在部署身份下观察到；继续扩独立状态（deployment 身份），
   不再沿用 synchronous 旧结论。
