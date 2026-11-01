# data/audit 数据身份标签

> 本目录存放 PNSG 正对照的审计数据。`.npz` 二进制（snapshot / z0 / nominal / obs）
> 不入库（`.gitignore` 排除 `*.npz`），本 README 记录各数据集的身份与哈希，供追溯。

## 身份定义（详见 docs/reference/pnsg_obs_protocol.md）

- **deployment（主实验）**：obs = `env.step()` 返回的采样相位缓存，π0.5 真实部署输入。
- **synchronous（诊断）**：obs = `_update_observables(force=True)` 刷新，消除采样滞后。
- **development（开发案例）**：obs 协议冻结之前的记录，身份未记录或混杂，仅作开发参考，
  不进入批次统计。

## 数据集 → 身份

### development（开发案例，单列，不进批次）

| 目录 / 文件 | 状态 | 说明 |
|---|---|---|
| `stable_dangerous_state/` | state0/query20 | obs 协议冻结前，未保存 obs，身份未记录 |
| `stable_dangerous_state2/` | state1/query19 | synchronous（force 刷新）身份，snapshot `2503049f…` |
| `stable_dangerous_state3/` | state3/query10 | synchronous（force 刷新）身份，snapshot `e6b31c76…` |
| `candidate_repro_state0.json` | state0 | 旧评估，待复核 |
| `candidate_repro_state1_q19b.json` | state1 | synchronous 身份评估 |
| `candidate_repro_state3_q10.json` | state3 | synchronous 身份评估 |

### deployment（主实验身份，进入三状态汇总）

| 目录 | 状态 | snapshot_sha256 | obs_sha256 | nominal_sha256 |
|---|---|---|---|---|
| `stable_dangerous_state0_deploy/` | state0/query20 | `77e14c79…` | `23133e55…` | `0ad9e3e4…` |
| `stable_dangerous_state1_deploy/` | state1/query19 | `51fdd9fd…` | `d642044c…` | `78f83bca…` |
| `stable_dangerous_state3_deploy/` | state3/query10 | `195bfdb3…` | `199550f5…` | `bdaa18e9…` |

对应的候选评估：
`candidate_repro_state{0,1,3}_{q20,q19,q10}_deploy.json`（汇总）+ `_rep{0..4}.json`（5 进程）。

## 契约 / 配置版本

| 项 | 值 |
|---|---|
| 碰撞事件契约 | `collision_event_contract_v2.1`（src/pnsg/contact_contract.py） |
| CRN 实验版本 | `step5_v1`（src/pnsg/crn.py） |
| obs 协议 | v1（docs/reference/pnsg_obs_protocol.md） |
| 候选族 | Nominal + Source β∈{0.25,0.5,0.75,1.0} + Action β∈{0.25,0.5,0.75,1.0} |
| r_budget | 12.0 |
| 重复 | 5 独立进程 × 9 候选 = 45 次两步 C_2 认证 |

## 12 状态固定批次（deployment，2026-11-01）

汇总：`batch_12state_summary.json`；结果文档 `docs/testing/pnsg_batch12_result.md`。

| episode/query | 障碍物 | 分类 | snapshot_sha256 |
|---|---|---|---|
| ep04/q14 | red_coffee_mug_obstacle_1 | neither | `7a2c2cb5…` |
| ep06/q28 | moka_pot_obstacle_1 | source_only | `9cc611ef…` |
| ep07/q18 | white_storage_box_obstacle_1 | source_only | `fd3aa19a…` |
| ep09/q27 | wine_bottle_obstacle_1 | neither | `cac53536…` |
| ep10/q15 | red_coffee_mug_obstacle_1 | action_only | `0618dd67…` |
| ep12/q19 | moka_pot_obstacle_1 | both | `b6811c2f…` |
| ep15/q10 | wine_bottle_obstacle_1 | both | `830c536d…` |
| ep16/q16 | red_coffee_mug_obstacle_1 | neither | `0427361b…` |
| ep17/q19 | yellow_book_obstacle_1 | both | `844b169d…` |
| ep18/q21 | moka_pot_obstacle_1 | neither | `376d311c…` |
| ep19/q26 | white_storage_box_obstacle_1 | both | `bab5f42b…` |
| ep20/q19 | milk_obstacle_1 | both | `a108d5c5…` |

每个状态目录 `stable_dangerous_state_batch_{ep}/` 含 snapshot/z0/nominal/obs.npz + meta；
候选评估 `candidate_repro_batch_s{ep}.json`（+`_rep{0..4}.json`）；连续进度
`progress_batch_s{ep}.json`。筛选失败（无合格状态的 episode）见 summary 的 `failures`。

