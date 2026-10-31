# PNSG 观测输入协议（obs protocol）

> 版本：v1（2026-10-31 冻结）
> 范围：PNSG 正对照实验的「危险状态发现 + 候选评估」全链路
> 地位：实验身份定义，改动即版本升级，旧数据不合并

## 1. 背景：为什么需要冻结 obs 协议

PNSG 正对照需要「冻结一个真实危险状态」，在该状态上评估 Source / Action 候选。
「冻结状态」必须同时冻结两样东西，且**明确二者的时序关系**：

1. **物理快照（snapshot）**：`sim.get_state().flatten()` = `time(1)+qpos(97)+qvel(85)`，
   是认证分支（`Certifier.certify_prefix → env.step`）的起点。
2. **策略输入（obs）**：π0.5 前向实际消费的观测 dict（含 image + 8 维 state + prompt）。

关键事实：**robosuite 的 `env.step()` 返回的 obs 是带采样相位的缓存**，不是 step
边界物理状态的纯函数。`Observable.update(force=False)` 按 `sampling_rate`（默认 20Hz）
降采样，且每个 control step 内只采样一次。因此：

- `env.step()` 返回的 obs 可能滞后于 step 结束的物理状态（实测 eef 差 ~9 mm）。
- `_update_observables(force=True)` 才强制把 observable 刷新到与 `sim.data` 同刻。

这两者是**两种不同的输入协议**，不能混用。混用会导致「同一 z0 生成不同动作」，
破坏候选评估的可比性。

## 2. 两种身份定义

### 身份 A：`deployment`（部署复现）—— **正对照主实验身份**

- obs 来源：`env.step()` 返回的 obs（与 π0.5 部署 rollout 完全一致，无 force 刷新）。
- snapshot：生成动作时刻的物理状态（`save_full_state`）。
- 语义：忠实复现「策略基于带传感相位的观测决策，决策作用于当前物理状态」——
  这正是 π0.5 真实部署的行为。
- 允许 obs 与 snapshot 不同刻，**量化并记录传感延迟**（见 §4）。
- 复现方式：加载保存的 obs → 策略生成动作；加载 snapshot → 恢复物理 → 执行 → 认证。

### 身份 B：`synchronous`（同步输入诊断）—— 诊断用途，非主实验

- obs 来源：`_update_observables(force=True)` 后取到的 obs，与 snapshot 严格同刻。
- 语义：研究「消除缓存滞后」后策略输入对危险性的影响。
- 复现方式：从 snapshot 恢复后 force 刷新重建 obs（因为同刻 obs 是 snapshot 的确定性函数）。

> 主实验**只用 deployment**。synchronous 仅用于诊断「传感延迟是否影响了危险性」，
> 其结论不与 deployment 结果合并。

## 3. 每个状态必须记录的字段

| 字段 | 说明 |
|---|---|
| `obs_protocol` | `deployment` 或 `synchronous` |
| `obs_source` | obs 来自 `env.step` 返回值 / `force=True` 刷新 |
| `snapshot_sha256` | 物理快照（flatten 183 维）哈希 |
| `obs_sha256` | 策略实际消费的 obs（按固定字段序列化）哈希 |
| `z0_sha256` | 源噪声向量哈希 |
| `nominal_sha256` | 名义动作哈希 |
| `sensing_lag` | obs 与 snapshot 的量化差异（见 §4） |

## 4. 传感延迟量化

deployment 身份下，obs 与 snapshot 不同刻是真实行为，需要量化而非掩盖。量化方式：

- 取 obs 中的本体字段 `robot0_eef_pos`（sensor 读 `site_xpos[grip_site]`）；
- 从 snapshot 恢复后，直接读**同一个 site** 的最新位置（`site_xpos[eef_site_id]`，
  不经 observable 缓存）；
- 记录二者差。**必须读同一 site**，读 body `xpos` 会混入 site-vs-body 几何偏差
  （~0.16–0.47 mm），不是传感延迟。

实测（本次 rollout）：`env.step` 返回的 obs 与最新 site 差 ~0.4 mm（sub-mm 级），
且状态相关——快速接近阶段可达数 mm。force 刷新后与最新 site 完全一致（0.000 mm）。

> 注：早期诊断记录过 ~9.3 mm 的「滞后」，那是某次 rollout 发散到快速接近阶段
> 的采样相位滞后，叠加了诊断代码时序问题；不是通用量级。

## 5. 版本与兼容性

- 本协议 v1 冻结后，所有后续状态用 `deployment` 身份。
- 早期记录的身份分类：
  - state0（`stable_dangerous_state/`）：旧筛选器，未保存 obs，**身份未记录**，待复核。
  - state1（`stable_dangerous_state2/`）：当前 obs.npz 是 `synchronous`（force 刷新），
    **不是主实验身份**，待用 deployment 复核。
  - state3（`stable_dangerous_state3/`）：当前 obs.npz 是 `synchronous`，
    **不是主实验身份**，其「两组均失败」结论需按 deployment 复核。
- 复核重跑**单独编号**（如 `state0_deploy`），不与旧结果合并。
