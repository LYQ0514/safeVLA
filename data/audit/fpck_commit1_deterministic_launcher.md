# FPCK/CKG Commit 1 — 确定性实验启动器

> 日期：2026-11-06。裁决 §一/§二/§十一 执行完成。

## 一、实现内容

### 1. 显式确定性 factory（`src/pnsg/deterministic_factory.py`）

- **不静默修改 PnsgPolicy 默认**：`PnsgPolicy.__init__` 的 `xla_flags` 参数默认仍为
  `None`（非确定配置），测试 `TestPnsgPolicyDefaultUnchanged` 锁定这一语义。
- `create_deterministic_policy()`：显式工厂，固定
  `XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_gpu_exclude_nondeterministic_ops=true"`。
- **硬门禁**：worker 启动后读 `xla_flags_at_import`（worker 在 `import jax` 前记录），
  缺确定性 flag → 抛 `DETERMINISM_CONFIGURATION_MISSING`（`DeterminismGateError`），
  不 warning 后继续。

### 2. XLA_FLAGS 生效机制

XLA_FLAGS 通过 `PnsgPolicy` 的 subprocess `env` 传入 worker 进程；worker 是独立进程，
在 `import jax` 之前读 `os.environ["XLA_FLAGS"]` 并记录为 `_XLA_FLAGS_AT_IMPORT`。
这满足裁决「XLA flags 必须在首次导入 JAX/backend 初始化之前设置」。

### 3. 门禁记录（写入 run manifest）

`determinism_gate_record(gate)` 输出：gate_version、passed、xla_flags、
xla_flags_at_import、jax/jaxlib/xla_client 版本、default_backend、devices。

## 二、验证

### 单元测试（7 纯逻辑 + 1 集成）

- 硬门禁：`None` / 空串 / 缺 flag → `DETERMINISM_CONFIGURATION_MISSING`；
- 完整配置 + 额外 flag → 通过；
- PnsgPolicy 默认语义不变；
- 集成：真实 worker + 硬门禁（13s，`default_backend=gpu` 确认）。

### 推理延迟对照（裁决 §十一）

| 配置 | p50 | p95 | p99 | mean |
|---|---|---|---|---|
| 组 A（默认 autotuned） | 101.08ms | 101.62ms | 102.01ms | 97.53ms |
| 组 C（确定性） | 141.46ms | 151.56ms | 151.60ms | 145.35ms |
| 代价 | 1.40x | 1.49x | 1.49x | 1.49x |

确定性配置推理延迟约 1.4–1.5x，可接受但**必须在 H0 重新测量并报告 p50/p95/p99**
（裁决 §十一：确定性配置可能降低或改变推理速度）。

## 三、与前序确定性矩阵的关系

`fpck_determinism_matrix` 已验证：组 B（autotune_level=0）10 冷启动进程 raw/deployed
chunk 唯一；组 C（+exclude_nondeterministic_ops）结果与组 B 完全相同。本启动器固定
使用组 C 完整配置，且增加了运行时硬门禁（防止未来 graph/分支引入非确定 kernel 而
未被察觉）。

## 四、历史结果分层（裁决 §三）

历史默认配置（`LEGACY_XLA_AUTOTUNED`）与确定性配置（`FPCK_DETERMINISTIC_XLA_V1`）
的数值轨迹不同（autotuned kernel ≠ autotune-level-0 kernel），不得逐数值合并。
历史结果仅用于发现机制/审计，不用于新主表、配对比较、v1.3 成功率或 value 训练。

## 五、Commit 1 完成状态

- ✅ 确定性启动器（JAX import 前设 flags + 硬门禁）；
- ✅ 运行时 manifest 字段（determinism gate record）；
- ✅ 10 进程 determinism 测试（前序 matrix）；
- ✅ 推理延迟对照；
- ✅ 不改 PnsgPolicy 默认语义。

## 六、后续 Commit（裁决 §十）

- Commit 2：DUAL_ENDPOINT_HARNESS_V1（native 不 break + integrity endpoint + 超时路由）；
- Commit 3：terminal contract manifest（17 任务逐项声明终态类型）；
- Commit 4：物理阈值表征（open 响应 / contact gone / 稳定速度 → H_post + N_stable）；
- Commit 5：factory freeze（完整 hash）；
- 之后才开始 state3 重跑。
