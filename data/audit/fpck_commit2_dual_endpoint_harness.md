# FPCK/CKG Commit 2 — DUAL_ENDPOINT_HARNESS_V1 结构

> 日期：2026-11-06。裁决 §四~§九 执行完成。
> 只实现结构，不冻结科学数值；所有科学参数必填、无默认值、生产硬门禁。

## 一、实现内容

### 1. `src/fpck/dual_endpoint_harness.py`（纯逻辑状态机）

- 状态机：`RUNNING_PRE_NATIVE → RUNNING_POST_NATIVE → TERMINATED_*`
  （`TERMINATED_INTEGRITY` / `TERMINATED_TIMEOUT` / `TERMINATED_RUNTIME_FAILURE`）；
- `NativeEndpointRecord`（frozen dataclass）：首次 native success 的不可变快照，
  含 goal truth / mode / lifecycle / 线速度 / 角速度 / contact / chunk hash /
  query index / buffer index / full_state hash（裁决 §四，防信息丢失）；
- `DualEndpointHarness.observe()`：首次 native_success 只记录、不覆盖、不终止；
- `terminate_integrity()`：固定优先级（identity failure > q_bad > q_unknown > q_acc）；
- `terminate_timeout()`：四分类路由（裁决 §八）；
- `terminate_runtime_failure()`。

### 2. `TerminalValidationConfig`（frozen dataclass，所有科学参数必填）

无默认值的参数：task_horizon、post_native_seconds、min_full_policy_chunks、
release_contact_gone_substeps、stable_control_steps、linear_velocity_max_mps、
angular_velocity_max_radps、control_hz、chunk_len、provenance、
characterization_sha256、verified。

- `require_verified()`：生产入口硬门禁，`verified=False` →
  `UNVERIFIED_TERMINAL_VALIDATION_CONFIG`；
- `__post_init__`：characterization_sha256 空 → `MISSING_TERMINAL_VALIDATION_CONFIG`。

### 3. horizon 计算（裁决 §五/§六）

- `hard_cap() = H_task + H_post`；
- `post_steps()`：物理窗口（秒→控制步）；
- `post_chunk_steps(r) = r + K*L`（完成当前 chunk 剩余 + K 个完整 chunk）；
- `post_window(r) = max(H_physical, H_chunk)`；
- `integrity_deadline(t_n, r) = min(H_hard, max(H_task, t_n + post_window))`。

## 二、测试（25 个，裁决 §九 全覆盖）

| 组 | 覆盖 |
|---|---|
| 配置安全 | 缺配置拒绝 / 未验证拒绝 / min_chunks≥1 / 科学参数无默认值 |
| Native endpoint | 不 break / 首次只写一次 / 记录速度 / 不终止 |
| Integrity endpoint | q_acc→SAFE_REENTRY / q_bad 安全→PREHANDOFF / q_bad 完整性→TASK_INTEGRITY / q_unknown→UNCERTAIN / runtime failure / identity 优先 |
| Timeout | native 无 witness / progressing / stall / not completed |
| Horizon | hard cap / 早 native 不缩短 / 晚 native 全窗口 / 不超 cap / chunk 剩余进 deadline |

## 三、明确不做的事（裁决 §二）

- **没有**为 post_native_seconds / stable_control_steps / linear_velocity_max 等
  提供任何默认值（2.0s / 5 步 / 0.05 m/s 等仅存在于测试 helper 的 synthetic 配置，
  且 `verified=False`）；
- **没有**在 `DualEndpointHarness` 提供 `allow_unverified_for_test` 绕过选项
  （裁决 §三：生产类本身不提供绕过；测试用 `unverified_config()` + 直接构造）。

## 四、测试配置与生产配置的隔离

- 测试用 `unverified_config()`（provenance=TEST_ONLY_SYNTHETIC, verified=False），
  断言 `DualEndpointHarness(unverified_config())` 抛
  `UNVERIFIED_TERMINAL_VALIDATION_CONFIG`；
- 生产配置只能来自 Commit 4 物理表征 artifact（verified=True +
  characterization_sha256 非空）。

## 五、下一步：Commit 4 物理表征

Commit 4 输出不可变 artifact（`terminal_validation_characterization_v1`），字段：
open_response（p50/p95/max 步）、contact_gone（p95/max substep）、rest_velocity_noise
（linear/angular p99）、placement_settling（p95/max 步）、selected_thresholds、
raw_data_sha256、artifact_sha256。这些数值回填 `TerminalValidationConfig` 并标
verified=True，引用 characterization_artifact_sha256 证明阈值非 state3/state4 反拟合。

## 六、Commit 2 完成状态

- ✅ 双 endpoint 状态机；
- ✅ native endpoint 不可变快照；
- ✅ integrity 终止路由；
- ✅ timeout 调度结构（horizon 计算接口）；
- ✅ 显式配置与验证门禁（生产硬门禁）；
- ✅ 25 测试（裁决 §九 全覆盖）；
- ⏸ 科学数值未冻结（Commit 4 回填）。
