# FPCK/CKG 确定性验证矩阵 — VLA 推理非确定性根因最终归因

> 日期：2026-11-06。裁决 §一~§九（确定性门禁）执行完成。
> 结论：根因 = **XLA GPU autotuning 编译期 kernel 选择非确定性**，非运行期算子、
> 非图像渲染、非输入张量、非 MuJoCo。`--xla_gpu_autotune_level=0` 根治。

## 一、先决条件证明（裁决 §二：输入张量逐字节一致）

在 worker 中新增规范化 hash 记录（`openpi_worker.py`）：
- 输入张量逐键 hash（image/proprio/lang token/attention mask/state）；
- noise_z_sha256；
- raw_chunk_sha256（模型原始输出，H×32 归一化空间）；
- deployed_chunk_sha256（后处理 + adapter 后，H×7 external）。

**结果**：固定 obs + 固定 z 下，9 个输入张量键跨进程 hash 全一致，noise_z 唯一。
这排除了裁决 §二 列举的所有输入侧差异（图像渲染、预处理、proprio、token、mask、字典序、noise）。

补充：独立实验（6 次 make_env）证明 EGL 图像渲染跨进程 hash 一致（`fpck_image_determinism.py`）。

## 二、根因定位链

| 实验 | 结果 | 结论 |
|---|---|---|
| settled baseline 确定性 | 4 次 make_env 差 0.000000mm | 排除 env 初始化 |
| 图像渲染确定性 | 6 次 image hash 一致 | 排除渲染 |
| 单次推理（q=0）确定性 | 6 进程 raw/deployed 唯一 | 单次 GPU 推理确定 |
| **8-query 连续查询** | **出现 2 个稳定簇**（proc0/1 vs proc2/3） | **编译期 autotune kernel 选择分簇** |
| 组 B（autotune_level=0） | 6 进程单次 + 8-query 全唯一 | autotune 是根因 |
| 组 C（+exclude_nondeterministic_ops） | 与组 B 相同 | 无运行期非确定算子 |

**关键证据**（`fpck_multiquery_determinism.py`，默认配置 4 进程 8-query）：

```
proc0/proc1: q0=0d639eed q1=131faf53 ...（簇 1）
proc2/proc3: q0=4d3a1b30 q1=1f8a3308 ...（簇 2）
```

每次冷启动，XLA autotune 基于当时 GPU 负载/状态（多用户共享 GPU）选择不同 kernel，
产生 bit 级不同的输出。同一进程内重复推理稳定（autotune 结果已缓存），跨进程分簇。

## 三、解决方案：`--xla_gpu_autotune_level=0`

`--xla_gpu_exclude_nondeterministic_ops` 与 `--xla_gpu_autotune_level=0` 在 jaxlib 0.5.3
均被接受（无 unknown flag error）。实验证明：
- 组 B（仅 `autotune_level=0`）已足够消除分簇；
- 组 C（`exclude_nondeterministic_ops + autotune_level=0`）结果与组 B **完全相同**
  （raw=87c3390bf83a6373，deployed=d5f741abe5394a38），说明此模型无运行期非确定算子，
  autotune 是唯一非确定来源。

## 四、门禁 PASS（裁决 §五）

| 门禁项 | 结果 |
|---|---|
| 输入张量 hash 跨进程一致 | ✅ 9 键全一致 |
| noise_z 唯一 | ✅ |
| raw_chunk 唯一（10 冷启动进程） | ✅ |
| deployed_chunk 唯一（10 冷启动进程） | ✅ |
| 8-query 跨进程一致 | ✅ |
| **轨迹级：3 冷启动进程 step36/37/38 精确一致** | ✅ 5.0561 / 10.5521 / 13.2947 |
| chunk hash 跨进程一致 | ✅ |
| 无 NaN/Inf | ✅ |
| XLA flags 生效（xla_flags_at_import 记录） | ✅ |

## 五、重要方法论结论：确定性改变了数值基线

组 B（autotune_level=0）选择了**不同但确定**的 kernel，因此轨迹数值与组 A（默认配置）
**不同**：

| 配置 | step37 disp | 跨进程 |
|---|---|---|
| 组 A（默认） | 10.1941 / 8.7162 / 9.7850（3 值） | 非确定 |
| 组 B（autotune_level=0） | **10.5521**（唯一） | 确定 |

这意味着：
1. **历史结果（组 A 默认配置）不能与组 B 结果直接数值对比**；
2. 审计 B/C/D 结论不受影响（各自在「同一轨迹内」的结论成立，不依赖跨进程复现）；
3. 未来所有评测必须统一用组 B 配置，重跑历史 anchor 以建立新基线。

## 六、对既有结论的影响（裁决 §十）

**保留**：
- 审计 B 原始 contact-pair（left finger + support，无 robot link）；
- state1 旧 5mm 规则假阴性；
- 审计 C 在线/离线一致、审计 D 运行身份一致；
- v1.3 抓取建立修复、state1 非完整 SAFE_REENTRY。

**撤回/修改**：
- 「固定 seed 跨进程完全确定」→ 仅在 autotune_level=0 下成立；
- 「run_idx 不影响随机性、重复完全等价」→ 仅在确定性 flags 下成立；
- 「轨迹差异仅来自 wall time」→ 实为 autotune kernel 选择。

## 七、落地

1. `PnsgPolicy` 增加 `xla_flags` 参数（已实现）；
2. 建议将 `--xla_gpu_autotune_level=0` 设为默认（下一步冻结 factory hash 前）；
3. 重跑所有受影响 anchor 时统一用组 B 配置。

## 八、Gate 状态

| Gate | 状态 |
|---|---|
| 位移复现性 | **已根治：autotune_level=0，门禁 PASS** |
| 审计 B/C/D | PASS（不变） |
| 冻结前剩余 | 双终止 harness 规格、terminal lifecycle manifest、H_post 数值、factory hash |
