# FPCK/CKG v1.1 解阻塞审计 + E1a 依赖评估

> 日期：2026-11-05。v1.1 包 SHA-256 校验通过（A683839ADD...）。
> 状态：manual 缺失 BLOCK 已解除，但 E1a 暴露新的依赖缺口（任务自动机未实现）。

## 1. v1.1 交付核对

| 项 | 状态 |
|---|---|
| `fpck_ckg_complete_release_v1.1.zip` SHA-256 | ✅ 匹配（A683839ADD619EE6AFDC5B253176B615A3154D52CA8FBAD98E61E5FAD269A4EC） |
| `fpck_ckg_experiment_manual.md` | ✅ 已在包内 + install 复制到 `docs/method/` |
| 四类标签判据 + 优先级 | ✅ 手册 §4 |
| 交还时刻 τ + 反事实规则 | ✅ 手册 §2 |
| forbidden collision / task-integrity / valid progress | ✅ 手册 §3 |
| JSONL 完整契约 | ✅ 手册 §5 |
| anchor 级切分 + 防泄漏 | ✅ 手册 §6 E1c |
| budget 下溢修复 | ✅ policy.py `max(0, budget-intervention)` + 新增字段 |

## 2. v1.1 安装与验证

- install 脚本新增：复制 manual 到 `docs/method/fpck_ckg_experiment_manual.md`；
- 重跑回归测试：**5 tests OK**（新增 `test_budget_never_underflows`）；
- 重跑 smoke：budget_remaining 从 v1.0 的 -263 → v1.1 的 **0**（零下界），
  负 budget 行数 = 0，新增 budget_before/intervention/outside_guarantee 字段正确。

## 3. 兼容性修复（v1.1 仍需要，已重新应用）

| 修复 | 原因 |
|---|---|
| `governor.py` 加 `from __future__ import annotations` | v1.1 仍用 `np.ndarray \| None`（py3.10+），safeVLA 是 py3.8 |
| `run_exact_e0.py`/`train_value.py` 加 `sys.path.insert(src)` | `from fpck` 路径与 install 布局（src/fpck）不一致 |

v1.1 的 install 覆盖了 v1.0 时的这 3 处修复，需重新应用（已做）。

## 4. E1a 依赖评估（新暴露的缺口）

### 已具备

| 依赖 | 状态 |
|---|---|
| 完整 MuJoCo 状态恢复 | ✅ `src/pnsg/full_state.py`（save/restore，验证差 0.0） |
| CRN 随机流 | ✅ `src/pnsg/crn.py`（nominal_seed / sample_z） |
| 角色清单 | ✅ `data/task_object_role_manifest.json` |
| 接触契约 | ✅ `data/collision_event_contract_v2.json` |

### 缺失 / 需澄清

| 依赖 | 状态 | 影响 |
|---|---|---|
| **任务自动机**（q_bad/q_acc/阶段状态机） | ❌ 仓库无实现 | 四类标签判据的核心依赖（§3.2/§4.1/§4.3 都要求「自动机转移/接受态」） |
| **S01/S04 witness 工具** | ⚠️ 无明确命名工具 | 手册 §6 说「仓库已有」，实际散落在 m1_t0_task_integrity.py / s05_residual_diag.py |

### 关于「任务自动机」的现状

- `contract_objects.py` 有 **TaskIntegrityMonitor**（PLANNING/PREGRASP/GRASP/TRANSPORT/PLACE
  阶段 + 位移阈值），是简化版阶段模型；
- 但它**不是**手册要求的完整任务自动机：没有 q_bad/q_acc 状态集、没有阶段顺序破坏的
  转移判定、没有「接受态」定义、没有 `automaton_before/after` 字段的语义；
- `m1_t0_task_integrity.py` 用「规划/预抓取阶段 soup/cream 位移 >5mm」判 TASK_INTEGRITY_FAIL，
  这是**单个任务的特定判据**，不是通用的任务自动机。

## 5. 诚实结论

1. **manual 缺失 BLOCK 已解除**：四类标签判据、契约、切分、门槛均已明确；
2. **budget 下溢已修复**：5 tests OK + smoke 验证 budget_remaining=0；
3. **但 E1a 有新的依赖缺口**：手册要求「任务自动机」判定 q_bad/q_acc/阶段转移，仓库
   只有简化版 `TaskIntegrityMonitor`，无完整自动机。这需要先实现（或确认自动机的
   确切规格），否则四类标签（尤其 TASK_INTEGRITY_FAIL / SAFE_REENTRY）无法严格判定。

## 6. 下一步建议（需用户决策）

E1a 生死验证需要先解决「任务自动机」：

**方案 A**：手册方补充「任务自动机」的规格（状态集、转移条件、q_bad/q_acc 定义、
阶段顺序规则），实验侧据此实现；

**方案 B**：实验侧基于 `TaskIntegrityMonitor` + 契约文件，实现一个**预注册的**简化
任务自动机（明确记录「这是简化版，非手册原始规格」），并在 E1 报告中如实标注其
与手册的差异；

**方案 C**：E1a 先用「不依赖自动机的三类」（UNSAFE_RETURN / SAFE_STALL / UNCERTAIN，
它们主要靠 forbidden collision + valid progress + stall 判据），TASK_INTEGRITY_FAIL /
SAFE_REENTRY 依赖自动机的部分待规格补齐后再做。

三种方案都不违反「不伪造」红线，但需要明确选择。默认建议 **方案 A**（向手册方要
自动机规格，最严谨），因为自动机的状态集和转移是判据正确性的根本，不宜自行发明。
