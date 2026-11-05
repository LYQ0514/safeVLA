# FPCK/CKG 实验：E 阶段 BLOCK 报告

> 日期：2026-11-05。判定：**BLOCK（前置文件缺失，无法严格执行）**

## BLOCK 原因

E1 反事实数据采集要求「生成符合 `fpck_ckg_experiment_manual.md` 数据契约的 JSONL」，
但该 manual **未随实验包提供**，导致以下硬阻塞：

### 1. `fpck_ckg_experiment_manual.md` 不存在

- 实验包 `/data/fpck_ckg_release`（解压自 `fpck_ckg_complete_release.zip`）共 42 个文件，
  **无 `fpck_ckg_experiment_manual.md`**；
- 全盘 `find /data -name "*manual*"` 也无此文件；
- 数据契约的**完整字段要求**、四类状态的**可操作判据**、train/cal/test 的**切分规则细节**
  均无权威定义。

### 2. 四类状态判据无定义

任务书 E 要求复现 `SAFE_REENTRY`、`UNSAFE_RETURN`、`SAFE_STALL`、`TASK_INTEGRITY_FAIL`
四类状态，但：

- 实验包 paper（main.tex）只给抽象描述：「safe continuation, safe stall, unsafe
  return, recoverable danger」+ 有限状态例（nominal "right" 进入 failure、"hold" 安全
  停滞、"left" 需要预算）；
- README 只说「S01/S04-style witnesses」+「recoverable danger」；
- **没有给出**：在 SafeLIBERO + 冻结 π0.5 上，这四类状态各对应什么**可操作的现象判据**
  （如 SAFE_STALL 是「末端停滞 + 无碰撞 + 预算未耗尽」？UNSAFE_RETURN 是「返回路径
  碰撞」？），以及每类的 `target` 值如何定义。

### 3. demo.jsonl 不能当真实数据

- `code/demo.jsonl` 是 `make_demo_dataset.py` 生成的**合成数据**（`synthetic: true`）；
- 缺 schema 要求的 6 个字段：`sample_id`、`task`、`episode`、`chunk_hash`、
  `buffer_index`、`label_source`；
- 实验包 README 明确：「The demo dataset is synthetic and may only be used to
  validate the software path」——不能当 E1 反事实数据。

### 4. 仓库 full_state/CRN 工具存在，但缺「采集什么状态」的定义

- 仓库有 `src/pnsg/full_state.py`、`certifier.py`、`pnsg_policy.py` + 多个 replay 工具；
- 但「复现四类状态」需要先有每类状态的**判据**（哪些 (task, init, stream) 组合、
  哪个时刻、什么观测签名算 SAFE_STALL 等），这依赖缺失的 manual。

## 为什么不能自行补全

任务书 §8 明确：
- 「不得将缺失结果填为 0，不得伪造 CKG 性能」；
- 「不通过时停止扩大实验，不得包装成正结果」；
- 「生成符合 fpck_ckg_experiment_manual.md 数据契约的 JSONL」。

在 manual 缺失下自行发明四类状态判据 = 伪造数据契约 = 违反上述红线。**正确做法是如实
报告 BLOCK，等待提供 manual 后再继续。**

## 已完成的阶段（A/B/C/D，均 PASS）

| 阶段 | 状态 | 产出 |
|---|---|---|
| A 环境审计 | ✅ | `data/audit/fpck_ab_audit.md` |
| B E0 精确实验 | ✅ | `data/audit/fpck_e0_b0.json` / `b1.json` |
| C 单元测试 | ✅ | 4 tests OK |
| D 接口冒烟 | ✅ | `data/eval/fpck_smoke/`（proxy_smoke_only） |

## 发现并修复/记录的问题

| # | 问题 | 处理 |
|---|---|---|
| 1 | Python 3.8 不支持 `X\|Y` union 注解 | 修复（`from __future__ import annotations`） |
| 2 | `from fpck` import 路径与 install 布局不一致 | 修复（加 `sys.path.insert(src)`) |
| 3 | budget_remaining 下溢（-263） | 记录（H 前必须修复） |
| 4 | smoke 缺 FPCK 专属字段 | 记录（E1 需按契约补） |
| 5 | **experiment manual 缺失** | **BLOCK（本报告）** |

## 需要提供才能继续 E 阶段

1. `fpck_ckg_experiment_manual.md`（数据契约全文 + 四类状态判据 + 切分规则）；
2. 四类状态在 SafeLIBERO + π0.5 上的可操作判据（现象 → 字段 → target 定义）；
3. 确认 E1 数据采集的范围（哪些 suite/level/task/init/stream）。

## 结论

**E 阶段 BLOCK**，不扩大实验、不伪造数据。A/B/C/D 阶段的产物和全部证据（含 2 处
兼容性修复 + 1 处 budget bug + 1 处字段缺口）已如实记录并保留。待 manual 提供后，
可从 E 阶段继续。
