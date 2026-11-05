# FPCK/CKG 实验：A/B/C 阶段审计记录

> 日期：2026-11-05
> 实验包：`/data/fpck_ckg_release`（解压自 `fpck_ckg_complete_release.zip`）
> 目标仓库：`/data/safeVLA`

## A. 环境与身份审计

| 项 | 值 |
|---|---|
| git commit | `148b54d1691992c7768582ccdc2c3c277fb3e016` |
| 当前分支 | `aegis-stg-v1` |
| π0.5 checkpoint | `/data/2608.SDL/MojocoProj/checkpoints/pi05/pi05_libero` |
| π0.5 hash（历史记录） | `b838d8b85a3f317c91950738172ee7f174afe7f2d36d9d8ed80f7110a085e90b` |
| SafeLIBERO | `/data/safeVLA/dataset/SafeLIBERO/safelibero` |
| MuJoCo | 2.3.7 |
| robosuite | 1.4.1 |
| openpi worker | `/opt/venv/openpi/bin/python`（存在） |
| EGL | `MUJOCO_GL=egl` |
| GPU | RTX 4090, 49140 MiB |
| Python（safeVLA .venv） | 3.8.18 |

**工作树状态**：`src/pnsg/openpi_worker.py` + `policy_adapter.py` 有未提交改动（历史探索，
非本轮 FPCK 相关）。核心评测文件无未提交改动。

**基线冒烟**：`tools/run_eval_aegis.py --help` 正常，评测链路未变。

## 安装

执行 `python /data/fpck_ckg_release/code/tools/install_into_safevla.py /data/safeVLA`：
- 新增 `src/fpck/`（8 个文件）
- 新增 `tools/run_exact_e0.py`、`train_value.py`、`aggregate_results.py`、`run_safelibero_ckg.py`
- **未覆盖任何现有文件**（安装前确认 src/fpck 和 4 个 tools 均不存在）

## 发现并修复的兼容性问题（如实报告）

### 问题 1：Python 3.8 不支持 `X | Y` union 注解

- `governor.py:14` 的 `action: np.ndarray | None` 在 Python 3.8 **运行时求值**报
  `TypeError: unsupported operand type(s) for |`（`ast.parse` 能过，但 import 时求值失败）；
- 根因：实验包按 Python 3.12/3.13 编写（zip 内含 cpython-312/313 的 pyc），safeVLA 是 3.8；
- 修复：`governor.py` 顶部加 `from __future__ import annotations`（延迟注解求值，
  不改算法逻辑）；
- 修复位置：`/data/safeVLA/src/fpck/governor.py` + `/data/fpck_ckg_release/code/fpck/governor.py`。

### 问题 2：`from fpck` import 路径与 install 布局不一致

- `run_exact_e0.py`、`train_value.py` 用 `from fpck import ...`（假设 fpck 在仓库根）；
- install 脚本把 fpck 放到 `src/fpck`，导致 `from fpck` 找不到（`run_safelibero_ckg.py`
  用的是 `from src.fpck`，是正确的）；
- 根因：实验包原始 `code/` 布局里 fpck 与 tools 同级（`code/fpck` + `code/tools`），
  `ROOT=parents[1]` 指向 code/ 能找到；install 到 safeVLA 后布局变了；
- 修复：`run_exact_e0.py`、`train_value.py` 加 `sys.path.insert(0, str(ROOT/"src"))`。

## B. E0 精确实验

| 配置 | physical_viability | fpck | strict_difference |
|---|---|---|---|
| horizon=3, budget=0 | ["A","S"] | ["A"] | **["S"]** |
| horizon=3, budget=1 | ["A","S"] | ["A","S"] | [] |

**验证通过**：
1. budget=0 时 physical viability 严格包含 FPCK（S 在 physical 但不在 fpck）✅；
2. budget=1 时恢复动作 `left` 使 S 进入 FPCK（witness action="left"）✅。

输出：`data/audit/fpck_e0_b0.json`、`data/audit/fpck_e0_b1.json`。

## C. 单元测试

```
Ran 4 tests in 0.000s — OK
  test_budget_recovers ... ok
  test_calibration ... ok
  test_governor ... ok
  test_strict_separation ... ok
```

命令：`cd /data/fpck_ckg_release/code && PYTHONPATH=... .venv/bin/python -m unittest discover -s tests -v`
（用 safeVLA Python 3.8 venv + 修复后的 governor.py）

## 结论（A/B/C PASS）

环境身份已记录，E0 精确实验 + 单元测试均通过。存在两处**兼容性修复**（Python 3.8
语法 + import 路径），均不改变算法逻辑、不覆盖现有结果，已如实记录。
