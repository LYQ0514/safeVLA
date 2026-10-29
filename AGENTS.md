# safeVLA — 项目规则（AI 工具入口）

## 项目定位
安全约束下的 VLA 研究 / 工程。项目目录 `/data/safeVLA`，所有产出留在本项目内。

## 工作区约定
- 在本目录（`/data/safeVLA`）启动 notebook / AI 工具，**不要**在 `/data/2608.SDL` 里启动（那是只读共享区，缓存会污染）。
- 缓存 / 数据一律落 `./.cache-local/<工具>/`，不写 `~/.cache`、`~/.huggingface`、`~/.torch`。
- 依赖隔离：自建 venv（`uv venv .venv` 或 conda），**不装进** `/opt/miniconda3` base 或 `/opt/venv/*`。

## 参考资源
- `/data/2608.SDL/MojocoProj`：π0.5 + LIBERO 共享代码与权重（只读，可读可用，不可改）
- 其排障文档：`/data/test/doc/operations/RUNBOOK.md`

## 评测口径（2026-10-29 定版，勿擅自更改）

改动口径会让历史结果失去可比性，要改先讨论并记录原因。

| 指标 | 定义 |
|---|---|
| TSR | 任务成功数 / 总数 |
| **CAR**（主） | 障碍物位移 ≤ **1mm**，官方口径，便于对标论文 |
| **CAR>5mm**（辅） | 位移 ≤ 5mm，即"无严重碰撞率"。**报告时必须与 CAR 同时给出** |
| ETS | 含超时的全部 episode 平均（AEGIS 口径）；另存仅成功 episode 口径（VLPSA） |
| SSR | 成功且无碰撞 / 总数 |

- **max_steps**：short 套件 300、long 套件 550。
- **必须两个碰撞指标一起报**：实测存在 `object I`（CAR 6.5% 但最大碰撞 4.1mm）
  与 `spatial II`（CAR 13.0% 但 31.5% 超 5mm）这种**结论相反**的情况，
  单看 CAR 会误导。详见 `docs/testing/2026-10-29-pi05-full-baseline-calibration.md`。
- 当前 π0.5 基线：TSR 59.9% / CAR 13.1% / CAR>5mm 88.1% / ETS 258.6（1600 条）。

## 文档
遵循团队文档体系，见 `README.md` 的目录表。踩坑按「现象 → 根因 → 修复 → 验证」写进 `docs/operations/RUNBOOK.md` 并注明日期。

## 提交
只在你被明确要求时 `git commit` / `push`。密钥不入库、不入日志。
