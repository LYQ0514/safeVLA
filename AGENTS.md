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

## 文档
遵循团队文档体系，见 `README.md` 的目录表。踩坑按「现象 → 根因 → 修复 → 验证」写进 `docs/operations/RUNBOOK.md` 并注明日期。

## 提交
只在你被明确要求时 `git commit` / `push`。密钥不入库、不入日志。
