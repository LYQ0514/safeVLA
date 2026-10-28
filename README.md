# safeVLA

安全约束下的 VLA（Vision-Language-Action）项目工作区。

- 归属：liuyiqian（容器内 unix 用户显示为 `dp3`）
- 路径：`/data/safeVLA`（持久，重启 / 重 spawn 不丢）
- 创建日期：2026-10-28

## 目录

| 路径 | 放什么 |
|---|---|
| `src/` | 项目代码 |
| `tools/` | 辅助脚本 / CLI |
| `examples/` | notebook、最小可运行样例 |
| `checkpoints/` | 权重（大文件，注意空间） |
| `docs/tutorial/` | 入门教程、逐步教学 |
| `docs/user-guide/` | 面向使用者的操作说明（按角色分册） |
| `docs/operations/RUNBOOK.md` | 部署 / 运维 / 踩坑 / 排障（注明日期） |
| `docs/reference/` | 术语、角色权限、状态机、模块清单 |
| `docs/explanation/` | 架构、设计取舍、决策记录（ADR） |
| `docs/testing/` | 需求 / 方案 / 用例 / 测试报告 |
| `.cache-local/` | 本项目相关缓存（HF / torch / pip 等），**不放用户 home** |

## 相关环境（已有，勿污染）

- `/opt/venv/openvla` — libero + robosuite 1.4.1 + mujoco 3.3.7（Python 3.11）
- `/opt/venv/openpi` — openpi + jax 0.5.3（GPU 可见）
- 共享参考项目：`/data/2608.SDL/MojocoProj`（π0.5 + LIBERO，**只读**）

本项目自有依赖请装进本目录下的独立虚拟环境（`uv venv` / `conda`），不要装进基础环境。

## 约定

- 跑东西前在本目录设置缓存指向项目内，避免落进 home：
  ```bash
  export HF_HOME=$PWD/.cache-local/huggingface
  export TORCH_HOME=$PWD/.cache-local/torch
  export XDG_CACHE_HOME=$PWD/.cache-local/cache
  ```
- 外网访问先 `proxyOn`，用完 `proxyOff`。
- 经验就近沉淀：项目规则写 `AGENTS.md`，排障写 `docs/operations/RUNBOOK.md`。
