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
| `docs/testing/` | 评测报告（需求 / 方案 / 用例 / 结果） |
| `src/policies.py` | π0.5 策略适配层（`Pi05Policy.act()`） |
| `src/eval_safelibero.py` | rollout 循环 + CAR / TSR / ETS 指标 |
| `tools/run_eval.py` | 评测 CLI |
| `tools/activate_env.sh` | 环境入口（`source` 后即可用 python / main_demo.py） |
| `requirements-lock.txt` | `.venv` 的完整包清单（`pip freeze`） |
| `docs/reference/` | 术语、角色权限、状态机、模块清单 |
| `docs/explanation/` | 架构、设计取舍、决策记录（ADR） |
| `docs/testing/` | 需求 / 方案 / 用例 / 测试报告 |
| `.cache-local/` | 本项目相关缓存（HF / torch / pip 等），**不放用户 home** |

## 评测环境（本项目自建，2026-10-28）

**`./.venv`**（Python 3.8.18）— SafeLIBERO 专用：`robosuite==1.4.1` + `mujoco==2.3.7` +
`bddl==1.0.1` + `torch==2.4.1+cu121`，EGL 离屏渲染。完整清单见 `requirements-lock.txt`。

```bash
cd /data/safeVLA && source tools/activate_env.sh   # 每个新 shell 都要
python tools/check_all_scenes.py                   # 32 个场景健康检查
```

搭建过程与 4 个踩坑详见 `docs/operations/RUNBOOK.md`（含「重新下载数据集后要重跑
`tools/fix_safelibero_asset_paths.py`」这条必读注意事项）。

## 相关环境（团队共享，已有，勿污染）

| 路径 | 内容 |
|---|---|
| `/opt/venv/openvla` | **原版** libero + robosuite 1.4.1 + mujoco 3.3.7（Python 3.11） |
| `/opt/venv/openpi` | openpi + jax 0.5.3（GPU 可见） |
| `/data/2608.SDL/MojocoProj` | π0.5 + LIBERO 共享参考项目（**只读**） |

本项目自有依赖装在 `./.venv`，已与上面隔离。不要装进 `/opt/miniconda3` 或 `/opt/venv/*`。

## 约定

- 跑东西前在本目录设置缓存指向项目内，避免落进 home：
  ```bash
  export HF_HOME=$PWD/.cache-local/huggingface
  export TORCH_HOME=$PWD/.cache-local/torch
  export XDG_CACHE_HOME=$PWD/.cache-local/cache
  ```
- 外网访问先 `proxyOn`，用完 `proxyOff`。
- 经验就近沉淀：项目规则写 `AGENTS.md`，排障写 `docs/operations/RUNBOOK.md`。
