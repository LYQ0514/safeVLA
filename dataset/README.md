# SafeLIBERO 评测集

下载日期：2026-10-28
来源：HuggingFace `THURCSCT/SafeLIBERO`（论文 AEGIS / VLSA，arXiv 2512.11891）
论文项目页：https://vlsa-aegis.github.io/
代码仓库：https://github.com/THU-RCSCT/vlsa-aegis

## 目录

```
dataset/SafeLIBERO/
├── main_demo.py                 官方评测入口
├── README.md                    官方说明
└── safelibero/
    ├── libero/libero/
    │   ├── assets/              场景 + 物体 + 障碍物模型（含 SafeLIBERO 新增 obstacle_objects）
    │   ├── bddl_files/          17 个任务定义（safelibero_{spatial,goal,object,long}）
    │   └── init_files/          32 个 .pruned_init（每场景 50 条 episode 初始状态）
    ├── benchmark_scripts/       下载 / 渲染 / 校验脚本
    └── notebooks/               官方 walkthrough
```

体量：618 MB，985 个文件。

## 评测集构成

- 4 套件（spatial / goal / object / long）× 4 任务 × 2 安全等级（I / II）= **32 个场景**
- 每场景 **50 条 episode**（位置随机化）→ 共 **1600 条评测 episode**
- 安全等级：
  - **Level I**：障碍物紧贴目标物体
  - **Level II**：障碍物较远，但挡在运动路径上
- 评测指标：**CAR**（无碰撞率）、**TSR**（任务成功率）、**ETS**（平均执行步数）

## 验证结果（2026-10-28）

- `.pruned_init` 32 个全部可 `torch.load`，形状统一为 `(50, D)`、`float64`
  - 例：`open_the_top_drawer_and_put_the_bowl_inside_level_I.pruned_init` → `(50, 170)`
  - D 随场景物体数不同（机器人与障碍物 qpos+qvel 拼接）
- 注意：**不是** README 里描述的纯文本，实际是 torch pickle 文件；用 `wc -l` 行数无意义
- `.bddl` 17 个（16 个任务 + 1 个 goal 套件的第 4 任务变体）

## 运行方式

```bash
cd dataset/SafeLIBERO
export PYTHONPATH=$PYTHONPATH:$PWD/safelibero
python main_demo.py \
    --task-suite-name safelibero_spatial \
    --safety-level I \
    --task-index 0 \
    --episode-index 0 1 2 3 4 5 \
    --video-out-path ../../data/libero/videos
```

依赖：官方要求 conda python 3.8.13 + `safelibero/requirements.txt`。

## 注意

- 官方仓库 README 的安装方式需要 `git clone` 代码（本目录已含全套代码，无需再 clone）。
- `safelibero/libero` 是**改版 LIBERO**（加了障碍物资产与 bddl），与 `robosuite` 版本强耦合；
  本机 `/opt/venv/openvla` 里的 libero 是原版，**不能直接混用**，需按 requirements 装独立环境。
- 大文件（贴图）占了大部分体积；`dataset/*` 已在 `.gitignore` 中忽略（只跟踪 `dataset/README.md`），不入 git。
