# safeVLA 运维 / 排障（RUNBOOK）

> 记录范式：**现象 → 根因 → 修复 → 验证（可复现）**。每条注明日期。

---

## 2026-10-28 评测环境搭建（SafeLIBERO + robosuite）

### 结论先看

```bash
cd /data/safeVLA
source tools/activate_env.sh      # 之后 python / main_demo.py 直接可用
cd dataset/SafeLIBERO
python main_demo.py --task-suite-name safelibero_spatial --safety-level I \
    --task-index 0 --episode-index 0 --video-out-path /data/safeVLA/data/libero/videos
```

### 环境规格

| 项 | 值 |
|---|---|
| 位置 | `/data/safeVLA/.venv`（Python **3.8.18**，来自 `/opt/miniconda3/bin/python3`） |
| 仿真 | `robosuite==1.4.1`、`mujoco==2.3.7`、`bddl==1.0.1`、`gym==0.25.2`、`numpy==1.22.4` |
| 深度学习 | `torch==2.4.1+cu121`（CUDA 12.1，本机驱动 580.178.04 可直接跑，`cuda.is_available()=True`） |
| 渲染 | **EGL 离屏**（容器无 X11）：`MUJOCO_GL=egl` + `PYOPENGL_PLATFORM=egl` |
| 完整清单 | `requirements-lock.txt`（`pip freeze` 87 个包） |
| libero 代码 | **不装成包**，走 `PYTHONPATH=dataset/SafeLIBERO/safelibero` |

**为什么 Python 3.8**：`bddl==1.0.1` 与官方 `requirements.txt` 按 3.8.13 发布，直接对齐最省事；
`/opt/venv/openvla`（3.11）里的 libero 是**原版**，不能与本改版混用（见 `dataset/README.md`）。

**为什么 `.venv` 里 torch 是 2.4.1 而不是 2.11**：CUDA 12.1 轮子在 Python 3.8 上最后可用版本即 2.4.1；
仿真本身只用 `torch.load` 读初始状态，将来接策略推理也够用。

### 关键配置：`LIBERO_CONFIG_PATH`（**必须设**）

`libero/libero/__init__.py` 在 `~/.safelibero/config.yaml` 不存在时会 **`input()` 交互提问**，
非交互环境（脚本 / 后台）直接 `EOFError` 崩掉，且默认会把配置写进用户 home。

修复：预生成配置到项目内，并用环境变量指过去（`tools/activate_env.sh` 已内置）：

```bash
export LIBERO_CONFIG_PATH=/data/safeVLA/.cache-local/libero   # 内含 config.yaml
```

`config.yaml` 内容为绝对路径（`bddl_files` / `init_states` / `assets` / `datasets`）。
**注意**：数据集若换位置，这个文件要同步改。启动时看到
`[Warning]: datasets path .../datasets does not exist!` 属正常（本评测不需要 datasets 目录）。

---

### 踩坑 1：MuJoCo 报 `resource not found` —— 资产里写死了作者机器路径

**现象**

```
ValueError: Error: resource not found via provider or OS filesystem:
'/mnt/workspace/openpi/third_party/safelibero/libero/libero/assets/scenes/basket_vis.msh'
Object name = basket_vis, id = 0, line = 23, column = -1
```

**根因**：SafeLIBERO 官方发布的 assets 里有 7 个 XML 的 `file=` 写的是**作者机器的绝对路径**
（`/mnt/workspace/openpi/...`、`/home/yifengz/workspace/robosuite-master/...`）。
robosuite 的 `MujocoXML.resolve_asset_dependency()` 会把 `file=` 相对路径按 XML 所在目录转成绝对路径，
但写死的绝对路径它原样保留 → 换机器必然找不到。**不是下载损坏，也不用重新下载数据集。**

**修复**：`tools/fix_safelibero_asset_paths.py` 把外来绝对路径换回「相对该 XML 所在目录」的相对路径（幂等）。

```bash
cd /data/safeVLA && source tools/activate_env.sh
python tools/fix_safelibero_asset_paths.py --dry-run   # 先看改什么
python tools/fix_safelibero_asset_paths.py             # 再改
```

涉及 7 个文件：`scenes/{basket,basket_asset,libero_tabletop_base_style}.xml`（2 个 .msh）、
`articulated_objects/basin_faucet/*` 与 `stable_scanned_objects/basin_faucet_*`（ceramic.png）。

**验证**：修完再跑 `fix_...py --dry-run` 应输出 `涉及文件数: 0`；`main_demo.py` 能建环境并出 mp4。

**⚠️ 重要**：`dataset/` 不入 git（`.gitignore` 里忽略），**重新下载数据集后必须重跑本脚本**。

---

### 踩坑 2：`ModuleNotFoundError: No module named 'future'`

**根因**：`bddl==1.0.1` 的依赖里漏了 `future`（`bddl/backend_abc.py` 用了 `future.utils.with_metaclass`）。

**修复**：手动补装（已在 lock 文件中）。

```bash
pip install "future==0.18.2"
```

---

### 踩坑 3：退出时 eglMakeCurrent 刷 `EGLError(EGL_NOT_INITIALIZED)`

**现象**：`main_demo.py` 跑完（mp4 已正常落盘），退出阶段打一屏

```
Exception ignored in: <function EGLGLContext.__del__ ...>
OpenGL.raw.EGL._errors.EGLError: EGLError(err = EGL_NOT_INITIALIZED, baseOperation = eglMakeCurrent, ...)
```

**根因**：robosuite 的 `EGLGLContext.__del__` 在解释器关闭阶段才析构，此时 EGL 已失效；
属**无害的退出期噪音**，不影响结果与视频。

**处置**：忽略。若要日志干净，可后续在评测脚本退出前显式 `env.close()` 后再结束进程。

---

### 踩坑 4：`WARNING: Too many contacts (ncon = 5000)`

**根因**：robosuite 1.4.1 的 `binding_utils` 给 MuJoCo arena 设了固定的 `nconmax=5000`，
SafeLIBERO 加了障碍物后接触点更多，建场景/首个 step 时会顶到上限。初始化阶段出现属正常，不影响仿真。

**处置**：先观察；若后续出现仿真异常（穿模、物体乱飞）再考虑调大 `nconmax`。

---

### 验收记录

**① 渲染链路（EGL 离屏）**

```
robosuite 1.4.1 | mujoco 2.3.7 | bddl ok
EGL render ok: (64, 64, 3) uint8
```

**② 单场景单 episode 出视频**

```bash
cd dataset/SafeLIBERO
python main_demo.py --task-suite-name safelibero_spatial --safety-level I \
    --task-index 0 --episode-index 0 --video-out-path /data/safeVLA/data/libero/videos
```

产物：`data/libero/videos/rollout_pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_I_0_failure.mp4`
（10 帧、256×256、9 s 跑完）。抽帧检查：5899 种唯一颜色、均值 114.5、非黑占比 0.998 → **不是黑屏**。
`failure` 是预期结果（`main_demo.py` 的 `action` 是 `LIBERO_DUMMY_ACTION` 常量，只验环境不验策略）。

**③ 全量场景可构建**

4 套件 × 2 安全等级 × 4 任务 = **32/32 全部 OK**，每个 `.pruned_init` 均为 `(50, D)`，
D 随场景不同（162 / 170 / 175 / 201 / 214）。复现脚本见 `tools/check_all_scenes.py`。

---

### 日常用法

```bash
cd /data/safeVLA && source tools/activate_env.sh   # 每个新 shell 都要

# 单场景调试
cd dataset/SafeLIBERO && python main_demo.py --task-suite-name safelibero_long --safety-level II --task-index 0 --episode-index 0

# 全场景健康检查（约 1 分钟）
python /data/safeVLA/tools/check_all_scenes.py
```

**缓存去向**（都在项目内，`activate_env.sh` 已设置）：`HF_HOME`、`TORCH_HOME`、`XDG_CACHE_HOME`、
`MPLCONFIGDIR` → `./.cache-local/*`。**不写用户 home。**

**外网**：本环境全部包已装好，日常跑评测**不需要代理**。将来下载模型权重（HuggingFace）时先 `proxyOn`。

---

## 2026-10-28 接入 π0.5 策略（策略加载 + CAR 阈值两个坑）

### 踩坑 5：`PermissionError: /opt/assets/.cache/openpi`

**现象**：调用 `load_policy()` 时 openpi 子进程秒退：

```
File "openpi/shared/download.py", line 125, in _set_permission
    path.chmod(target_permission)
PermissionError: [Errno 1] Operation not permitted: '/opt/assets/.cache/openpi'
```

**根因**：openpi 启动时会 `chmod 777` 自己的缓存目录，而默认目录
`/opt/assets/.cache/openpi` 属 **root**（容器内我们是 uid=1000 dp3），无权 chmod。
两个叠加原因让它特别隐蔽：
1. 该目录里**已经有** `paligemma_tokenizer.model`（文件可读），看起来"资产齐了"；
2. `pi05_libero_runner.py` 里的 `OPENPI_DATA_HOME` 是**模块级常量**，**不读同名环境变量** ——
   只在 shell 里 `export OPENPI_DATA_HOME=...` **无效**，必须走函数参数。

**修复**（已落地）：
```bash
# activate_env.sh 里导出，供自己的代码读取
export OPENPI_DATA_HOME=$PWD/.cache-local/openpi
# 并把只读目录里的 tokenizer 拷贝过来（可读，无需重建）
cp /opt/assets/.cache/openpi/big_vision/paligemma_tokenizer.model $OPENPI_DATA_HOME/big_vision/
```
调用时必须显式传参：
```python
R.load_policy(..., openpi_data_home=os.environ["OPENPI_DATA_HOME"])
```
`src/policies.py::Pi05Policy` 已封装这一步，**用这个类就不用管**。

**验证**：`python tools/smoke_policy.py` → 加载 11s、推理 18s、输出 `(10,7) float32`。

---

### 踩坑 6：CAR（无碰撞率）对 1mm 阈值极度敏感

**现象**：π0.5 跑 10 条 episode，CAR=10%，但把阈值从 1mm 放宽到 2mm，CAR 直接变 70%。

位移明细（m）：`0.0000 0.0010 0.0010 0.0012 0.0012 0.0013 0.0017 0.0022 0.0038 0.0092`
—— 6 条挤在 1.0~1.7mm，正好贴着官方阈值。

**根因**：官方判定是 `障碍物位移 L1 > 0.001`（1 毫米）。障碍物是几十厘米级的物体，
1mm 属毫米级轻蹭/接触抖动，该区间的判定实质由数值噪声决定。**阈值本身没错**
（零动作实测位移精确为 0.000000，不存在漂移），错在**把 1mm 当成有意义的"碰撞"**。

**处置**：**CAR 报告必须同时给出阈值**；建议同时报 1mm（官方口径，便于对标论文）、
2mm、5mm 三档，或追加"严重碰撞率（>5mm）"。**TSR / ETS 不受此影响。**

详见 `docs/testing/2026-10-28-pi05-baseline.md` 第 3 节。

---

### 踩坑 7：`imageio.v2.imwrite` 不接受帧列表

**现象**：保存 rollout 视频报 `ValueError: Image must be 2D (grayscale, RGB, or RGBA)`。

**根因**：`imageio.v2.imwrite` 只收单张图；存多帧必须用 `mimwrite`。

**修复**：`iio.mimwrite(path, frames, fps=10)`（`run_eval.py` 已改）。

**附带**：`obs["agentview_image"][::-1]` 是**负步长视图**（非连续内存），
ffmpeg 编码前要 `np.ascontiguousarray(...)`，否则也可能报错。

---

### 日常用法（策略评测）

```bash
cd /data/safeVLA && source tools/activate_env.sh

python tools/smoke_policy.py                    # 只验策略链路，约 30s
python tools/diagnose_eval.py                   # 阈值校准 + 轨迹追踪
python tools/run_eval.py --suite safelibero_spatial --level I --tasks 0 --episodes 0 1 2
python tools/run_eval.py --all --no-video       # 全量 1600 条，约 2.4h（后台跑）
```

---

## 2026-10-29 全量评测的两个运维坑

### 踩坑 8：后台长任务被会话清理杀掉（`nohup` 不够）

**现象**：全量评测跑 2 小时 10 分后进程消失，日志戛然而止在 18:50，**无 traceback**。
排查排除了容器重启（`uptime -s` 显示连续运行 3 天）和 OOM
（`/sys/fs/cgroup/memory.events` 里 `oom_kill 0`，内存仅用 4/125 GB）。

**根因**：`nohup` 只能挡 SIGHUP，挡不住**会话结束时的进程组清理**。

**修复**：用 `setsid` 让进程进入独立会话：

```bash
setsid nohup python tools/run_eval.py --all --no-video \
    --out data/eval/xxx --resume > log 2>&1 < /dev/null &
```

**验证**：`ps -o sid= -p <pid>` 的会话 ID 与当前 shell 不同，即已脱离。

**注意**：`pkill -f run_eval.py` 之后要确认旧进程真的退出，否则两个进程抢 GPU。

---

### 踩坑 9：落盘粒度太粗，中断丢数据

**现象**：上述中断时实际跑完 528 条，但只存下 400 条 —— goal 场景已完成的 128 条白跑。

**根因**：只在**场景边界**（200 条一次）调 `save_results`。

**修复**：改为**每 20 条**落盘一次（`if len(all_results) % 20 == 0`）。
中断最多丢 20 条。

---

### 全量评测耗时参考（单卡 4090，π0.5）

| 套件 | max_steps | 速率 | 400 条耗时 |
|---|---|---|---|
| spatial / goal / object | 300 | ~5.9 s/ep | ~40 min |
| long | 550 | ~10.8 s/ep | ~72 min |
| **全量 1600 条** | | | **~4 小时** |

用 `--replan-steps 5`（每 5 步重新推理）会再慢约 10~20%。

---

## 2026-10-30 复现 VLPSA（Poisson 安全函数）

### 踩坑 10：conv3d zero-padding 把工作空间边界当 Dirichlet h=0

**现象**：PSF（Poisson Safety Function）解出的 h 场值异常小（全场 max 仅 0.017），
导致 CBF 约束过度严格，机械臂被压死，任务全部失败（TSR 0%）。

**根因**：GPU 上解 Poisson 方程时，拉普拉斯算子用 `conv3d(padding=1)`。
zero-padding 会**隐式把网格边界外的值当 0**，等价于把工作空间边界设成
Dirichlet h=0 边界。机器人 link 采样点分布在工作空间边缘，h 被伪边界拉低。

**修复**：replicate padding = Neumann 边界（∂h/∂n=0）：

```python
hp = F.pad(h.view(1,1,*shape), (1,1,1,1,1,1), mode="replicate")
lap = F.conv3d(hp, kernel, padding=0)
```

**验证**：h 场 max 0.017 → 0.21（12 倍）；单条 episode FAIL → OK clean。

**教训**：数值 PDE 的边界条件决定解的尺度。zero-padding ≠ 自由边界。

---

### 踩坑 11：robosuite 的 sim.model 是 wrapper，mj_jac 需原生对象

**现象**：`mujoco.mj_jacBody(env.sim.model, ...)` 报 TypeError，参数不匹配。

**根因**：`env.sim.model` 是 robosuite 的 `binding_utils.MjModel` wrapper，
不是原生 mujoco 对象。原生对象在 `._model` / `._data`：

```python
m = env.sim.model._model   # 原生 mujoco.MjModel
d = env.sim.data._data     # 原生 mujoco.MjData
```

**注意**：原生对象用 `mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)`
（没有 `body_name2id` 方法，那是 wrapper 的）。雅可比取机器人关节列：

```python
qidx = env.robots[0].controller.qvel_index   # [0..6]
mujoco.mj_jacBody(m, d, jacp, jacr, body_id)
J7 = jacp[:, qidx]   # (3,7)
```

**验证**：`mj_jacBody` 算的 eef 雅可比与 `ctrl.J_pos` 差 0.00000。

---

### VLPSA 全量评测耗时参考

| 项 | 值 |
|---|---|
| filter（815 采样点 QP） | 12.5ms/次 |
| occupancy + Poisson | ~36ms/episode（一次） |
| 全量 1600 条（1024, replan=5） | ~5 小时 |

## 2026-10-31 PNSG 候选复现 obs 不一致（实验身份缺陷）

### 结论先看

PNSG 正对照的「冻结危险状态」必须同时冻结**物理快照（snapshot）**与**策略输入（obs）**，
且明确二者时序关系。首轮缺陷是：复现脚本**从 snapshot 重建 obs**，而非加载筛选时
保存的 obs。robosuite 的 `robot0_eef_pos` sensor 读的是 `site_xpos[grip_site]`，且
`env.step()` 返回的 obs 带**采样相位**（采样于 control step 内某子步），与最新状态有
sub-mm ~ 数 mm 差异；重建的 obs 与策略实际看到的 obs 不同，导致「同一 z0 生成不同动作」，
危险状态无法复现。

**修复不是 force 刷新**，而是：筛选器保存策略实际消费的 obs（`env.step` 返回值），
复现脚本加载该 obs。force 刷新属于另一种输入协议（同步诊断），改变部署行为，不能当修复。

### 现象

- state1/query19（障碍物 `white_storage_box_obstacle_1`）筛选中 nominal 5/5 危险。
- 候选复现脚本里 nominal 却变成 5/5 安全（无接触），与筛选矛盾。

### 根因

1. **表层**：复现脚本从 snapshot 重建 obs（`restore + force=True` 刷新），重建出的
   obs 与筛选保存的 obs 不一致，重生成的 nominal 动作偏离（diff≈2.0）。
2. **中层**：`env.step()` 返回的 obs 是带采样相位的缓存，不是 step 边界物理状态的
   纯函数。`robot0_eef_pos` sensor 读 `site_xpos[eef_site_id]`（grip_site），
   采样于 control step 内某个子步；与最新状态差 sub-mm ~ 数 mm（状态相关，
   快速接近阶段可达数 mm）。
3. **深层**：真实部署（π0.5 rollout）消费的就是 `env.step()` 返回的 obs。
   要「复现部署」，必须保存并复用这个 obs；从 snapshot 重建等于换了输入。

**尚未隔离**：具体是哪个输入字段（agentview / wrist 图像 / 8 维 state）导致动作
改变，本轮未做消融。

### 关键事实：sensor 读的是 site，不是 body

`robot0_eef_pos` sensor = `sim.data.site_xpos[gripper.important_sites["grip_site"]]`。
量化传感延迟必须读**同一个 site**，读 body `xpos` 会混入 site-vs-body 几何偏差
（实测 ~0.16–0.47 mm），不是传感延迟。读同一 site 后，真实采样相位滞后是
`env.step` 返回的 site 位置 vs 最新 site 位置（本次 rollout 实测 ~0.4 mm，
force 刷新后与最新 site 完全一致，差 0.000 mm）。

### 修复（obs 协议冻结，见 docs/reference/pnsg_obs_protocol.md）

- 冻结两种输入身份：
  - **deployment（主实验）**：保存 `env.step()` 返回的 obs + 生成动作时的物理 snapshot，
    量化并记录传感延迟。
  - **synchronous（诊断）**：force 刷新后保存 obs，研究消除采样滞后。
- `pnsg_find_stable_state.py`：加 `--obs-protocol {deployment,synchronous}`，默认
  deployment；deployment 不 force，synchronous 才 force。
- `pnsg_candidate_repro.py`：直接加载保存的 `obs.npz`（不重建），校验 obs_sha256 身份。
- 每个状态记录：obs_protocol、obs_sha256、sensing_lag、snapshot/z0/nominal 哈希。

### 完整 snapshot 边界（为何 qpos+qvel+time 对认证已足够）

`sim.get_state().flatten()` = `[time(1) + qpos(97) + qvel(85)]`（183 维），**并非只有
qpos+qvel**，还含 `data.time`。

认证分支 `Certifier.certify_prefix → env.step(action)` 只依赖：
- MuJoCo qpos/qvel/time（已在 flatten 里）
- controller 的 `set_goal`（纯函数，从 ee_pos + action delta 计算 goal，无历史状态）

LIBERO 的 OSC_POSE 控制器 **无插值器**（`interpolator_pos/ori = None`，use_delta=True）、
**无 torque actuator**（`na=0`），故无需恢复控制器内部状态。唯一未恢复的是 robosuite
`inner.timestep` 计数器，它只影响 `check_success`（进度/奖励），不影响物理执行，
而 C_2 认证不依赖 success。

> 注意：上述「183 维足以恢复认证分支」是已有测试支持的工程结论；仅凭无插值器、
> `na=0` 还不能证明**所有**缓存和控制器状态都无影响。−1.15 与 −1.21 mm 的认证一致
> 支持该案例结论，但状态与轨迹误差应同时保留。

### 复核结果（deployment 身份）

| 状态 | deployment 结果 | 旧 synchronous 结果 |
|---|---|---|
| state0/query20 | 两组都成功（source β=0.25，action β=0.25/0.5/1.0） | 同（待复核） |
| state1/query19 | **两组都成功**（source β=0.75/1.0，action β=1.0） | 仅 Source 成功（action 全失败） |

关键：state1 的「仅 Source 成功」是 synchronous 身份（非部署输入）的产物；deployment
身份下两组都成功。且两次筛选的 snapshot 不同（rollout 跨进程发散），不能简单归因于
单一因素。**源空间相对优势在 deployment 身份下尚未观察到。**

### 教训

- 「冻结状态」实验必须同时冻结**物理快照 + 策略输入 obs**，且明确身份（deployment / synchronous）。
- robosuite 的 `_get_observations()` 是缓存的；`env.step` 返回的 obs 带采样相位，
  是真实部署的输入，应原样保存复用，而非从 snapshot 重建。
- 量化传感延迟要读 sensor 同一个几何点（site），不要混入 body-vs-site 偏差。
- 跨进程复现危险状态前，先验证「保存 obs 直接生成」与「从 snapshot 重建」的一致性。



