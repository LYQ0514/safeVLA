# AEGIS vendored 代码 — 改动与差异记录

上游：https://github.com/THU-RCSCT/vlsa-aegis （MIT License）
论文：VLSA / AEGIS, arXiv 2512.11891, IROS 2026
vendored 日期：2026-10-29

`main_aegis.py` / `utils.py` / `AEGIS_requirements.txt` **原样保留，未改一行**。
所有适配写在 `../../src/safety/aegis_adapted.py` 与
`../../tools/run_eval_aegis.py`，便于对照和审计。

---

## 四处必要改动（影响可比性，报告须注明）

### 1. 删除 `eef_marker` 依赖

**上游行为**（`main_aegis.py` ~138 行）：
```python
eef_body_id = model.body_name2id("eef_marker")
env.sim.model.body_pos[eef_body_id] = ball_pos
```
用来在仿真里画一个球，可视化末端椭球位置。

**问题**：公开版 SafeLIBERO **没有这个 body**，直接抛
`ValueError: No "body" with name eef_marker exists.`

**根因**：AEGIS 仓库自带的 `safelibero/libero/libero/assets/scenes/libero_tabletop_base_style.xml`
比 HuggingFace 公开版多一行 `<body name="eef_marker" pos="0 0 1.3">`。

> ⚠️ 这意味着**论文数字来自一个与公开基准略有差异的环境**。
> 差异本身很小（该 body 只用于可视化，CBF 实际用 Python 变量 `p1`），
> 但"用官方仓库复现论文数字"这个前提并不成立。

**我们的做法**：不调用该 body，`p1` 由 `eef_pos + R1 @ [0,0,-0.08]` 算得（与上游同一公式）。

### 2. 障碍物识别：云端 VLM → oracle 真值

**上游行为**（`utils.py::obstacle_detection`）：
```python
api_key = ""
client = ZhipuAiClient(api_key=api_key)     # 智谱 GLM-4.5v
model = "glm-4.5v"
```
每个 episode 开头调用一次付费云端多模态 API，让模型看图指出哪个障碍物最可能被撞；
再用 GroundingDINO 按该名称出框，结合深度图反投影成点云，拟合最小体积包围椭球。

**问题**：`api_key` 为空，直接 `raise ValueError`。需要付费 key + 外网。

**我们的做法**：用仿真真值替代整条链路：

| 需要 | 上游 | 我们 |
|---|---|---|
| 哪个障碍物 | GLM-4.5v 看图判断 | `sim.model.joint_names` 筛 `obstacle` + 工作空间判定 |
| 障碍物位姿 | 深度点云反投影 + 椭球拟合 | `obs[f"{name}_pos"]` / `_quat` |
| 障碍物尺寸 | 最小体积包围椭球（mvee） | `model.geom_size` 取该 body 全部 geom 最大值 |

**先例**：VLPSA（arXiv 2609.22462）重评 VLSA 时明确采用同一做法 ——
"we replace online vision language model (VLM) obstacle identification with ground-truth labels"。
用 oracle 感知隔离控制层贡献是已被采用的标准协议。

**后果**：我们的 AEGIS 结果**不受感知误差影响**，因此：

- 若我们的 CAR **高于**论文的 77.85%，不能直接说"我们实现更好" —— 是感知差异
- 这恰好能验证论文那句自述："残余碰撞来自上游感知局限，不是控制逻辑"

### 3. π0.5 调用方式：websocket → 文件 IPC

上游用 `openpi_client.websocket_client_policy.WebsocketClientPolicy(host, port)`，
需要另起 policy server。我们复用本项目 `src/policies.py::Pi05Policy`（文件 IPC）。

**影响**：无。送入模型的观测构造与上游一致（agentview + wrist + 8 维 state + prompt），
动作取前 7 维。

### 4. 运行框架：单脚本 → 统一评测循环

上游是独立脚本，自带一套统计。我们接入 `src/eval_safelibero.py`，
与基线**完全同一套**场景初始化、碰撞判定（1mm）、指标计算。

**影响**：这是**提高**可比性的改动 —— 两个方法的唯一差异只剩安全层。

---

## 保持与上游一致的部分（不要改）

| 参数 | 值 | 出处 |
|---|---|---|
| CBF gain | 约束里 `10 * h` | `main_aegis.py` |
| 动作缩放 | `0.2 * R1 @ u_v` | 同上 |
| 参考速度增益 | `u_v_ref = 5 * v_ref` | 同上 |
| z 方向演化 | `dt = 0.05` | 同上 |
| 末端椭球半轴 | `[0.06,0.12,0.11]`；含 orange juice/milk/alphabet soup 时 `[0.06,0.12,0.20]` | 同上 |
| 末端偏移 | 局部 z 方向 `-0.08` | 同上 |
| QP 权重 | `diag([1/25]*6 + [1,1,1])` | 同上 |
| 求解器 | `cp.OSQP` | 同上 |
| replan_steps | 5 | 上游默认 |
| 分辨率 | 1024 | `LIBERO_ENV_RESOLUTION` |

---

## 我们额外做的两处改进（可选关闭）

这两处**偏离上游**，但方向是修 bug，需在报告中说明：

### ① 障碍物位姿每步刷新（⚠️ 已回退，见下）

上游只在 episode 开头算一次 `p2`（障碍物中心）。障碍物一旦被碰动，椭球就是错的。

~~我们的 `filter()` 末尾会更新 `self.p2 = obs[f"{name}_pos"]`。~~

**2026-11-04 更新**：为**严格复现上游**，此改动已回退——`filter()` 不再每步刷新 `p2`，
与上游一致（fit_ellipse 一次，episode 内固定）。见
`docs/research/aegis_reproduction_audit.md` 与 `docs/research/aegis_strict_reproduction_gaps.md`。

### ② 记录 QP 不可行率

上游在 `u.value is None`（QP 不可行 = 无法满足安全约束）时：

```python
u_v = action[:3]      # 回退到原始动作 —— 即最危险的动作
```

**逻辑上是反的**：安全约束无法满足时，反而执行最不安全动作。

为保持"复现"，我们**保留该行为**，但统计 `infeasible_rate` 并在输出里打印，
以便量化这一缺陷。若要改成"急停"回退，属改进而非复现，需单列实验。

### ③ 四元数约定 bug（已修复，2026-11-04）

**发现**：`aegis_adapted.py` 的 `_quat_wxyz_to_rot` 曾把 `robot0_eef_quat`（实为
**xyzw**，robosuite `convert_quat(to="xyzw")`）误当 wxyz，做 `[x,y,z,w]=[q1,q2,q3,q0]`
重排，导致末端姿态 R1 的 y/z 轴符号翻转、末端椭球中心 p1 偏移方向反了 **16cm**。

**影响**：污染所有 AEGIS 复现（TSR 被严重低估，long task0 从错误的 0% 修复为 45%）。

**修复**：改为 `R.from_quat`（xyzw 直通）。详见
`docs/research/aegis_reproduction_audit.md`。

### ④ 策略输入分辨率对齐（2026-11-04）

上游「渲染 1024 + resize 224 喂策略」，π0.5 训练分辨率 224。此前评测曾用 256/1024
直接喂策略（OOD），已对齐为「渲染 1024 + 策略输入 224」（新增 `--policy-resize 224`）。
