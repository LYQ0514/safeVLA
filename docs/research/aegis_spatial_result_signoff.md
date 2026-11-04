# AEGIS Spatial perception 变体结果收口

> 日期：2026-11-04
> 代码基线：几何/四元数修复 + perception 链路（VLM+GroundingDINO+MVEE）
> 评测命令：`run_eval_aegis.py --suite safelibero_spatial --level I/II --tasks 0 1 2 3 --episodes 0..49 --perception --no-video`

---

## 1. 结果数字（overall，每 level 200 episode）

| 套件 | TSR | CAR | CAR>5mm | SSR | ETS |
|---|---|---|---|---|---|
| Spatial **level II** (perception) | 76.0% | 70.0% | 72.5% | 64.5% | 180.6 |
| Spatial **level I** (perception) | 58.0% | 68.0% | 71.5% | 46.0% | 213.2 |

（overall 口径，`summary.json` 的 `overall` 字段，各 200 episode）

## 2. 与官方对标

官方 Spatial（论文 Table 1 / 项目页）：

| 口径 | CAR | TSR |
|---|---|---|
| Translational Only（论文 Table 1） | ~75.5% | ~73.25% |
| Full Action Space（项目页） | ~68.9% | — |

我们的结果：
- **LII**：CAR 70% 落在官方 [68.9, 75.5] 区间内，TSR 76% 略高于 73.25%；
- **LI**：CAR 68% 接近 full action space 下限，TSR 58% 明显低于 LII。

> ⚠️ level I 与 level II 是 SafeLIBERO 的不同 safety level（II 更难），官方是否分开
> 报 LI/LII 需再核实；不能直接把「LII 76%」当作「全 Spatial 76%」与官方单数对标。

## 3. 必须诚实报告的 4 个问题

### 3.1 不是端到端严格复现（已接受的审计判决）

perception 链路有三处与上游不同：
1. VLM：上游 `from zai import ZhipuAiClient`（私有封装）→ 我们改 `zhipuai.ZhipuAI` 官方 SDK；
2. GroundingDINO：复用本地 `groundingdino_loader.py`（BERT + 权重前缀剥离）；
3. oracle 椭球：碰撞几何顶点拟合 MVEE（oracle 变体用，perception 变体用 VLM 点云 MVEE）。

**结论：这是「近似/部分复现」，不能声称端到端严格复现论文。**

### 3.2 VLM 识别正确率未审计（关键缺口）

`summary.json` 的 `obstacle` 字段 = **真值 `active_obstacle`**（CAR 追踪对象，第一个
workspace 内障碍物），**不是 VLM 识别结果**。VLM 识别的 `vlm_name` 未写入 summary。

后果：
- 无法审计「VLM 识别是否命中真值 obstacle」；
- CAR 追踪真值 obstacle，AEGIS 椭球来自 VLM 识别物体——**两者可能不一致**（VLM 识别错
  → AEGIS 保护错物体 → 真值 obstacle 被撞 → CAR 下降），这是 perception 变体的固有
  特性，但当前无法量化识别正确率。

> 注：SafeLIBERO spatial 每个 task 的 50 个 init state 里，`active_obstacle` 真值
> 确实是 5 种障碍物各 10 个（物体摆放位置不同），这是 benchmark 设计，不是 VLM 随机。

### 3.3 并发写文件 bug 的污染风险（LII 全程 + LI 前 40）

时间线：
- 最初 LI + LII **并行**启动（代码 d8bfafd，`/tmp/aegis_obstacle_detection.png` 共享路径）；
- LI 在 ep38 后崩溃（并发写文件致图片损坏）；
- 修复并发 bug（b2226d9，进程唯一路径）后，LI `--resume` 续跑（此时 LII 已结束，单进程）。

污染范围：
- **LI 前 40 episode**（d8bfafd，与 LII 并行）：有竞态污染风险；
- **LII 全程 200 episode**（d8bfafd，前 ~40 个与 LI 并行）：前 ~40 个有竞态污染风险。

竞态机制：`plt.imsave` → `open().read()` 之间的微秒窗口，另一进程可能覆盖图片，导致
VLM 识别**错误的障碍物**（静默污染，不崩溃）或图片损坏（崩溃，已发生在 LI ep38）。

**影响**：被污染的 episode，AEGIS 可能保护了错误物体，真值 obstacle 未被保护 → CAR
被低估。这是**概率性**污染，无法从 summary 直接定位哪些 episode 被污染。

### 3.4 per_scenario 统计 bug（resume 合并）

LI 的 `summary.json`：
- `overall` = 200 episode（正确）；
- `per_scenario[safelibero_spatial_LI]` = 160 episode（**错误**，只算 resume 本次新跑的，
  未加载旧的 40 个）。

根因：`run_eval_aegis.py:229-233`，resume 时 `suite_results` 只累积本次进程新跑的
episode，未合并旧 episode。**episode 明细（200 条）本身正确，只是 per_scenario 汇总
数字偏小。**

## 4. 诚实结论

1. **数字落在官方 Spatial 区间内**（LII CAR 70% ∈ [68.9, 75.5]，TSR 76% ≈ 73.25%），
   但**不能据此声称「已严格复现」**；
2. **三项缺口**：VLM 识别正确率未审计、并发污染范围未定位、per_scenario 统计 bug；
3. 当前结果的正确表述是：**「AEGIS-QP (Perception) 变体在 SafeLIBERO-Spatial 上的
   近似复现，LII TSR 76% / CAR 70%、LI TSR 58% / CAR 68%，数字接近官方但存在未审计的
   VLM 识别与并发污染风险」**。

## 5. 若要进一步收紧（待决定）

1. **补记录 VLM 识别**：perception 链路把 `vlm_name` + 识别置信写入 summary，重跑时
   可审计「VLM 命中真值 obstacle 的正确率」；
2. **重跑污染 episode**：用 b2226d9 单进程重跑 LI 前 40 + LII 前 40（或全量 LII），
   排除并发污染；
3. **修 per_scenario bug**：resume 时合并旧 episode 到 `suite_results`；
4. **补 oracle 变体**：1600 episode 全量（真值 MVEE，作为 perception 变体的上界参照）。
