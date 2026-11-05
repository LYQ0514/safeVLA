# AEGIS perception 变体：SafeLIBERO 四套件完整结果

> 日期：2026-11-04（补跑完成 2026-11-05）
> 代码基线：几何/四元数修复 + perception 链路（VLM+GroundingDINO+MVEE）+ 并发修复（b2226d9）
> 变体：**perception**（AEGIS-QP Perception），非 oracle
> 口径：CAR = 障碍物位移 ≤1mm（官方）；TSR / SSR / ETS（含超时）同 `AGENTS.md` 定版

---

## 1. 四套件完整结果（每 cell 200 episode）

| 套件 | level | TSR | CAR | CAR>5mm | SSR | ETS |
|---|---|---|---|---|---|---|
| **spatial** | I | 56.5% | 68.5% | 71.5% | 44.5% | 216.0 |
| **spatial** | II | 77.5% | 70.5% | 73.0% | 65.0% | 178.7 |
| **goal** | I | 84.5% | 94.5% | 96.0% | 80.0% | 173.8 |
| **goal** | II | 84.0% | 61.0% | 71.0% | 56.0% | 160.5 |
| **object** | I | 58.0% | 40.0% | 52.5% | 24.0% | 242.6 |
| **object** | II | 72.5% | 71.5% | 76.5% | 57.0% | 205.2 |
| **long** | I | 50.0% | 45.0% | 53.5% | 21.0% | 438.8 |
| **long** | II | 41.5% | 74.0% | 79.0% | 31.5% | 464.2 |

## 2. 按套件聚合（level I + II 合并，共 400 episode/套件）

| 套件 | TSR | CAR | SSR |
|---|---|---|---|
| spatial | 67.0% | 69.5% | 54.8% |
| goal | 84.3% | 77.8% | 68.0% |
| object | 65.3% | 55.8% | 40.5% |
| long | 45.8% | 59.5% | 26.3% |

## 3. 观察

1. **套件难度梯度清晰**：goal（桌面，障碍物远离任务路径）最易，CAR 94.5%（LI）；
   long（living room，多阶段长任务）最难，TSR 45.8%、SSR 26.3%；
   object（floor，末端下探）CAR 最低（LI 40%）。

2. **level I vs II 分化方向不一致**：
   - spatial / object：LII 反而**更好**（TSR 77.5% vs 56.5%；72.5% vs 58%）；
   - goal：LI 更好（CAR 94.5% vs 61%，TSR 持平）；
   - long：LII TSR 更低但 CAR 更高（45%→41.5% TSR，45%→74% CAR）。
   
   ⚠️ 这与「LII 更难」的直觉不完全一致，说明 SafeLIBERO 的 level I/II 差异是
   **障碍物摆放**不同，不必然是难度单调递增——部分任务 LII 反而更利于完成。

3. **ETS 套件差异大**：short 套件（spatial/goal/object）170-240，long 套件 438-464
   （max_steps 550，大量超时）。

## 4. 与官方对标（诚实边界）

- 官方 AEGIS 论文 Table 1 是 **Translational Only** 口径，项目页另有 Full Action Space
  口径；本实现是 **full action space（含旋转 u_omega）**，严格对标 Full Action Space；
- 官方未按 level I/II 分开报所有套件（论文主表集中在 spatial 等特定套件），
  本结果按 level 分开报，便于逐 cell 核对；
- **不能声称端到端严格复现**：perception 链路的 VLM（zhipuai 官方 SDK）、GroundingDINO
  （本地复用）、MVEE 三处与上游实现不同（详见 `aegis_spatial_result_signoff.md`）；
- VLM 识别正确率仍未审计（summary 的 obstacle 字段 = 真值 active_obstacle，非 VLM 结果）。

## 5. 数据与复现

- 结果目录：`data/eval/aegis_perception_{suite}_{level}/summary.json`（8 个）；
- 评测命令：`run_eval_aegis.py --suite <s> --level <l> --tasks 0 1 2 3 --episodes 0..49
  --perception --no-video --resume`；
- 串行 runner：`tools/run_remaining_aegis.sh`（goal/object/long 六评测串行，全部 exit=0）；
- 全程 GPU 稳定（~20GB），无并发污染（串行），无崩溃。

## 6. 结论

perception 变体的四套件完整结果已跑通（共 1600 episode，含 spatial 400）。数字呈现
清晰的套件难度梯度，但 level I/II 的分化方向因套件而异。这些数字可作为「AEGIS-QP
(Perception) 变体在 SafeLIBERO 全四套件的近似复现基线」，**但受 VLM 识别正确率未审计
的限制，不能作为严格复现结论**。
