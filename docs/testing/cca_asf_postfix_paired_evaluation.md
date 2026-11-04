# CCA-ASF 正式 Pilot（修复后 32-episode 配对对照）

> 日期：2026-11-04
> 规格：`CCA_ASF_PostFix_Strict_Implementation_Spec_20261104.md` §8.1
> 代码基线：HEAD `b2226d9`（四元数修复后）
> 脚本：`tools/cca_asf_pilot.py`；manifest/结果：`data/audit/cca_asf_pilot_*.json`

---

## 1. 协议

| 项 | 值 |
|---|---|
| 场景 | `safelibero_long` Long-I task0，S01/S02/S04/S09（state_id ∈ {1,2,4,9}） |
| 方法 | `euclidean` vs `cca_asf(beta_max=4)` |
| 样本 | 4 state × 4 stream × 2 method = **32 episodes** |
| policy | PnsgPolicy + CRN seed（预期确定性） |
| 障碍物 | 动态 `find_active_obstacle` + `_main`（不硬编码） |
| 指标 | 分离 TSR_native / CAR_native / SSR_native |

## 2. 结果（指标分离）

### 2.1 总体

| 指标 | euclidean (n=16) | cca_asf (n=16) |
|---|---|---|
| TSR_native | 13/16 | 15/16 |
| **CAR_native（≤1mm）** | **1/16** | **0/16** |
| **SSR_native** | **0/16** | **0/16** |
| mug 位移均值 | **11.5mm** | **17.9mm** |
| 介入次数均值 | 164.1 | 129.5 |

### 2.2 逐 state mug 位移均值（mm）

| state | euclidean | cca_asf | Δ |
|---|---|---|---|
| S01 | 15.1 | 18.2 | +3.1 |
| S02 | 20.9 | 30.1 | +9.2 |
| S04 | 8.1 | 16.6 | +8.5 |
| S09 | 1.8 | 6.7 | +4.9 |

**4 个 state 的 cca_asf mug 位移均值全部更高**（碰撞更严重）。

### 2.3 逐 state TSR

| state | euclidean | cca_asf |
|---|---|---|
| S01 | 2/4 | 4/4 |
| S02 | 4/4 | 4/4 |
| S04 | 4/4 | 4/4 |
| S09 | 3/4 | 3/4 |

TSR 仅在 S01 有差异（cca 4/4 vs euclidean 2/4），其余持平。

### 2.4 配对 Δ（euclidean → cca_asf 的 mug 位移差，mm）

16 对里 **13 对 Δ>0（cca 更差）**，3 对 Δ<0（cca 更好）。Δ 均值 ≈ **+6.4mm**。

## 3. 核心发现

### 3.1 「CCA 灾难性恶化」判决已推翻（确认）

修复后 euclidean 本身 mug 位移均值 1.8-20.9mm（S09 最低 1.8mm），不再是 bug 时代
的「euclidean=0 vs CCA 27-30mm」。原判决是四元数 bug 假象，**已作废**。

### 3.2 但发现一个相反方向的趋势（重要）

**CCA 在 4 个 state 上 mug 位移均值全部更高**（碰撞更严重），同时**介入次数更少**
（129.5 vs 164.1），TSR 持平或略高（S01 +2）。

这是一个**物理上可解释**的结果：

> CCA 的设计目标是把安全修正「从保护障碍物方向」重分配到「保持任务方向」。
> 结果是：介入更少、任务方向保留更好（TSR 不降），但代价是**障碍物被撞得更厉害**
> （mug 位移更大）。

这与规格 §12 的禁令完全吻合：「CCA 每次干预更少所以更安全」是**错误**的——
Pilot 证明**干预少 ≠ 更安全，反而更危险**。

### 3.3 非确定性验证：同进程确定、跨进程漂移

requick（n=4，仅 S01）与 pilot（n=16，含 S01）的**同条件结果不一致**：

| 条件 | requick | pilot | nondet_check |
|---|---|---|---|
| S01/st0 euclidean mug | 25.6mm | 7.8mm | 25.6/25.6/25.6mm |

为此专门跑了 `tools/cca_asf_nondet_check.py`（同一 `S01/st0 euclidean` 连续 3 次）：

```
run0/1/2: mug=25.6mm, first_chunk_hash 完全一致
均值=25.6mm  标准差=0.00mm  极差=0.0mm
```

**关键发现**：

1. **同进程内完全确定**：3 次重跑 mug 位移标准差 0.00mm、首个 chunk 哈希一致 →
   PnsgPolicy + CRN 在**同一进程内**是确定性可复现的；
2. **跨进程漂移**：requick(25.6)、nondet(25.6) 与 pilot(7.8) 是**三个不同进程**，
   结果不同。pilot 是唯一与 LI 续跑（Pi05Policy 的 openpi worker）**并发**启动的，
   疑 GPU 并发负载改变了 openpi worker 的初始化/推理数值路径。

**对 Pilot 结论的影响**：

- pilot 的配对 Δ（euclidean vs cca_asf）在**同一 pilot 进程内**算得，同进程确定，
  **配对对照本身自洽**；
- 但**绝对基线会随进程漂移**（euclidean 基线 25.6 ↔ 7.8），因此「cca mug 均值
  更高（17.9 vs 11.5）」的**跨进程稳健性未验证**；
- requick「cca 略好」与 pilot「cca 更差」的矛盾，部分源于跨进程基线漂移，
  部分源于 requick 的 n=4 太小。

## 4. 诚实结论

1. **「CCA 有害（灾难性）」**：已洗清——那是四元数 bug 假象；
2. **「CCA 有效」**：**未证实**。TSR 差异（15 vs 13）在 n=16 且受跨进程漂移困扰下无统计意义；
3. **「CCA 无害」**：**反而出现不利信号**——单次 pilot 内 4 个 state 上 mug 位移均值
   全部更高，与「少干预 → 更安全」的直觉相反；
4. **机制解释**：CCA 的「少干预保任务方向」在物理上**确实以碰撞更严重为代价**，
   这是方法设计本身的 trade-off，而非 bug；
5. **稳健性未验证**：跨进程基线漂移（25.6 ↔ 7.8mm）使得「cca 更差」的结论
   只对单次 run 成立，需跨进程重跑才能确认为稳健。

**总裁决**：CCA-ASF 在修复后 baseline 上，**既未证明有效，也未证明无害**，且单次
pilot 的初步证据显示「少干预」与「更危险」相伴。该结论受跨进程非确定性困扰，
在下强结论前需（a）定位跨进程漂移根因（疑 GPU 并发），（b）跨进程重跑 pilot
验证配对 Δ 的方向稳健性。

## 5. 待办（若要继续）

1. **定位跨进程漂移**：nondet_check 已证明同进程确定；下一步在「GPU 空闲 vs 被
   并发任务占用」两种状态下各跑一次 S01/st0 euclidean，确认漂移是否由 GPU 并发导致；
2. **跨进程重跑 pilot**：若漂移根因可控（如独占 GPU），则在干净 GPU 下重跑 32
   episode，验证「13/16 cca 更差」的方向是否稳健；
3. 若方向稳健，则「CCA 少干预 → mug 位移更大」是一个可报告的科学结论；
   若方向不稳，则 CCA 与 euclidean 在 mug 保护上无显著差异，结论退回「无效」。
