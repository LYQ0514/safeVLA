# M3 最终归档：State-Dependent Value of Non-Minimal Safety Interventions

> 日期：2026-11-02
> 状态：研究线 ARCHIVE。不追加实验。本文件是未来重新立题的研究证据档案。
> 前置：`m3_p0_intervention_burden_hypothesis.md`、`m3_p0_ranking_reversal_report.md`、
> `m3_p1a_frozen_vla_burden_transfer.md`、`m3_p1a_stage_conditioned_reanalysis.md`。

---

## §1 Research Question

> 最小当前安全修正（传统 QP 解），是否必然带来更小的未来安全干预负担？

形式化：

- 未来累计修正负担 $D_H = \sum_{k=1}^H \gamma^{k-1} \|\Delta u_{t+k}\|_W^2$；
- 未来干预次数 $N_H = \sum_{k=1}^H \mathbb I[\|\Delta u_{t+k}\| > \epsilon]$；
- 排序反转：$\|u^A - u^\pi\| < \|u^B - u^\pi\|$ 但 $D_H(u^A) > D_H(u^B)$。

候选 $u^A$ = QP 解（最小修正，贴 CBF 边界）；$u^B$ = 切向绕行（当前修正更大）。

## §2 M3-P0：最小机制实验（2D 单积分器）

- 系统：2D 单积分器 + 圆形障碍（CBF 相对度 1）+ 冻结 PD 朝目标；
- 候选：QP 解 vs 切向绕行（沿障碍切向，预注册规则选取）；
- **结果**：一般场景（目标随机）反转率 ~80-83%；正对穿障族 100%（构造退化）；
- 参数稳健：K/α/r/H 全组合稳定；
- 机制：QP 贴边界滑行 → 未来持续干预（3.22 次/8 步）；切向绕行 → 一次绕开（2.15 次）；
- 任务进度：绕行 0.754 vs QP 0.041（绕行**改善**任务）；
- **适用边界**：单积分器、单障碍、直线穿障是简化假设。

## §3 M3-P1a：冻结 VLA 迁移实验（真实 π0.5 + AEGIS）

- 任务 Long-I task0；状态 S01/S09 × stream 3/4；检查点 @0/@5（首次/第五次干预）；
- 候选：AEGIS QP 解 u_A vs 9D 切向绕行 u_B（Gate 1 探测：53/53 可构造）；
- 配对：共享父状态 + 首步不同 + 后续都恢复 AEGIS，H=2 重规划周期；
- **结果**：8 对，4/8 反转，且反转全部集中在 @0（首次干预），@5 全不反转；
- 任务进度：P_A/P_B 均为负（检查点在「远离准备」阶段，eef-soup 距离先增后减）。

## §4 M3-P1a-R：机制澄清（离线细粒度重分析）

- 用 run_len（CBF 连续激活长度）分组，替代「第几次干预」：
  - Entry（run_len=1，首次干预）：反转 $D_H$ **4/4**；
  - Persistent（run_len=6，持续干预）：反转 $D_H$ **0/4**；
- **关键分离**：反转 $D_H$ 4/4，但反转 $N_H$ **0/4**（$N_A = N_B = 9$）——
  切向绕行**未减少干预次数**，而是**减少了单次修正幅度**；
- 任务进度：8 对全部 `TASK_PROGRESS_INCONCLUSIVE`（H=2 内无 soup 抬升/成功事件）。

## §5 Evidence Boundaries（已证实 vs 相关性 vs 未证实）

| 命题 | 状态 |
|---|---|
| 2D 最小模型存在排序反转 | ✅ 已证实（~80%，参数稳健）|
| 真实 π0.5 存在排序反转 | ⚠️ 条件性（Entry 4/4，Persistent 0/4）|
| 反转作用于修正**幅度**（非次数）| ✅ 已证实（N_H 反转 0/4）|
| run_len 是反转的**因果变量** | ❌ 未证实（只是相关性，未隔离其他状态差异）|
| 「未来名义动作更贴近安全集」是机制 | ⚠️ 机制假说（未直接比较未来名义动作的约束违反程度）|
| 替代动作不损任务 | ❌ 未证实（INCONCLUSIVE）|
| 替代动作改善任务 | ❌ 未证实（INCONCLUSIVE）|

## §6 Final Verdict & Reopening Conditions

**暂停理由**：
1. 任务价值指标失效（eef-soup 距离在接近初期先增后减，且 H=2 无任务事件）；
2. 反转高度依赖 run_len，但 run_len 只是相关性，非已证因果；
3. 「幅度 vs 次数」的分离提示研究对象需重新定义。

**重启必要条件（三者缺一不可）**：
1. **可信任务价值指标**：区分「准备绕行 / 任务停滞 / 合法抓取 / 任务完成」，
   不能只用 EEF-目标距离；
2. **预注册状态条件**：按 CBF 连续激活长度、约束违反程度、任务阶段分层，
   不挑选已有结果有利的 Entry 状态；
3. **负担-价值联合比较**：仅在「未来 D_H 下降 + 任务不降 + 物理安全合格」同时
   成立时，才考虑学习未来负担预测器。

## §7 原始资产索引

| 资产 | 提交 |
|---|---|
| 假设文档 | `docs/research/m3_p0_intervention_burden_hypothesis.md`（9159c71）|
| 最小实验 | `experiments/m3_p0_future_burden_test.py`（9159c71）|
| 迁移实验 | `tools/m3_p1a_burden_transfer.py`（42ef5bb）|
| Gate1 探测 | `tools/m3_p1a_gate1_probe.py`（42ef5bb）|
| 重分析 | `tools/m3_p1a_stage_reanalysis.py`（b449119）|
| 原始结果 | `data/audit/m3_p0_*.json`、`m3_p1a_*.json` |

## §8 最终定位

阶段性研究命题：

$$\boxed{\text{State-Dependent Value of Non-Minimal Safety Interventions}}$$

核心发现：

$$\boxed{\begin{gathered}
\text{更大的当前安全修正}\\
\Downarrow\\
\text{可能产生更小的未来修正负担}\\
\Downarrow\\
\text{但收益具有状态依赖性，}\\
\text{真实任务价值仍有待验证}
\end{gathered}}$$

可作为未来论文的研究动机，但**不足以**单独支撑一篇具算法贡献与完整验证的 T-RO 论文。

## §9 与 CCA-ASF 的互补认识（长期研究问题）

- **CCA-ASF**：当前安全修正的**方向分配**，影响未来闭环物理安全；
- **M3**：当前安全修正的**幅度最优性**，不等于未来修正负担的最优性，且非最小
  修正的收益依赖状态。

二者共同指向：

$$\boxed{\text{如何评价一次安全干预对冻结 VLA 后续闭环行为的真实价值？}}$$

在形成可验证的任务价值定义之前，保留为研究问题，不急于包装成新算法。
