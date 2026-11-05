# G2 病例选择判据（S1-S4，预注册冻结）

> 日期：2026-11-05。在 B0 病例收集**之后**、病例表**冻结之前**预注册。
> 判据是**可操作**的，由 B0 每步日志字段计算，不凭主观印象。

## 字段来源（g2_b0_episodes.jsonl 每 step）

- `h`：AEGIS CBF 值（椭圆归一化距离，h>0 安全，越小越接近约束边界）
- `intervened`：本步是否被 filter 修正
- `mug_disp_mm`：受保护物体（mug）累计最大 L1 位移
- `eef_pos` / `mug_pos`：末端 / mug 世界系位置
- `success` / `steps`：任务成功 / 步数
- `forbidden_steps`：robot↔protected/support/distractor 接触步数
- `task_final_disp_mm`：任务物体（soup/cream/tomato）最终位移

## S1 停滞（CONTROL_DEADLOCK 候选）

**定义**：CBF 长期活跃且末端进展停滞，但名义动作有任务进展线索。

**可操作判据（全部满足）**：
1. `success == False` 且 `steps == MAX_STEPS`（550，跑满超时）；
2. 后 30% 步（step ≥ 385）中 `intervened=True` 的比例 ≥ 0.5（CBF 持续活跃）；
3. 末端位置在后 100 步内的位移 < 5cm（`eef_pos` 首尾差 L1 < 0.05）；
4. 名义动作非零（排除「VLA 本身停住」：后 30% 步 nominal_7d[:3] 的平均范数 > 0.05）。

## S2 振荡（跨周期修正方向交替）

**定义**：CBF 修正方向交替/返回，目标或物体状态变化异常（疑似跨周期参考不一致）。

**可操作判据（满足 1+2 或 1+3）**：
1. 存在 ≥ 3 次「修正方向翻转」：相邻 intervened 步的 `exec_7d[:3] - nominal_7d[:3]`
   方向与上一次翻转（点积 < 0），且翻转间隔 < 20 步；
2. `success == False`（振荡未收敛到成功）；
3. 或 `max_mug_disp_mm > 1`（振荡伴随碰撞）。

## S3 抓取失败 / 初始惯性推走（VLA_LIMITATION 候选）

**定义**：π0.5 本身抓取失败，或受保护物体被初始惯性推走（控制器无法挽回）。

**可操作判据（满足任一）**：
1. 前 10 步（空转后、任务刚开始）`mug_disp_mm > 1` 且后续 `intervened` 比例 < 0.3
   （初始惯性推走，非 CBF 干预导致）；
2. `success == False` 且后 30% 步 `intervened` 比例 < 0.3 且名义动作范数均值 < 0.05
   （VLA 本身无有效动作，不是被 CBF 卡住）。

## S4 间接碰撞 / 任务完整性违规（审计限制）

**定义**：已存在任务完整性违规或间接碰撞（机器人推底座带动 mug）。

**可操作判据（满足任一）**：
1. `forbidden_steps > 0`（robot↔protected/support/distractor 接触）；
2. `task_final_disp_mm` 任一任务物体在**非抓取**状态下位移 > 5mm（但成功，疑似推入）；
3. `max_mug_disp_mm > 1` 但无对应 `intervened` 记录（碰撞非 CBF 可防）。

## 归类纪律（任务书 §6）

- **保留无法归类的案例**：不满足任何 S1-S4 的失败案例，标 `UNCLASSIFIED`，不强行归类；
- **负对照**：成功案例（`success=True`）单独列出，作为「正常」参照；
- 一个案例可同时满足多类，按优先级 S1 > S2 > S3 > S4 标记主类，其余记副类；
- 选择规则、全部候选、排除原因写入冻结病例表 `G2_CASE_SELECTION.csv`。

## 判据的边界（诚实）

- 这些是**探索性启发式判据**，用于机制鉴别，不是统计检验；
- 末端「进展停滞」用 eef_pos 位移近似，未用任务物体抓取阶段精确定义；
- `forbidden_steps > 0` 是保守口径（含未分类接触），可能高估 S4。
