# AEGIS 严格复现的可行性结论与 A 方案执行路径

> 日期：2026-11-04。回答「A 方案（彻底对齐偏差）怎么执行」，以及「哪些能严格复现、
> 哪些不能」。

---

## 1. 关键区分：过滤器算法 vs 障碍物感知

AEGIS 分两部分，它们的「严格复现」可行性不同：

| 部分 | 内容 | 能否严格复现 |
|---|---|---|
| **CBF-QP 过滤器** | `compute_h_coeffs_3d`、QP 求解、z_fixed 演化、动作反映射 | ✅ **能**（纯数学，无外部依赖） |
| **障碍物感知** | VLM 命名 + GroundingDINO 检测 + 深度点云 + MVEE 拟合 | ❌ **不能**（GLM-4.5V 需 api_key，本机空） |

**结论**：A 方案的目标应修正为——**严格复现「过滤器算法」的全部数学，障碍物感知用
「明确的 oracle 替代」并完整记录差异**，而不是追求「感知也严格复现」（本机不可行）。

## 2. 过滤器算法的严格复现清单（可行，必须做）

| # | 维度 | 上游 | 目标 | 现状 |
|---|---|---|---|---|
| 1 | 四元数约定 | `R.from_quat`(xyzw) | xyzw 直通 | ✅ 已修复 |
| 2 | p2 固定 | fit_ellipse 一次，episode 内不更新 | p2 固定 | ❌ 当前每步刷新，需改 |
| 3 | num_steps_wait | 20 | 20 | ❌ 当前 10，需改 |
| 4 | Q1 选择规则 | orange juice/milk/soup → 大椭球 | 一致 | ✅ 已一致 |
| 5 | QP 数学（W/a_v/u_z_nom/QP_H_GAIN） | 见 main_aegis.py 304-354 | 一致 | ✅ 已一致 |
| 6 | z_fixed 演化 | dz=(I-zz^T)@u_z，dt=0.05 | 一致 | ✅ 已一致 |
| 7 | 动作反映射 | out[:3]=0.2*R1@u_v 等 | 一致 | ✅ 已一致 |
| 8 | 不可行回退 | u=action（原始动作） | 一致 | ✅ 已一致 |

**策略输入分辨率**（上一轮发现的问题）：
- 上游渲染 1024 → `resize_with_pad` 224 → 喂 π0.5；
- openpi 内部 `preprocess_observation` 也会 `resize_with_pad` 到 224；
- 两者最终都到 224，但「谁做 resize」「用 resize 还是 resize_with_pad」需对齐。
- **结论**：渲染分辨率对齐上游（1024），策略输入路径对齐上游（渲染 1024 → 显式
  resize_with_pad 224 → 喂策略），让 openpi 内部不触发二次 resize。

## 3. 障碍物感知的 oracle 替代（不可严格复现，需明确）

上游感知链（VLM+GroundingDINO+MVEE）无法复现。替代方案：

| 上游 | oracle 替代 | 差异说明 |
|---|---|---|
| VLM 命名 obstacle | `find_active_obstacle`（工作空间内第一个 obstacle joint） | 上游选「最可能被碰」的，oracle 选「第一个」 |
| GroundingDINO 点云 | 无 | — |
| MVEE 拟合 p2/R2/Q2 | `obstacle_ellipsoid`（MuJoCo geom 真值） | 真值椭球 vs 视觉拟合椭球 |

**诚实声明**：oracle 替代使「过滤器数学」严格复现，但「感知」不是。评测结果应表述为
「AEGIS 过滤器（oracle 感知）在本机的表现」，不能声称「AEGIS 论文复现」。

## 4. A 方案执行顺序

1. **对齐过滤器参数**（纯代码，快）：
   - p2 固定（reset 后不刷新）；
   - num_steps_wait=20；
   - 渲染 1024 + 显式 resize_with_pad 224（策略输入路径对齐上游）；
2. **跑对齐后的 AEGIS baseline**（全量 1600，oracle 感知，明确标注）；
3. **跑对齐后的 vanilla baseline**（同样 1024 渲染 + 224 策略输入，与 AEGIS 同配置，
   才能公平对比）；
4. **对比 + 归因**：在「过滤器数学严格复现 + 感知 oracle 化 + 双方同配置」的基础上，
   重新回答「AEGIS 的真实缺陷」。

## 5. 诚实边界

- 即使做完 A 方案，「感知」仍是 oracle 化，不是 AEGIS 论文的完整复现；
- 但「过滤器算法」能严格复现，这是 AEGIS 的**核心贡献**（CBF-QP 安全过滤），感知只是
  其工程依赖；
- 因此 A 方案能回答的科学问题是：「**AEGIS 的 CBF-QP 过滤器算法**（用真值感知）在
  SafeLIBERO 上的真实安全/任务表现」，这是可发表、可复现的严格口径。
