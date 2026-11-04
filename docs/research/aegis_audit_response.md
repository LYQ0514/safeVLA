# AEGIS 复现审计响应与修正记录

> 日期：2026-11-04。响应第三方审计报告（「AEGIS / SafeLIBERO 严格复现审计报告」）。
> 立场：**接受审计，不再辩解**。审计指出的问题真实存在，逐一核实并修正。

---

## 0. 审计核心判决（接受）

> 当前是 AEGIS CBF-QP 控制层的**近似/部分复现**（oracle obstacle perception），
> 不是 AEGIS 端到端严格复现；尚不能认定修复后完成了可审计的 1600-episode 结果复现。

**我承认**：之前我声称「过滤器数学已严格复现」是正确的（精确对照差异 0.00），但
**「完整 AEGIS 复现」还远未完成**——感知、几何、评测统计、对照公平性都存在问题。

## 1. 已修复的问题（本次响应）

| # | 问题 | 修复 | 状态 |
|---|---|---|---|
| P0-D | baseline 对比 replan-steps 不一致（vanilla None vs AEGIS 5） | `run_eval.py` 默认改 5 | ✅ |
| P1-A | 碰撞后 max_displacement 停止更新 | `CollisionTracker.update` 改为继续追踪 | ✅ |
| P1-B | QP 计数未按 episode 重置 | `AegisSafetyLayer.reset` 清零计数器 | ✅ |
| P1-C | 干预计数只比 u_v 漏旋转 | 改为比 u_v + u_omega | ✅ |

（另有更早已修复的四元数 bug、p2 固定、num_steps_wait、策略输入分辨率，见
`aegis_reproduction_audit.md` / `aegis_strict_reproduction_gaps.md`）

## 2. 未修复的问题（需继续，不能声称完成）

### P0-A 感知链缺失（已知，不可本机复现）

上游 VLM(GLM-4.5V) + GroundingDINO + RGB-D 点云 + MVEE。本机 api_key 为空，
**无法复现感知链**。当前是 oracle（MuJoCo 真值）。结论必须表述为
「AEGIS-QP (Oracle Perception)」，不能称「完整 AEGIS 复现」。

### P0-B 真值椭球几何不具可靠包围性（**新发现，严重，待修**）

`obstacle_ellipsoid` 用「body 原点 + 所有 geom 的 geom_size 逐轴 max」构造椭球，
**忽略 geom 相对位移、局部旋转、不同 type 的 size 语义、mesh 缩放因子**。

实测 mug 有 12 个 geom，相对位置从 -0.047 到 +0.054，相对旋转各不相同。当前椭球
中心 = body 原点，与真实几何中心不重合，**包围性无法保证**。

**待修**：需逐个障碍物用世界系碰撞几何（geom 顶点 / AABB）验证覆盖率，或用
「全体几何顶点拟合包含性椭球」。这是审计第 3 条「几何有效性」的核心。

### P0-E 模型预处理/动作截断未对拍

`src/policies.py` 的 `obs_to_policy_obs` 在 `pi05_libero_runner`（压缩包未含），
无法核实图像翻转 / resize / 8D state / prompt / dtype 与官方一致；
`clip(action,-1,1)` 官方未显式做。需保存同初始状态下的原始输入与前 N 个名义动作块对拍。

### P1-D 文档与输出语义不一致

- ETS 注释「只算成功」vs 实际 `ets_mode='all'`（含失败/超时）；
- `CAR>5mm=` 标签实际是 CAR_severe，易反向误读；
- 多份审计文档（`aegis_reproduction_audit.md`、`aegis_strict_reproduction_gaps.md`）
  的 p2/num_steps_wait 陈述已过期，需标注并统一为单一权威 manifest。

### P2-A 启动轨迹小差异

上游在 wait 前初始化 p1/R1、wait 后按先前 p1 初始化 z_fixed；我的实现 wait 后才
`layer.reset(obs)`。更合理但不完全同构，需首帧对拍。

### P2-B 续跑污染 + 可追溯性不足

`--resume` 只按 (suite,level,task,episode) 跳过，无代码 SHA / 模型 hash / 配置校验。
历史存在被 bug 污染的实验目录。**禁止复用旧目录续跑新基线**。

## 3. 已停止的评测（必须重做）

本轮已停止两个后台评测（`pi05_aligned`、`aegis_aligned`），因为它们：
1. vanilla 用了 open-loop（replan=None），与 AEGIS closed-loop 不可比（P0-D）；
2. CollisionTracker 有 max_displacement 停止更新 bug（P1-A）；
3. QP 计数未重置（P1-B）。

**这些目录的结果作废，不得引用。**

## 4. 对齐后正式复现的最低验收标准（采纳审计第 4 节）

1. **协议身份锁定**：上游 Git SHA、适配 Git SHA、SafeLIBERO init hash、π0.5 checkpoint
   hash、Python/CVXPy/OSQP/MuJoCo/LIBERO 版本、全部 CLI 参数与 seed；
2. **数学对拍**：相同输入下，官方代码 vs 适配代码逐步比较 h / QP 系数 / u_ref / u_sol /
   动作 / 下一帧 z_fixed，≥100 组真实轨迹状态；
3. **几何有效性**：逐障碍物验证 oracle 椭球对世界系碰撞几何的包围性，oracle 与
   视觉拟合严格分为不同方法变体（**当前 P0-B 未修，此项未达标**）；
4. **公平对照**：baseline 与 AEGIS 均 resolution=1024、policy-resize=224、wait=20、
   replan=5、相同权重/init/seed/步数预算；
5. **统计正确性**：修正 max displacement 与逐 episode QP 计数（已修 P1-A/P1-B）；
6. **完整评测**：全新目录 1600 episode，逐条唯一键 + 失败日志 + 8 场景分组 + 配对 CI；
7. **结论边界**：oracle 感知时称「AEGIS-QP (Oracle Perception)」；只有重建 VLM+
   GroundingDINO+depth+MVEE 链路并完成对拍，才可主张端到端复现。

## 5. 当前准确表述（采纳审计第 5 节）

> 已移植 AEGIS CBF-QP 内核并实现 oracle-perception 评测框架；
> 正在完成几何、协议、统计与全量实验复核。

**不能在论文中声称「已严格复现 AEGIS 论文」。**

## 6. 下一步（按优先级）

1. **修 P0-B**（几何包围性）：重写 `obstacle_ellipsoid`，用世界系 geom 顶点/AABB
   构造包含性椭球，并验证覆盖（审计第 3 条）；
2. **修 P1-D**：统一 ETS 语义注释、CAR 标签、过期文档标注；
3. **写单一权威 manifest**：合并过期的多份审计文档；
4. **数学对拍**：≥100 组状态，官方 vs 适配逐项比较（审计第 2 条）——此前只做了
   单状态对拍（差异 0.00），需扩展到多状态；
5. **协议身份锁定 + 完整评测**：修完上述后，用全新目录重跑 1600 episode（AEGIS 与
   baseline 均 replan=5 等对齐参数），并保存 manifest。

## 7. 诚实声明

审计是专业且正确的。我之前的工作存在两个层面的不足：
1. **层面一（已承认）**：四元数 bug、分辨率、p2、wait 等——已修复；
2. **层面二（本轮审计新暴露）**：P0-B 几何包围性、P0-D 对照公平性、P1-A/P1-B 统计
   bug——其中 P0-B 和 P0-D 是**我之前完全没意识到的根本性问题**，直接导致「即使过滤器
   数学一致，评测结果仍不可信」。

在 P0-B、P0-D、P1-A、P1-B 全部修复并通过审计验收标准前，**不启动任何新的 1600-episode
评测，也不引用任何历史 AEGIS 定量结论**。
