# 复现边界（reproduction_boundary）

> 日期：2026-11-05。G0 审计产物。回答「什么被复现、什么没被复现、边界在哪」。

## 1. 命名：AEGIS-Oracle vs AEGIS-Perception（必须严格分开）

| 变体 | 障碍物椭球来源 | 状态 |
|---|---|---|
| **AEGIS-QP (Oracle Perception)** | 真值 `find_active_obstacle` + 碰撞几何顶点拟合 MVEE | ✅ 已跑通，本 STG 实验的基线 B0 |
| **AEGIS (Perception)** | VLM(GLM-4.5V) + GroundingDINO + 深度点云 + MVEE | ⚠️ 已跑通，但**非端到端严格复现**（见 §2） |

**结论**：`AEGIS-Oracle` 与论文的视觉感知 AEGIS 是**两个不同方法**，必须分开命名。
本 STG 实验（B0-B3）的基线是 **AEGIS-QP (Oracle Perception)**，不是视觉感知 AEGIS。

## 2. 与上游 AEGIS 的三处实现差异（非严格复现）

| # | 差异点 | 上游 | 本仓库 |
|---|---|---|---|
| 1 | VLM 客户端 | `from zai import ZhipuAiClient`（私有封装） | 官方 `zhipuai.ZhipuAI`（pypi 的 zai 是空包） |
| 2 | GroundingDINO | 上游自带加载 | 本地 `groundingdino_loader.py`（BERT + 权重 module. 前缀剥离） |
| 3 | oracle 椭球 | 上游 `fit_ellipse`（对 VLM 点云） | 碰撞几何顶点拟合 MVEE（+1.001 保守缩放，保证包围性） |

## 3. 已知修复（本仓库相对最早版本）

| 修复 | 影响 | 验证 |
|---|---|---|
| 四元数 xyzw 直通 | p1 偏移 16cm → 0.036mm | geometry_audit.csv PASS |
| 几何 MVEE 包围性 | 障碍物椭球未包住碰撞几何 → 0 越界 | ellipsoid_coverage_verify |
| 并发写临时文件 | 图片损坏 crash → 进程唯一路径 | 串行 runner |
| replan_steps 对齐上游（5） | baseline vs AEGIS 对照公平 | — |

## 4. 当前无权的认定

1. **无权认定「原版 AEGIS 已严格复现」**：Perception 变体有三处实现差异（§2），
   VLM 识别正确率未审计（summary 的 obstacle 字段 = 真值 active_obstacle，非 VLM 结果）；
2. **SafeLIBERO fork 未锁定上游 commit SHA**：vendored 本机，需补 init states hash 与
   上游 THU-RCSCT/vlsa-aegis 的确切 commit；
3. **`full action space` 口径**（含旋转 u_omega），对标官方项目页（68.9%），
   不是论文 Table 1 的 Translational Only（77.85%）。

## 5. STG 实验的边界（本任务书 §0 硬约束）

- 只允许：原 QP 目标加软引导项 + 外部小状态参考管理器；
- 禁止：在线多候选搜索、多分支仿真、额外 VLA 查询、重训练、改 CBF 几何/阈值、
  重写 3D/6D QP、把 proxy margin 当 certified barrier；
- 硬在线预算：每控制周期 ≤1 次 VLA 获取、≤1 次 QP。
