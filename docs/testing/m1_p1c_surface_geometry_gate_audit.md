# M1-P1c：Surface-Geometry Candidate Gate 审计

> 日期：2026-11-01
> 状态：G0 几何清单 ✅ / G1 表面间隙 ✅ / G2 方向响应 PASS（带非光滑性保留）。
> 前置：`m1_p1_design_gate_audit.md`（v3 的中心距离生成器失败）。

---

## 0. 结论

> 用 MuJoCo 原生碰撞几何间隙（凸 mesh ↔ box，经独立查询模型 + `geom_margin`）
> 构造的动作空间有限差分梯度，在 Step54 上使 link6–mug 最小表面间隙
> **+5.79mm**（18.93→24.72mm），无接触、历史 CAR 合格。G2 核心验收通过，
> 但最近 geom 对从 g203 切换到 g205，存在 min 函数的非光滑性，需在 G3 中继续观察。

这标志着 M1 第一次拥有了**由真实碰撞几何自动生成**（而非手工指定固定方向）的
避让候选，且方向经真实 OSC 响应验证有效。

## 1. G0：碰撞几何清单（决定性）

| 对象 | 碰撞几何 | 参数 |
|---|---|---|
| link6 | 单个**凸 mesh**（geom 64 `robot0_g47_vis`）| 102 顶点 / 200 面，rbound=**132.9mm** |
| mug | 5 个**薄盒** | g202 把手 g7；g203–g206 四墙 g8–g11 |

**首次接触对**（step59 substep 内，同进程真实动力学）：`g204`（mug 墙 g9）↔ `g64`
（link6 mesh），substep22→24 穿透递进 −0.028→−0.399mm。

> 关键：link6 是凸 mesh（非胶囊），mug 是薄盒（非球），故解析胶囊/球公式不适用；
> 采用 MuJoCo 原生凸 mesh↔盒碰撞检测。

## 2. G1：表面间隙查询（`geom_margin` + `mj_collision`）

MuJoCo 2.3.7 无 `mj_geomDistance`，但通过**独立查询模型**（`deepcopy`）设大 margin
后跑 `mj_collision`，可得到 link6–mug 的最小带符号表面间隙。

**三项核对结果**：

1. **`deepcopy` 独立查询模型可行**（ngeom=220 完整复制），margin 只设在查询模型，
   真实 model 的 margin 恒为 0，不污染真实动力学。
2. **margin 语义**：`contact.dist` = 真实带符号距离（正=分离，负=穿透），margin
   **只扩大宽相位检测范围，不改变已检测 geom 对的 dist 数值**（margin=0/10/20mm
   时 g204 均为 −1.2246mm）。
3. **同状态一致性**：同一 qpos 下 margin=0 与 margin>0 的 dist 一致；`mj_step`
   内部记录（−1.07mm，积分前）与碰撞检测（−1.22mm，积分后）差 0.15mm，属
   「substep 内积分前/后几何演化」，非查询口径误差。G2 统一用「积分后 qpos」。

**G0b(−0.399mm) 与 G1(−1.72mm) 差异**：跨进程 GPU 非确定（π0.5 推理非位精确）+ 采样
时点差异，非 margin 污染。

**误差边界**：同一 qpos 重复查询确定（<1e-6mm）；margin 不引入 dist 偏差；积分时点
口径统一后无系统偏差。Δf 显著性判断阈值取独立重复的数值噪声（远小于本次 5.79mm）。

**措辞**：此距离为「MuJoCo 原生碰撞几何间隙估计」，非具数学严格误差界的精确距离
（mesh 接触位置存在 MuJoCo 近似）。

## 3. G2：Surface-Gap 梯度方向响应

$$
f(a)=d_{surf}(\mathcal F_{OSC}(s_{54},a)),\quad
d_{surf}=\min_{j\in\{g202..g206\}} d_{MuJoCo}(g64,j)
$$

有限差分（实际执行值作分母，6 个 pose 维，gripper 维不差分）：

$$ g_i = \frac{f(a_{nom}+h e_i)-f(a_{nom}-h e_i)}{2h},\quad h=0.05 $$

$$ a_{new}=\Pi_\mathcal{U}\left(a_{nom}+\delta\frac{g}{\|g\|}\right),\quad \delta=0.2 $$

| 指标 | 名义 | 新候选 |
|---|---|---|
| f(a) 末端间隙 | 18.929mm | **24.723mm** |
| Δf | — | **+5.794mm** |
| 最小 substep 间隙 | 18.929mm @substep24 | 24.723mm @substep24 |
| 最近 geom 对 | (g203, g64) | (g205, g64) ⚠️切换 |
| 原始接触 | 无 | 无 |
| mug 历史 Dmax | 0 | 0 |
| 差分分母（实际）| [0.1]×6（无裁剪）| — |

**判决**：Δf>0、无接触、CAR 合格 → **G2 PASS**。

**保留项（非光滑性）**：最近 geom 对从 g203 切换到 g205，说明 min 函数的 argmin
发生切换，梯度在该点不光滑。但 min 值单调改善（18.93→24.72），即**所有 mug 部位
的最小间隙整体增大**，方向本身有效。未因此调整梯度参数。

## 4. 判决表（用户预设）

| 结果 | 本实验 | 后续 |
|---|---|---|
| Δf>0、无接触、CAR 合格 | ✅ 成立 | 进入 H=10 的 G3 |
| 最近 geom 对切换 | ⚠️ 观察到 | 报告非光滑性，不调参，G3 继续观察 |

## 5. 关键验证：Surface-gap improvement ⟹ 物理安全避让？

G2 只在**单步**（step54→55）验证了方向有效。首次接触在 step59（还需 5 步），
单步表面间隙改善**不足以**证明 H=10 内能避免接触。真正的检验在 G3：
- H=10 闭环，表面梯度候选在避让阶段（step54–55）后接续冻结 π0.5；
- 检查是否避免 step59 的 link6–mug 接触 + 历史 CAR 合格 + 任务进度。

## 6. 工程产物

- `tools/m1_p1c_g0_geom.py` / `tools/m1_p1c_g0b_mesh.py`：几何清单 + substep 接触对。
- `tools/m1_p1c_g2_consistency.py`：同状态一致性 + margin 语义核对。
- `tools/m1_p1c_surface_gradient_gate.py`：G2 表面梯度方向响应。
- `data/audit/m1_p1c_g0_geom_inventory.json` / `m1_p1c_g0b_mesh_substep.json` / `m1_p1c_g2_surface_gradient.json`。
