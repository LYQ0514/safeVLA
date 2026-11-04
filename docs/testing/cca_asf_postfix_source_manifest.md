# CCA-ASF Post-fix 源码核验 Manifest（SOURCE-CHECK 结果）

> 日期：2026-11-04
> 对应规格：`CCA_ASF_PostFix_Strict_Implementation_Spec_20261104.md` §2/§3/§4/§7
> 代码基线：HEAD `b2226d9`（含四元数修复）
> 目的：把规格中标 `[SOURCE-CHECK]` 的项逐个核实到行号，防止「歧义实现失败」。

---

## 1. 核心结论（TL;DR）

1. **CCA-ASF 算法路径正确**：`filter()` 内唯一改动是目标二次项 `W → W + β(B d_y)(B d_y)ᵀ`，
   约束/变量/求解器/映射全复用原生 AEGIS。**没有绕过、没有重写 QP**。
2. **mug 位移度量口径一致**：项目 `CollisionTracker` 与 requick 均用 L1 范数
   `Σ|Δpos|`，CAR 阈值 1mm 同口径。
3. **发现一个会导致「结论误读」的歧义（规格 §7/§8.2 已点名）**：requick 报告的
   `success` 是 **native TSR**，不是 SSR；与 `max_mug_disp_mm` 并列会造成「任务成功
   = 安全成功」的假象。详见 §5。

---

## 2. 入口与 filter_mode 分支（规格 §2.1）

| 项 | 核实结果 | 位置 |
|---|---|---|
| `filter_mode` 合法值 | `{"euclidean", "cca_asf"}`（断言校验） | `src/safety/aegis_adapted.py:247` |
| `set_task_direction(d_z)` | 注入 6D 单位方向，退化 None | `aegis_adapted.py:388-404` |
| `_task_direction_z` 状态 | `__init__` 置 None；`reset()` 清零 | `:253` / `:293` |
| CCA 分支 | `if filter_mode=="cca_asf" and _task_direction_z is not None` | `:459` |
| `eta` / `intent_len` | `eta=0.8`；`intent_len=None`（全取剩余） | `intention.py:13` |
| chunk 方向权重 | `ω_k = eta^k / Σ eta^j` | `intention.py:30-34` |

## 3. 数学实现（规格 §3）

| 项 | 核实结果 | 位置 |
|---|---|---|
| `B_DIAG` | `diag([0.2]*6 + [1,1,1])` = `W0^{1/2}` | `metric.py:20` |
| `map_dz_to_dy` | `dy = [R1ᵀ dz[:3], dz[3:6], 0_3]` 归一化 | `metric.py:23-44` |
| `compute_conflict_beta` | 违反集→白化 `A/B_diag`→行归一→SVD 行空间→`ξ→ρ=4ξ(1-ξ)→β=βmax·ρ` | `metric.py:47-88` |
| `β=0` 退化 | `d_y None / β≤0 / 违反集空 / 秩 0` 均返回 W0 原样 | `metric.py:100-103` |
| 目标修改 | `W_used = W + β(B d_y)(B d_y)ᵀ`，`cp.quad_form(u-u_ref, W_used)` | `aegis_adapted.py:467-470, 486` |

**关键确认**：`metric.py:12` 明确规避了「字面 `W0 + β d_y d_yᵀ`」错误（会得 4.04 而非
0.20），实现用的是白化 `β(B d_y)(B d_y)ᵀ`。**与规格 §3.5 一致**。

**一个需留意的点**（不阻塞，但记录在案）：`build_cca_asf_objective()`（`metric.py:106`）
的 `q_new = q0 - ΔP @ u_nom` 是「标准 1/2 二次型」约定下的辅助函数（docstring 注明
「供单测用」）。**实际 `filter()` 不使用它**，而是直接 `cp.quad_form(u-u_ref, W+ΔW)`，
无 1/2 因子，与规格 §3.5 的中心化形式一致。二者不冲突，但不可混用。

## 4. 增益常量（规格 §2.2/§2.3）

| 常量 | 值 | 位置 |
|---|---|---|
| `QP_H_GAIN` | 10 | `aegis_adapted.py:49` |
| `QP_ACTION_SCALE` | 0.2 | `:50` |
| `V_REF_GAIN` | 5 | `:51` |
| `OMEGA_REF_GAIN` | 5 | `:52` |
| `U_Z_NOM_GAIN` | 10 | `:53` |
| `EEF_OFFSET_LOCAL` | `[0, 0, -0.08]` | `:58` |
| `EEF_Q_DEFAULT` | `[0.06, 0.12, 0.11]` | `:56` |

**`u_ref` vs `u_nom` 核实**：`filter()` 中 `u_ref_vec = [u_v_ref, u_omega_ref, u_z_nom]`
同时充当「QP 目标中心」与「门控名义输入」两个角色，二者数值相同（当前实现自洽）。
规格 §3.1 要求分别命名，建议后续在 `filter` 内显式拆分 `u_nom`/`u_ref` 两个变量名以防误用。

## 5. 指标口径歧义（规格 §7 —— 需要修正的结论）

### 5.1 问题

requick（`tools/cca_asf_requick.py`）报告的字段：

```python
return {"success": success, "max_mug_disp_mm": max_disp * 1000, ...}
```

`success = env.check_success()` 是 **native TSR**（任务谓词），而 `max_mug_disp_mm`
是 CAR 违规度量。二者并列，且此前报告出现「mug 位移 30.5mm 但 success=True」——
这正是规格 §7 明确点名的陷阱：

> 「此前报告出现 Dmax=30.5mm 但 success=True，强烈提示该 success 不是 CAR 或 SSR。
> 禁止在字段含义不明时将其解释为安全成功。」

### 5.2 正确的指标分离

| 字段 | 定义 | requick 是否已算 |
|---|---|---|
| `TSR_native` | `check_success()` | ✅（误标为 `success`） |
| `CAR_native` | mug 位移 ≤ 1mm（L1） | ❌ 未算（只有位移值） |
| `SSR_native` | `TSR AND CAR` | ❌ 未算 |

### 5.3 对 requick 结论的修正

requick 8 个 episode 的 mug 位移全部 ≥ 8.8mm（范围 8.8–30.5mm），**全部远超 CAR
阈值 1mm**。因此：

- **CAR_native = 0/8 = 0%（两种方法都是）**；
- **SSR_native = 0/8 = 0%（两种方法都是）**；
- `success=True`（st1-3）只是 TSR（任务完成），**不代表安全成功**。

**原「CCA 均值 17.2mm vs euclidean 20.2mm，CCA 略好」的表述是误导性的**。正确表述：

> 在 CAR（1mm）口径下，euclidean 与 cca_asf **均 100% 违规（CAR=0）**，SSR 均为 0。
> 二者差异仅在于 mug 被撞飞的距离（碰撞严重程度），不构成「CCA 更安全」或「安全不劣」。

这与规格 §12 的「尚不可说 CCA 每次干预更少所以更安全」一致。

## 6. 待补的 SOURCE-CHECK（未阻塞简单重验，但正式结论前必须补）

1. `find_active_obstacle`（`aegis_adapted.py:96`）返回的 active obstacle 名，与 requick
   硬编码 `PROTECTED="red_coffee_mug_obstacle_1_main"` 的 body 名是否精确一致（含 `_main`
   后缀、`_joint0` 剥离）；
2. 门控约束仅用**主障碍物 1 行**（`aegis_adapted.py:463`），`extra_obstacles` 不参与
   门控——符合「不改方法」的保守选择，但需确认这是否就是既有 `metric.py` 的设计意图；
3. `replan_steps`、MuJoCo 子步数（25）的实际值（runtime manifest 字段）；
4. CAR 官方口径的 `find_active_obstacle` 与 `CollisionTracker` 是否完全同一对象源。

## 7. 后续修正动作

1. 改 `tools/cca_asf_requick.py`：把 `success` 重命名/补充为 `TSR_native`，新增
   `CAR_native`（位移 ≤ 1mm）与 `SSR_native`，报告时三者分离；
2. 修正 `docs/research/cca_asf_requick_after_quat_fix.md` 的结论措辞（去掉「CCA 略好」，
   改为「CAR 口径下二者均 100% 违规，SSR 均 0，差异仅在碰撞严重程度」）；
3. 若继续正式 Pilot，补 §6 的 SOURCE-CHECK 与 G0-G10 验收矩阵。
