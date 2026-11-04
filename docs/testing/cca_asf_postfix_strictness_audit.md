# CCA-ASF 严格性审计（实现 vs 规格逐条核对）

> 日期：2026-11-04
> 规格：`CCA_ASF_PostFix_Strict_Implementation_Spec_20261104.md`
> 代码基线：HEAD `b2226d9`（aegis_adapted.py 未在本轮改动）
> 结论：**实现不严格，发现 1 个严重协议缺陷 + 若干需记录的轻缺陷。**

---

## 1. 数学层（✅ 正确，逐条核对）

| 项 | 规格要求 | 实现 | 判定 |
|---|---|---|---|
| 白化低秩项 | `ΔW=β(B d_y)(B d_y)ᵀ`，禁字面 `β d_y d_yᵀ` | `metric.py:102-103` `Bd=B_diag*d_y; W0+β·outer(Bd,Bd)` | ✅ |
| `map_dz_to_dy` | `[R1ᵀ d_p, d_ω, 0_3]` 归一化 = `B J_t d_z` | `metric.py:40` `concat([R1.T@dz[:3], dz[3:6], zeros(3)])/n` | ✅ |
| 冲突门控 | 违反集→`A/B_diag`→行归一→SVD 行空间→`ξ→ρ=4ξ(1-ξ)→β` | `metric.py:69-88` | ✅ |
| 目标替换 | `cp.quad_form(u-u_ref, W+ΔW)`，无 1/2 | `aegis_adapted.py:486` | ✅ |
| β=0 退化 | 与 euclidean 目标逐位一致 | `if beta>0.0: W_used=W+ΔP`，否则 `W_used=W` | ✅ |
| 门控 (A,b) | 与 QP 约束一致 | `A_row=[a_u_v,a_u_omega,a_uz]`, `b=-QP_H_GAIN·h` 与 line 487 约束同 | ✅ |

## 2. 协议层（❌ 发现严重缺陷）

### 2.1 【严重】方向提取跨 replan 窗口（违反 §3.2）

**规格 §3.2 明确**：
> 设当前 act() 关联的未过滤 chunk 为 Z=[z_0,...,z_{K-1}]，当前消费索引 k_t。
> 不包含……未来尚未生成的 chunk……

**事实链**：
1. `action_horizon H = 10`（实测 `PnsgPolicy.H`）；
2. `policy.generate(z, obs)` 返回 `10×7` 完整 chunk；
3. `replan_steps = 5`；
4. 上游 AEGIS 正确截断：`action_plan.extend(action_chunk[:replan_steps])`（`third_party/aegis/main_aegis.py:294`），第 6-10 步丢弃，下一轮重新 infer；
5. 但 `tools/cca_asf_requick.py` 与 `tools/cca_asf_pilot.py`：

```python
chunk = np.clip(policy.generate(z0, po), -1, 1)   # 10×7，未截断！
...
d_z = chunk_direction(chunk[step % REPLAN:])       # 用了 chunk[step%5 : 10]
```

`chunk_direction(chunk[step%5:])` 在 step=0 时取 `chunk[0:10]`，包含第 6-10 步——
这些动作**下一轮 replan 会被丢弃并重新生成**，属于「未来尚未生成的 chunk」。

**污染量**：`chunk_direction` 用 eta 加权（eta=0.8），未来 5 步权重
`Σ_{j=5}^{9} eta^j ≈ 1.102`，当前窗口剩余权重 `Σ_{j=0}^{4} eta^j ≈ 3.362`，
未来动作占比约 **25%**，不可忽略。

**影响**：requick 与 pilot 的 CCA 方向 `d_z` 被未来动作污染 → β、ΔW、最终修正全被
影响。**requick 与 pilot 的 CCA 结果作废，需修复后重跑。**

### 2.2 【轻微】u_ref 与 u_nom 未分别命名（违反 §3.1）

规格 §3.1：u_ref（QP 目标中心）与 u_nom（门控名义输入）即便数值相同也必须分别
命名。当前 `aegis_adapted.py:453` 的 `u_ref_vec` 同时充当两者。

### 2.3 【需记录】门控只含主障碍物 1 行

`aegis_adapted.py:463` 的 `A_row` 只取主障碍物约束 1 行，不含 extra_obstacles 行。
当前 CCA 实验（S01 mug，无 extra）自洽，但规格 §3.4 要求「记录选中行数」。

### 2.4 【需记录】门控阈值用默认值未记录

`compute_conflict_beta(..., violation_tol=1e-7, rank_tol=1e-8)` 用默认值，未显式
传入或写进 manifest。

## 3. 指标层（✅ 已修正）

`tools/cca_asf_requick.py` 已改为 TSR_native / CAR_native / SSR_native 分离（规格 §7）。

## 4. 修复方案

**缺陷 2.1 修复**（必须）：requick/pilot 里截断 chunk 到 replan_steps：

```python
chunk = np.clip(policy.generate(z0, po), -1, 1)[:REPLAN]   # 截断到 5 步
```

修复后 `chunk_direction(chunk[step%REPLAN:])` 只在当前窗口内（`chunk[step%5 : 5]`）。

**缺陷 2.2 修复**（建议）：`aegis_adapted.py` 内显式拆分 `u_nom` / `u_ref` 两个变量。

**缺陷 2.3/2.4**（记录）：写入 runtime manifest（`n_gate_rows=1`、`violation_tol`、
`rank_tol`）。

## 5. 对已有结果的处置

| 产物 | 状态 |
|---|---|
| `data/audit/cca_asf_requick.json` | ⚠️ CCA 分支受跨窗口污染，作废重跑 |
| `data/audit/cca_asf_pilot_results.json` | ⚠️ 同上，作废重跑 |
| `docs/testing/cca_asf_postfix_paired_evaluation.md` | ⚠️ 结论需加注「受缺陷 2.1 污染」 |

> euclidean 分支不用方向提取，不受缺陷 2.1 影响；但 requick/pilot 是配对对照，
> 单边污染即可使配对 Δ 失效，故整体作废。
