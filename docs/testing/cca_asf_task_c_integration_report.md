# CCA-ASF TASK C 集成报告

> 日期：2026-11-02
> 状态：真实 AEGIS QP 集成完成，验收通过。
> 前置：`cca_asf_runtime_manifest.md`（TASK A）、规格书 §11 TASK B。

---

## 0. 结论

CCA-ASF 已接入真实 `AegisSafetyLayer.filter()`。实现仅替换 QP 目标函数的二次项
（白化各向异性低秩更新），**完全保留** AEGIS 的约束、变量、求解器、OSC 参数与
gripper 语义。22 个测试通过，β=0 逐位等价、β>0 非零各向异性、约束一致、辅助
维度完整求解均验证。

## 1. 三个 filter_mode

| mode | 行为 |
|---|---|
| `euclidean` | 原始 AEGIS，目标 `min (u-u_ref)ᵀ W (u-u_ref)`，`W=diag([1/25]×6,1,1,1)`，完全不变 |
| `cca_asf` | 原始 AEGIS 约束 + 白化各向异性目标 `W+ΔP` |

## 2. 唯一的算法级修改

在 `filter()` 里，`cca_asf` 模式且已注入任务方向时：

```
d_y = [R1ᵀ d_p, d_ω, 0₃]                    # 白化 9D 任务方向
A_row = [a_u_v, a_u_omega, a_uz]            # 原 CBF 约束行
beta = conflict_beta(A_row, -10h, u_ref, d_y, beta_max)
ΔP = beta · (B d_y)(B d_y)ᵀ，B = diag([0.2]×6, 1,1,1)
W_used = W + ΔP
objective = minimize quad_form(u - u_ref, W_used)
```

约束、`u` 变量（9D 无 box 约束）、`cp.OSQP`、回退分支、z 方向演化、输出映射
（`out[:3]=0.2·R1·u_v` 等）**全部不动**。

## 3. 关键实现决策

1. **中心化二次形对齐**：cvxpy 的 `quad_form(x,P)` 是 `xᵀPx`（不含 1/2）。
   AEGIS 原目标就是 `quad_form(u-u_ref, W)`。CCA-ASF 同样用
   `quad_form(u-u_ref, W+ΔP)`，系数约定天然一致，β 的物理意义直接对应
   「任务方向上相对名义的惩罚强度」，无需处理 1/2 系数（规格书 §4.3 的
   `P=W, q=-Wz` 是含 1/2 的标准形式，接入 cvxpy 时统一为中心化二次形）。

2. **冲突门控在白化空间**：约束法向 `Ã = A_row · B⁻¹`，SVD 行空间投影，
   `ξ = ||P_row d_y||²`，`ρ = 4ξ(1-ξ)`，`β = β_max·ρ`。单约束时 `ρ = sin²(2θ)`
   （θ 为任务方向与白化约束法向夹角）。

3. **复用而非重写**：`map_dz_to_dy` / `compute_conflict_beta` / `B_DIAG` 直接
   从 `src/safety/cca_asf/metric.py` 导入，与 TASK B 单测同一实现。

## 4. 验收矩阵

| 测试 | 结果 |
|---|---|
| T9：β=0 与 euclidean 逐位等价 | ✅ atol=1e-8 |
| C-T1：约束冲突时 β>0 且输出偏离 | ✅ beta>0，Δ输出>1e-6 |
| C-T2：β>0 时解仍满足同一 CBF 约束 | ✅ lhs ≥ -1e-5 |
| C-T3：辅助维度完整求解 | ✅ feasible=True |
| C-T4：零/None 方向退化 | ✅ beta=0，输出不变 |

（TASK B 的 17 个数学/接口测试也全部保持通过，共 22 个。）

## 5. 尚未验证（TASK D 起）

- T8（policy 一致性）/ D1-T1（baseline regression）/ D1-T3（β=0 全路径等价）：
  需真实 π0.5 worker + MuJoCo rollout 才能验证（D1 接口逻辑已在
  `test_cca_asf_policy_interface.py` 用 fake policy 验证，但真实 worker 的
  chunk 一致性待 TASK D）；
- 真实 MuJoCo 下的 CBF 激活频率、动作改变、物理安全。

## 6. 工程产物

- `src/safety/aegis_adapted.py`：`filter_mode` / `set_task_direction` / 白化低秩目标。
- `tests/safety/test_cca_asf_integration.py`：5 个集成测试。
