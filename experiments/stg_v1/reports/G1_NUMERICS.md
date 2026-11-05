# G1_NUMERICS：STG 数学与接口测试

> 日期：2026-11-05。门禁 G1。判定：**PASS**

## 0. 判定

| G1 验收项（任务书 §5 + 用户指令） | 结论 |
|---|---|
| 1. λ=0 时与 B0 原始 QP 输出等价 | ✅ PASS |
| 2. B1/B2/B3 硬安全可行集与 B0 完全一致 | ✅ PASS |
| 3. 切向几何正交性 + 退化 + 方向切换 | ✅ PASS |
| 4. Governor reset 不串 episode | ✅ PASS |
| 5. 引导项映射 9D QP + Hessian PSD | ✅ PASS |
| 6. 不可行 QP / 异常输入 / solver 失败一致性 | ✅ PASS |
| 7. 模块 CPU 延迟 + QP 延迟 | ✅ PASS（p95 比值 1.133 ≤ 1.2） |
| 8. 相同安全约束与执行预算 | ✅ PASS（软项不改约束，VLA query=0，QP=1/step） |

**G1 = PASS**：全部数值/接口测试通过，仅产生固定维度向量计算 + 原有一次 QP，无在线
候选池、无二次策略查询。可进入 G2（机制鉴别，GPU rollout）。

## 1. 实现（最小侵入）

### 1.1 分支与文件

- 分支：`aegis-stg-v1`（从 G0 的 `082349a` 切出）
- 新模块：`src/safety/stg.py`（STGGuide，纯 NumPy，无 MuJoCo）
- 最小 patch：`src/safety/aegis_adapted.py`（加 `stg_mode` + 软项，不改约束）
- 严格口径接入：`src/safety/contract_objects.py`（StrictAuditTracker）+
  `src/eval_safelibero.py`（run_episode `strict_audit` 参数 + summarize）

### 1.2 2×2 因子设计

| 方法 | 环流（切向 t̂） | 参考迟滞（r_t 记忆） | 软项 |
|---|---|---|---|
| B0 = aegis_native | ✗ | ✗ | λ=0 |
| B1 = circulation_only | ✓（瞬时，无记忆） | ✗ | λ‖P_t u - g_t‖² |
| B2 = governor_only | ✗ | ✓（r_t 平滑） | λ‖P_t u - g_t‖² |
| B3 = joint_stg | ✓（r_t 演化含切向） | ✓ | λ‖P_t u - g_t‖² |

### 1.3 关键映射（G0 已验证）

- 法向 `n_t = ∇h/‖∇h‖ = R1·(a_v/‖a_v‖)`（远离障碍物方向）
- 切向 `t̂_t = (I - n n^T) d_t` 归一化（意图 d_t 的切向投影）
- 参考 `r_t ∈ R³`（世界系米），风险进入时初始化 = 末端位置
- 引导 `g_t = (r_{t+1} - p_ee)/OSC_POS_SCALE`，`OSC_POS_SCALE=0.05`（归一化平移增量）
- 软项 `λ‖P_t u - g_t‖²_Q`，`P_t = 0.2·R1·[I₃,0,0]`（G0 有限差分已验证）

### 1.4 量纲纪律（不把 OSC 增量当连续速度）

- `z[:3]` 是「归一化平移增量」∈[-1,1]，OSC 映射 `0.05 m`（robosuite osc.py output_max）；
- `g_t` 与 `z[:3]` 同量纲，`P_t u` 输出同量纲；
- 参考 `r_t` 是「米」，`r_des = p_ee + z[:3]·0.05`；
- h 是椭圆归一化距离（无量纲），**未**代入 `v²/(2a)`（任务书 §2.2 禁令）。

## 2. 测试结果

| 测试文件 | 用例数 | 覆盖 |
|---|---|---|
| tests/stg/test_tangent.py | 11 | 切向正交性、退化、方向切换、旋转等变 |
| tests/stg/test_governor_reset.py | 10 | reset 不串 episode、风险迟滞、2×2 差异 |
| tests/stg/test_mapping.py | 6 | P_t 结构、Hessian PSD、量纲 |
| tests/stg/test_qp_equivalence.py | 6 | λ=0 等价、约束不变、不可行、NaN |
| tests/stg/test_safety_contract.py | 6 | TSR_valid/StrictSSR_valid 组合逻辑 |
| **合计** | **39**（含 circulation 15 = 49 全过） | — |

## 3. 延迟微基准（g1_latency_bench.py，N=1000）

| 项 | 值 |
|---|---|
| governor.step（纯几何/参考） | 14.98 µs |
| B0 filter p95 | 2.97 ms |
| B3 filter+governor p95 | 3.36 ms |
| **p95 比值 B3/B0** | **1.133（≤1.2 PASS）** |

**优化记录**：软项初版用 3D 中间表达式 `quad_form(P_t u - g_t, Q)`，p95 比值 1.244 超
门槛；改为预计算 9D 二次型 + 线性项后降到 1.133。governor 本身仅 15µs，可忽略。

## 4. G0 遗留项补齐

1. **SafeLIBERO fork 上游 SHA**：vendored（无 .git），已记录关键文件 hash：
   `env_wrapper.py a7a39afabe58...`、`benchmark/__init__.py 0f00c9e717...`、
   `mu_creation.py 1aca2195e1b5...`（与历史 upstream_sources.json 一致，证明未变）。
   上游 THU-RCSCT/vlsa-aegis 的 git SHA 仍需从源仓库确认（本机无法从 vendored 推断）。
2. **TSR_valid / StrictSSR_valid 接入 summary**：`StrictAuditTracker` +
   `run_episode(strict_audit=True)` + `summarize` 输出 `TSR_valid`/`StrictSSR_valid`/
   `forbidden_contact_episode_rate`/`task_integrity_rate`。

## 5. 参数冻结（Pilot 前预注册，测试集不再调）

| 参数 | 值 |
|---|---|
| stg_mode | B0/B1/B2/B3 |
| stg_lam_max | 1.0 |
| stg_h_enter / stg_h_exit | 0.01 / 0.02（迟滞） |
| stg_alpha（参考吸引速率） | 0.5 |
| stg_gamma（切向幅度，米/步） | 0.01 |
| OSC_POS_SCALE | 0.05（固定，来自 OSC） |

**诚实声明**：`governor` 是 ERG 启发的 dynamic reference heuristic，**无正式 ERG 安全
保证**（无 certified DSM、无稳定性证明），不声称继承 ERG 安全证书。h 无米单位，α 是
固定速率（首轮先验证记忆机制，未做动态裕度）。

## 6. 下一步（G2 机制鉴别）

G1 PASS，可进入 G2（小规模机制鉴别，GPU rollout）：
1. 用 B0 结果冻结病例选择表（S1 停滞/S2 振荡/S3 抓取失败/S4 间接碰撞）；
2. 配对运行 B0-B3，记录每步几何 proxy、CBF 活动、参考 r_t、方向翻转、进展；
3. 归因等级：CONTROL_DEADLOCK_SUPPORTED / VLA_LIMITATION / CONTACT_OR_GEOMETRY /
   INCONCLUSIVE。
