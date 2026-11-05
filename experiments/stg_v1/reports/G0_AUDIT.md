# G0_AUDIT：源码真实性、物理几何与评测契约审计

> 日期：2026-11-05。门禁 G0。判定：**PASS（含 2 项非致命待补）**

## 0. 判定

| 门禁项 | 结论 |
|---|---|
| 源码身份（HEAD/dirty/版本/checkpoint） | ✅ PASS |
| π0.5 chunk 消费与 replan 机制 | ✅ PASS |
| AEGIS QP 完整 9D 导出 | ✅ PASS |
| 四元数 + 椭球几何独立核验 | ✅ PASS（p1 偏差 0.036mm） |
| 动作映射（7D→9D→7D + P_t） | ✅ PASS（有限差分 < 1e-9） |
| 评测契约（TSR/CAR/SSR） | ✅ PASS（native 口径明确） |
| 零引导数值等价 | ✅ PASS（确定性 + W 匹配） |
| SafeLIBERO fork 上游 SHA | ⚠️ 待补（非致命，vendored 环境） |
| TSR_valid/StrictSSR_valid 接入 summary | ⚠️ 待 G1 接入（组件已实现） |

**G0 = PASS**：物理几何、策略时序、评测身份无未解决的致命冲突。可进入 G1。

## 1. 交付物清单（全部存在）

| 交付物 | 路径 | 状态 |
|---|---|---|
| runtime_manifest.json | `experiments/stg_v1/audit/` | ✅ |
| reproduction_boundary.md | 同上 | ✅ |
| geometry_audit.csv | 同上 | ✅（50 行，50 init state） |
| safety_contract.md | 同上 | ✅ |
| action_mapping_test.json | 同上 | ✅ |
| baseline_equivalence.json | 同上 | ✅ |

## 2. 关键证据摘要

### 2.1 源码身份

- HEAD = `21e9e494d9c9ceee2b3d418c9e6fa76bf51e3e39`（`21e9e49`）
- dirty 仅 `src/pnsg/openpi_worker.py` + `policy_adapter.py`（无关历史探索，非 AEGIS 路径）
- 依赖：numpy 1.22.4 / cvxpy 1.5.2 / osqp 1.1.3 / mujoco 2.3.7 / robosuite 1.4.1 /
  open3d 0.19.0 / torch 2.4.1+cu121
- checkpoint hash = `b838d8b85a3f317c91950738172ee7f174afe7f2d36d9d8ed80f7110a085e90b`

### 2.2 几何核验（geometry_audit.csv，50 init state）

| 指标 | 最大值 |
|---|---|
| eef_pos 与 grip_site 真值偏差 | 0.025 mm |
| R1（四元数→旋转）与 body_xmat 偏差 | 1.56e-4 |
| **p1 椭球中心偏差** | **0.036 mm** |

**关键**：`robot0_eef_pos` 来自 `grip_site` site，`robot0_eef_quat` 来自
`robot0_right_hand` body（robosuite 源码 single_arm.py:306-312），这是 AEGIS 上游的
固有定义。p1 = eef_pos + R1@[0,0,-0.08] 已独立核验，16cm 级偏差**不可再现**。

### 2.3 动作映射（action_mapping_test.json）

- `P_t = 0.2·R1·[I₃, 0₃, 0₃] ∈ R^{3×9}`，有限差分与解析一致（<1e-9）；
- `z_p,ref` 匹配世界系平移（`0.2·5·(R1 R1ᵀ)z_p = z_p`，<1e-9）；
- 夹爪直通（`out[6]=z[6]`）。

### 2.4 零引导等价（baseline_equivalence.json）

- 同输入 3 次调用输出 bit 级一致（确定性）；
- `W0 = diag(1/25×6 + 1×3)` 与期望一致（λ=0 时 STG 软项应消失回归此目标）。

### 2.5 评测契约（safety_contract.md）

- `TSR_native = check_success`，`CAR_native = L1 位移≤1mm`，`SSR_native = TSR∧CAR`；
- `forbidden_contact` / `task_integrity_ok` 组件在 `contract_objects.py` 已实现；
- `box_small_base → mug` 间接位移链已在支撑感知线查清（PROTECTED_SUPPORT 角色）。

## 3. 失败归因 / 待补项（非致命）

1. **SafeLIBERO fork 上游 commit SHA 未锁定**：vendored 本机（非 git），需从
   THU-RCSCT/vlsa-aegis 锁定确切 commit + 计算 init states hash；
2. **TSR_valid/StrictSSR_valid 未接入 summary**：组件已实现但 `eval_safelibero.py`
   只输出 native 口径，G1 需接入以支撑 G2/G3/G4 的逐 episode 严格口径日志。

## 4. 计算开销

本 G0 阶段全部为只读审计 + 纯几何/映射数值测试，**零 GPU rollout**：
- geometry_audit：50 次 env 构造（无推理），数分钟；
- action_mapping / baseline_equivalence：单 env 单次 QP，秒级。

## 5. 下一阶段（G1）建议

G0 PASS，进入 G1（纯数学与接口测试，不消耗 GPU rollout）：
1. P_t 已在本 G0 验证（有限差分 + 解析一致），G1 补旋转/辅助维耦合的有限差分；
2. 实现 B1/B2/B3 最小侵入模块，验证 λ=0 恢复 B0、QP 9D 保真、chunk 查询数不变；
3. 接入 TSR_valid/StrictSSR_valid 到评测日志（契约字段）；
4. 2D toy + 3D 偏置障碍的切向性/方向连续性/退化测试；
5. governor reset 不串 episode、风险退出恢复 B0、非有限防护。

## 6. 可复现命令

```bash
cd /data/safeVLA
# 几何审计（50 init state）
PYTHONPATH=/data/safeVLA .venv/bin/python tools/g0_geometry_audit.py
# 动作映射 + P_t 有限差分
PYTHONPATH=/data/safeVLA .venv/bin/python tools/g0_action_mapping_test.py
# 零引导数值等价 + 确定性
PYTHONPATH=/data/safeVLA .venv/bin/python tools/g0_baseline_equivalence.py
```
