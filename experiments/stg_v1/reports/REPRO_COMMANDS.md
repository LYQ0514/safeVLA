# REPRO_COMMANDS：AEGIS-STG 可复现命令清单

> 日期：2026-11-05。记录 G0（及后续各阶段）的真实执行命令与数据路径。

## 环境

```bash
cd /data/safeVLA
source tools/activate_env.sh   # 激活 .venv（Python 3.8）
```

## G0 审计命令

```bash
# 几何审计（p1/R1 独立真值核验，50 init state）
PYTHONPATH=/data/safeVLA .venv/bin/python tools/g0_geometry_audit.py
#   输出: experiments/stg_v1/audit/geometry_audit.csv

# 动作映射 + P_t 有限差分
PYTHONPATH=/data/safeVLA .venv/bin/python tools/g0_action_mapping_test.py
#   输出: experiments/stg_v1/audit/action_mapping_test.json

# 零引导数值等价 + 确定性
PYTHONPATH=/data/safeVLA .venv/bin/python tools/g0_baseline_equivalence.py
#   输出: experiments/stg_v1/audit/baseline_equivalence.json
```

## 基线 B0（AEGIS-QP Oracle Perception）评测

```bash
# 单套件/level（perception 变体；oracle 用 --no-perception 去掉 --perception）
PYTHONPATH=/data/safeVLA .venv/bin/python tools/run_eval_aegis.py \
  --suite safelibero_long --level I --tasks 0 1 2 3 --episodes 0..49 \
  --resolution 1024 --policy-resize 224 --replan-steps 5 --num-steps-wait 20 \
  --perception --no-video --resume --out <输出目录>
```

## 关键数据路径

| 数据 | 路径 |
|---|---|
| π0.5 checkpoint | `/data/2608.SDL/MojocoProj/checkpoints/pi05/pi05_libero` |
| SafeLIBERO fork | `/data/safeVLA/dataset/SafeLIBERO` |
| 上游 AEGIS (vendored) | `/data/safeVLA/third_party/aegis/` |
| AEGIS 内核 | `src/safety/aegis_adapted.py` |
| perception 链 | `src/safety/aegis_perception.py` |
| circulation（严格实现） | `src/safety/circulation.py` |
| 评测循环 | `src/eval_safelibero.py` |
| 评测入口 | `tools/run_eval_aegis.py` |
| 安全契约对象 | `src/safety/contract_objects.py` |

## 评测口径

- CAR = obstacle 位移 L1 ≤ 1mm（官方）
- TSR = check_success
- SSR = TSR ∧ CAR
- ETS = 含超时全部 episode 平均步数
- max_steps: spatial/object/goal=300, long=550
