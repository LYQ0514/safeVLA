#!/usr/bin/env bash
# AEGIS 消融：在 long I 上试不同约束强度，量化「过度约束」的缓解空间。
#
# 基线参照（已跑完，data/eval/aegis_1024 的 long_LI）：
#   TSR 13.5%  CAR 68.0%  CAR>5mm 98.5%
# π0.5 纯基线（data/eval/pi05_1024_baseline 的 long_LI）：
#   TSR 62.0%  CAR 14.0%  CAR>5mm 93.5%
#
# 三个消融：
#   smalleef  忽略上游「大椭球」规则（orange juice/milk/alphabet soup 用 1.8 倍椭球）
#   gain5     CBF h 增益 10 → 5（约束更松）
#   gain3     CBF h 增益 10 → 3（更松）
#
# 用法：bash tools/run_ablation.sh
set -u

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
source tools/activate_env.sh

run_one() {
    local name="$1"; shift
    local flag="$1"; shift
    echo "########## $name  extra=[$flag] ##########"
    python tools/run_eval_aegis.py \
        --suite safelibero_long --level I \
        --tasks 0 1 2 3 --episodes $(seq 0 49) \
        --resolution 1024 --replan-steps 5 --no-video \
        $flag \
        --out "data/eval/ablation_${name}" --resume
}

run_one smalleef "--small-eef"
run_one gain5   "--h-gain 5"
run_one gain3   "--h-gain 3"
