#!/usr/bin/env bash
# 同轨迹双影子监视器 A/B：只重跑受 acquisition 重入 bug 影响的 5 个 anchor 的
# 完整 sibling sets（每 anchor 4 candidate = 20 branch）。
#
# 与「跨时刻重跑」不同，本脚本在每条物理轨迹上同时运行 legacy（含 bug）与
# fixed（修复）两个自动机（见 collect 脚本的 compare_automaton），因此不依赖
# 历史轨迹的 bitwise 复现；因果归因由同轨迹 A/B 直接给出。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -z "${LIBERO_CONFIG_PATH:-}" ]]; then
  echo "LIBERO_CONFIG_PATH is missing; run: source tools/activate_env.sh" >&2
  exit 2
fi

RUN="data/eval/fpck_main_shadow_ab"
mkdir -p "$RUN/corpus"

# 5 个受影响 (suite, level, episode) 组合，均 task 0 / stream 0。
python tools/collect_fpck_counterfactual.py \
  --suite safelibero_goal --level II --tasks 0 \
  --episodes 0 --streams 0 --candidates 4 \
  --out "$RUN/corpus/goal_LII_s0"

python tools/collect_fpck_counterfactual.py \
  --suite safelibero_goal --level II --tasks 0 \
  --episodes 1 --streams 0 --candidates 4 \
  --out "$RUN/corpus/goal_LII_s1"

python tools/collect_fpck_counterfactual.py \
  --suite safelibero_long --level II --tasks 0 \
  --episodes 0 --streams 0 --candidates 4 \
  --out "$RUN/corpus/long_LII_s0"

python tools/collect_fpck_counterfactual.py \
  --suite safelibero_long --level II --tasks 0 \
  --episodes 1 --streams 0 --candidates 4 \
  --out "$RUN/corpus/long_LII_s1"

python tools/collect_fpck_counterfactual.py \
  --suite safelibero_long --level II --tasks 0 \
  --episodes 2 --streams 0 --candidates 4 \
  --out "$RUN/corpus/long_LII_s2"

echo "same-trace shadow A/B collection complete"
