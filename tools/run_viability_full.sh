#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -z "${LIBERO_CONFIG_PATH:-}" ]]; then
  echo "Run: source tools/activate_env.sh" >&2; exit 2
fi
RUN="${1:-data/eval/counterfactual_resume_viability_full}"
TASKS=(0 1 2 3 4 5 6 7 8 9)
EPISODES=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19)
mkdir -p "$RUN/baseline" "$RUN/viability"

for suite in safelibero_spatial safelibero_goal safelibero_object safelibero_long; do
  for level in I II; do
    cell="${suite}_L${level}"
    python tools/run_product_space_shield.py --method baseline \
      --suite "$suite" --level "$level" --tasks "${TASKS[@]}" \
      --episodes "${EPISODES[@]}" --streams 0 --resume \
      --execute-steps 2 --out "$RUN/baseline/$cell"
    python tools/run_product_space_shield.py --method viability \
      --suite "$suite" --level "$level" --tasks "${TASKS[@]}" \
      --episodes "${EPISODES[@]}" --streams 0 --resume \
      --horizon 5 --resume-horizon 5 --execute-steps 2 \
      --min-intent-ratio 0.30 --goal-regression-tolerance 0.02 \
      --out "$RUN/viability/$cell"
  done
done


find "$RUN/baseline" -name episodes.jsonl -print0 | sort -z | xargs -0 cat > "$RUN/baseline_all.jsonl"
find "$RUN/viability" -name episodes.jsonl -print0 | sort -z | xargs -0 cat > "$RUN/viability_all.jsonl"
python tools/compare_fpck_methods.py "$RUN/baseline_all.jsonl" "$RUN/viability_all.jsonl" \
  --out "$RUN/comparison.json"
echo "Complete: $RUN/comparison.json"
