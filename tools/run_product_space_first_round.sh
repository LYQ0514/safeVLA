#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -z "${LIBERO_CONFIG_PATH:-}" ]]; then
  echo "Run: source tools/activate_env.sh" >&2; exit 2
fi
RUN="${1:-data/eval/product_space_shield_first}"
mkdir -p "$RUN/baseline" "$RUN/shield"

for suite in safelibero_spatial safelibero_goal safelibero_object safelibero_long; do
  for level in I II; do
    cell="${suite}_L${level}"
    python tools/run_product_space_shield.py --method baseline \
      --suite "$suite" --level "$level" --tasks 0 \
      --episodes 0 1 2 3 4 --streams 0 --resume \
      --out "$RUN/baseline/$cell"
    python tools/run_product_space_shield.py --method shield \
      --suite "$suite" --level "$level" --tasks 0 \
      --episodes 0 1 2 3 4 --streams 0 --resume \
      --out "$RUN/shield/$cell"
  done
done

find "$RUN/baseline" -name episodes.jsonl -print0 | sort -z | xargs -0 cat > "$RUN/baseline_all.jsonl"
find "$RUN/shield" -name episodes.jsonl -print0 | sort -z | xargs -0 cat > "$RUN/shield_all.jsonl"
python tools/compare_fpck_methods.py "$RUN/baseline_all.jsonl" "$RUN/shield_all.jsonl" \
  --out "$RUN/comparison.json"
echo "Complete: $RUN/comparison.json"
