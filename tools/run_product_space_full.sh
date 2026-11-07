#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -z "${LIBERO_CONFIG_PATH:-}" ]]; then
  echo "Run: source tools/activate_env.sh" >&2
  exit 2
fi

# Full locked SafeLIBERO evaluation: 4 suites x 2 levels x 10 tasks x 20
# initialization states = 1600 paired cells per method. Every invocation is
# resumable; rerunning the script skips completed episode identities.
RUN="${1:-data/eval/product_space_shield_full}"
TASKS=(0 1 2 3 4 5 6 7 8 9)
EPISODES=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19)
mkdir -p "$RUN/baseline" "$RUN/shield"

for suite in safelibero_spatial safelibero_goal safelibero_object safelibero_long; do
  for level in I II; do
    cell="${suite}_L${level}"
    python tools/run_product_space_shield.py --method baseline \
      --suite "$suite" --level "$level" --tasks "${TASKS[@]}" \
      --episodes "${EPISODES[@]}" --streams 0 --resume \
      --out "$RUN/baseline/$cell"
    python tools/run_product_space_shield.py --method shield \
      --suite "$suite" --level "$level" --tasks "${TASKS[@]}" \
      --episodes "${EPISODES[@]}" --streams 0 --resume \
      --out "$RUN/shield/$cell"
  done
done

find "$RUN/baseline" -name episodes.jsonl -print0 | sort -z | xargs -0 cat > "$RUN/baseline_all.jsonl"
find "$RUN/shield" -name episodes.jsonl -print0 | sort -z | xargs -0 cat > "$RUN/shield_all.jsonl"
python tools/compare_fpck_methods.py "$RUN/baseline_all.jsonl" "$RUN/shield_all.jsonl" \
  --out "$RUN/comparison.json"
echo "Complete: $RUN/comparison.json"
