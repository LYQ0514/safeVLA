#!/usr/bin/env bash
# One-command FPCK/CKG decision experiment on frozen pi0.5 + SafeLIBERO.
#
# This deliberately runs only the preregistered first round.  It stops if the
# restored-branch corpus cannot support non-trivial continuation learning.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -z "${LIBERO_CONFIG_PATH:-}" ]]; then
  echo "LIBERO_CONFIG_PATH is missing; run: source tools/activate_env.sh" >&2
  exit 2
fi

RUN="${1:-data/eval/fpck_main_first}"
if [[ -e "$RUN" ]] && [[ -n "$(find "$RUN" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Refusing to append to non-empty run directory: $RUN" >&2
  echo "Choose a new directory so raw evidence remains append-only." >&2
  exit 2
fi

mkdir -p "$RUN/corpus" "$RUN/baseline" "$RUN/ckg"

for suite in safelibero_spatial safelibero_goal safelibero_object safelibero_long; do
  for level in I II; do
    python tools/collect_fpck_counterfactual.py \
      --suite "$suite" --level "$level" --tasks 0 \
      --episodes 0 1 2 3 4 --streams 0 --candidates 4 \
      --out "$RUN/corpus/${suite}_L${level}"
  done
done

python tools/train_fpck_value.py \
  "$RUN"/corpus/*/samples.jsonl \
  --out "$RUN/continuation_model.json"

for suite in safelibero_spatial safelibero_goal safelibero_object safelibero_long; do
  for level in I II; do
    python tools/run_fpck_main.py --method baseline \
      --suite "$suite" --level "$level" --tasks 0 \
      --episodes 5 6 7 8 9 --streams 0 \
      --out "$RUN/baseline/${suite}_L${level}"
    python tools/run_fpck_main.py --method ckg \
      --model "$RUN/continuation_model.json" \
      --suite "$suite" --level "$level" --tasks 0 \
      --episodes 5 6 7 8 9 --streams 0 \
      --out "$RUN/ckg/${suite}_L${level}"
  done
done

find "$RUN/baseline" -name episodes.jsonl -print0 | sort -z | xargs -0 cat > "$RUN/baseline_all.jsonl"
find "$RUN/ckg" -name episodes.jsonl -print0 | sort -z | xargs -0 cat > "$RUN/ckg_all.jsonl"
python tools/compare_fpck_methods.py \
  "$RUN/baseline_all.jsonl" "$RUN/ckg_all.jsonl" \
  --out "$RUN/comparison.json"

echo "FPCK first round complete: $RUN/comparison.json"
