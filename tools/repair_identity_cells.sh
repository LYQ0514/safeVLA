#!/usr/bin/env bash
# Re-collect the 9 identity-failed cells with the signed-zero fix.
# Output goes to a separate repair directory; original evidence is preserved.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

REPAIR="data/eval/fpck_main_first_identity_repair"
mkdir -p "$REPAIR/corpus"

python tools/collect_fpck_counterfactual.py \
  --suite safelibero_goal --level I --tasks 0 \
  --episodes 0 --streams 0 --candidates 4 \
  --out "$REPAIR/corpus/goal_LI_s0"

python tools/collect_fpck_counterfactual.py \
  --suite safelibero_goal --level II --tasks 0 \
  --episodes 0 --streams 0 --candidates 4 \
  --out "$REPAIR/corpus/goal_LII_s0"

python tools/collect_fpck_counterfactual.py \
  --suite safelibero_long --level II --tasks 0 \
  --episodes 4 --streams 0 --candidates 4 \
  --out "$REPAIR/corpus/long_LII_s4"

python tools/collect_fpck_counterfactual.py \
  --suite safelibero_object --level I --tasks 0 \
  --episodes 2 3 --streams 0 --candidates 4 \
  --out "$REPAIR/corpus/object_LI_s2_s3"

python tools/collect_fpck_counterfactual.py \
  --suite safelibero_object --level II --tasks 0 \
  --episodes 0 1 3 4 --streams 0 --candidates 4 \
  --out "$REPAIR/corpus/object_LII_s0_s1_s3_s4"

echo "identity repair collection complete"
