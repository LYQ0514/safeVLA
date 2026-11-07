# Counterfactual-Resume Viability Shield (CRVS)

## Why the first shield failed

The original product-space shield certified only a short intervention prefix.
Its sparse task score could not distinguish a safe continuation from a safe
stop, and its zero-pose backup made safety freeze the easiest solution. CRVS
changes the acceptance set rather than retuning that objective.

## Decision operator

At a frozen pi0.5 chunk boundary, CRVS constructs task-intent-preserving
repairs of the remaining chunk. The candidate family contains the nominal
chunk, minimum-norm action-space clearance projections, and non-local
task-retaining retreat/tangent macros. The gripper dimension is never changed.

Every candidate is evaluated from the same restored state in two connected
segments:

1. execute the repaired remainder of the current chunk;
2. render the resulting observation, query the unchanged frozen pi0.5 using
   the registered next CRN query identity, and execute its resume chunk.

A candidate is deployable only when all of the following hold:

- neither segment enters `q_bad` or `q_unknown`;
- the product-state progress score does not regress;
- stage-conditioned task geometry does not regress by more than 20 mm: before
  grasp control this is EEF-to-target distance, and after control it is
  target-to-receptacle distance;
- the repaired pose sequence retains at least 30% of nominal motion energy;
- the repaired pose sequence has non-negative cosine alignment with the
  nominal pose sequence;
- episode termination, controller state, sticky gripper state, automaton state,
  and the live pi0.5 chunk cursor remain identical after planning restore.

Candidate selection is lexicographic: acceptance, product-state progress,
task geometry, proxy clearance, then nominal deviation. No scalar weight may
trade task viability against the earlier gates. If no candidate passes, CRVS
executes the longest-safe task-retaining candidate for one receding-horizon
step and records `unverified_best_effort_nonstop`; it never substitutes a
zero-motion hold.

The implementation executes two actions at most before replanning. It never
pads a partial VLA chunk across a query boundary; the next chunk is generated
from the simulated terminal observation.

## Smoke test

Run this only after the current first-shield job has finished so its evidence
remains immutable.

```bash
cd /data/safeVLA
source tools/activate_env.sh
python tools/run_product_space_shield.py \
  --method viability \
  --suite safelibero_spatial --level I \
  --tasks 0 --episodes 0 --streams 0 \
  --horizon 5 --resume-horizon 5 --execute-steps 2 \
  --min-intent-ratio 0.30 --goal-regression-tolerance 0.02 \
  --out data/eval/crvs_smoke
```

## Pre-registered first round

```bash
bash tools/run_viability_first_round.sh \
  data/eval/counterfactual_resume_viability_first
```

This runs the same 40 paired cells as the first shield. Inspect
`comparison.json` and the decision logs before any full benchmark.

Required gates for proceeding to the full run:

- at least one task-valid success (`q_acc`), unless the paired baseline also
  has zero support in the exact same cells;
- no increase in `UnsafeReturn` or `IntegrityFail` relative to paired baseline;
- StrictSSR is not lower than paired baseline;
- `unverified_best_effort_nonstop` is reported, not hidden;
- SAFE_STALL does not dominate the viability episodes.

## Full benchmark

```bash
bash tools/run_viability_full.sh \
  data/eval/counterfactual_resume_viability_full
```

This evaluates 1600 paired cells per method and supports `--resume` through
the suite scripts.

## Scientific boundary

CRVS establishes finite-horizon empirical viability under the registered
SafeLIBERO simulator and task automaton. It does not establish model-independent
or real-world forward invariance. A resume query demonstrates that the frozen
VLA can continue safely for the registered horizon; it is not a proof of
eventual task completion.
