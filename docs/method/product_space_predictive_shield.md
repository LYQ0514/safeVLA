# Product-Space Predictive Shield: SafeLIBERO execution

This is the direct, training-free main method for frozen pi0.5.  At every VLA
chunk boundary it optimizes a continuous five-action sequence, shortlists with
the geometric proxy, and verifies candidate prefixes by restoring MuJoCo and
the v1.3 task automaton. A verified sequence is preferred. If the registered
simulator cannot find one, the runner executes a zero-pose backup while
preserving the VLA gripper command and reports it explicitly as unverified.

The implementation provides an empirical simulator-verified result, not a
model-independent real-world certificate.

## Smoke test

```bash
cd /data/safeVLA
source tools/activate_env.sh
python tools/run_product_space_shield.py --method shield \
  --suite safelibero_spatial --level I --tasks 0 --episodes 0 --streams 0 \
  --population 16 --rollout-budget 4 --iterations 2 --elite 2 \
  --out data/eval/psps_smoke
```

## Preregistered first round

```bash
bash tools/run_product_space_first_round.sh data/eval/product_space_shield_first
```

This first round contains 40 paired cells (task 0, initialization states 0--4,
all four suites and both safety levels). It is the recommended integration and
feasibility gate before committing GPU time to the complete benchmark.

## Full locked SafeLIBERO benchmark

```bash
bash tools/run_product_space_full.sh data/eval/product_space_shield_full
```

The full entry point runs 10 tasks x 20 initialization states x four suites x
two safety levels = 1600 paired cells for each method. It is resumable. The
baseline and shield use the same deterministic frozen pi0.5 factory and the
same episode identities.

The output contains append-only executed event logs, planning decisions,
per-episode results, paired baseline/shield summaries, CAR at 1 mm and CAR at
5 mm, native success, task-valid success, strict SSR, intervention counts,
unverified backup counts, and planning latency.

`TSR` in this runner is task success before an integrity-terminal state. The
runner intentionally stops on `q_bad` / `q_unknown`; it therefore must not be
substituted for the repository's already locked 1600-episode nominal pi0.5 TSR
calibration. The paired baseline is the valid comparison for this method.
