#!/usr/bin/env python3
"""Train/calibrate the direct FPCK candidate continuation model."""
from __future__ import annotations
import argparse, json, pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.fpck.continuation_model import fit_calibrated_ensemble
from src.fpck.product_features import FEATURE_NAMES


POSITIVE = {"SAFE_REENTRY"}
DECISIVE = {"SAFE_REENTRY", "UNSAFE_RETURN", "SAFE_STALL", "TASK_INTEGRITY_FAIL"}


def _load(paths):
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    return rows


def _metric(y, p, threshold=.5):
    y = np.asarray(y, int); p = np.asarray(p, float); pred = p >= threshold
    tp = int(np.sum(pred & (y == 1))); fp = int(np.sum(pred & (y == 0)))
    tn = int(np.sum(~pred & (y == 0))); fn = int(np.sum(~pred & (y == 1)))
    return {"n": int(len(y)), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "false_optimism_rate": fp/max(1, fp+tn),
            "recall": tp/max(1, tp+fn),
            "precision": tp/max(1, tp+fp),
            "brier": float(np.mean((p-y)**2)) if len(y) else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+"); ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=.05); ap.add_argument("--members", type=int, default=16)
    ap.add_argument("--threshold", type=float, default=.5); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    raw = _load(a.inputs)
    rows = [r for r in raw if r.get("label") in DECISIVE and r.get("identity_verified", True)]
    for r in rows:
        if len(r.get("features", [])) != len(FEATURE_NAMES):
            raise SystemExit(f"feature contract mismatch: {r.get('sample_id')}")
    tr = [r for r in rows if r.get("split") == "train"]
    ca = [r for r in rows if r.get("split") == "calibration"]
    te = [r for r in rows if r.get("split") == "test"]
    if not tr or not ca:
        raise SystemExit("need nonempty train and calibration anchors")
    x = lambda rr: np.asarray([r["features"] for r in rr], float)
    y = lambda rr: np.asarray([r["label"] in POSITIVE for r in rr], float)
    if len(np.unique(y(tr))) < 2 or len(np.unique(y(ca))) < 2:
        raise SystemExit(
            "NO_SAFE_REENTRY_SUPPORT_OR_SPLIT_COVERAGE: train and calibration "
            "must each contain SAFE_REENTRY and decisive failure; do not train a trivial model")
    model = fit_calibrated_ensemble(
        x(tr), y(tr), x(ca), y(ca), feature_names=FEATURE_NAMES,
        groups=[r.get("anchor_id", r["sample_id"]) for r in tr],
        members=a.members, alpha=a.alpha, seed=a.seed,
        metadata={"target": "continuable iff SAFE_REENTRY",
                  "excluded_labels": ["UNCERTAIN", "UNCERTAIN_TIMEOUT_PROGRESSING"],
                  "runtime_threshold": a.threshold})
    model.save(a.out)
    report = {"model": str(a.out), "correction": model.correction,
              "train": _metric(y(tr), model.lower(x(tr)), a.threshold),
              "calibration": _metric(y(ca), model.lower(x(ca)), a.threshold),
              "test": _metric(y(te), model.lower(x(te)), a.threshold) if te else None,
              "rows_raw": len(raw), "rows_used": len(rows)}
    report_path = pathlib.Path(a.out).with_suffix(".metrics.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
