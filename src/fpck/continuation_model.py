"""Calibrated empirical continuation action-value model.

The exact FPCK recursion lives in :mod:`fpck.governor`.  SafeLIBERO cannot
enumerate its continuous predecessor online, so the deployed governor learns a
direct candidate action-value from fully restored counterfactual branches.

This module deliberately calls the result an *empirical lower score*.  A split
one-sided correction controls measured false optimism; it is not promoted to a
uniform mathematical certificate without the stronger assumptions in the
paper.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


def _sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def _quantile_higher(values, q: float) -> float:
    values = np.sort(np.asarray(values, dtype=np.float64))
    if values.size == 0:
        raise ValueError("empty calibration residuals")
    index = min(values.size - 1, max(0, math.ceil(q * values.size) - 1))
    return float(values[index])


@dataclass
class CalibratedContinuationModel:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    correction: float
    ensemble_quantile: float
    feature_names: tuple[str, ...]
    metadata: dict

    def _design(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.shape[-1] != self.mean.size:
            raise ValueError(f"feature width {x.shape[-1]} != {self.mean.size}")
        z = (x - self.mean) / self.scale
        return np.concatenate([z, np.ones((*z.shape[:-1], 1))], axis=-1)

    def member_probabilities(self, x):
        d = self._design(x)
        return _sigmoid(np.einsum("...d,md->...m", d, self.weights))

    def predict(self, x):
        return np.mean(self.member_probabilities(x), axis=-1)

    def lower(self, x):
        probs = self.member_probabilities(x)
        # NumPy <1.22 compatibility: interpolation="lower" has stable semantics.
        try:
            base = np.quantile(probs, self.ensemble_quantile, axis=-1, method="lower")
        except TypeError:
            base = np.quantile(probs, self.ensemble_quantile, axis=-1,
                               interpolation="lower")
        return np.clip(base - self.correction, 0.0, 1.0)

    def save(self, path):
        payload = {
            "format": "fpck-continuation-ensemble-v1",
            "mean": self.mean.tolist(), "scale": self.scale.tolist(),
            "weights": self.weights.tolist(), "correction": self.correction,
            "ensemble_quantile": self.ensemble_quantile,
            "feature_names": list(self.feature_names), "metadata": self.metadata,
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if d.get("format") != "fpck-continuation-ensemble-v1":
            raise ValueError("unsupported continuation model format")
        return cls(np.asarray(d["mean"], float), np.asarray(d["scale"], float),
                   np.asarray(d["weights"], float), float(d["correction"]),
                   float(d["ensemble_quantile"]), tuple(d["feature_names"]),
                   dict(d.get("metadata", {})))


def _fit_logistic(x, y, l2, steps, lr, rng):
    n, d = x.shape
    w = np.zeros(d, dtype=np.float64)
    # Class balancing prevents a failure-heavy discovery pool from learning a
    # trivial all-failure score.
    pos = max(1, int(np.sum(y == 1)))
    neg = max(1, int(np.sum(y == 0)))
    sw = np.where(y == 1, n / (2.0 * pos), n / (2.0 * neg))
    for k in range(steps):
        p = _sigmoid(x @ w)
        grad = x.T @ (sw * (p - y)) / n
        grad[:-1] += l2 * w[:-1]
        # Smooth decay is deterministic and avoids an optional optimizer dep.
        eta = lr / math.sqrt(1.0 + k / 100.0)
        w -= eta * grad
    return w


def fit_calibrated_ensemble(
        train_x, train_y, calibration_x, calibration_y, *,
        feature_names: Sequence[str], groups: Iterable[str] | None = None,
        members: int = 16, l2: float = 1e-3, steps: int = 1500,
        lr: float = 0.08, alpha: float = 0.05,
        ensemble_quantile: float = 0.1, seed: int = 0, metadata=None):
    """Fit bootstrap logistic members and a one-sided optimism correction.

    Bootstrap sampling is by ``groups`` (normally anchor_id), so candidate
    siblings never masquerade as independent samples.
    """
    x = np.asarray(train_x, float); y = np.asarray(train_y, float)
    xc = np.asarray(calibration_x, float); yc = np.asarray(calibration_y, float)
    if x.ndim != 2 or y.shape != (len(x),) or xc.ndim != 2 or yc.shape != (len(xc),):
        raise ValueError("bad train/calibration shapes")
    if len(x) == 0 or len(xc) == 0 or x.shape[1] != xc.shape[1]:
        raise ValueError("nonempty equal-width train/calibration required")
    if set(np.unique(y)) - {0.0, 1.0} or set(np.unique(yc)) - {0.0, 1.0}:
        raise ValueError("continuation targets must be binary")
    mean = x.mean(0); scale = x.std(0); scale[scale < 1e-8] = 1.0
    z = (x - mean) / scale
    design = np.column_stack([z, np.ones(len(z))])
    rng = np.random.default_rng(seed)
    groups = np.asarray(list(groups) if groups is not None else np.arange(len(x)))
    if groups.shape != (len(x),):
        raise ValueError("groups must align with training rows")
    unique = np.unique(groups)
    ws = []
    for _ in range(int(members)):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == g) for g in sampled])
        ws.append(_fit_logistic(design[idx], y[idx], l2, steps, lr, rng))
    weights = np.asarray(ws)
    provisional = CalibratedContinuationModel(
        mean, scale, weights, 0.0, ensemble_quantile,
        tuple(feature_names), dict(metadata or {}))
    base = np.asarray(provisional.lower(xc), float)
    # Positive residual means the empirical lower score was still optimistic.
    correction = max(0.0, _quantile_higher(base - yc, 1.0 - alpha))
    provisional.correction = correction
    provisional.metadata.update({
        "members": int(members), "alpha": float(alpha), "l2": float(l2),
        "steps": int(steps), "seed": int(seed),
        "train_n": int(len(x)), "calibration_n": int(len(xc)),
        "train_anchor_n": int(len(unique)),
    })
    return provisional
