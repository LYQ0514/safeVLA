"""Runtime CKG over calibrated direct candidate continuation values."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Sequence
import numpy as np
from .governor import Candidate, Decision


class DirectContinuationGovernor:
    """Choose minimum-deviation candidate whose empirical lower score passes.

    ``feature_builder(state, nominal, candidate, budget, horizon)`` must return
    the exact feature contract used during counterfactual training.
    """
    def __init__(self, model, feature_builder: Callable, threshold: float = 0.5,
                 intervention_eps: float = 1e-7, distance_weight=None):
        self.model = model; self.feature_builder = feature_builder
        self.threshold = float(threshold); self.intervention_eps = float(intervention_eps)
        self.distance_weight = (np.ones(7) if distance_weight is None
                                else np.asarray(distance_weight, float))

    def choose(self, state, nominal, candidates: Sequence[Candidate], budget: int,
               horizon: int) -> Decision:
        nominal = np.asarray(nominal, float)
        if horizon <= 0:
            return Decision(None, "horizon_exhausted", float("inf"), 0.0, ())
        rows, feasible = [], []
        for c in candidates:
            action = np.asarray(c.action, float)
            distance = float(np.linalg.norm((action-nominal)*self.distance_weight))
            cost = int(distance > self.intervention_eps)
            if cost > budget:
                rows.append((c.source, distance, cost, 0.0, False, "budget")); continue
            phi = self.feature_builder(state, nominal, action, budget-cost, horizon-1)
            lower = float(np.asarray(self.model.lower(np.asarray(phi)[None]))[0])
            ok = bool(lower >= self.threshold)
            rows.append((c.source, distance, cost, lower, ok, "score"))
            if ok:
                feasible.append((distance, -lower, c.source, action, lower))
        if not feasible:
            return Decision(None, "outside_empirical_domain", float("inf"), 0.0,
                            tuple(rows))
        feasible.sort(key=lambda x: (x[0], x[1], x[2]))
        dist, _, _, action, lower = feasible[0]
        return Decision(action.astype(np.float32), "empirically_admissible",
                        dist, lower, tuple(rows))
