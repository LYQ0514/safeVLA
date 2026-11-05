"""Runtime CKG and calibration utilities; independent of SafeLIBERO."""
from __future__ import annotations  # py3.8 兼容：延迟注解求值，允许 X | None 语法

from dataclasses import dataclass
import math
from typing import Any, Callable, Iterable, Sequence
import numpy as np

@dataclass(frozen=True)
class Candidate:
    action: np.ndarray
    source: str = "unspecified"

@dataclass(frozen=True)
class Decision:
    action: np.ndarray | None
    status: str
    nominal_distance: float
    worst_value: float
    evaluated: tuple

def one_sided_correction(predicted: Sequence[float], truth: Sequence[float], alpha: float) -> float:
    """Split-calibration upper quantile of over-estimation residuals."""
    p, y = np.asarray(predicted, float), np.asarray(truth, float)
    if p.shape != y.shape or p.ndim != 1 or len(p) == 0:
        raise ValueError("predicted/truth must be equal nonempty vectors")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    r = np.sort(p - y)
    rank = min(len(r), math.ceil((len(r) + 1) * (1 - alpha))) - 1
    return float(max(0.0, r[rank]))

class ContinuationKernelGovernor:
    def __init__(self, transition: Callable[[Any, np.ndarray, Any], Any],
                 disturbances: Callable[[Any], Iterable[Any]],
                 lower_value: Callable[[Any, int, int], float],
                 deployed_distance: Callable[[np.ndarray, np.ndarray], float],
                 intervention_eps: float = 1e-7):
        self.transition, self.disturbances = transition, disturbances
        self.lower_value, self.distance = lower_value, deployed_distance
        self.intervention_eps = float(intervention_eps)

    def choose(self, state, nominal, candidates: Sequence[Candidate], budget: int, horizon: int) -> Decision:
        nominal = np.asarray(nominal, float)
        if horizon <= 0:
            return Decision(None, "horizon_exhausted", float("inf"), -float("inf"), ())
        rows, feasible = [], []
        for c in candidates:
            action = np.asarray(c.action, float)
            dist = float(self.distance(action, nominal))
            cost = int(dist > self.intervention_eps)
            if cost > budget:
                rows.append((c.source, dist, cost, -float("inf"), False)); continue
            values = [self.lower_value(self.transition(state, action, w), budget - cost, horizon - 1)
                      for w in self.disturbances(state)]
            if not values:
                raise ValueError("disturbance enumerator returned no values")
            worst = float(min(values)); ok = worst >= 0.0
            rows.append((c.source, dist, cost, worst, ok))
            if ok: feasible.append((dist, -worst, c.source, action, worst))
        if not feasible:
            return Decision(None, "outside_certified_domain", float("inf"), -float("inf"), tuple(rows))
        feasible.sort(key=lambda z: (z[0], z[1], z[2]))
        dist, _, _, action, worst = feasible[0]
        return Decision(action.astype(np.float32), "certified", dist, worst, tuple(rows))
