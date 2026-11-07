"""Registered, authority-matched SafeLIBERO candidate action family."""
from __future__ import annotations
import numpy as np
from .governor import Candidate


def _match_authority(action, nominal, max_delta):
    action = np.asarray(action, float).copy(); nominal = np.asarray(nominal, float)
    delta = action[:6] - nominal[:6]
    norm = np.linalg.norm(delta)
    if norm > max_delta > 0:
        action[:6] = nominal[:6] + delta * (max_delta / norm)
    action[6] = nominal[6]  # safety candidates never invent grasp/release intent
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def registered_candidates(nominal, eef_pos=None, obstacle_pos=None,
                          max_delta=0.75, include_tangent=True):
    nominal = np.asarray(nominal, dtype=np.float32).reshape(-1)[:7]
    if nominal.size != 7:
        raise ValueError("nominal action must be seven dimensional")
    rows = [Candidate(nominal.copy(), "nominal")]
    for scale in (0.75, 0.5, 0.25):
        a = nominal.copy(); a[:6] *= scale
        rows.append(Candidate(_match_authority(a, nominal, max_delta), f"damped_{scale:g}"))
    hold = nominal.copy(); hold[:6] = 0.0
    rows.append(Candidate(_match_authority(hold, nominal, max_delta), "hold"))
    if eef_pos is not None and obstacle_pos is not None:
        away = np.asarray(eef_pos, float) - np.asarray(obstacle_pos, float)
        n = np.linalg.norm(away)
        if n > 1e-8:
            direction = away / n
            a = nominal.copy(); a[:3] = direction
            rows.append(Candidate(_match_authority(a, nominal, max_delta), "retreat"))
            if include_tangent:
                tangent = np.array([-direction[1], direction[0], 0.0])
                nt = np.linalg.norm(tangent)
                if nt > 1e-8:
                    tangent /= nt
                    for sign, name in ((1.0, "tangent_left"), (-1.0, "tangent_right")):
                        a = nominal.copy(); a[:3] = sign*tangent
                        rows.append(Candidate(_match_authority(a, nominal, max_delta), name))
    # Stable de-duplication keeps nominal tie-breaking deterministic.
    unique = []
    for row in rows:
        if not any(np.array_equal(row.action, x.action) for x in unique):
            unique.append(row)
    return unique
