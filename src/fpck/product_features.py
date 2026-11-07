"""Fixed-width product-state and candidate feature contract for FPCK/CKG."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence
import numpy as np


MODES = ("q_init", "q_engaged", "q_acquisition", "q_controlled",
         "q_progress", "q_released", "q_acc", "q_bad", "q_unknown")
LIFECYCLES = ("RESTING", "GRASP_ACQUISITION", "TASK_CONTACT", "CONTROLLED",
              "TRANSPORTED", "RELEASED")

FEATURE_NAMES = (
    *(f"eef_pos_{a}" for a in "xyz"),
    *(f"gripper_qpos_{i}" for i in range(2)),
    *(f"target_rel_{a}" for a in "xyz"), "target_distance",
    *(f"goal_rel_{a}" for a in "xyz"), "goal_distance",
    *(f"obstacle_rel_{a}" for a in "xyz"), "obstacle_distance",
    *(f"mode_{m}" for m in MODES),
    *(f"lifecycle_frac_{m}" for m in LIFECYCLES),
    "goal_fraction", "budget_fraction", "horizon_fraction", "buffer_fraction",
    *(f"nominal_{i}" for i in range(7)),
    *(f"candidate_{i}" for i in range(7)),
    *(f"delta_{i}" for i in range(7)),
    "candidate_l2", "intervention_cost",
)


def _vec(obs, key, n):
    value = np.asarray(obs.get(key, np.zeros(n)), dtype=np.float64).reshape(-1)
    out = np.zeros(n, dtype=np.float64); out[:min(n, value.size)] = value[:n]
    return out


def _first_role_position(obs, manifest: Mapping, role: str):
    objects = manifest.get("objects", {}) if manifest else {}
    for name in sorted(objects):
        if role in objects[name].get("roles", []):
            for key in (name + "_pos", name.replace("_main", "") + "_pos"):
                if key in obs:
                    return _vec(obs, key, 3)
    return np.zeros(3)


@dataclass(frozen=True)
class ProductFeatureContext:
    manifest: Mapping
    automaton_state: object
    budget: int
    initial_budget: int
    horizon: int
    initial_horizon: int
    buffer_index: int = 0
    chunk_length: int = 1
    obstacle_name: str | None = None


def encode_product_candidate(obs, nominal: Sequence[float], candidate: Sequence[float],
                             context: ProductFeatureContext):
    nominal = np.asarray(nominal, float).reshape(-1)[:7]
    candidate = np.asarray(candidate, float).reshape(-1)[:7]
    if nominal.size != 7 or candidate.size != 7:
        raise ValueError("SafeLIBERO actions must have seven deployed dimensions")
    eef = _vec(obs, "robot0_eef_pos", 3)
    grip = _vec(obs, "robot0_gripper_qpos", 2)
    target = _first_role_position(obs, context.manifest, "MANIPULATED_TARGET")
    goal = _first_role_position(obs, context.manifest, "GOAL_RECEPTACLE")
    obstacle = (_vec(obs, context.obstacle_name + "_pos", 3)
                if context.obstacle_name and context.obstacle_name + "_pos" in obs
                else np.zeros(3))
    target_rel, goal_rel, obstacle_rel = target-eef, goal-eef, obstacle-eef
    state = context.automaton_state
    mode = getattr(state, "mode", "q_unknown")
    mode_onehot = [float(mode == m) for m in MODES]
    lifecycle = getattr(state, "lifecycle", {}) or {}
    denom = max(1, len(lifecycle))
    life = [sum(v == name for v in lifecycle.values()) / denom for name in LIFECYCLES]
    achieved = getattr(state, "achieved_mask", {}) or {}
    goal_fraction = sum(bool(v) for v in achieved.values()) / max(1, len(achieved))
    delta = candidate - nominal
    values = [
        *eef, *grip,
        *target_rel, np.linalg.norm(target_rel),
        *goal_rel, np.linalg.norm(goal_rel),
        *obstacle_rel, np.linalg.norm(obstacle_rel),
        *mode_onehot, *life, goal_fraction,
        context.budget/max(1, context.initial_budget),
        context.horizon/max(1, context.initial_horizon),
        context.buffer_index/max(1, context.chunk_length),
        *nominal, *candidate, *delta, np.linalg.norm(delta),
        float(np.linalg.norm(delta) > 1e-7),
    ]
    out = np.asarray(values, dtype=np.float32)
    if out.size != len(FEATURE_NAMES):
        raise AssertionError((out.size, len(FEATURE_NAMES)))
    if not np.all(np.isfinite(out)):
        raise ValueError("non-finite product feature")
    return out
