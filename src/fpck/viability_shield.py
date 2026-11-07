"""Counterfactual-resume viability shield for a frozen VLA.

Unlike a short-horizon collision filter, this method validates two connected
segments from the same restored product state:

1. a task-intent-preserving geometric repair of the current pi0.5 chunk; and
2. a fresh frozen-pi0.5 chunk queried at the repaired terminal observation.

A prefix is deployable only if the prefix and resume segment remain outside
q_bad/q_unknown, task progress does not regress, task geometry does not regress
beyond a registered tolerance, and the repair retains a minimum fraction of
the nominal motion.  This explicitly rejects the zero-motion safety-freeze
solution that dominated the first product-space shield.

The guarantee is simulator- and horizon-relative.  It is not a universal
real-world safety certificate.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Optional

import numpy as np

from src.fpck.predictive_shield import RuntimeSnapshot, ProductSpacePredictiveShield
TERMINAL_FAILURE = {"q_bad", "q_unknown"}


def _body_pos(env, logical_name):
    from tools.g1a_soup_trace import body_state
    body = logical_name if logical_name.endswith("_main") else logical_name + "_main"
    return np.asarray(body_state(env, body)["pos"], dtype=np.float64)


def goal_pairs_from_manifest(manifest):
    """Return registered (target, receptacle) pairs for On/In clauses."""
    regions = manifest.get("regions", [])
    objects = manifest.get("objects", {})
    pairs = []
    for pred, args in manifest.get("goal_clauses", []):
        if str(pred).lower() not in ("on", "in") or len(args) < 2:
            continue
        target, second = args[0], args[1]
        receptacle = second if second in objects else None
        if receptacle is None:
            for region in regions:
                if second.endswith("_" + region.get("name", "")):
                    receptacle = region.get("target")
                    break
        if target in objects and receptacle in objects:
            pairs.append((target, receptacle))
    return pairs


def task_geometry_potential(env, goal_pairs):
    """Negative mean target-to-receptacle distance; None if not applicable."""
    distances = []
    for target, receptacle in goal_pairs:
        try:
            distances.append(float(np.linalg.norm(
                _body_pos(env, target) - _body_pos(env, receptacle))))
        except Exception:
            continue
    return None if not distances else -float(np.mean(distances))


def stage_task_potential(env, bridge, goal_pairs):
    """Dense task potential matched to the monitored manipulation stage.

    Before control is established, progress is EEF-to-target approach.  After
    control, progress is target-to-receptacle transport.  This prevents an
    obstacle retreat from looking task-neutral merely because the object has
    not moved yet.
    """
    try:
        from tools.g1a_soup_trace import eef_pos
        eef = np.asarray(eef_pos(env), dtype=np.float64)
    except Exception:
        eef = None
    lifecycle = getattr(bridge.auto.state, "lifecycle", {}) or {}
    values = []
    for target, receptacle in goal_pairs:
        try:
            target_pos = _body_pos(env, target)
            life = lifecycle.get(target, "RESTING")
            if life in ("RESTING", "TASK_CONTACT", "GRASP_ACQUISITION") and eef is not None:
                values.append(-float(np.linalg.norm(eef - target_pos)))
            else:
                values.append(-float(np.linalg.norm(
                    target_pos - _body_pos(env, receptacle))))
        except Exception:
            continue
    return None if not values else float(np.mean(values))


@dataclass(frozen=True)
class ViabilityEvaluation:
    source: str
    actions: np.ndarray
    verified: bool
    prefix_safe: bool
    resume_safe: bool
    motion_retained: bool
    final_mode: str
    violation_code: Optional[str]
    prefix_steps: int
    resume_steps: int
    safe_steps: int
    progress_delta: float
    goal_potential_delta: Optional[float]
    intent_ratio: float
    intent_cosine: float
    min_proxy_clearance: float
    deviation: float
    resume_chunk_hash: Optional[str]


@dataclass(frozen=True)
class ViabilityDecision:
    actions: np.ndarray
    source: str
    status: str
    verified: bool
    evaluations: int
    runtime_ms: float
    final_mode: str
    violation_code: Optional[str]
    progress_delta: float
    goal_potential_delta: Optional[float]
    intent_ratio: float
    intent_cosine: float
    min_proxy_clearance: float
    resume_chunk_hash: Optional[str]
    candidate_summaries: tuple


class CounterfactualResumeViabilityShield:
    """Task-conditioned shield with an explicit frozen-VLA resume test."""

    def __init__(self, *, horizon=5, resume_horizon=5, execute_steps=2,
                 h_floor=0.002, constraint_margin=0.015,
                 max_constraints=6, correction_scales=(0.75, 1.0, 1.25, 1.5),
                 min_intent_ratio=0.30, min_intent_cosine=0.0,
                 goal_regression_tolerance=0.02,
                 progress_regression_tolerance=0.0,
                 normal_macro_strengths=(0.35, 0.60)):
        self.horizon = int(horizon)
        self.resume_horizon = int(resume_horizon)
        self.execute_steps = int(execute_steps)
        self.h_floor = float(h_floor)
        self.constraint_margin = float(constraint_margin)
        self.max_constraints = int(max_constraints)
        self.correction_scales = tuple(float(x) for x in correction_scales)
        self.min_intent_ratio = float(min_intent_ratio)
        self.min_intent_cosine = float(min_intent_cosine)
        self.goal_regression_tolerance = float(goal_regression_tolerance)
        self.progress_regression_tolerance = float(progress_regression_tolerance)
        self.normal_macro_strengths = tuple(float(x) for x in normal_macro_strengths)
        if self.horizon < 1 or self.resume_horizon < 1:
            raise ValueError("horizons must be positive")
        if not 1 <= self.execute_steps <= self.horizon:
            raise ValueError("execute_steps must lie in [1,horizon]")
        if not 0.0 < self.min_intent_ratio <= 1.0:
            raise ValueError("min_intent_ratio must lie in (0,1]")
        if not -1.0 <= self.min_intent_cosine <= 1.0:
            raise ValueError("min_intent_cosine must lie in [-1,1]")

    @staticmethod
    def _progress(bridge):
        return ProductSpacePredictiveShield._progress(bridge)

    @staticmethod
    def _proxy_clearance(cps, obstacles, sequence):
        if not cps or not obstacles:
            return float("inf")
        from src.pnsg.geometry_proxy import compute_clearance
        values = compute_clearance(cps, obstacles, sequence[:, :6])
        return float(np.min(values)) if values.size else float("inf")

    @staticmethod
    def _deduplicate(rows):
        out, seen = [], set()
        for source, seq in rows:
            key = np.round(np.asarray(seq, dtype=np.float64), 6).tobytes()
            if key not in seen:
                seen.add(key); out.append((source, np.asarray(seq, np.float32)))
        return out

    def _candidate_sequences(self, env, obs, nominal, cps, obstacles,
                             obstacle_name):
        rows = [("nominal", nominal.copy())]
        from src.pnsg.geometry_proxy import selected_constraints
        from src.pnsg.source_solver import minimum_norm_halfspaces
        constraints = selected_constraints(
            cps, obstacles, nominal[:, :6], h_floor=self.h_floor,
            max_constraints=self.max_constraints, margin=self.constraint_margin)
        if constraints:
            C = np.vstack([c for c, _ in constraints])
            b = np.asarray([x for _, x in constraints], dtype=np.float64)
            delta, _ = minimum_norm_halfspaces(C, b)
            if delta is not None:
                delta = delta.reshape(len(nominal), 6)
                for alpha in self.correction_scales:
                    seq = nominal.copy()
                    seq[:, :6] = np.clip(
                        nominal[:, :6] + alpha * delta, -1.0, 1.0)
                    rows.append((f"min_norm_projection_{alpha:g}", seq))

        # Non-local escape candidates provide control authority when the local
        # linear repair is sub-threshold.  Nominal components are retained, and
        # the exact two-segment viability test decides whether they are usable.
        if (obstacle_name and "robot0_eef_pos" in obs
                and obstacle_name + "_pos" in obs):
            away = (np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
                    - np.asarray(obs[obstacle_name + "_pos"], dtype=np.float64))
            away[2] = max(0.0, away[2])
            n = float(np.linalg.norm(away))
            if n > 1e-9:
                away /= n
                tangent = np.asarray([-away[1], away[0], 0.0])
                tangent /= max(1e-9, float(np.linalg.norm(tangent)))
                for strength in self.normal_macro_strengths:
                    seq = nominal.copy()
                    seq[:, :3] = np.clip(
                        0.70 * nominal[:, :3] + strength * away, -1.0, 1.0)
                    rows.append((f"task_retaining_retreat_{strength:g}", seq))
                    for sign in (-1.0, 1.0):
                        seq = nominal.copy()
                        correction = 0.55 * strength * away + sign * strength * tangent
                        seq[:, :3] = np.clip(
                            0.75 * nominal[:, :3] + correction, -1.0, 1.0)
                        rows.append((f"task_retaining_tangent_{sign:+g}_{strength:g}", seq))
        return self._deduplicate(rows)

    def _evaluate(self, *, env, bridge, snapshot, policy, prompt, manifest,
                  target_bodies, nominal, sequence, source, start_step,
                  cps, obstacles):
        snapshot.restore(env, bridge, planning=True)
        initial_progress = self._progress(bridge)
        pairs = goal_pairs_from_manifest(manifest)
        initial_potential = stage_task_potential(env, bridge, pairs)
        min_clearance = self._proxy_clearance(cps, obstacles, sequence)
        nominal_energy = float(np.linalg.norm(nominal[:, :6]))
        selected_energy = float(np.linalg.norm(sequence[:, :6]))
        intent_ratio = (1.0 if nominal_energy < 1e-6
                        else selected_energy / nominal_energy)
        intent_cosine = (1.0 if nominal_energy < 1e-6 or selected_energy < 1e-9
                         else float(np.sum(nominal[:, :6] * sequence[:, :6]) /
                                    (nominal_energy * selected_energy)))
        motion_retained = (nominal_energy < 1e-6
                           or (intent_ratio >= self.min_intent_ratio and
                               intent_cosine >= self.min_intent_cosine))
        prefix_safe = True; resume_safe = False
        prefix_steps = resume_steps = safe_steps = 0
        mode = bridge.auto.state.mode; obs = None
        resume_hash = None
        try:
            for k, action in enumerate(sequence):
                obs, _reward, done, _info = env.step(np.asarray(action, np.float32))
                mode = bridge.observe_control_step(
                    env, obs, "VIABILITY_PREFIX", start_step + k,
                    nominal[k], action, target_bodies=target_bodies)
                prefix_steps = k + 1
                if mode in TERMINAL_FAILURE or (done and mode != "q_acc"):
                    prefix_safe = False; break
                safe_steps += 1
                if mode == "q_acc": break

            if prefix_safe and mode == "q_acc":
                resume_safe = True
            elif prefix_safe and obs is not None:
                resume, resume_meta = policy.counterfactual_chunk(obs, prompt)
                resume_hash = resume_meta.get("chunk_hash")
                for j, action in enumerate(resume[:self.resume_horizon]):
                    obs, _reward, done, _info = env.step(np.asarray(action, np.float32))
                    mode = bridge.observe_control_step(
                        env, obs, "FROZEN_VLA_RESUME",
                        start_step + len(sequence) + j, action, action,
                        chunk_hash=resume_hash, buffer_index=j,
                        target_bodies=target_bodies)
                    resume_steps = j + 1
                    if mode in TERMINAL_FAILURE or (done and mode != "q_acc"):
                        break
                    safe_steps += 1
                    if mode == "q_acc": break
                resume_safe = mode not in TERMINAL_FAILURE and not (
                    done and mode != "q_acc")

            progress_delta = self._progress(bridge) - initial_progress
            final_potential = stage_task_potential(env, bridge, pairs)
            potential_delta = (None if initial_potential is None or final_potential is None
                               else float(final_potential - initial_potential))
            progress_ok = progress_delta >= -self.progress_regression_tolerance
            # A lifecycle transition changes the potential's reference from
            # EEF-to-target to target-to-goal.  Positive symbolic progress is
            # therefore sufficient at that discontinuity; otherwise enforce
            # the dense geometric non-regression bound.
            geometry_ok = (progress_delta > 0.0 or potential_delta is None or
                           potential_delta >= -self.goal_regression_tolerance)
            verified = bool(prefix_safe and resume_safe and motion_retained
                            and progress_ok and geometry_ok)
            deviation = float(np.sum((sequence[:, :6] - nominal[:, :6]) ** 2))
            return ViabilityEvaluation(
                source, sequence.copy(), verified, prefix_safe, resume_safe,
                motion_retained, mode, bridge.auto.state.violation_code,
                prefix_steps, resume_steps, safe_steps, float(progress_delta),
                potential_delta, float(intent_ratio), float(intent_cosine),
                min_clearance, deviation,
                resume_hash)
        except Exception as exc:
            return ViabilityEvaluation(
                source, sequence.copy(), False, False, False, motion_retained,
                "runtime_failure", f"{type(exc).__name__}: {exc}", prefix_steps,
                resume_steps, safe_steps, -math.inf, None, float(intent_ratio),
                float(intent_cosine), min_clearance, float("inf"), resume_hash)

    @staticmethod
    def _verified_rank(ev):
        # Lexicographic priorities prevent scalar weights from buying safety or
        # task viability with a small numerical improvement elsewhere.
        return (
            1 if ev.final_mode == "q_acc" else 0,
            ev.progress_delta,
            -math.inf if ev.goal_potential_delta is None else ev.goal_potential_delta,
            ev.min_proxy_clearance,
            -ev.deviation,
        )

    @staticmethod
    def _fallback_rank(ev):
        return (
            1 if ev.prefix_safe else 0,
            1 if ev.resume_safe else 0,
            ev.safe_steps,
            ev.progress_delta,
            -math.inf if ev.goal_potential_delta is None else ev.goal_potential_delta,
            ev.intent_ratio,
            -ev.deviation,
        )

    def plan(self, *, env, bridge, obs, nominal_chunk, policy, prompt,
             manifest, target_bodies, obstacle_name, start_step=0):
        t0 = time.perf_counter()
        nominal = np.asarray(nominal_chunk, dtype=np.float32)
        if nominal.ndim != 2 or nominal.shape[1] != 7 or len(nominal) < 1:
            raise ValueError("nominal_chunk must have shape (H,7), H>=1")
        # Never pad across a VLA query boundary.  The counterfactual resume
        # query is precisely the policy response after the remaining chunk.
        nominal = nominal[:self.horizon].copy()
        snapshot = RuntimeSnapshot.capture(env, bridge)
        from src.pnsg.geometry_proxy import build_control_points, build_obstacles
        cps = build_control_points(env, obs)
        obstacles = build_obstacles(env, obs)
        rows = self._candidate_sequences(
            env, obs, nominal, cps, obstacles, obstacle_name)
        evaluations = []
        try:
            for source, sequence in rows:
                evaluations.append(self._evaluate(
                    env=env, bridge=bridge, snapshot=snapshot, policy=policy,
                    prompt=prompt, manifest=manifest,
                    target_bodies=target_bodies, nominal=nominal,
                    sequence=sequence, source=source, start_step=start_step,
                    cps=cps, obstacles=obstacles))
            viable = [x for x in evaluations if x.verified]
            if viable:
                best = max(viable, key=self._verified_rank)
                status = ("viable_nominal" if best.source == "nominal"
                          else "viable_projection")
                verified = True
            else:
                # No zero-motion backup: choose the longest safe,
                # task-retaining candidate and expose that it is unverified.
                best = max(evaluations, key=self._fallback_rank)
                status = "unverified_best_effort_nonstop"
                verified = False
            take = min(self.execute_steps, len(best.actions))
            summaries = tuple({
                "source": x.source,
                "verified": x.verified,
                "prefix_safe": x.prefix_safe,
                "resume_safe": x.resume_safe,
                "motion_retained": x.motion_retained,
                "final_mode": x.final_mode,
                "violation_code": x.violation_code,
                "safe_steps": x.safe_steps,
                "progress_delta": x.progress_delta,
                "goal_potential_delta": x.goal_potential_delta,
                "intent_ratio": x.intent_ratio,
                "intent_cosine": x.intent_cosine,
                "min_proxy_clearance": x.min_proxy_clearance,
                "deviation": x.deviation,
                "resume_chunk_hash": x.resume_chunk_hash,
            } for x in evaluations)
            return ViabilityDecision(
                best.actions[:take].copy(), best.source, status, verified,
                len(evaluations), (time.perf_counter() - t0) * 1000.0,
                best.final_mode, best.violation_code, best.progress_delta,
                best.goal_potential_delta, best.intent_ratio,
                best.intent_cosine, best.min_proxy_clearance,
                best.resume_chunk_hash, summaries)
        finally:
            snapshot.restore(env, bridge, planning=False)
