"""Product-space predictive safety projection for frozen VLA action chunks.

The shield optimizes a continuous action sequence around the frozen VLA chunk,
then verifies shortlisted sequences by restoring and rolling MuJoCo together
with the task automaton.  It requires no learned continuation classifier.

The implementation is an empirical approximation of the product-space MPC
operator.  ``verified=True`` means that the complete proposed prefix passed the
registered simulator and automaton checks; it is not a model-independent
real-world certificate.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import math
import time
from typing import Callable, Optional

import numpy as np

from src.fpck.control_state import restore_control_state, snapshot_control_state
from src.pnsg.full_state import restore_full_state, save_full_state


TERMINAL_MODES = {"q_acc", "q_bad", "q_unknown"}
MODE_VALUE = {
    "q_init": 0.0, "q_engaged": 0.5, "q_acquisition": 1.0,
    "q_controlled": 2.0, "q_progress": 3.0, "q_released": 4.0,
    "q_acc": 6.0, "q_bad": -8.0, "q_unknown": -6.0,
}
LIFECYCLE_VALUE = {
    "RESTING": 0.0, "TASK_CONTACT": 0.5, "GRASP_ACQUISITION": 1.0,
    "CONTROLLED": 2.0, "TRANSPORTED": 3.0, "RELEASED": 4.0,
}


def _wrapper_chain(env):
    seen, rows, cur = set(), [], env
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur)); rows.append(cur); cur = getattr(cur, "env", None)
    return rows


def _snapshot_episode_fields(env):
    fields = []
    for obj in _wrapper_chain(env):
        row = {}
        for name in ("timestep", "done", "cur_time", "_elapsed_steps"):
            if hasattr(obj, name):
                row[name] = copy.deepcopy(getattr(obj, name))
        fields.append(row)
    return fields


def _restore_episode_fields(env, fields):
    chain = _wrapper_chain(env)
    if len(chain) != len(fields):
        raise RuntimeError("environment wrapper chain changed during planning")
    for obj, row in zip(chain, fields):
        for name, value in row.items():
            setattr(obj, name, copy.deepcopy(value))


@dataclass
class RuntimeSnapshot:
    env: dict
    control: list
    episode: list
    bridge: dict
    raw_log_path: Optional[str]

    @classmethod
    def capture(cls, env, bridge):
        return cls(save_full_state(env), snapshot_control_state(env),
                   _snapshot_episode_fields(env), copy.deepcopy(bridge.snapshot()),
                   bridge.raw_log_path)

    def restore(self, env, bridge, *, planning=False):
        restore_full_state(env, self.env)
        restore_control_state(env, self.control)
        _restore_episode_fields(env, self.episode)
        bridge.restore(copy.deepcopy(self.bridge))
        bridge.raw_log_path = None if planning else self.raw_log_path


@dataclass(frozen=True)
class SequenceEvaluation:
    objective: float
    verified: bool
    final_mode: str
    violation_code: Optional[str]
    steps: int
    min_proxy_clearance: float
    progress_score: float


@dataclass(frozen=True)
class ShieldDecision:
    actions: np.ndarray
    status: str
    verified: bool
    objective: float
    nominal_objective: float
    evaluations: int
    runtime_ms: float
    final_mode: str
    violation_code: Optional[str]
    min_proxy_clearance: float


class ProductSpacePredictiveShield:
    """CEM sequence optimizer with exact restored-prefix verification."""

    def __init__(self, *, horizon=5, population=32, rollout_budget=8,
                 iterations=3, elite=4, sigma=0.45, min_sigma=0.08,
                 clearance_margin=0.002, deviation_weight=1.0,
                 smoothness_weight=0.08, clearance_weight=20.0,
                 progress_weight=0.25, seed=0):
        self.horizon = int(horizon); self.population = int(population)
        self.rollout_budget = int(rollout_budget); self.iterations = int(iterations)
        self.elite = int(elite); self.sigma = float(sigma)
        self.min_sigma = float(min_sigma); self.clearance_margin = float(clearance_margin)
        self.deviation_weight = float(deviation_weight)
        self.smoothness_weight = float(smoothness_weight)
        self.clearance_weight = float(clearance_weight)
        self.progress_weight = float(progress_weight); self.seed = int(seed)
        if not (1 <= self.elite <= self.rollout_budget <= self.population):
            raise ValueError("require elite <= rollout_budget <= population")

    @staticmethod
    def _progress(bridge):
        state = bridge.auto.state
        mode = MODE_VALUE.get(state.mode, -2.0)
        life = sum(LIFECYCLE_VALUE.get(x, 0.0)
                   for x in (getattr(state, "lifecycle", {}) or {}).values())
        achieved = getattr(state, "achieved_mask", {}) or {}
        goal = sum(bool(x) for x in achieved.values()) / max(1, len(achieved))
        return float(mode + life + 2.0 * goal)

    def _proxy_clearance(self, cps, obstacles, sequence):
        if not cps or not obstacles:
            return float("inf")
        from src.pnsg.geometry_proxy import compute_clearance
        values = compute_clearance(cps, obstacles, sequence[:, :6])
        return float(np.min(values)) if values.size else float("inf")

    def _analytic_cost(self, sequence, nominal, min_clearance):
        delta = sequence[:, :6] - nominal[:, :6]
        deviation = float(np.sum(delta * delta))
        smooth = float(np.sum(np.diff(sequence[:, :6], axis=0) ** 2))
        deficit = max(0.0, self.clearance_margin - min_clearance)
        return (self.deviation_weight * deviation + self.smoothness_weight * smooth
                + self.clearance_weight * deficit * deficit)

    def _evaluate(self, env, bridge, snapshot, target_bodies, nominal, sequence,
                  start_step, cps, obstacles):
        snapshot.restore(env, bridge, planning=True)
        min_clear = self._proxy_clearance(cps, obstacles, sequence)
        initial_progress = self._progress(bridge)
        mode = bridge.auto.state.mode; steps = 0
        try:
            for k, action in enumerate(sequence):
                obs, _reward, done, _info = env.step(np.asarray(action, np.float32))
                mode = bridge.observe_control_step(
                    env, obs, "PREDICTIVE_VERIFY", start_step + k,
                    nominal[k], action, target_bodies=target_bodies)
                steps = k + 1
                if mode in TERMINAL_MODES or done:
                    break
            progress = self._progress(bridge) - initial_progress
            clearance_fail = (math.isfinite(min_clear)
                              and min_clear < self.clearance_margin)
            hard_fail = (mode in ("q_bad", "q_unknown")
                         or (done and mode != "q_acc") or clearance_fail)
            objective = self._analytic_cost(sequence, nominal, min_clear)
            objective -= self.progress_weight * progress
            if hard_fail:
                objective += 1e5 + 100.0 * (self.horizon - steps)
            verified = not hard_fail
            return SequenceEvaluation(
                float(objective), bool(verified), mode,
                (bridge.auto.state.violation_code
                 or ("PROXY_CLEARANCE_MARGIN" if clearance_fail else None)),
                steps, min_clear, progress)
        except Exception:
            return SequenceEvaluation(1e9, False, "runtime_failure",
                                      "ROLLOUT_RUNTIME_FAILURE", steps,
                                      min_clear, -1.0)

    def _seed_sequences(self, nominal, obs, obstacle_name):
        seeds = [nominal.copy()]
        hold = nominal.copy(); hold[:, :6] = 0.0; seeds.append(hold)
        if obstacle_name and "robot0_eef_pos" in obs and obstacle_name + "_pos" in obs:
            away = (np.asarray(obs["robot0_eef_pos"], float)
                    - np.asarray(obs[obstacle_name + "_pos"], float))
            n = np.linalg.norm(away)
            if n > 1e-8:
                away /= n
                retreat = nominal.copy(); retreat[:, :3] = away; seeds.append(retreat)
                tangent = np.array([-away[1], away[0], 0.0])
                nt = np.linalg.norm(tangent)
                if nt > 1e-8:
                    tangent /= nt
                    for sign in (1.0, -1.0):
                        seq = nominal.copy(); seq[:, :3] = sign * tangent; seeds.append(seq)
        return seeds

    def plan(self, env, bridge, obs, nominal_chunk, target_bodies,
             obstacle_name, start_step=0):
        t0 = time.perf_counter()
        nominal = np.asarray(nominal_chunk, np.float32)
        if nominal.ndim != 2 or nominal.shape[1] != 7:
            raise ValueError("nominal chunk must have shape (H,7)")
        if len(nominal) < self.horizon:
            pad = np.repeat(nominal[-1:], self.horizon-len(nominal), axis=0)
            nominal = np.concatenate([nominal, pad], axis=0)
        nominal = nominal[:self.horizon].copy()
        snapshot = RuntimeSnapshot.capture(env, bridge)
        from src.pnsg.geometry_proxy import build_control_points, build_obstacles
        cps = build_control_points(env, obs); obstacles = build_obstacles(env, obs)
        rng = np.random.default_rng(self.seed + int(start_step))
        mean = nominal[:, :6].astype(float).copy()
        std = np.full_like(mean, self.sigma)
        evaluated = []

        try:
            seed_sequences = self._seed_sequences(nominal, obs, obstacle_name)
            for iteration in range(self.iterations):
                samples = np.empty((self.population, self.horizon, 7), np.float32)
                noise = rng.normal(size=(self.population, self.horizon, 6))
                samples[:, :, :6] = np.clip(mean + noise * std, -1.0, 1.0)
                samples[:, :, 6] = nominal[None, :, 6]
                if iteration == 0:
                    for i, seq in enumerate(seed_sequences[:self.population]):
                        samples[i] = seq
                proxy = []
                for i, seq in enumerate(samples):
                    clear = self._proxy_clearance(cps, obstacles, seq)
                    proxy.append((self._analytic_cost(seq, nominal, clear), i))
                shortlist = [i for _, i in sorted(proxy)[:self.rollout_budget]]
                round_eval = []
                for i in shortlist:
                    ev = self._evaluate(env, bridge, snapshot, target_bodies,
                                        nominal, samples[i], start_step, cps, obstacles)
                    evaluated.append((ev, samples[i].copy()))
                    round_eval.append((ev.objective, i))
                elite_idx = [i for _, i in sorted(round_eval)[:self.elite]]
                elite_actions = samples[elite_idx, :, :6].astype(float)
                mean = elite_actions.mean(axis=0)
                std = np.maximum(elite_actions.std(axis=0), self.min_sigma)

            nominal_eval = self._evaluate(env, bridge, snapshot, target_bodies,
                                          nominal, nominal, start_step, cps, obstacles)
            evaluated.append((nominal_eval, nominal.copy()))
            feasible = [(ev.objective, ev, seq) for ev, seq in evaluated if ev.verified]
            if feasible:
                _, best, actions = min(feasible, key=lambda x: x[0])
                status = ("verified_nominal" if np.array_equal(actions, nominal)
                          else "verified_projection")
                verified = True
            else:
                # Registered backup: zero pose motion while preserving the VLA's
                # gripper intent. Evaluate it even when proxy shortlisting did
                # not select it; only call it verified when the full restored
                # prefix and clearance gate pass.
                actions = nominal.copy(); actions[:, :6] = 0.0
                backup = self._evaluate(env, bridge, snapshot, target_bodies,
                                        nominal, actions, start_step, cps, obstacles)
                evaluated.append((backup, actions.copy()))
                if backup.verified:
                    best = backup; status = "verified_backup_hold"; verified = True
                else:
                    best = backup; status = "unverified_backup_hold"; verified = False
            return ShieldDecision(
                np.asarray(actions, np.float32), status, verified,
                float(best.objective), float(nominal_eval.objective), len(evaluated),
                (time.perf_counter()-t0)*1000.0, best.final_mode,
                best.violation_code, best.min_proxy_clearance)
        finally:
            snapshot.restore(env, bridge, planning=False)

