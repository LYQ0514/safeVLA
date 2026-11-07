"""Snapshot the non-MuJoCo actuation state needed by restored branches.

``sim.get_state()`` contains qpos/qvel but not robosuite's sticky gripper
target.  Candidate siblings must therefore restore both layers.  OSC goals are
also copied when present; the next ``env.step`` overwrites them, but restoring
them keeps the anchor identity explicit and robust to controller variants.
"""
from __future__ import annotations

import copy
import numpy as np


_CONTROLLER_FIELDS = (
    "goal_pos", "goal_ori", "relative_ori", "ori_ref", "initial_joint",
)


def _robots(env):
    base = getattr(env, "env", env)
    return list(getattr(base, "robots", ()))


def snapshot_control_state(env):
    rows = []
    for robot in _robots(env):
        row = {"gripper": {}, "controller": {}}
        gripper = getattr(robot, "gripper", None)
        if gripper is not None and hasattr(gripper, "current_action"):
            row["gripper"]["current_action"] = np.asarray(
                gripper.current_action).copy()
        controller = getattr(robot, "controller", None)
        if controller is not None:
            for name in _CONTROLLER_FIELDS:
                if hasattr(controller, name):
                    value = getattr(controller, name)
                    if value is not None:
                        row["controller"][name] = copy.deepcopy(value)
        rows.append(row)
    return rows


def restore_control_state(env, snapshot):
    robots = _robots(env)
    if len(robots) != len(snapshot):
        raise RuntimeError("robot count changed across counterfactual restore")
    for robot, row in zip(robots, snapshot):
        gripper = getattr(robot, "gripper", None)
        if "current_action" in row["gripper"]:
            if gripper is None:
                raise RuntimeError("gripper missing during counterfactual restore")
            gripper.current_action = np.asarray(
                row["gripper"]["current_action"]).copy()
        controller = getattr(robot, "controller", None)
        for name, value in row["controller"].items():
            if controller is None or not hasattr(controller, name):
                raise RuntimeError(f"controller field missing during restore: {name}")
            setattr(controller, name, copy.deepcopy(value))


def control_state_payload(snapshot):
    """Return a deterministic JSON-compatible identity payload.

    Signed zero is canonicalized to +0.0 so that ``0.0`` and ``-0.0`` (physically
    identical, but different JSON / bytes representations) hash the same.
    """
    out = []
    for row in snapshot:
        out.append({
            kind: {name: _canonical_list(np.asarray(value))
                   for name, value in fields.items()}
            for kind, fields in row.items()
        })
    return out


def _canonical_list(arr):
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr.copy()
    arr[arr == 0.0] = 0.0  # signed-zero canonicalization
    return arr.tolist()
