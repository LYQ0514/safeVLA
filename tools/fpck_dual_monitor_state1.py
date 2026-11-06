"""FPCK/CKG 新旧监视器离线对比（裁决 §八）。

在 spatial/task1/state1（已知假阴性回归测试）上跑一条原始轨迹，同时驱动
legacy v1.2 自动机与 v1.3 GRASP_ACQUISITION 自动机，输出逐控制步对比。

预期（裁决 §八）：
- legacy 在 ~5.16mm 位移处：q_engaged → q_bad（UNAUTHORIZED_TARGET_MOTION）；
- v1.3：Resting → GraspAcquisition(left contact, support contact)
       → GraspAcquisition(bilateral) → Controlled(lift/support unloading)。

用法：
  python tools/fpck_dual_monitor_state1.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

ROOT = "/data/safeVLA"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "dataset", "SafeLIBERO", "safelibero"))
sys.path.insert(0, "/data/2608.SDL/MojocoProj/src/pi05")
sys.path.insert(0, os.path.join(ROOT, "third_party", "aegis"))

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

import pi05_libero_runner as R  # noqa: E402
from src.pnsg.policy_adapter import PnsgPolicy  # noqa: E402
from src.pnsg.crn import nominal_seed, sample_z  # noqa: E402
from src.fpck.task_automaton import TaskAutomaton  # noqa: E402
from src.fpck.task_automaton_acq import GraspAcquisitionAutomaton  # noqa: E402
from src.fpck.task_event_bridge import TaskEventBridge  # noqa: E402
from src.fpck.goal_evaluator import canonical_clause  # noqa: E402
from src.pnsg.role_manifest import build_role_manifest  # noqa: E402
from src.safety.aegis_adapted import find_active_obstacle  # noqa: E402

LEVEL = "I"
IMAGE_SIZE = 224
REPLAN = 5
NUM_STEPS_WAIT = 20
SUITE = "safelibero_spatial"
TASK_ID = 1
STATE_ID = 1
MAX_STEPS = 300


def make_env_local(ts, task, task_id, state_id):
    import pathlib
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256,
                             controller="OSC_POSE")
    env.seed(7)
    env.reset()
    env.set_init_state(ts.get_task_init_states(task_id)[state_id])
    dummy = np.zeros(7, dtype=np.float32); dummy[-1] = -1.0
    for _ in range(NUM_STEPS_WAIT):
        env.step(dummy)
    return env


def main():
    from libero.libero import benchmark, get_libero_path
    ts = benchmark.get_benchmark_dict()[SUITE](safety_level=LEVEL)
    task = ts.get_task(TASK_ID)
    prompt = task.language

    import pathlib
    bddl_path = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    manifest = build_role_manifest(bddl_path)
    manipulated_targets = [inst for inst, info in manifest["objects"].items()
                           if "MANIPULATED_TARGET" in info["roles"]]
    goal_keys = [canonical_clause(s) for s in manifest["goal_clauses"]]
    target_bodies = [t + "_main" for t in manipulated_targets]

    out_dir = os.path.join(ROOT, "data", "audit", "fpck_dual_monitor",
                           f"{SUITE}_t{TASK_ID}_s{STATE_ID}")
    os.makedirs(out_dir, exist_ok=True)
    raw_log_path = os.path.join(out_dir, "raw_events.jsonl")
    # 清空旧原始事件（append 模式会叠加多次运行）
    if os.path.exists(raw_log_path):
        os.remove(raw_log_path)

    # 双监视器：v1.3 为主，legacy v1.2 为对比
    legacy = TaskAutomaton(manipulated_targets, goal_keys)
    v13 = GraspAcquisitionAutomaton(manipulated_targets, goal_keys)

    env = make_env_local(ts, task, TASK_ID, STATE_ID)
    obs = env.env._get_observations()
    obstacle = find_active_obstacle(env, obs)
    obstacle_body = obstacle + "_main" if obstacle else None

    bridge = TaskEventBridge(v13, raw_log_path, manipulated_targets, goal_keys,
                             obstacle_body=obstacle_body, compare_automaton=legacy)
    from tools.g1a_soup_trace import body_state
    settled_baseline = {t: body_state(env, t)["pos"].copy() for t in target_bodies}
    bridge.reset(env, settled_baseline, target_bodies, obstacle_body)

    policy = PnsgPolicy()
    t0 = time.time()
    chunk = None
    query_idx = 0
    buf_idx = 0
    success_native = False
    native_step = None

    try:
        for step in range(MAX_STEPS):
            if chunk is None or buf_idx >= len(chunk):
                z0, _ = sample_z(nominal_seed(SUITE, TASK_ID, STATE_ID, 0, query_idx))
                po = R.obs_to_policy_obs(obs, prompt, image_size=IMAGE_SIZE)
                chunk = np.clip(policy.generate(z0, po), -1, 1)[:REPLAN]
                query_idx += 1
                buf_idx = 0
            action = chunk[buf_idx]
            obs, _r, _d, _i = env.step(action)
            chunk_hash = hashlib.sha256(np.asarray(chunk).tobytes()).hexdigest()[:16]
            bridge.observe_control_step(
                env=env, obs=obs, phase="ROLLOUT", control_step=step,
                nominal_action=action, executed_action=action,
                chunk_hash=chunk_hash, buffer_index=buf_idx,
                target_bodies=target_bodies)
            buf_idx += 1
            # 记录 native_success 但不 break：让策略继续运行到释放/终态，
            # 否则「未释放」可能是循环过早截断的产物（裁决 §一 需要完整 release 时间线）
            if env.check_success() and not success_native:
                success_native = True
                native_step = step
            # 自动机终态即停止
            if v13.state.mode in ("q_acc", "q_bad", "q_unknown"):
                break
    finally:
        policy.close()
    env.close()

    # 读取原始事件，提取双监视器时间线
    rows = [json.loads(l) for l in open(raw_log_path, encoding="utf-8")]

    print(f"=== 双监视器对比 {SUITE} task{TASK_ID} state{STATE_ID} ===")
    print(f"native_success={success_native}@{native_step}  事件数={len(rows)}")
    print(f"{'step':>4} {'disp_mm':>8} {'close':>5} {'L_f':>4} {'R_f':>4} "
          f"{'sweep_mm':>8} {'lift_mm':>7} {'legacy':>16} {'v1.3':>16}")
    for r in rows:
        if r.get("event_type") != "control_step":
            continue
        disp = max(r.get("target_displacement_mm", {}).values()) if r.get("target_displacement_mm") else 0.0
        print(f"{r['control_step']:>4} {disp:>8.2f} "
              f"{str(bool(r.get('deployed_close_intent'))):>5} "
              f"{str(bool(r.get('left_finger_contact'))):>4} "
              f"{str(bool(r.get('right_finger_contact'))):>4} "
              f"{float(r.get('lateral_sweep_mm') or 0):>8.2f} "
              f"{float(r.get('target_lifted_mm') or 0):>7.2f} "
              f"{str(r.get('compare_automaton_after')):>16} "
              f"{str(r.get('automaton_after')):>16}")

    # 关键差异点
    legacy_killed = next((r for r in rows
                          if r.get("event_type") == "control_step"
                          and r.get("compare_automaton_after") == "q_bad"), None)
    v13_killed = next((r for r in rows
                       if r.get("event_type") == "control_step"
                       and r.get("automaton_after") == "q_bad"), None)
    v13_controlled = next((r for r in rows
                           if r.get("event_type") == "control_step"
                           and r.get("automaton_after") in ("q_controlled", "q_released", "q_acc")), None)
    print("\n--- 关键差异 ---")
    if legacy_killed:
        d = max(legacy_killed.get("target_displacement_mm", {}).values()) if legacy_killed.get("target_displacement_mm") else 0.0
        print(f"legacy 首次 q_bad: step={legacy_killed['control_step']} disp={d:.2f}mm "
              f"violation={legacy.state.violation_code}")
    if v13_killed:
        print(f"v1.3 首次 q_bad: step={v13_killed['control_step']} "
              f"violation={v13_killed.get('violation_code')}")
    if v13_controlled:
        print(f"v1.3 首次 controlled: step={v13_controlled['control_step']}")
    print(f"\nlegacy 最终: {legacy.state.mode} / {legacy.state.violation_code}")
    print(f"v1.3 最终: {v13.state.mode} / {v13.state.violation_code}")
    print(f"v1.3 lifecycle: {v13.state.lifecycle}")
    print(f"→ {raw_log_path}")


if __name__ == "__main__":
    main()
