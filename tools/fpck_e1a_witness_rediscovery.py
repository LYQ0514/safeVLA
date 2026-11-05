"""FPCK/CKG E1a witness 重发现 runner（事件级 instrumented 版本）。

复用历史 RRT 机制（reconstruct_nvs / plan_main_path），用 instrumented 循环
（execute_joint_path_instrumented / run_pv_branch_instrumented）+ TaskEventBridge
接入任务自动机，按 v1.2 规则生成事件级标签。

正式流程：
  create one TaskAutomaton + one TaskEventBridge
  record settled baseline
  reconstruct_nvs
  plan_main_path（RRT）
  execute_joint_path_instrumented(bridge)   # Joint-PD 预抓取
  save handoff snapshot
  if bridge q_bad/q_unknown: 终止（不跑 P-V）
  else: run_pv_branch_instrumented(bridge)  # π0.5 rollout
  save terminal snapshot
  finalize trace + summary

关键：
- 自动机跨 PREGRASP→REENTRY 不重置；
- HANDOFF_PRE_ACTION_CHECK：交还快照后、首个 π0.5 动作前读 goal/native/自动机，
  捕获历史 State02 的 success@step0；
- classify_from_history 仅保留用于旧结果对照，legacy_diagnostic_used_for_label=false。

用法：
  python tools/fpck_e1a_witness_rediscovery.py --case State02 --run 0 --dry-run
  python tools/fpck_e1a_witness_rediscovery.py --case State02 --run 0
"""

from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess
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
import mujoco  # noqa: E402
from scipy.spatial.transform import Rotation as Rot  # noqa: E402

from src.pnsg.policy_adapter import PnsgPolicy  # noqa: E402
from src.pnsg.full_state import save_full_state  # noqa: E402
from src.fpck.task_automaton import TaskAutomaton, bind_four_labels, route_e1_label  # noqa: E402
from src.fpck.task_event_bridge import TaskEventBridge  # noqa: E402
from src.fpck.goal_evaluator import canonical_clause  # noqa: E402
from src.fpck.instrumented_loops import (  # noqa: E402
    execute_joint_path_instrumented, run_pv_branch_instrumented)
from src.pnsg.role_manifest import build_role_manifest  # noqa: E402
from src.safety.contract_objects import robot_geoms, forbidden_contact_geoms  # noqa: E402
from tools.g1a_soup_trace import body_state  # noqa: E402

# 历史机制（import 复用，不改行为）
from tools.m1_p4_p1_reentry import plan_main_path  # noqa: E402
from tools.m1_p4_p0e_joint_execution import reconstruct_nvs, SOUP_BODY, MUG_BODY  # noqa: E402
from tools.m1_p4_p0_reachability import get_arm_q, set_arm_q  # noqa: E402
from tools.m1_p2_p1a_pilot import SUITE, LEVEL, TASK_ID, REPLAN_STEPS, IMAGE_SIZE  # noqa: E402

CASES = {
    "State02": {"state_id": 2, "stream": 0},
    "S01": {"state_id": 1, "stream": 3},
    "S04": {"state_id": 4, "stream": 3},
}

SPEC_VERSION = "fpck-task-automaton-v1.2"
CHECKPOINT_HASH = "b838d8b85a3f317c91950738172ee7f174afe7f2d36d9d8ed80f7110a085e90b"
MAX_STEPS = 550
N_RUNS = 3
IMPLEMENTATION_STATUS = "EVENT_LEVEL_V1_2"

# 历史源哈希（复用声明）
HISTORICAL_SOURCE_HASHES = {
    "m1_p4_p1_reentry.py": "a98e558f78ac0818834b0f06ba5b0247",
    "m1_p4_p0e_joint_execution.py": "abd7f68da6d312b3cb811d43b32170a2",
}

# 契约 hash（读取实际文件）
def _file_sha256(path):
    try:
        return hashlib.sha256(open(path, "rb").read()).hexdigest()
    except Exception:
        return None


def sha256_hex(obj):
    if isinstance(obj, np.ndarray):
        obj = np.asarray(obj).tobytes()
    elif isinstance(obj, (list, tuple, dict)):
        obj = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    elif not isinstance(obj, bytes):
        obj = str(obj).encode()
    return hashlib.sha256(obj).hexdigest()


def make_automaton(bddl_path):
    manifest = build_role_manifest(bddl_path)
    targets = [inst for inst, info in manifest["objects"].items()
               if "MANIPULATED_TARGET" in info["roles"]]
    goal_keys = [canonical_clause(s) for s in manifest["goal_clauses"]]
    return TaskAutomaton(targets, goal_keys), manifest


def make_manifest(a, manifest, bddl_sha):
    """run_manifest.json（启动前静态检查）。"""
    role_manifest_hash = manifest.get("role_manifest_sha256")
    contract_hash = _file_sha256(os.path.join(ROOT, "data", "collision_event_contract_v2.json"))
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).decode()
    # 只报告 tracked 文件的 dirty（忽略历史无关的 untracked）
    dirty_tracked = [l for l in dirty.splitlines() if not l.startswith("??")]
    return {
        "repo_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
        "dirty_worktree": dirty_tracked,
        "checkpoint_hash": CHECKPOINT_HASH,
        "suite": SUITE, "level": LEVEL, "task_id": TASK_ID,
        "state_id": a.case_map["state_id"], "stream": a.case_map["stream"],
        "run_idx": a.run_idx,
        "case": a.case,
        "nominal_seed_scheme": "CRN nominal_seed(suite,task,state,stream,query)",
        "rrt_seed_schedule": "historical fixed (plan_main_path: range(5) + default_rng(oi*100+off))",
        "historical_source_hashes": HISTORICAL_SOURCE_HASHES,
        "collector_version": "event-level TaskEventBridge v1.2",
        "automaton_spec_version": SPEC_VERSION,
        "role_manifest_hash": role_manifest_hash,
        "contact_contract_hash": contract_hash,
        "bddl_sha256": bddl_sha,
        "max_steps": MAX_STEPS,
        "replan_steps": REPLAN_STEPS,
        "implementation_status": IMPLEMENTATION_STATUS,
        "parent_code_commit": subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT).decode().strip(),
        "label_router": "route_e1_label",
        "label_router_version": "v1",
        "legacy_diagnostic_used_for_label": False,
        "State02_closed": True,
        "historical_candidate": {
            "State02": "State02_NATIVE_FALSE_POSITIVE_CANDIDATE",
            "S01": "S01_UNSAFE_RETURN",
            "S04": "S04_TASK_INTEGRITY_OR_STALL",
        }.get(a.case, None),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=list(CASES))
    ap.add_argument("--run", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    a.case_map = CASES[a.case]
    a.run_idx = a.run

    from libero.libero import benchmark, get_libero_path
    ts = benchmark.get_benchmark_dict()[SUITE](safety_level=LEVEL)
    task = ts.get_task(TASK_ID)
    prompt = task.language

    import pathlib
    bddl_path = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    bddl_sha = hashlib.sha256(bddl_path.read_bytes()).hexdigest()
    auto, manifest = make_automaton(bddl_path)

    out_dir = os.path.join(ROOT, "data", "audit", "fpck_e1a_witness",
                           a.case, f"run{a.run_idx}")
    snapshots_dir = os.path.join(out_dir, "snapshots")
    os.makedirs(snapshots_dir, exist_ok=True)

    if a.dry_run:
        m = make_manifest(a, manifest, bddl_sha)
        print("[dry-run] manifest 字段:")
        for k in ("case", "state_id", "stream", "run_idx", "implementation_status"):
            print(f"  {k} = {m[k]}")
        print(f"[dry-run] manipulated_targets = {[k for k,v in manifest['objects'].items() if 'MANIPULATED_TARGET' in v['roles']]}")
        print(f"[dry-run] goal_clauses = {manifest['goal_clauses']}")
        print("[dry-run] OK：未执行重型 rollout")
        return

    # ---- 正式执行 ----
    state_id = a.case_map["state_id"]
    stream = a.case_map["stream"]
    run_idx = a.run_idx

    manifest = make_manifest(a, manifest, bddl_sha)
    manifest_path = os.path.join(out_dir, "run_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # 自动机 + bridge（各创建一次）
    auto, mf = make_automaton(bddl_path)
    manipulated_targets = [inst for inst, info in mf["objects"].items()
                           if "MANIPULATED_TARGET" in info["roles"]]
    goal_keys = [canonical_clause(s) for s in mf["goal_clauses"]]
    target_bodies = [t + "_main" for t in manipulated_targets]

    raw_log_path = os.path.join(out_dir, "raw_events.jsonl")
    bridge = TaskEventBridge(auto, raw_log_path, manipulated_targets, goal_keys,
                             obstacle_body=MUG_BODY)

    policy = PnsgPolicy()
    t0 = time.time()
    summary = {
        "case": a.case, "state_id": state_id, "stream": stream, "run_idx": run_idx,
        "implementation_status": IMPLEMENTATION_STATUS,
        "legacy_diagnostic_used_for_label": False,
        "historical_source_hashes": HISTORICAL_SOURCE_HASHES,
        "identity_verified": False,
        "started_at": time.time(),
    }

    try:
        # 1. reconstruct_nvs → q_start
        env, layer, scratch, layer_s, rg, og, p0, soup_pos, nvs_step, Dmax0 = \
            reconstruct_nvs(ts, task, prompt, policy, state_id, stream)
        summary["nvs_step"] = nvs_step
        summary["Dmax0_mm"] = Dmax0 * 1000

        # settled baseline（reconstruct 后，Joint-PD 前）
        settled_baseline = {t: body_state(env, t)["pos"].copy() for t in target_bodies}
        bridge.reset(env, settled_baseline, target_bodies, MUG_BODY)

        q_start = get_arm_q(env)
        summary["q_start_hash"] = sha256_hex(q_start)

        # 2. plan_main_path（RRT）
        rg2 = robot_geoms(env)
        fg2 = forbidden_contact_geoms(env)
        obs = env.env._get_observations()
        iq = np.asarray(obs["robot0_eef_quat"])
        base_ori = Rot.from_quat([iq[1], iq[2], iq[3], iq[0]]).as_matrix()
        q_goal, path, gap_mm = plan_main_path(env, rg2, fg2, q_start, soup_pos, base_ori)
        if q_goal is None:
            summary["termination"] = "NO_PATH"
            summary["final_e1_label"] = "NO_PATH"
        else:
            summary["selected_path_hash"] = sha256_hex(np.asarray(path))
            summary["gap_mm"] = gap_mm

            # 3. execute_joint_path_instrumented（Joint-PD 预抓取，bridge 接入）
            set_arm_q(env, q_start)
            mujoco.mj_forward(env.sim.model._model, env.sim.data._data)
            env.sim.data._data.qvel[:] = 0.0
            Dmax_exec = execute_joint_path_instrumented(
                env, path, p0, bridge, target_bodies, settled_baseline, MUG_BODY)
            summary["Dmax_pregrasp_mm"] = max(Dmax0, Dmax_exec) * 1000

            # 4. handoff snapshot（bridge 状态 + env 状态）
            handoff_bridge_snap = bridge.snapshot()
            handoff_env_snap = save_full_state(env)
            summary["handoff_env_hash"] = sha256_hex(handoff_env_snap["flattened"])
            summary["handoff_auto_mode"] = auto.state.mode
            handoff_snap_path = os.path.join(snapshots_dir, "handoff.json")
            with open(handoff_snap_path, "w") as f:
                json.dump({"bridge": _jsonable(handoff_bridge_snap),
                           "env_hash": summary["handoff_env_hash"]}, f,
                          ensure_ascii=False, indent=2)

            # 5. 判断 handoff eligible
            handoff_eligible = auto.state.mode not in ("q_bad", "q_unknown", "q_acc")
            summary["handoff_eligible"] = handoff_eligible

            if not handoff_eligible:
                summary["termination"] = f"pregrasp_{auto.state.mode}"
                # 正式标签在 finalize 阶段由 route_e1_label 统一生成
                summary["pregrasp_verdict"] = auto.state.mode
            else:
                # OSC 接管（不 reset bridge）
                c = env.robots[0].controller
                q_now = get_arm_q(env)
                c.update_initial_joints(q_now)
                obs = env.env._get_observations(force_update=True)

                # HANDOFF_PRE_ACTION_CHECK：交还后、首个 π0.5 动作前
                bridge._emit({
                    "phase": "HANDOFF", "control_step": -1,
                    "event_type": "HANDOFF_PRE_ACTION_CHECK",
                    "goal_clause_truth": eval_goal_clauses_raw(env),
                    "native_success": bool(env.check_success()),
                    "automaton_mode": auto.state.mode,
                })

                # 6. run_pv_branch_instrumented（π0.5 rollout，bridge 接入）
                r = run_pv_branch_instrumented(
                    env, layer, scratch, layer_s, obs, Dmax0,
                    policy, prompt, state_id, stream, bridge, target_bodies,
                    max_steps=MAX_STEPS)
                summary.update(r)

                # 7. terminal snapshot
                terminal_env_snap = save_full_state(env)
                summary["terminal_env_hash"] = sha256_hex(terminal_env_snap["flattened"])
                summary["final_automaton_mode"] = auto.state.mode
                summary["pregrasp_verdict"] = "eligible"
                summary["reentry_verdict"] = auto.state.mode

        env.close()
        scratch.close() if scratch is not None else None
    except Exception as e:
        summary["termination"] = "RUNTIME_FAILURE"
        summary["error"] = str(e)
        import traceback
        summary["traceback"] = traceback.format_exc()
    finally:
        policy.close()

    # finalize
    fin = bridge.finalize()
    summary.update(fin)
    summary["wall_s"] = time.time() - t0
    # raw trace hash
    try:
        summary["raw_trace_sha256"] = hashlib.sha256(
            open(raw_log_path, "rb").read()).hexdigest()
    except Exception:
        summary["raw_trace_sha256"] = None
    # semantic trace hash（排除 wall_time 等易变字段，可跨 run 对比）
    try:
        from src.fpck.canonical_hash import semantic_trace_hash, SEMANTIC_HASH_VERSION
        summary["semantic_trace_sha256"] = semantic_trace_hash(raw_log_path)
        summary["semantic_hash_version"] = SEMANTIC_HASH_VERSION
    except Exception:
        summary["semantic_trace_sha256"] = None
    summary["identity_verified"] = (summary.get("q_start_hash") is not None
                                    and summary.get("raw_trace_sha256") is not None)

    # ---- 正式标签路由（route_e1_label，结合违规阶段/handoff/违规码/q_acc）----
    decisive_phase, accepted_before_violation = _derive_from_trace(raw_log_path)
    primary_code = None
    if fin.get("primary_violation"):
        primary_code = fin["primary_violation"][3]
    handoff_elig = summary.get("handoff_eligible", False)
    final_mode = summary.get("final_automaton_mode", auto.state.mode)
    timed_out = not summary.get("success", False)
    safe_valid = final_mode not in ("q_bad", "q_unknown")
    final_label, subreason = route_e1_label(
        handoff_eligible=handoff_elig,
        decisive_phase=decisive_phase,
        primary_violation_code=primary_code,
        accepted_before_violation=accepted_before_violation,
        final_mode=final_mode,
        timed_out=timed_out,
        safe_and_integrity_valid=safe_valid,
        stalled=False,  # 停滞窗判定留待 E1b 完整实现
        identity_verified=summary.get("identity_verified", False),
    )
    summary["final_e1_label"] = final_label
    summary["label_subreason"] = subreason
    summary["decisive_phase"] = decisive_phase
    summary["accepted_before_violation"] = accepted_before_violation
    summary["label_router"] = "route_e1_label"

    summary_path = os.path.join(out_dir, "run_summary.json")
    with open(summary_path, "w") as f:
        json.dump(_jsonable(summary), f, ensure_ascii=False, indent=2)
    print(f"[{a.case} run{run_idx}] → {summary_path}")
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2))


def _derive_from_trace(raw_log_path):
    """从 raw trace 推导 decisive_phase（首次 q_bad 的 phase）和
    accepted_before_violation（q_bad 前是否已 q_acc）。"""
    decisive_phase = None
    accepted_before_violation = False
    try:
        for l in open(raw_log_path, encoding="utf-8"):
            d = json.loads(l)
            after = d.get("automaton_after")
            if after == "q_acc":
                accepted_before_violation = True
            if after == "q_bad" and decisive_phase is None:
                decisive_phase = d.get("phase")
    except Exception:
        pass
    return decisive_phase, accepted_before_violation


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    return o


def eval_goal_clauses_raw(env):
    from src.fpck.goal_evaluator import eval_goal_clauses
    return eval_goal_clauses(env)


if __name__ == "__main__":
    main()
