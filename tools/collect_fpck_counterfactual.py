#!/usr/bin/env python3
"""Collect restored SafeLIBERO candidate continuations for FPCK training.

One task/init/stream produces one independent anchor.  Candidate siblings share
the exact MuJoCo state, v1.3 automaton memory, π0.5 chunk and CRN cursor.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, os, pathlib, subprocess, sys, time
import numpy as np

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"dataset"/"SafeLIBERO"/"safelibero"))
from src.fpck.candidate_actions import registered_candidates
from src.fpck.control_state import (control_state_payload, restore_control_state,
                                    snapshot_control_state)
from src.fpck.frozen_policy import DeterministicFrozenPolicy
from src.fpck.product_features import ProductFeatureContext,encode_product_candidate,FEATURE_NAMES
from src.fpck.task_automaton import route_e1_label
from src.fpck.task_automaton_acq import GraspAcquisitionAutomaton,SPEC_VERSION
from src.fpck.task_event_bridge import TaskEventBridge
from src.fpck.goal_evaluator import canonical_clause
from src.pnsg.deterministic_factory import create_deterministic_policy,determinism_gate_record
from src.pnsg.full_state import save_full_state,restore_full_state
from src.pnsg.role_manifest import build_role_manifest
from src.eval_safelibero import CollisionTracker,active_obstacle

MAX_STEPS={"safelibero_spatial":300,"safelibero_goal":300,"safelibero_object":300,"safelibero_long":550}


def _hash_state(snap): return hashlib.sha256(np.asarray(snap["flattened"],np.float64).tobytes()).hexdigest()
def _hash_anchor_state(env_snap,control_snap):
    h=hashlib.sha256(np.asarray(env_snap["flattened"],np.float64).tobytes())
    h.update(json.dumps(control_state_payload(control_snap),sort_keys=True).encode())
    return h.hexdigest()
def _append(path,row):
    with open(path,"a",encoding="utf-8") as f: f.write(json.dumps(row,ensure_ascii=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else int(x) if isinstance(x,np.integer) else float(x))+"\n")
def _split(anchor):
    x=int(hashlib.sha256(anchor.encode()).hexdigest(),16)%100
    return "train" if x<60 else "calibration" if x<80 else "test"


def _setup_bridge(env,bddl,obs,raw_path):
    from tools.g1a_soup_trace import body_state
    manifest=build_role_manifest(bddl)
    targets=[n for n,v in manifest["objects"].items() if "MANIPULATED_TARGET" in v.get("roles",[])]
    bodies=[n if n.endswith("_main") else n+"_main" for n in targets]
    keys=[canonical_clause(x) for x in manifest["goal_clauses"]]
    auto=GraspAcquisitionAutomaton(targets,keys); obstacle=active_obstacle(env,obs)
    bridge=TaskEventBridge(auto,str(raw_path),targets,keys,
                           obstacle_body=None if obstacle is None else obstacle+"_main")
    bridge.reset(env,{b:body_state(env,b)["pos"].copy() for b in bodies},bodies,
                 None if obstacle is None else obstacle+"_main")
    return manifest,bodies,obstacle,auto,bridge


def _choose_candidates(all_rows,n):
    priority=("nominal","hold","retreat","tangent_left","damped_0.5","tangent_right","damped_0.25")
    by={r.source:r for r in all_rows}; out=[by[k] for k in priority if k in by]
    return out[:n]


def _roll_branch(env,base,bridge,target_bodies,nominal,candidate,max_steps,start_step,
                 tracker,stall_window,prompt):
    last_progress=start_step; prev=bridge.auto.state.mode; native=False
    # consume the nominal cursor slot replaced by candidate
    base.buffer_index += 1
    obs,_r,_d,_i=env.step(candidate.action); tracker.update(obs,start_step)
    meta=base.get_chunk_metadata()
    mode=bridge.observe_control_step(env,obs,"REENTRY",start_step,nominal,candidate.action,
                                     chunk_hash=meta.get("chunk_hash"),buffer_index=base.buffer_index-1,
                                     target_bodies=target_bodies)
    native=bool(env.check_success())
    if mode!=prev and mode not in ("q_bad","q_unknown"): last_progress=start_step
    final_step=start_step
    for step in range(start_step+1,max_steps):
        if mode in ("q_acc","q_bad","q_unknown"): break
        action=base.act(obs,prompt)
        obs,_r,_d,_i=env.step(action); tracker.update(obs,step)
        meta=base.get_chunk_metadata(); prev=mode
        mode=bridge.observe_control_step(env,obs,"REENTRY",step,action,action,
                                         chunk_hash=meta.get("chunk_hash"),buffer_index=base.buffer_index-1,
                                         target_bodies=target_bodies)
        native = native or bool(env.check_success()); final_step=step
        if mode!=prev and mode not in ("q_bad","q_unknown"): last_progress=step
    fin=bridge.finalize(); timed=mode not in ("q_acc","q_bad","q_unknown")
    stalled=timed and final_step-last_progress>=stall_window
    label,sub=route_e1_label(
        handoff_eligible=True,decisive_phase="REENTRY" if fin.get("violation_code") else None,
        primary_violation_code=fin.get("violation_code"),accepted_before_violation=False,
        final_mode=mode,timed_out=timed,safe_and_integrity_valid=mode not in ("q_bad","q_unknown"),
        stalled=stalled,identity_verified=True)
    return label,sub,native,fin,final_step,tracker


def _snapshot_step(env, base, bridge):
    """Snapshot one pre-action state into a ring-buffer entry."""
    return {
        "env": save_full_state(env),
        "control": snapshot_control_state(env),
        "policy": base.snapshot(),
        "bridge": copy.deepcopy(bridge.snapshot()),
        "step": None, "nominal": None, "obs": None,
    }


def _restore_entry(env, base, bridge, entry):
    restore_full_state(env, entry["env"])
    restore_control_state(env, entry["control"])
    base.restore(entry["policy"])
    bridge.restore(copy.deepcopy(entry["bridge"]))


def _collect_anchor(env, base, bridge, manifest, bodies, obstacle, auto,
                    env_snap, control_snap, env_hash, policy_snap, bridge_snap,
                    obs, nominal, step, out, rows_path, a, max_steps, task_id,
                    state_id, stream, repo, prompt, nominal_log, anchor_source):
    """Collect candidate siblings from a restored anchor state.

    Returns (rows_written:int, anchor_id:str).  Restores the nominal state before
    returning so the caller can continue or terminate the nominal rollout.
    """
    eef = np.asarray(obs.get("robot0_eef_pos", np.zeros(3)), float)
    opos = np.asarray(obs[obstacle + "_pos"], float) if obstacle and obstacle + "_pos" in obs else None
    anchor_payload = f"{repo}|{a.suite}|{a.level}|{task_id}|{state_id}|{stream}|{env_hash}|{policy_snap['query_idx']}|{policy_snap['buffer_index']}"
    anchor = hashlib.sha256(anchor_payload.encode()).hexdigest()
    split = _split(anchor)
    candidates = _choose_candidates(
        registered_candidates(nominal, eef, opos, a.max_action_delta), a.candidates)
    written = 0
    for ci, candidate in enumerate(candidates):
        restore_full_state(env, env_snap); restore_control_state(env, control_snap)
        base.restore(policy_snap); bridge.restore(copy.deepcopy(bridge_snap))
        branch_log = out / "events" / f"{anchor[:12]}_c{ci}_{candidate.source}.jsonl"
        bridge.raw_log_path = str(branch_log)
        restore_hash = _hash_anchor_state(save_full_state(env), snapshot_control_state(env))
        identity = restore_hash == env_hash
        context = ProductFeatureContext(manifest, auto.state, a.budget, a.budget,
                                        max_steps - step, max_steps,
                                        policy_snap["buffer_index"],
                                        len(policy_snap["chunk"]), obstacle)
        phi = encode_product_candidate(obs, nominal, candidate.action, context)
        tracker = CollisionTracker(env, obs)
        label, sub, native, fin, last, tr = _roll_branch(
            env, base, bridge, bodies, nominal, candidate, max_steps,
            step, tracker, a.stall_window, prompt)
        row = {"sample_id": f"{anchor}/c{ci}", "anchor_id": anchor, "candidate_id": ci,
               "split": split, "features": phi.tolist(), "label": label,
               "continuable": int(label == "SAFE_REENTRY"),
               "target": float(label == "SAFE_REENTRY"),
               "label_source": "restored_counterfactual_rollout",
               "suite": a.suite, "level": a.level, "task_id": task_id, "task": prompt,
               "episode": state_id, "stream_id": stream,
               "candidate_source": candidate.source,
               "candidate_action": candidate.action.tolist(),
               "nominal_action": nominal.tolist(),
               "intervention_cost": int(np.linalg.norm(candidate.action - nominal) > 1e-7),
               "native_success": native, "valid_success": label == "SAFE_REENTRY",
               "collision_1mm": tr.collided, "collision_5mm": tr.max_displacement > .005,
               "first_failure_step": fin.get("first_decisive_event_index"),
               "remaining_horizon": max_steps - step, "identity_verified": identity,
               "full_state_hash": env_hash,
               "chunk_hash": policy_snap.get("meta", {}).get("chunk_hash"),
               "buffer_index": policy_snap["buffer_index"],
               "automaton_before": bridge_snap["automaton"], "automaton_after": fin,
               "label_subreason": sub, "anchor_source": anchor_source,
               "event_log": str(branch_log.relative_to(out))}
        _append(rows_path, row)
        written += 1
        print(f"{anchor[:8]} {candidate.source}: {label}", flush=True)
    # restore nominal so the caller can continue/terminate deterministically
    restore_full_state(env, env_snap); restore_control_state(env, control_snap)
    base.restore(policy_snap); bridge.restore(copy.deepcopy(bridge_snap))
    bridge.raw_log_path = str(nominal_log)
    return written, anchor


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--suite",required=True); ap.add_argument("--level",default="I")
    ap.add_argument("--tasks",type=int,nargs="+",default=[0,1,2,3])
    ap.add_argument("--episodes",type=int,nargs="+",default=list(range(10)))
    ap.add_argument("--streams",type=int,nargs="+",default=[0])
    ap.add_argument("--anchor-step",type=int,default=50,
                    help="fallback anchor; earlier q_acquisition/q_controlled state wins")
    ap.add_argument("--lead-steps",type=int,default=5,
                    help="pre-failure lead: anchor H_lead control steps before early q_bad")
    ap.add_argument("--candidates",type=int,default=4); ap.add_argument("--budget",type=int,default=20)
    ap.add_argument("--replan-steps",type=int,default=5); ap.add_argument("--policy-resize",type=int,default=224)
    ap.add_argument("--num-steps-wait",type=int,default=20); ap.add_argument("--stall-window",type=int,default=50)
    ap.add_argument("--max-action-delta",type=float,default=.75); ap.add_argument("--out",required=True)
    a=ap.parse_args()
    if not os.environ.get("LIBERO_CONFIG_PATH"): raise SystemExit("source tools/activate_env.sh first")
    os.environ.setdefault("MUJOCO_GL","egl"); os.environ.setdefault("PYOPENGL_PLATFORM","egl")
    from libero.libero import benchmark,get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    out=pathlib.Path(a.out); out.mkdir(parents=True,exist_ok=True); rows_path=out/"samples.jsonl"
    worker,gate=create_deterministic_policy(num_steps=10); repo=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT).decode().strip()
    (out/"run_manifest.json").write_text(json.dumps({"repo_commit":repo,"args":vars(a),"automaton":SPEC_VERSION,"features":list(FEATURE_NAMES),"determinism":determinism_gate_record(gate)},indent=2),encoding="utf-8")
    try:
        suite=benchmark.get_benchmark_dict()[a.suite](safety_level=a.level)
        for task_id in a.tasks:
            task=suite.get_task(task_id); prompt=task.language
            bddl=pathlib.Path(get_libero_path("bddl_files"))/task.problem_folder/task.bddl_file
            env=OffScreenRenderEnv(bddl_file_name=bddl,camera_heights=1024,camera_widths=1024,camera_depths=False,controller="OSC_POSE"); env.seed(7)
            try:
                for state_id in a.episodes:
                    for stream in a.streams:
                        env.reset(); obs=env.set_init_state(suite.get_task_init_states(task_id)[state_id])
                        dummy=np.zeros(7,np.float32); dummy[-1]=-1
                        for _ in range(a.num_steps_wait): obs,*_=env.step(dummy)
                        nominal_log=out/"events"/f"nominal_t{task_id}_s{state_id}_r{stream}.jsonl"; nominal_log.parent.mkdir(exist_ok=True)
                        manifest,bodies,obstacle,auto,bridge=_setup_bridge(env,bddl,obs,nominal_log)
                        base=DeterministicFrozenPolicy(worker,suite=a.suite,task_id=task_id,state_id=state_id,stream=stream,replan_steps=a.replan_steps,image_size=a.policy_resize)
                        max_steps=MAX_STEPS[a.suite]; anchor_done=False
                        # Ring buffer of pre-action snapshots for pre-failure anchors.
                        ring=[]; H_lead=max(1,int(a.lead_steps))
                        for step in range(max_steps):
                            nominal=base.peek(obs,prompt)
                            # Capture the pre-action state for every step.
                            ring.append({
                                "env":save_full_state(env),
                                "control":snapshot_control_state(env),
                                "policy":base.snapshot(),
                                "bridge":copy.deepcopy(bridge.snapshot()),
                                "obs":obs, "nominal":nominal, "step":step,
                            })
                            if len(ring)>H_lead+1: ring.pop(0)
                            risky=auto.state.mode in ("q_acquisition","q_controlled","q_progress","q_released")
                            # Anchor selection priority: progress anchor, then fixed
                            # fallback step; early failure is handled after the loop body.
                            if not anchor_done and (risky or step>=a.anchor_step):
                                e=ring[-1]
                                env_snap=e["env"]; control_snap=e["control"]
                                policy_snap=e["policy"]; bridge_snap=e["bridge"]
                                env_hash=_hash_anchor_state(env_snap,control_snap)
                                _collect_anchor(env,base,bridge,manifest,bodies,obstacle,auto,
                                    env_snap,control_snap,env_hash,policy_snap,bridge_snap,
                                    e["obs"],e["nominal"],e["step"],out,rows_path,a,max_steps,
                                    task_id,state_id,stream,repo,prompt,nominal_log,
                                    "progress" if risky else "fixed_step")
                                anchor_done=True
                            # Continue the nominal episode.
                            action=base.act(obs,prompt); obs,*_=env.step(action)
                            meta=base.get_chunk_metadata(); bridge.observe_control_step(env,obs,"NOMINAL",step,action,action,chunk_hash=meta.get("chunk_hash"),buffer_index=base.buffer_index-1,target_bodies=bodies)
                            if auto.state.mode in ("q_acc","q_bad","q_unknown"):
                                # Early terminal before any anchor: use pre-failure lead.
                                if not anchor_done and auto.state.mode=="q_bad":
                                    idx=max(0,len(ring)-1-H_lead)
                                    e=ring[idx]
                                    env_snap=e["env"]; control_snap=e["control"]
                                    policy_snap=e["policy"]; bridge_snap=e["bridge"]
                                    env_hash=_hash_anchor_state(env_snap,control_snap)
                                    # Verify the lead anchor is strictly before the decisive event.
                                    lead_mode=e["bridge"]["automaton"]["mode"]
                                    if lead_mode not in ("q_bad","q_unknown","q_acc"):
                                        _collect_anchor(env,base,bridge,manifest,bodies,obstacle,auto,
                                            env_snap,control_snap,env_hash,policy_snap,bridge_snap,
                                            e["obs"],e["nominal"],e["step"],out,rows_path,a,max_steps,
                                            task_id,state_id,stream,repo,prompt,nominal_log,
                                            f"pre_failure_lead{H_lead}")
                                        anchor_done=True
                                    else:
                                        print(f"[anchor] state{state_id} lead snapshot at step{e['step']} already terminal ({lead_mode}); skip",flush=True)
                                elif not anchor_done:
                                    print(f"[anchor] state{state_id} terminal mode {auto.state.mode} before anchor; no anchor",flush=True)
                                break
            finally: env.close()
    finally: worker.close()

if __name__=="__main__": main()
