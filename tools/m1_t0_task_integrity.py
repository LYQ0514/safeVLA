"""M1-T0：Task-Object Integrity Audit（TASK 物体操作完整性审计）。

审计四件事：
1. P4-P1 的四条路径执行前缀（S01/S04/S09/State02），到达接管状态时 soup/cream 是否完整；
2. 重新判定 State02 的「假成功」：定位首次非预期任务物体位移；
3. HVNP-P1b 完整安全成功轨迹的操作过程复核（它是不同轨迹，不因 P4 失败撤回）；
4. 固化 TASK_INTEGRITY_FAIL / PREGRASP_VALID 事件判决。

核心：规划/预抓取阶段 soup/cream 位移 > 5mm 视为 TASK_INTEGRITY_FAIL。
"""

from __future__ import annotations
import json, os, sys, time, warnings
warnings.filterwarnings("ignore")
ROOT = "/data/safeVLA"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "dataset", "SafeLIBERO", "safelibero"))
sys.path.insert(0, "/data/2608.SDL/MojocoProj/src/pi05")
sys.path.insert(0, os.path.join(ROOT, "third_party", "aegis"))
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as Rot
from libero.libero import benchmark
from src.pnsg.policy_adapter import PnsgPolicy
from src.safety.contract_objects import (task_integrity_ok, TASK_STILL_DISP_THRESHOLD,
                                         robot_geoms, forbidden_contact_geoms)
from tools.g1a_soup_trace import make_env, body_state
from tools.m1_p4_p0e_joint_execution import (reconstruct_nvs, SOUP_BODY, MUG_BODY,
                                            trapezoidal, path_at_s, KP, KD, V_MAX, A_MAX,
                                            SETTLE_S, TORQUE_LO, TORQUE_HI)
from tools.m1_p4_p0_reachability import joint_limits, set_arm_q, get_arm_q, ik
from tools.m1_p4_p0r2_reachability import (build_orientations, APPROACH_OFFSETS,
                                           classify_ik, min_gap_full, rrt_full)

CASES = [(1, 3), (4, 3), (9, 3), (2, 0)]
TASK_BODIES = ("alphabet_soup_1_main", "cream_cheese_1_main")
OUT_PATH = os.path.join(ROOT, "data", "audit", "m1_t0_task_integrity.json")


def atomic_write(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, path)


def execute_joint_path_tracked(env, path, p0, baseline):
    """关节 PD 执行，逐 substep 检测 TASK 物体位移，返回首次超阈详情。"""
    m = env.sim.model._model; d = env.sim.data._data
    dt = env.env.model_timestep
    seg = [float(np.linalg.norm(path[i+1]-path[i])) for i in range(len(path)-1)]
    L = sum(seg); cumlen = np.concatenate([[0.0], np.cumsum(seg)])
    s_arr, t_arr, T = trapezoidal(L, V_MAX, A_MAX, dt)
    n_settle = int(SETTLE_S / dt)
    q_goal = path[-1].copy()
    M_full = np.ndarray((m.nv, m.nv), dtype=np.float64, order="C")
    def p_at(s):
        idx = int(np.searchsorted(cumlen, s))-1
        idx = max(0, min(idx, len(path)-2))
        seglen = cumlen[idx+1]-cumlen[idx]
        frac = (s-cumlen[idx])/seglen if seglen>1e-12 else 0.0
        return path[idx] + frac*(path[idx+1]-path[idx])
    first_viol = None
    for i in range(len(s_arr)+n_settle):
        q_ref = p_at(s_arr[i]) if i < len(s_arr) else q_goal
        env.env.sim.forward()
        q = np.asarray(d.qpos[:7]); qvel = np.asarray(d.qvel[:7])
        mujoco.mj_fullM(m, M_full, d.qM)
        tau = np.clip(M_full[:7,:7] @ (KP*(q_ref-q)-KD*qvel) + np.asarray(d.qfrc_bias[:7]), TORQUE_LO, TORQUE_HI)
        d.ctrl[:7] = tau; d.ctrl[7:] = 0.0
        env.env.sim.step()
        if i % 25 == 0:  # 每 25 substep（1 控制步）检查一次
            ok, disps = task_integrity_ok(env, baseline, TASK_BODIES)
            if not ok and first_viol is None:
                first_viol = {"substep": i, "disps_mm": disps}
    ok, disps = task_integrity_ok(env, baseline, TASK_BODIES)
    return ok, disps, first_viol


def audit_case(ts, task, prompt, pol, state_id, stream):
    """审计一个状态的预抓取路径执行前/后 TASK 物体完整性。"""
    env, layer, scratch, layer_s, rg_old, og, p0, soup_pos, nvs_step, Dmax0 = \
        reconstruct_nvs(ts, task, prompt, pol, state_id, stream)
    rg = robot_geoms(env); fg = forbidden_contact_geoms(env)
    q_start = get_arm_q(env)
    obs = env.env._get_observations()
    iq = np.asarray(obs["robot0_eef_quat"])
    base_ori = Rot.from_quat([iq[1], iq[2], iq[3], iq[0]]).as_matrix()

    # 基线：soup/cream 初始位置（关节执行前）
    baseline = {b: body_state(env, b)["pos"].copy() for b in TASK_BODIES}

    # 规划主路径
    lo, hi = joint_limits(env)
    oris = build_orientations(base_ori)
    safe_pregrasps = []
    for oi, (axis, ang, R_tilt_abs) in enumerate(oris):
        for oi_off, (off, off_desc) in enumerate(APPROACH_OFFSETS):
            tgt = soup_pos + off; tgt_ori = R_tilt_abs
            rng = np.random.default_rng(oi*100+oi_off)
            best = None
            for _ in range(30):
                qinit = rng.uniform(lo, hi); set_arm_q(env, qinit)
                mujoco.mj_forward(env.sim.model._model, env.sim.data._data)
                q, ok, perr = ik(env, tgt, tgt_ori, rg, fg)
                r = classify_ik(env, q, ok, perr, rg, fg)
                if r == "SAFE_PREGRASP":
                    g = min_gap_full(env, q, rg, fg); gv = 0.05 if g is None else g
                    if best is None or gv > best[1]: best = (q.copy(), gv)
            if best is not None:
                safe_pregrasps.append((best[0], best[1], tgt_ori))
    cands = []
    for qg, gap, tgt_ori in safe_pregrasps:
        for seed in range(5):
            p, f = rrt_full(env, rg, fg, q_start, qg, max_iter=20000, seed=seed)
            if f: cands.append((gap*1000, len(p), seed, qg, p))
    if not cands:
        env.close(); scratch.close()
        return {"state": f"{state_id}/{stream}", "error": "no_path"}
    cands.sort(key=lambda c: (-c[0], c[1], c[2]))
    _, _, _, q_goal, path = cands[0]

    # 执行 + 完整性检测
    set_arm_q(env, q_start); mujoco.mj_forward(env.sim.model._model, env.sim.data._data)
    env.sim.data._data.qvel[:] = 0.0
    ok, disps, first_viol = execute_joint_path_tracked(env, path, p0, baseline)

    # 执行后 success 谓词（复现假成功）
    success_pred = env.check_success()

    env.close(); scratch.close()
    return {
        "state": f"{state_id}/{stream}",
        "task_integrity_ok": ok,
        "disps_mm": {b: round(d, 3) for b, d in disps.items()},
        "first_violation": first_viol,
        "success_predicate_after_exec": success_pred,
        "verdict": "PREGRASP_VALID" if ok else "TASK_INTEGRITY_FAIL",
    }


def main():
    from libero.libero import benchmark
    ts = benchmark.get_benchmark_dict()['safelibero_long'](safety_level='I')
    task = ts.get_task(0)
    prompt = task.language
    pol = PnsgPolicy()

    print("=== M1-T0：TASK 物体操作完整性审计 ===", flush=True)
    out_all = {}

    for state_id, stream in CASES:
        print(f"\n--- S{state_id:02d}/st{stream} ---", flush=True)
        t0 = time.time()
        r = audit_case(ts, task, prompt, pol, state_id, stream)
        out_all[f"{state_id}/{stream}"] = r
        atomic_write(out_all, OUT_PATH)
        if "error" in r:
            print(f"  {r['error']}", flush=True)
        else:
            print(f"  完整性={'✓' if r['task_integrity_ok'] else '✗'} "
                  f"disps={r['disps_mm']} 判决={r['verdict']}", flush=True)
            if r["first_violation"]:
                print(f"  首次违规: substep{r['first_violation']['substep']} disps={r['first_violation']['disps_mm']}", flush=True)
            print(f"  执行后 success 谓词={r['success_predicate_after_exec']}", flush=True)
        print(f"  [耗时 {time.time()-t0:.1f}s]", flush=True)

    print(f"\n结果: {OUT_PATH}", flush=True)
    pol.close()

if __name__ == "__main__":
    main()
