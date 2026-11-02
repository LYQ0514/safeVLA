"""M1-P4-P1：Frozen-VLA Re-entry（冻结 π0.5 安全接管）。

R0（已通过 probe 验证）：Joint PD → OSC 用 update_initial_joints(q_current)，零瞬态。

R1：两条配对接续分支，从相同预抓取状态出发：
- P-V：冻结 π0.5（原始 OSC），检验冻结策略能否从侧倾预抓取姿态自主完成抓取；
- P-H：冻结 π0.5 + HVNP（已冻结），检验安全过滤下能否完成抓取。

CAR 历史不归零（含关节执行阶段位移）。query-index 接管后统一从固定起点，
replan_steps=5。不重训、不改任务指令、不提供特权 waypoint。
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
from src.safety.aegis_adapted import AegisSafetyLayer
from src.safety.contract_objects import robot_geoms, forbidden_contact_geoms, monitor_contacts
from tools.g1a_soup_trace import make_env, body_state, eef_pos, subgoal_values
from tools.m1_p1_design_gate_v2 import build_body_geoms
from tools.m1_p4_p0e_joint_execution import (reconstruct_nvs, SOUP_BODY, MUG_BODY,
                                            trapezoidal, path_at_s, KP, KD, V_MAX, A_MAX,
                                            SETTLE_S, TORQUE_LO, TORQUE_HI, generate_block,
                                            select_action)
from tools.m1_p4_p0_reachability import joint_limits, set_arm_q, get_arm_q, ik
from tools.m1_p4_p0r2_reachability import (build_orientations, APPROACH_OFFSETS,
                                           classify_ik, min_gap_full, rrt_full)
from tools.m1_p2_p1a_pilot import (Z_DIR, L_DIR, AVOID_MAG, AVOID_STEPS,
                                   SUITE, LEVEL, TASK_ID, REPLAN_STEPS, IMAGE_SIZE, SWITCH,
                                   H, PROTECTED, DELTA_CAR, EPS)

CASES = [(1, 3), (4, 3), (9, 3), (2, 0)]
MAX_STEPS = 550
OUT_PATH = os.path.join(ROOT, "data", "audit", "m1_p4_p1_reentry.json")


def atomic_write(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, path)


def execute_joint_path(env, path, p0):
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
    Dmax = 0.0
    for i in range(len(s_arr)+n_settle):
        q_ref = p_at(s_arr[i]) if i < len(s_arr) else q_goal
        env.env.sim.forward()
        q = np.asarray(d.qpos[:7]); qvel = np.asarray(d.qvel[:7])
        mujoco.mj_fullM(m, M_full, d.qM)
        tau = np.clip(M_full[:7,:7] @ (KP*(q_ref-q)-KD*qvel) + np.asarray(d.qfrc_bias[:7]), TORQUE_LO, TORQUE_HI)
        d.ctrl[:7] = tau; d.ctrl[7:] = 0.0
        env.env.sim.step()
        Dmax = max(Dmax, float(np.sum(np.abs(body_state(env, MUG_BODY)["pos"]-p0))))
    return Dmax


def plan_main_path(env, rg, fg, q_start, soup_pos, base_ori):
    """重跑 IK + RRT 找主路径，返回 (q_goal, path, gap_mm)。"""
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
        return None, None, None
    cands.sort(key=lambda c: (-c[0], c[1], c[2]))
    return cands[0][3], cands[0][4], cands[0][0]


def reach_and_takeover(ts, task, prompt, pol, state_id, stream):
    """确定性重跑：关节 PD 到预抓取 → OSC 接管。返回 (env, layer, scratch, layer_s, obs, Dmax_hist)。"""
    env, layer, scratch, layer_s, rg_old, og, p0, soup_pos, nvs_step, Dmax0 = \
        reconstruct_nvs(ts, task, prompt, pol, state_id, stream)
    rg = robot_geoms(env); fg = forbidden_contact_geoms(env)
    q_start = get_arm_q(env)
    obs = env.env._get_observations()
    iq = np.asarray(obs["robot0_eef_quat"])
    base_ori = Rot.from_quat([iq[1], iq[2], iq[3], iq[0]]).as_matrix()
    q_goal, path, gap_mm = plan_main_path(env, rg, fg, q_start, soup_pos, base_ori)
    if q_goal is None:
        env.close(); scratch.close()
        return None, None, None, None, None, None

    # 关节 PD 执行
    set_arm_q(env, q_start); mujoco.mj_forward(env.sim.model._model, env.sim.data._data)
    env.sim.data._data.qvel[:] = 0.0
    Dmax_exec = execute_joint_path(env, path, p0)

    # 接管
    c = env.robots[0].controller
    q_now = get_arm_q(env)
    c.update_initial_joints(q_now)
    obs = env.env._get_observations(force_update=True)

    Dmax_hist = max(Dmax0, Dmax_exec)  # CAR 历史不归零
    return env, layer, scratch, layer_s, obs, Dmax_hist


def run_branch(ts, task, prompt, pol, state_id, stream, branch):
    """从接管状态执行 P-V 或 P-H，返回完整结果。"""
    t0 = time.time()
    env, layer, scratch, layer_s, obs, Dmax_hist = \
        reach_and_takeover(ts, task, prompt, pol, state_id, stream)
    if env is None:
        return {"branch": branch, "state": f"{state_id}/{stream}", "error": "no_main_path"}

    rg = robot_geoms(env)
    fg = forbidden_contact_geoms(env)
    rg_old, og = build_body_geoms(env)  # HVNP 决策用的旧接触集合（保持 HVNP 行为）
    p0 = body_state(env, MUG_BODY)["pos"].copy()
    Dmax = Dmax_hist
    first_forbidden = None
    forbidden_events = []
    n_nvs = 0
    nvs_step = None
    soup_done = None
    success = False
    success_step = None
    nominal = None
    n_generate = 0

    # 接管后 query-index 起点固定（从当前 step 对应 index 继续）
    # reconstruct_nvs 已跑到 nvs_step；接管后续从新 query 开始
    # 简化：接管后的第一个动作块用当前 obs 生成，query 从 0 重新编号
    # （这里用连续 query，保证两分支一致）

    avoid_remaining = 0
    avoid_dir = None
    avoid_name = None

    for step in range(MAX_STEPS):
        if step % REPLAN_STEPS == 0:
            nominal = generate_block(pol, obs, prompt, step // REPLAN_STEPS, state_id, stream)
            n_generate += 1
        nom_a = nominal[step % REPLAN_STEPS]

        if branch == "P-V":
            act = nom_a
        elif branch == "P-H":
            if avoid_remaining > 0:
                act = np.clip(nom_a + AVOID_MAG * avoid_dir, -1, 1)
                avoid_remaining -= 1
            else:
                chosen, verdicts = select_action(scratch, layer_s, pol, prompt,
                                                 env, layer, nominal, step, Dmax,
                                                 rg_old, og, p0, state_id, stream)
                if chosen == "NO_VERIFIED_SEQUENCE":
                    n_nvs += 1; nvs_step = step
                    break
                if chosen == "N":
                    act = nom_a
                elif chosen == "A":
                    act = layer.filter(nom_a, obs)[0]
                elif chosen == "Z+A":
                    act = np.clip(nom_a + AVOID_MAG * Z_DIR, -1, 1)
                    avoid_remaining = AVOID_STEPS - 1; avoid_dir = Z_DIR; avoid_name = chosen
                elif chosen == "L+A":
                    act = np.clip(nom_a + AVOID_MAG * L_DIR, -1, 1)
                    avoid_remaining = AVOID_STEPS - 1; avoid_dir = L_DIR; avoid_name = chosen
                elif chosen == "Hold+N":
                    act = np.zeros(7)
        else:
            raise ValueError(branch)

        # 正常 env.step（OSC 路径）
        obs, _r, _d, _i = env.step(act)
        # 完整契约审计
        mon = monitor_contacts(env)
        for ev in mon["forbidden"] + mon["unclassified"]:
            forbidden_events.append((step, ev["body1"], ev["body2"], ev["dist"]))
            if first_forbidden is None:
                first_forbidden = step
        Dmax = max(Dmax, float(np.sum(np.abs(body_state(env, MUG_BODY)["pos"] - p0))))

        sg = subgoal_values(env)
        if sg.get("alphabet_soup_1") and soup_done is None:
            soup_done = step
        if env.check_success():
            success = True; success_step = step
            break

    r = {
        "branch": branch, "state": f"{state_id}/{stream}",
        "success": success, "success_step": success_step,
        "soup_done": soup_done,
        "Dmax": Dmax, "car_ok": Dmax < DELTA_CAR - EPS,
        "n_forbidden_events": len(forbidden_events),
        "first_forbidden": first_forbidden,
        "forbidden_pair_types": _pair_types(forbidden_events),
        "n_nvs": n_nvs, "nvs_step": nvs_step,
        "n_generate": n_generate, "wall_s": time.time() - t0,
    }
    env.close(); scratch.close()
    return r


def _pair_types(events):
    d = {}
    for (st, b1, b2, dist) in events:
        k = f"{b1}↔{b2}"
        d[k] = d.get(k, 0) + 1
    return d


def main():
    from libero.libero import benchmark
    ts = benchmark.get_benchmark_dict()['safelibero_long'](safety_level='I')
    task = ts.get_task(0)
    prompt = task.language
    pol = PnsgPolicy()

    print("=== M1-P4-P1：冻结 π0.5 安全接管 ===", flush=True)
    out_all = {}
    if os.path.exists(OUT_PATH):
        out_all = json.load(open(OUT_PATH))

    for state_id, stream in CASES:
        for branch in ["P-V", "P-H"]:
            key = f"{state_id}/{stream}/{branch}"
            if key in out_all and out_all[key].get("done"):
                print(f"\n--- {key} 已完成，跳过 ---", flush=True)
                continue
            print(f"\n--- S{state_id:02d}/st{stream} {branch} ---", flush=True)
            t0 = time.time()
            r = run_branch(ts, task, prompt, pol, state_id, stream, branch)
            r["done"] = True
            r["wall_s"] = time.time() - t0
            out_all[key] = r
            atomic_write(out_all, OUT_PATH)
            if "error" in r:
                print(f"  {r['error']}", flush=True)
            else:
                print(f"  成功={r['success']}@step{r['success_step']} soup_done={r['soup_done']} "
                      f"Dmax={r['Dmax']*1000:.3f}mm CAR={'✓' if r['car_ok'] else '✗'} "
                      f"forbidden={r['n_forbidden_events']} NVS={r['n_nvs']}", flush=True)
            print(f"  [耗时 {r['wall_s']:.1f}s，已落盘]", flush=True)

    print(f"\n结果: {OUT_PATH}", flush=True)
    pol.close()

if __name__ == "__main__":
    main()
