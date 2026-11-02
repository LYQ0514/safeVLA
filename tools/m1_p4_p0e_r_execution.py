"""M1-P4-P0e-r：Complete-Contract Dynamic Path Execution（修订版）。

在完整契约（含底座）下，对 P0r2 找到的安全预抓取路径做真实动力学执行验证。

三个门禁：
- Gate E0：路径与初始状态一致性，重新规划 RRT 并保存完整路径，计算 ref 最小间隙
  d_min_ref（含 mug+底座+完整角色清单），记录最窄位置 geom 对/关节配置/路径长度/拐角；
- Gate E1：真实控制器执行（重力补偿关节 PD，kp=50/kd=14.14/v_max=0.5），记录
  全身接触/支撑物间接作用/历史CAR/跟踪误差，逐 substep 用 monitor_contacts 审计；
- Gate E2：预抓取终点有效性（无违规 + 到达 + 稳定 + 姿态满足入口）。

主路径选取规则（固定）：最大 min_gap 优先 → 路径长度次之 → seed 破平局。
不通过执行结果反向选路径。
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
from src.safety.contract_objects import (robot_geoms, forbidden_contact_geoms,
                                         monitor_contacts, classify_body)
from tools.g1a_soup_trace import make_env, body_state
from tools.m1_p4_p0e_joint_execution import (reconstruct_nvs, SOUP_BODY, MUG_BODY,
                                            trapezoidal, path_at_s, KP, KD, V_MAX, A_MAX,
                                            SETTLE_S, TORQUE_LO, TORQUE_HI)
from tools.m1_p4_p0_reachability import joint_limits, set_arm_q, get_arm_q, ik
from tools.m1_p4_p0r2_reachability import (build_orientations, APPROACH_OFFSETS,
                                           classify_ik, min_gap_full, rrt_full)

CASES = [(1, 3), (4, 3), (9, 3), (2, 0)]
DELTA_CAR = 0.001
EPS = 0.0001
RRT_SEEDS = 5  # 每目标尝试 seed 0..4

OUT_PATH = os.path.join(ROOT, "data", "audit", "m1_p4_p0e_r_execution.json")


def atomic_write(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, path)


def gap_and_pair(env, q, rg, fg):
    """计算 robot 与 forbidden 的最小间隙及对应 geom 对（margin 查询）。"""
    set_arm_q(env, q)
    m = env.sim.model._model
    d = env.sim.data._data
    mujoco.mj_forward(m, d)
    saved = {g: m.geom_margin[g] for g in list(rg) + list(fg)}
    for g in list(rg) + list(fg):
        m.geom_margin[g] = 0.05
    best = None
    pair = None
    try:
        mujoco.mj_collision(m, d)
        names = env.sim.model.body_names
        for c in range(d.ncon):
            g1, g2 = d.contact[c].geom1, d.contact[c].geom2
            if (g1 in rg and g2 in fg) or (g2 in rg and g1 in fg):
                dd = float(d.contact[c].dist)
                if best is None or dd < best:
                    best = dd
                    pair = (names[m.geom_bodyid[g1]], names[m.geom_bodyid[g2]])
        return best, pair
    finally:
        for g, v in saved.items():
            m.geom_margin[g] = v


def path_min_gap_scan(env, path, rg, fg):
    """沿路径扫描最小间隙，返回 (min_gap, min_gap_pair, min_gap_q)。"""
    min_g = None
    min_pair = None
    min_q = None
    for q in path:
        g, pr = gap_and_pair(env, q, rg, fg)
        gv = 0.05 if g is None else g
        if min_g is None or gv < min_g:
            min_g = gv
            min_pair = pr
            min_q = q.copy()
    return min_g, min_pair, min_q


def execute_path_dynamic(env, path, rg, fg, p0):
    """重力补偿关节 PD 执行，逐 substep 审计。返回完整指标。"""
    m = env.sim.model._model
    d = env.sim.data._data
    dt = env.env.model_timestep
    seg = [float(np.linalg.norm(path[i + 1] - path[i])) for i in range(len(path) - 1)]
    L = sum(seg)
    cumlen = np.concatenate([[0.0], np.cumsum(seg)])
    s_arr, t_arr, T = trapezoidal(L, V_MAX, A_MAX, dt)
    n_settle = int(SETTLE_S / dt)
    q_goal = path[-1].copy()
    M_full = np.ndarray((m.nv, m.nv), dtype=np.float64, order="C")

    def p_at(s):
        idx = int(np.searchsorted(cumlen, s)) - 1
        idx = max(0, min(idx, len(path) - 2))
        seglen = cumlen[idx + 1] - cumlen[idx]
        frac = (s - cumlen[idx]) / seglen if seglen > 1e-12 else 0.0
        return path[idx] + frac * (path[idx + 1] - path[idx])

    Dmax = 0.0
    track_errs = []
    track_err_max = 0.0
    tau_max = 0.0
    forbidden_events = []  # (substep, body1, body2, dist)
    first_forbidden_substep = None
    min_gap_actual = None

    for i in range(len(s_arr) + n_settle):
        q_ref = p_at(s_arr[i]) if i < len(s_arr) else q_goal
        env.env.sim.forward()
        q = np.asarray(d.qpos[:7])
        qvel = np.asarray(d.qvel[:7])
        mujoco.mj_fullM(m, M_full, d.qM)
        tau = np.clip(M_full[:7, :7] @ (KP * (q_ref - q) - KD * qvel)
                      + np.asarray(d.qfrc_bias[:7]), TORQUE_LO, TORQUE_HI)
        tau_max = max(tau_max, float(np.max(np.abs(tau))))
        d.ctrl[:7] = tau
        d.ctrl[7:] = 0.0
        env.env.sim.step()
        # 完整契约接触审计
        mon = monitor_contacts(env)
        for ev in mon["forbidden"] + mon["unclassified"]:
            forbidden_events.append((i, ev["body1"], ev["body2"], ev["dist"]))
            if first_forbidden_substep is None:
                first_forbidden_substep = i
        Dmax = max(Dmax, float(np.sum(np.abs(body_state(env, MUG_BODY)["pos"] - p0))))
        e = float(np.linalg.norm(q_ref - np.asarray(d.qpos[:7])))
        track_errs.append(e)
        track_err_max = max(track_err_max, e)
        # 实际最小间隙（每 100 substep 采样）
        if i % 100 == 0:
            g, _ = gap_and_pair(env, np.asarray(d.qpos[:7]), rg, fg)
            gv = 0.05 if g is None else g
            if min_gap_actual is None or gv < min_gap_actual:
                min_gap_actual = gv

    q_final = np.asarray(d.qpos[:7])
    qvel_final = np.asarray(d.qvel[:7])
    return {
        "T": T, "L": L, "Dmax": Dmax,
        "track_err_max": track_err_max,
        "track_err_mean": float(np.mean(track_errs)),
        "tau_max": tau_max,
        "pos_err": float(np.linalg.norm(q_final - q_goal)),
        "vel_norm": float(np.linalg.norm(qvel_final)),
        "n_forbidden_events": len(forbidden_events),
        "first_forbidden_substep": first_forbidden_substep,
        "forbidden_pair_types": _pair_types(forbidden_events),
        "min_gap_actual": min_gap_actual,
        "q_final": q_final.tolist(), "q_goal": q_goal.tolist(),
    }


def _pair_types(events):
    d = {}
    for (i, b1, b2, dist) in events:
        k = f"{b1}↔{b2}"
        d[k] = d.get(k, 0) + 1
    return d


def run_case(ts, task, prompt, pol, state_id, stream):
    env, layer, scratch, layer_s, rg_old, og, p0, soup_pos, nvs_step, Dmax0 = \
        reconstruct_nvs(ts, task, prompt, pol, state_id, stream)
    rg = robot_geoms(env)
    fg = forbidden_contact_geoms(env)
    q_start = get_arm_q(env)
    obs = env.env._get_observations()
    iq = np.asarray(obs["robot0_eef_quat"])
    base_ori = Rot.from_quat([iq[1], iq[2], iq[3], iq[0]]).as_matrix()
    oris = build_orientations(base_ori)
    lo, hi = joint_limits(env)

    # 重跑 IK 找安全解（含 tgt_ori），与 P0r2 一致
    safe_pregrasps = []
    for oi, (axis, ang, R_tilt_abs) in enumerate(oris):
        tilt_label = "基准朝下" if oi == 0 else f"{axis}{ang}°"
        for oi_off, (off, off_desc) in enumerate(APPROACH_OFFSETS):
            tgt = soup_pos + off
            tgt_ori = R_tilt_abs
            rng = np.random.default_rng(oi * 100 + oi_off)
            best_safe = None
            for _ in range(30):
                qinit = rng.uniform(lo, hi)
                set_arm_q(env, qinit)
                mujoco.mj_forward(env.sim.model._model, env.sim.data._data)
                q, ok, perr = ik(env, tgt, tgt_ori, rg, fg)
                r = classify_ik(env, q, ok, perr, rg, fg)
                if r == "SAFE_PREGRASP":
                    gap = min_gap_full(env, q, rg, fg)
                    gap_v = 0.05 if gap is None else gap
                    if best_safe is None or gap_v > best_safe[1]:
                        best_safe = (q.copy(), gap_v)
            if best_safe is not None:
                safe_pregrasps.append({"tilt": tilt_label, "offset_desc": off_desc,
                                       "q": best_safe[0], "gap_mm": best_safe[1] * 1000,
                                       "tgt_ori": tgt_ori})

    # 对每个安全解跑 RRT（保存路径），记录成功路径 + 其 gap
    candidates = []  # (gap_mm, path_len, seed, q_goal, path)
    for sp in safe_pregrasps:
        q_goal = np.array(sp["q"])
        for seed in range(RRT_SEEDS):
            p, f = rrt_full(env, rg, fg, q_start, q_goal, max_iter=20000, seed=seed)
            if f:
                candidates.append((sp["gap_mm"], len(p), seed, q_goal, p))
    # 主路径选取：max gap → min path_len → min seed
    if not candidates:
        env.close(); scratch.close()
        return {"state": f"{state_id}/{stream}", "n_safe": len(safe_pregrasps),
                "n_rrt_candidates": 0, "selected": None}

    candidates.sort(key=lambda c: (-c[0], c[1], c[2]))
    sel_gap, sel_len, sel_seed, sel_qgoal, sel_path = candidates[0]
    sel = {"gap_mm": sel_gap, "path_len": sel_len, "seed": sel_seed,
           "q_goal": sel_qgoal.tolist(), "path": [q.tolist() for q in sel_path]}

    # Gate E0：路径最小间隙扫描
    min_g, min_pair, min_q = path_min_gap_scan(env, sel_path, rg, fg)
    e0 = {"d_min_ref_mm": None if min_g is None else min_g * 1000,
          "min_gap_pair": min_pair,
          "min_gap_q": None if min_q is None else min_q.tolist(),
          "path_len_rad": float(sum(np.linalg.norm(sel_path[i+1]-sel_path[i]) for i in range(len(sel_path)-1)))}

    # Gate E1：真实执行（恢复起点）
    set_arm_q(env, q_start)
    mujoco.mj_forward(env.sim.model._model, env.sim.data._data)
    env.sim.data._data.qvel[:] = 0.0
    e1 = execute_path_dynamic(env, sel_path, rg, fg, p0)

    # Gate E2：验收
    e2 = {
        "forbidden_free": e1["n_forbidden_events"] == 0,
        "car_ok": e1["Dmax"] < DELTA_CAR - EPS,
        "reached": e1["pos_err"] < 0.02,
        "stable": e1["vel_norm"] < 0.05,
        "accept": (e1["n_forbidden_events"] == 0) and (e1["Dmax"] < DELTA_CAR - EPS)
                  and (e1["pos_err"] < 0.02) and (e1["vel_norm"] < 0.05),
    }

    env.close(); scratch.close()
    return {"state": f"{state_id}/{stream}", "nvs_step": nvs_step,
            "n_safe": len(safe_pregrasps), "n_rrt_candidates": len(candidates),
            "selected": sel, "e0": e0, "e1": e1, "e2": e2}


def main():
    from libero.libero import benchmark
    ts = benchmark.get_benchmark_dict()['safelibero_long'](safety_level='I')
    task = ts.get_task(0)
    prompt = task.language
    pol = PnsgPolicy()

    print("=== M1-P4-P0e-r：完整契约动力学执行 ===", flush=True)
    out_all = {}
    if os.path.exists(OUT_PATH):
        out_all = json.load(open(OUT_PATH))

    for state_id, stream in CASES:
        key = f"{state_id}/{stream}"
        if key in out_all and out_all[key].get("done"):
            print(f"\n--- {key} 已完成，跳过 ---", flush=True)
            continue
        print(f"\n--- S{state_id:02d}/st{stream} ---", flush=True)
        t0 = time.time()
        r = run_case(ts, task, prompt, pol, state_id, stream)
        r["done"] = True
        r["wall_s"] = time.time() - t0
        out_all[key] = r
        atomic_write(out_all, OUT_PATH)
        sel = r.get("selected")
        if sel:
            print(f"  主路径: gap={sel['gap_mm']:.1f}mm len={sel['path_len']} seed={sel['seed']}", flush=True)
            print(f"  E0: d_min_ref={r['e0']['d_min_ref_mm']}mm pair={r['e0']['min_gap_pair']}", flush=True)
            print(f"  E1: Dmax={r['e1']['Dmax']*1000:.3f}mm 跟踪max={r['e1']['track_err_max']*1000:.0f}mrad "
                  f"forbidden={r['e1']['n_forbidden_events']}", flush=True)
            print(f"  E2: {r['e2']}", flush=True)
        else:
            print(f"  无 RRT 候选", flush=True)
        print(f"  [耗时 {r['wall_s']:.1f}s，已落盘]", flush=True)

    print(f"\n结果: {OUT_PATH}", flush=True)
    pol.close()

if __name__ == "__main__":
    main()
