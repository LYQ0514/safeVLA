"""M1-P4-P1 R0 探针：控制器接管门禁（Joint PD → OSC_POSE）。

验证接管协议：
1. 关节 PD 执行到预抓取位姿（复用 P0e-r 的主路径）；
2. 用 controller.update_initial_joints(q_current) 初始化 OSC 零空间参考 + 重置目标位姿；
3. 执行几个 OSC 控制步（零动作），检查切换瞬态是否产生接触/位移/异常肘部运动。

关键：不允许 qpos 瞬移、不重置 episode。切换前后记录 q/q̇/EEF 位姿/接触。
"""

from __future__ import annotations
import json, os, sys, warnings, time
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
from src.safety.contract_objects import robot_geoms, forbidden_contact_geoms, monitor_contacts
from tools.g1a_soup_trace import make_env, body_state, eef_pos
from tools.m1_p4_p0e_joint_execution import (reconstruct_nvs, SOUP_BODY, MUG_BODY,
                                            trapezoidal, path_at_s, KP, KD, V_MAX, A_MAX,
                                            SETTLE_S, TORQUE_LO, TORQUE_HI)
from tools.m1_p4_p0_reachability import joint_limits, set_arm_q, get_arm_q, ik
from tools.m1_p4_p0r2_reachability import (build_orientations, APPROACH_OFFSETS,
                                           classify_ik, min_gap_full, rrt_full)

DELTA_CAR = 0.001; EPS = 0.0001

def execute_joint_path(env, path, p0):
    """关节 PD 执行到终点（复用 P0e-r 逻辑，返回终点状态）。"""
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

def main():
    from libero.libero import benchmark
    ts = benchmark.get_benchmark_dict()['safelibero_long'](safety_level='I')
    task = ts.get_task(0); prompt = task.language
    pol = PnsgPolicy()

    state_id, stream = 1, 3
    env, layer, scratch, layer_s, rg_old, og, p0, soup_pos, nvs_step, Dmax0 = \
        reconstruct_nvs(ts, task, prompt, pol, state_id, stream)
    rg = robot_geoms(env); fg = forbidden_contact_geoms(env)
    q_start = get_arm_q(env)
    obs = env.env._get_observations()
    iq = np.asarray(obs["robot0_eef_quat"])
    base_ori = Rot.from_quat([iq[1],iq[2],iq[3],iq[0]]).as_matrix()
    oris = build_orientations(base_ori)
    lo, hi = joint_limits(env)

    # 重跑 IK + RRT 找主路径
    safe_pregrasps = []
    for oi, (axis, ang, R_tilt_abs) in enumerate(oris):
        for oi_off, (off, off_desc) in enumerate(APPROACH_OFFSETS):
            tgt = soup_pos + off; tgt_ori = R_tilt_abs
            rng = np.random.default_rng(oi*100+oi_off)
            best = None
            for _ in range(30):
                qinit = rng.uniform(lo,hi); set_arm_q(env, qinit)
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
    cands.sort(key=lambda c: (-c[0], c[1], c[2]))
    _, _, _, q_goal, path = cands[0]
    print(f"主路径 gap={cands[0][0]:.1f}mm len={cands[0][1]} seed={cands[0][2]}", flush=True)

    # 关节 PD 执行
    set_arm_q(env, q_start); mujoco.mj_forward(env.sim.model._model, env.sim.data._data)
    env.sim.data._data.qvel[:] = 0.0
    Dmax_exec = execute_joint_path(env, path, p0)
    c = env.robots[0].controller
    q_before = get_arm_q(env)
    qvel_before = np.asarray(env.sim.data._data.qvel[:7]).copy()
    eef_before = eef_pos(env).copy()
    print(f"\n[R0] 关节 PD 执行后: Dmax_exec={Dmax_exec*1000:.4f}mm", flush=True)
    print(f"  q_before={np.round(q_before,3)}", flush=True)
    print(f"  eef_before={np.round(eef_before,4)}", flush=True)

    # 接管：update_initial_joints(q_current)
    c.update_initial_joints(q_before)
    print(f"  接管后 goal_pos={np.round(c.goal_pos,4)} initial_joint[0:3]={np.round(c.initial_joint[0:3],3)}", flush=True)

    # 执行 5 个 OSC 零动作控制步（保持 EEF 不动），检查瞬态
    Dmax_switch = Dmax_exec
    forbidden_switch = []
    obs_switch = obs
    for step in range(5):
        zero_act = np.zeros(7)
        # 用正常 env.step 路径（OSC）
        obs_switch, _r, _d, _i = env.step(zero_act)
        mon = monitor_contacts(env)
        for ev in mon["forbidden"] + mon["unclassified"]:
            forbidden_switch.append((step, ev["body1"], ev["body2"], ev["dist"]))
        Dmax_switch = max(Dmax_switch, float(np.sum(np.abs(body_state(env, MUG_BODY)["pos"]-p0))))

    q_after = get_arm_q(env)
    eef_after = eef_pos(env)
    print(f"\n[R0] 5 步 OSC 零动作后:", flush=True)
    print(f"  forbidden 事件={len(forbidden_switch)}", flush=True)
    print(f"  Dmax_switch={Dmax_switch*1000:.4f}mm（执行后 {Dmax_exec*1000:.4f}mm）", flush=True)
    print(f"  q 漂移={np.linalg.norm(q_after-q_before)*1000:.2f}mrad", flush=True)
    print(f"  eef 漂移={np.linalg.norm(eef_after-eef_before)*1000:.2f}mm", flush=True)

    r0 = {"Dmax_exec_mm": Dmax_exec*1000, "Dmax_switch_mm": Dmax_switch*1000,
          "n_forbidden_switch": len(forbidden_switch),
          "q_drift_mrad": float(np.linalg.norm(q_after-q_before)*1000),
          "eef_drift_mm": float(np.linalg.norm(eef_after-eef_before)*1000),
          "q_before": q_before.tolist(), "q_after": q_after.tolist()}
    out = json.dumps(r0, ensure_ascii=False, indent=2, default=str)
    with open(os.path.join(ROOT, "data", "audit", "m1_p4_p1_r0_probe.json"), "w") as f:
        f.write(out)
    print(f"\n结果: data/audit/m1_p4_p1_r0_probe.json", flush=True)
    pol.close(); env.close(); scratch.close()

if __name__ == "__main__":
    main()
