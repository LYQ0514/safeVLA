"""M1-P4-P0e：Joint-Space Path Execution Feasibility（S01 单路径动态执行）。

回答：RRT 关节路径能否由真实物理执行（关节 PD 跟踪，非 qpos 瞬移、非 OSC 位姿跟踪），
全程满足禁止接触 + mug 历史位移 < 1mm，稳定到达预抓取位姿。

关键修正（相对 P4-P0）：
- generate_block 显式传 state_id/stream（修复 CRN 硬编码 bug）；
- 从确定性重跑获得规划起点，保存状态身份；
- RRT 起点 = 真实执行起始关节配置；
- 关节 PD 跟踪（tau = kp*(q_ref-q) - kd*qvel，clip 到原始 actuator 限幅），
  qpos 不瞬移，力矩不超原 Panda 权限；
- 时间参数化（梯形速度剖面）尊重速度/加速度上限。

工程设定（保守，统一用于所有状态，不按轨迹调参）：
- v_max = 0.5 rad/s（弧长速率，模型无显式关节速度上限，保守取值）
- a_max = 1.0 rad/s^2
- kp = 50, kd = 14.14（匹配 robosuite JOINT_POSITION 默认，临界阻尼）
- 力矩限幅 = 原 actuator ctrlrange（j1-5 ±80, j6-7 ±12）
- gripper 在执行期间保持 NVS 状态不变（ctrl[7:]=0）
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
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
import pi05_libero_runner as R  # noqa: E402
from scipy.spatial.transform import Rotation as Rot  # noqa: E402

from src.pnsg.crn import nominal_seed, sample_z  # noqa: E402
from src.pnsg.policy_adapter import PnsgPolicy  # noqa: E402
from src.safety.aegis_adapted import AegisSafetyLayer  # noqa: E402
from tools.g1a_soup_trace import make_env, body_state  # noqa: E402
from tools.m1_p1_design_gate_v2 import step_tracked, build_body_geoms  # noqa: E402
from tools.m1_p1_design_gate_v3 import disp_from_p0  # noqa: E402
from tools.m1_p2_p1a_pilot import (  # noqa: E402
    save_full, restore_full, Z_DIR, L_DIR, AVOID_MAG, AVOID_STEPS,
    SUITE, LEVEL, TASK_ID, REPLAN_STEPS, IMAGE_SIZE, SWITCH, H, PROTECTED, DELTA_CAR, EPS,
)
from tools.m1_p4_p0_reachability import (  # noqa: E402
    ik, joint_limits, set_arm_q, get_arm_q, collision_free,
)
from tools.m1_p4_p0_reachability_v2 import ik_multistart, rrt_connect, SOUP_BODY, MUG_BODY  # noqa: E402

# 工程设定（保守，统一）
V_MAX = 0.5
A_MAX = 1.0
KP = 50.0
KD = 14.14  # 2*sqrt(50)，临界阻尼
TORQUE_LO = np.array([-80.0, -80.0, -80.0, -80.0, -80.0, -12.0, -12.0])
TORQUE_HI = np.array([80.0, 80.0, 80.0, 80.0, 80.0, 12.0, 12.0])
SETTLE_S = 1.0
RRT_ITER = 20000
IK_RESTARTS = 60

# 预抓取位姿（P4-P0 确认侧向接近可行）
PREGRASP_OFFSETS = [
    np.array([0.0, 0.0, 0.14]),
    np.array([0.0, 0.0, 0.09]),
    np.array([0.0, 0.06, 0.12]),
    np.array([0.0, 0.10, 0.14]),
    np.array([0.04, 0.08, 0.13]),
    np.array([0.06, 0.0, 0.12]),
]


def generate_block(policy, obs, prompt, query_idx, state_id, stream):
    """修正版：显式传 state_id/stream。"""
    z0, _ = sample_z(nominal_seed(SUITE, TASK_ID, state_id, stream, query_idx))
    po = R.obs_to_policy_obs(obs, prompt, image_size=IMAGE_SIZE)
    return np.clip(policy.generate(z0, po), -1, 1)


def noise_hash(state_id, stream, query_idx=0):
    z, _ = sample_z(nominal_seed(SUITE, TASK_ID, state_id, stream, query_idx))
    return hashlib.sha256(np.asarray(z).tobytes()).hexdigest()[:16]


def verify_candidate(scratch, layer_s, pol, prompt, snapshot, cand, nom_block,
                     start_step, Dmax0, rg, og, p0, state_id, stream):
    restore_full(scratch, layer_s, snapshot)
    obs = scratch.env._get_observations(force_update=True)
    nominal = nom_block.copy()
    first_contact = None
    first_step_contact = False
    contact_steps = 0
    Dmax = Dmax0
    J_dev = 0.0
    for t in range(H):
        step = start_step + t
        if step % REPLAN_STEPS == 0 and t > 0:
            nominal = generate_block(pol, obs, prompt, step // REPLAN_STEPS, state_id, stream)
        nom_a = nominal[step % REPLAN_STEPS]
        if cand == "N":
            act = nom_a
        elif cand == "A":
            act = layer_s.filter(nom_a, obs)[0]
        elif cand == "Z+A":
            act = np.clip(nom_a + AVOID_MAG * Z_DIR, -1, 1) if t < AVOID_STEPS else layer_s.filter(nom_a, obs)[0]
        elif cand == "L+A":
            act = np.clip(nom_a + AVOID_MAG * L_DIR, -1, 1) if t < AVOID_STEPS else layer_s.filter(nom_a, obs)[0]
        elif cand == "Hold+N":
            act = np.zeros(7) if t < AVOID_STEPS else nom_a
        else:
            raise ValueError(cand)
        J_dev += float(np.linalg.norm(np.asarray(act) - np.asarray(nom_a)))
        obs, _r, _d, _i, c_any, c_l6, _ = step_tracked(scratch, act, rg, og)
        if c_any:
            contact_steps += 1
            if first_contact is None:
                first_contact = step
            if t == 0:
                first_step_contact = True
        Dmax = max(Dmax, disp_from_p0(scratch, p0))
    safe = (contact_steps == 0) and (Dmax < DELTA_CAR - EPS)
    return {"cand": cand, "safe": safe, "first_contact": first_contact,
            "first_step_contact": first_step_contact, "contact_steps": contact_steps,
            "Dmax": Dmax, "J_dev": J_dev}


def select_action(scratch, layer_s, pol, prompt, live, layer_live, nom_block,
                  step, Dmax, rg, og, p0, state_id, stream):
    snapshot = save_full(live, layer_live)
    vN = verify_candidate(scratch, layer_s, pol, prompt, snapshot, "N", nom_block,
                          step, Dmax, rg, og, p0, state_id, stream)
    if vN["safe"]:
        return "N", {"N": vN}
    alts = ["A", "Z+A", "L+A", "Hold+N"] if step >= SWITCH else ["Hold+N"]
    verdicts = {"N": vN}
    for cand in alts:
        verdicts[cand] = verify_candidate(scratch, layer_s, pol, prompt, snapshot, cand,
                                          nom_block, step, Dmax, rg, og, p0, state_id, stream)
    safe_alts = {k: v for k, v in verdicts.items() if k != "N" and v["safe"]}
    if safe_alts:
        best = min(safe_alts, key=lambda k: safe_alts[k]["J_dev"])
        return best, verdicts
    return "NO_VERIFIED_SEQUENCE", verdicts


def reconstruct_nvs(ts, task, prompt, pol, state_id, stream):
    """确定性重跑 HVNP 到首次 NVS（修正 CRN）。返回完整环境状态。"""
    env = make_env(ts, task, state_id)
    layer = AegisSafetyLayer(env)
    scratch = make_env(ts, task, state_id)
    layer_s = AegisSafetyLayer(scratch)
    obs = env.env._get_observations()
    layer.reset(obs, prompt)
    rg, og = build_body_geoms(env)
    p0 = body_state(env, MUG_BODY)["pos"].copy()
    soup_pos = body_state(env, SOUP_BODY)["pos"].copy()
    Dmax = 0.0
    nominal = None
    avoid_remaining = 0
    avoid_dir = None
    nvs_step = None
    for step in range(550):
        if step == SWITCH:
            layer.reset(obs, prompt)
        if step % REPLAN_STEPS == 0:
            nominal = generate_block(pol, obs, prompt, step // REPLAN_STEPS, state_id, stream)
        nom_a = nominal[step % REPLAN_STEPS]
        if avoid_remaining > 0:
            act = np.clip(nom_a + AVOID_MAG * avoid_dir, -1, 1)
            avoid_remaining -= 1
        else:
            chosen, verdicts = select_action(scratch, layer_s, pol, prompt, env, layer,
                                             nominal, step, Dmax, rg, og, p0, state_id, stream)
            if chosen == "NO_VERIFIED_SEQUENCE":
                nvs_step = step
                break
            if chosen == "N":
                act = nom_a
            elif chosen == "Hold+N":
                act = np.zeros(7)
            elif chosen == "A":
                act = layer.filter(nom_a, obs)[0]
            else:
                act = np.clip(nom_a + AVOID_MAG * (Z_DIR if chosen == "Z+A" else L_DIR), -1, 1)
                avoid_remaining = AVOID_STEPS - 1
                avoid_dir = Z_DIR if chosen == "Z+A" else L_DIR
        obs, _r, _d, _i, ca, cl, _ = step_tracked(env, act, rg, og)
        Dmax = max(Dmax, disp_from_p0(env, p0))
    return env, layer, scratch, layer_s, rg, og, p0, soup_pos, nvs_step, Dmax


def trapezoidal(L, v_max, a_max, dt):
    """梯形速度剖面，返回 (s_arr, t_arr, T)。"""
    t_a = v_max / a_max
    d_a = 0.5 * a_max * t_a ** 2
    if 2 * d_a >= L:
        t_a = np.sqrt(L / a_max)
        T = 2 * t_a
    else:
        T = 2 * t_a + (L - 2 * d_a) / v_max
    n = int(np.ceil(T / dt)) + 1
    t = np.arange(n) * dt
    s = np.zeros(n)
    for i, ti in enumerate(t):
        if ti < t_a:
            s[i] = 0.5 * a_max * ti ** 2
        elif ti < T - t_a:
            s[i] = d_a + v_max * (ti - t_a)
        else:
            tr = max(T - ti, 0.0)
            s[i] = L - 0.5 * a_max * tr ** 2
    return np.clip(s, 0, L), t, T


def path_at_s(path, cumlen, s):
    idx = int(np.searchsorted(cumlen, s)) - 1
    idx = max(0, min(idx, len(path) - 2))
    seg = cumlen[idx + 1] - cumlen[idx]
    if seg < 1e-12:
        return path[idx + 1].copy()
    frac = (s - cumlen[idx]) / seg
    return path[idx] + frac * (path[idx + 1] - path[idx])


def check_contact_data(env, rg, og):
    """读当前 d.contact（sim.step 后已更新），判断 robot↔obstacle。"""
    m = env.sim.model._model
    d = env.sim.data._data
    for c in range(d.ncon):
        g1, g2 = d.contact[c].geom1, d.contact[c].geom2
        if (g1 in rg and g2 in og) or (g2 in rg and g1 in og):
            return True
    return False


def execute_joint_path(env, path, rg, og, p0, state_id, stream):
    """关节 PD 跟踪时间参数化路径，逐 substep 验证安全。"""
    m = env.sim.model._model
    d = env.sim.data._data
    dt = env.env.model_timestep

    seg = [float(np.linalg.norm(path[i + 1] - path[i])) for i in range(len(path) - 1)]
    L = sum(seg)
    cumlen = np.concatenate([[0.0], np.cumsum(seg)])
    s_arr, t_arr, T = trapezoidal(L, V_MAX, A_MAX, dt)
    n_settle = int(SETTLE_S / dt)
    q_goal = path[-1].copy()

    # 预分配全质量矩阵（robosuite 同款 computed-torque）
    M_full = np.ndarray((m.nv, m.nv), dtype=np.float64, order="C")

    first_contact_t = None
    Dmax = 0.0
    track_err_max = 0.0
    tau_max_abs = 0.0
    q_ref_hist = []
    q_actual_hist = []

    n_steps = len(s_arr) + n_settle
    for i in range(n_steps):
        if i < len(s_arr):
            q_ref = path_at_s(path, cumlen, s_arr[i])
        else:
            q_ref = q_goal

        # robosuite JOINT_POSITION 语义：computed-torque + 重力补偿
        # 顺序：forward（算 qM/qfrc_bias）→ 算 tau → 设 ctrl → step
        env.env.sim.forward()
        q = np.asarray(d.qpos[:7])
        qvel = np.asarray(d.qvel[:7])
        mujoco.mj_fullM(m, M_full, d.qM)
        M_arm = M_full[:7, :7]
        qfrc_bias = np.asarray(d.qfrc_bias[:7])
        desired_qdd = KP * (q_ref - q) - KD * qvel
        tau = M_arm @ desired_qdd + qfrc_bias
        tau = np.clip(tau, TORQUE_LO, TORQUE_HI)
        tau_max_abs = max(tau_max_abs, float(np.max(np.abs(tau))))
        d.ctrl[:7] = tau
        d.ctrl[7:] = 0.0  # gripper 保持
        env.env.sim.step()
        if check_contact_data(env, rg, og) and first_contact_t is None:
            first_contact_t = i * dt
        Dmax = max(Dmax, disp_from_p0(env, p0))
        track_err_max = max(track_err_max, float(np.linalg.norm(q_ref - np.asarray(d.qpos[:7]))))
        if i % 100 == 0:
            q_ref_hist.append(q_ref.copy())
            q_actual_hist.append(np.asarray(d.qpos[:7]).copy())

    q_final = np.asarray(d.qpos[:7])
    qvel_final = np.asarray(d.qvel[:7])
    return {
        "T": T, "L": L, "first_contact_t": first_contact_t, "Dmax": Dmax,
        "track_err_max": track_err_max, "tau_max_abs": tau_max_abs,
        "pos_err": float(np.linalg.norm(q_final - q_goal)),
        "vel_norm": float(np.linalg.norm(qvel_final)),
        "q_final": q_final.tolist(), "q_goal": q_goal.tolist(),
        "n_substeps": n_steps,
        "q_ref_hist": [q.tolist() for q in q_ref_hist],
        "q_actual_hist": [q.tolist() for q in q_actual_hist],
    }


def main():
    from libero.libero import benchmark
    ts = benchmark.get_benchmark_dict()[SUITE](safety_level=LEVEL)
    task = ts.get_task(TASK_ID)
    prompt = task.language
    pol = PnsgPolicy()

    print("=== M1-P4-P0e：关节路径执行可行性 ===", flush=True)

    # CRN 修复验证：不同 stream 噪声哈希不同
    print("[CRN 修复验证] 噪声哈希:", flush=True)
    for sid, st in [(1, 3), (1, 4), (1, 5), (2, 0), (4, 3)]:
        print(f"  state{sid}/stream{st}: {noise_hash(sid, st)}", flush=True)

    out_all = {}
    for state_id, stream in [(1, 3)]:  # P0e-1：先 S01
        print(f"\n--- S{state_id:02d}/st{stream} ---", flush=True)
        t0 = time.time()
        env, layer, scratch, layer_s, rg, og, p0, soup_pos, nvs_step, Dmax = \
            reconstruct_nvs(ts, task, prompt, pol, state_id, stream)
        q_start = get_arm_q(env)
        print(f"  NVS @ step {nvs_step}，Dmax={Dmax*1000:.3f}mm", flush=True)
        print(f"  q_start={np.round(q_start,3)}", flush=True)
        print(f"  soup={np.round(soup_pos,4)} mug={np.round(p0,4)}", flush=True)
        print(f"  collision_free(q_start)={collision_free(env, q_start, rg, og)}", flush=True)

        obs = env.env._get_observations()
        iq = np.asarray(obs["robot0_eef_quat"])
        tgt_ori = Rot.from_quat([iq[1], iq[2], iq[3], iq[0]]).as_matrix()

        # 多起点 IK → 无碰撞目标
        free_by_off = {}
        for off in PREGRASP_OFFSETS:
            free_sols, _, _ = ik_multistart(env, soup_pos + off, tgt_ori, rg, og, n=IK_RESTARTS)
            free_by_off[tuple(off.tolist())] = len(free_sols)
        best_off = max(PREGRASP_OFFSETS, key=lambda o: free_by_off[tuple(o.tolist())])
        n_free = free_by_off[tuple(best_off.tolist())]
        print(f"  [IK] 无碰撞解分布: {free_by_off}", flush=True)
        if n_free == 0:
            print("  无无碰撞 IK 解，跳过", flush=True)
            env.close(); scratch.close()
            continue
        free_sols, _, _ = ik_multistart(env, soup_pos + best_off, tgt_ori, rg, og, n=IK_RESTARTS, seed=7)
        q_goal = free_sols[0]

        # RRT（多 seed，记录失败次数）
        found = False
        path = None
        used_seed = None
        fail_count = 0
        for seed in range(8):
            p, f = rrt_connect(env, rg, og, q_start, q_goal, max_iter=RRT_ITER, seed=seed)
            if f:
                found = True
                path = p
                used_seed = seed
                break
            fail_count += 1
        print(f"  [RRT] found={found} seed={used_seed} 失败次数={fail_count} "
              f"路径长度={len(path) if path else 0}", flush=True)
        if not found:
            print("  RRT 未找到路径，跳过", flush=True)
            env.close(); scratch.close()
            continue

        # 恢复执行起点（RRT 规划污染了 env）
        set_arm_q(env, q_start)
        mujoco.mj_forward(env.sim.model._model, env.sim.data._data)
        # 清零臂速度（从静止开始执行）
        env.sim.data._data.qvel[:7] = 0.0
        env.sim.data._data.qvel[7:9] = 0.0
        print(f"  执行起点 collision_free={collision_free(env, q_start, rg, og)}", flush=True)

        # 关节 PD 执行
        print(f"  [执行] 关节 PD 跟踪（v_max={V_MAX}, a_max={A_MAX}, kp={KP}, kd={KD}）...", flush=True)
        res = execute_joint_path(env, path, rg, og, p0, state_id, stream)
        print(f"  [结果] T={res['T']:.2f}s 路径长={res['L']:.2f}rad  "
              f"首接触={res['first_contact_t']}  Dmax={res['Dmax']*1000:.3f}mm", flush=True)
        print(f"  跟踪: max_track_err={res['track_err_max']*1000:.1f}mrad  "
              f"max_tau={res['tau_max_abs']:.1f}N·m", flush=True)
        print(f"  终点: pos_err={res['pos_err']*1000:.1f}mrad  vel_norm={res['vel_norm']:.4f}rad/s", flush=True)

        # 验收
        ok = (res["first_contact_t"] is None) and (res["Dmax"] < DELTA_CAR - EPS) \
             and (res["pos_err"] < 0.02) and (res["vel_norm"] < 0.05)
        print(f"  [验收] {'PASS' if ok else 'FAIL'}", flush=True)

        out_all[f"{state_id}/{stream}"] = {
            "nvs_step": nvs_step, "q_start": q_start.tolist(),
            "free_by_off": {str(k): v for k, v in free_by_off.items()},
            "best_off": best_off.tolist(), "n_free": n_free,
            "rrt_found": found, "rrt_seed": used_seed, "rrt_fail_count": fail_count,
            "rrt_path_len": len(path), "exec": res, "accept": ok,
        }
        env.close(); scratch.close()
        print(f"  [耗时 {time.time()-t0:.1f}s]", flush=True)

    out = pathlib.Path(ROOT, "data", "audit", "m1_p4_p0e_s01_execution.json")
    out.write_text(json.dumps(out_all, ensure_ascii=False, indent=2, default=str))
    print(f"\n结果: {out}")
    pol.close()


if __name__ == "__main__":
    main()
