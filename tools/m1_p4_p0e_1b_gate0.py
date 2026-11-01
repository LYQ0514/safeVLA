"""P0e-1b Gate 0：定位 S01 执行中 2.115mm mug 位移的实际来源。

两个独立审计：
A. 静止基线：同一 NVS 状态，无机器人运动，mug 自身是否漂移 > 1mm？
B. 执行中：首次超过 1mm 的控制步/substep，记录 mug 位置/速度、全部 mug 接触对
   （不只 link5/link6）、接触 dist、约束状态。

不归因，只定位事实。
"""

from __future__ import annotations

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

from src.pnsg.policy_adapter import PnsgPolicy  # noqa: E402
from tools.g1a_soup_trace import body_state  # noqa: E402
from tools.m1_p1_design_gate_v2 import build_body_geoms  # noqa: E402
from tools.m1_p1_design_gate_v3 import disp_from_p0  # noqa: E402
from tools.m1_p4_p0e_joint_execution import (  # noqa: E402
    reconstruct_nvs, generate_block, MUG_BODY, SOUP_BODY,
    trapezoidal, path_at_s, check_contact_data, KP, KD, V_MAX, A_MAX,
    TORQUE_LO, TORQUE_HI, SETTLE_S, PREGRASP_OFFSETS, IK_RESTARTS, RRT_ITER,
)
from tools.m1_p4_p0_reachability import (  # noqa: E402
    joint_limits, set_arm_q, get_arm_q, collision_free,
)
from tools.m1_p4_p0_reachability_v2 import ik_multistart, rrt_connect  # noqa: E402


def mug_contacts(env, rg, og):
    """当前 d.contact 中所有涉及 mug 的接触对。"""
    m = env.sim.model._model
    d = env.sim.data._data
    out = []
    for c in range(d.ncon):
        g1, g2 = d.contact[c].geom1, d.contact[c].geom2
        b1 = env.sim.model.body_names[m.geom_bodyid[g1]]
        b2 = env.sim.model.body_names[m.geom_bodyid[g2]]
        if "mug" in b1 or "mug" in b2:
            out.append({
                "pair": f"{b1}↔{b2}",
                "geom1": g1, "geom2": g2,
                "dist": float(d.contact[c].dist),
                "efc_address": int(d.contact[c].efc_address),
            })
    return out


def baseline_stillness(env, p0, seconds=2.0):
    """无机器人运动，mug 静止基线。"""
    m = env.sim.model._model
    d = env.sim.data._data
    dt = env.env.model_timestep
    n = int(seconds / dt)
    Dmax = 0.0
    for i in range(n):
        d.ctrl[:7] = 0.0
        d.ctrl[7:] = 0.0
        env.env.sim.forward()
        env.env.sim.step()
        Dmax = max(Dmax, disp_from_p0(env, p0))
    return Dmax


def main():
    from libero.libero import benchmark
    ts = benchmark.get_benchmark_dict()['safelibero_long'](safety_level='I')
    task = ts.get_task(0)
    prompt = task.language
    pol = PnsgPolicy()

    print("=== P0e-1b Gate 0：S01 位移来源审计 ===", flush=True)
    state_id, stream = 1, 3

    env, layer, scratch, layer_s, rg, og, p0, soup_pos, nvs_step, Dmax0 = \
        reconstruct_nvs(ts, task, prompt, pol, state_id, stream)
    q_start = get_arm_q(env)
    print(f"NVS @ step {nvs_step}", flush=True)

    # ---- Gate 0-A：静止基线 ----
    set_arm_q(env, q_start)
    mujoco.mj_forward(env.sim.model._model, env.sim.data._data)
    env.sim.data._data.qvel[:] = 0.0
    Dstill = baseline_stillness(env, p0, seconds=2.0)
    print(f"[Gate0-A] 静止基线 2s：mug Dmax = {Dstill*1000:.3f}mm", flush=True)

    # ---- 重建 RRT 路径（固定 seed=1，同 P0e-1）----
    obs = env.env._get_observations()
    iq = np.asarray(obs["robot0_eef_quat"])
    from scipy.spatial.transform import Rotation as Rot
    tgt_ori = Rot.from_quat([iq[1], iq[2], iq[3], iq[0]]).as_matrix()
    free_by_off = {}
    for off in PREGRASP_OFFSETS:
        fs, _, _ = ik_multistart(env, soup_pos + off, tgt_ori, rg, og, n=IK_RESTARTS)
        free_by_off[tuple(off.tolist())] = len(fs)
    best_off = max(PREGRASP_OFFSETS, key=lambda o: free_by_off[tuple(o.tolist())])
    free_sols, _, _ = ik_multistart(env, soup_pos + best_off, tgt_ori, rg, og, n=IK_RESTARTS, seed=7)
    q_goal = free_sols[0]
    path = None
    for seed in range(8):
        p, f = rrt_connect(env, rg, og, q_start, q_goal, max_iter=RRT_ITER, seed=seed)
        if f:
            path = p
            break
    print(f"[路径] RRT 找到 {len(path)} 路点", flush=True)

    # ---- Gate 0-B：执行并逐步定位首次 >1mm ----
    set_arm_q(env, q_start)
    mujoco.mj_forward(env.sim.model._model, env.sim.data._data)
    env.sim.data._data.qvel[:] = 0.0
    m = env.sim.model._model
    d = env.sim.data._data
    dt = env.env.model_timestep
    seg = [float(np.linalg.norm(path[i+1] - path[i])) for i in range(len(path)-1)]
    L = sum(seg)
    cumlen = np.concatenate([[0.0], np.cumsum(seg)])
    s_arr, t_arr, T = trapezoidal(L, V_MAX, A_MAX, dt)
    n_settle = int(SETTLE_S / dt)
    M_full = np.ndarray((m.nv, m.nv), dtype=np.float64, order="C")

    Dmax = 0.0
    first_over_step = None
    contact_steps = []  # (substep_idx, contacts)
    for i in range(len(s_arr) + n_settle):
        q_ref = path_at_s(path, cumlen, s_arr[i]) if i < len(s_arr) else q_goal
        env.env.sim.forward()
        q = np.asarray(d.qpos[:7])
        qvel = np.asarray(d.qvel[:7])
        mujoco.mj_fullM(m, M_full, d.qM)
        M_arm = M_full[:7, :7]
        qfrc_bias = np.asarray(d.qfrc_bias[:7])
        tau = np.clip(M_arm @ (KP * (q_ref - q) - KD * qvel) + qfrc_bias, TORQUE_LO, TORQUE_HI)
        d.ctrl[:7] = tau
        d.ctrl[7:] = 0.0
        env.env.sim.step()
        mc = mug_contacts(env, rg, og)
        if mc:
            contact_steps.append((i, mc))
        Dmax = max(Dmax, disp_from_p0(env, p0))
        if first_over_step is None and Dmax > 0.001:
            first_over_step = i
            mug_state = body_state(env, MUG_BODY)
            print(f"[Gate0-B] 首次超 1mm @ substep {i} (t={i*dt:.2f}s)", flush=True)
            print(f"  mug pos={np.round(mug_state['pos'],4)}  vel={np.round(mug_state['vel'],4)}", flush=True)
            print(f"  p0={np.round(p0,4)}  disp={disp_from_p0(env, p0)*1000:.3f}mm", flush=True)
            print(f"  该 substep mug 接触对: {mc if mc else '无(可能前一 substep 已累积)'}", flush=True)
            # 附近 20 substep 的接触
            print(f"  总接触 substep 数={len(contact_steps)}", flush=True)
            break

    print(f"\n[Gate0-B] 执行结束 Dmax={Dmax*1000:.3f}mm", flush=True)
    print(f"  有 mug 接触的 substep 数={len(contact_steps)}", flush=True)
    # 汇总所有唯一接触对
    unique_pairs = set()
    for _, mcs in contact_steps:
        for c in mcs:
            unique_pairs.add(c["pair"])
    print(f"  涉及 mug 的唯一接触对: {unique_pairs if unique_pairs else '无'}", flush=True)

    out = pathlib.Path(ROOT, "data", "audit", "m1_p4_p0e_1b_gate0.json")
    out.write_text(json.dumps({
        "stillness_2s_Dmax_mm": Dstill * 1000,
        "exec_Dmax_mm": Dmax * 1000,
        "first_over_step": first_over_step,
        "first_over_t_s": None if first_over_step is None else first_over_step * dt,
        "contact_substep_count": len(contact_steps),
        "unique_mug_contact_pairs": sorted(unique_pairs),
        "mug_contacts_detail": contact_steps[:50],
    }, ensure_ascii=False, indent=2, default=str))
    print(f"\n结果: {out}")

    pol.close()
    env.close()
    scratch.close()


if __name__ == "__main__":
    main()
