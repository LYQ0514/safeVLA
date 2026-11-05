"""Instrumented 执行循环（E1a 事件级接入）。

复制历史函数的最小执行内核，接入 TaskEventBridge。除插入观测/快照/日志外，
不改控制计算、循环次数、路径插值、torque clipping、action 生成、OSC 初始化、
success/termination 顺序。

源函数（复制声明）：
- execute_joint_path_instrumented ← tools/m1_p4_p1_reentry.py:execute_joint_path
- run_pv_branch_instrumented       ← tools/m1_p4_p1_reentry.py:run_branch

source_file = tools/m1_p4_p1_reentry.py
source_commit = 85a0d1c（FPCK 安装前主线）
source_sha256 = a98e558f78ac0818834b0f06ba5b0247
local_modification = instrumentation_only
"""

from __future__ import annotations

import numpy as np
import mujoco

from tools.g1a_soup_trace import body_state, eef_pos, subgoal_values
from tools.m1_p4_p0e_joint_execution import (
    trapezoidal, generate_block, SOUP_BODY, MUG_BODY,
    KP, KD, V_MAX, A_MAX, SETTLE_S, TORQUE_LO, TORQUE_HI,
)
from tools.m1_p4_p1_reentry import _pair_types
from tools.m1_p2_p1a_pilot import (
    Z_DIR, L_DIR, AVOID_MAG, AVOID_STEPS, SWITCH, REPLAN_STEPS, IMAGE_SIZE,
    PROTECTED, DELTA_CAR, EPS,
)
from src.safety.contract_objects import robot_geoms, forbidden_contact_geoms, monitor_contacts
from src.safety.aegis_adapted import AegisSafetyLayer

# 每 25 个 substep 聚合一个控制步（Joint-PD 阶段）
SUBSTEPS_PER_CONTROL = 25


def execute_joint_path_instrumented(env, path, p0, bridge, target_bodies,
                                    settled_baseline, obstacle_body):
    """Joint-PD 执行器（instrumented）：逐 substep 观测 + 每 25 substep 聚合控制步。

    与 execute_joint_path 唯一区别：插入 bridge.observe_physics_substep /
    observe_control_step 观测，不改控制计算。
    """
    m = env.sim.model._model
    d = env.sim.data._data
    dt = env.env.model_timestep
    seg = [float(np.linalg.norm(path[i+1]-path[i])) for i in range(len(path)-1)]
    L = sum(seg)
    cumlen = np.concatenate([[0.0], np.cumsum(seg)])
    s_arr, t_arr, T = trapezoidal(L, V_MAX, A_MAX, dt)
    n_settle = int(SETTLE_S / dt)
    q_goal = path[-1].copy()
    M_full = np.ndarray((m.nv, m.nv), dtype=np.float64, order="C")

    def p_at(s):
        idx = int(np.searchsorted(cumlen, s))-1
        idx = max(0, min(idx, len(path)-2))
        seglen = cumlen[idx+1]-cumlen[idx]
        frac = (s-cumlen[idx])/seglen if seglen > 1e-12 else 0.0
        return path[idx] + frac*(path[idx+1]-path[idx])

    Dmax = 0.0
    total_substeps = len(s_arr) + n_settle
    for i in range(total_substeps):
        q_ref = p_at(s_arr[i]) if i < len(s_arr) else q_goal
        env.env.sim.forward()
        q = np.asarray(d.qpos[:7]); qvel = np.asarray(d.qvel[:7])
        mujoco.mj_fullM(m, M_full, d.qM)
        tau = np.clip(M_full[:7, :7] @ (KP*(q_ref-q)-KD*qvel) + np.asarray(d.qfrc_bias[:7]),
                      TORQUE_LO, TORQUE_HI)
        d.ctrl[:7] = tau; d.ctrl[7:] = 0.0
        env.env.sim.step()
        Dmax = max(Dmax, float(np.sum(np.abs(body_state(env, MUG_BODY)["pos"]-p0))))

        # ---- instrumentation：physics substep 观测 ----
        target_poses = {t: body_state(env, t)["pos"] for t in target_bodies}
        bridge.observe_physics_substep(
            env=env, phase="PREGRASP", control_step=i // SUBSTEPS_PER_CONTROL,
            physics_substep=i, target_bodies=target_bodies, target_poses=target_poses)

        # 每 25 substep 聚合控制步（goal truth + 自动机状态）
        if (i + 1) % SUBSTEPS_PER_CONTROL == 0 or i == total_substeps - 1:
            obs = env.env._get_observations()
            bridge.observe_control_step(
                env=env, obs=obs, phase="PREGRASP",
                control_step=i // SUBSTEPS_PER_CONTROL,
                nominal_action=None, executed_action=None,
                target_bodies=target_bodies)

    return Dmax


def run_pv_branch_instrumented(env, layer, scratch, layer_s, obs, Dmax_hist,
                               pol, prompt, state_id, stream, bridge,
                               target_bodies, max_steps=550):
    """P-V 分支（instrumented）：从接管状态执行冻结 π0.5，逐控制步观测。

    与 run_branch 的 P-V 分支唯一区别：接入 bridge.observe_control_step。
    """
    rg = robot_geoms(env)
    fg = forbidden_contact_geoms(env)
    rg_old, og = _build_body_geoms_compat(env)
    p0 = body_state(env, MUG_BODY)["pos"].copy()
    Dmax = Dmax_hist
    first_forbidden = None
    forbidden_events = []
    soup_done = None
    success = False
    success_step = None
    nominal = None
    n_generate = 0

    for step in range(max_steps):
        if step % REPLAN_STEPS == 0:
            nominal = generate_block(pol, obs, prompt, step // REPLAN_STEPS, state_id, stream)
            n_generate += 1
        nom_a = nominal[step % REPLAN_STEPS]
        act = nom_a  # P-V：直接执行 nominal

        obs, _r, _d, _i = env.step(act)

        # 完整契约审计（与 run_branch 一致）
        mon = monitor_contacts(env)
        for ev in mon["forbidden"] + mon["unclassified"]:
            forbidden_events.append((step, ev["body1"], ev["body2"], ev["dist"]))
            if first_forbidden is None:
                first_forbidden = step
        Dmax = max(Dmax, float(np.sum(np.abs(body_state(env, MUG_BODY)["pos"] - p0))))

        sg = subgoal_values(env)
        if sg.get("alphabet_soup_1") and soup_done is None:
            soup_done = step

        # ---- instrumentation：控制步观测 ----
        chunk_hash = None
        from src.fpck.collector import sha256_hex
        if nominal is not None:
            chunk_hash = sha256_hex(np.asarray(nominal).tobytes())[:16]
        bridge.observe_control_step(
            env=env, obs=obs, phase="REENTRY", control_step=step,
            nominal_action=nom_a, executed_action=act,
            chunk_hash=chunk_hash, buffer_index=step % REPLAN_STEPS,
            target_bodies=target_bodies)

        if env.check_success():
            success = True
            success_step = step
            break

    return {
        "success": success, "success_step": success_step, "soup_done": soup_done,
        "Dmax": Dmax, "car_ok": Dmax < DELTA_CAR - EPS,
        "n_forbidden_events": len(forbidden_events), "first_forbidden": first_forbidden,
        "forbidden_pair_types": _pair_types(forbidden_events),
        "n_generate": n_generate,
    }


def _build_body_geoms_compat(env):
    """HVNP 决策用旧接触集合（保持 run_branch 行为，来自 m1_p4_p1_reentry 的 import）。"""
    from tools.m1_p4_p1_reentry import build_body_geoms
    return build_body_geoms(env)
