"""M3-P1a：Frozen-VLA Intervention Burden Transfer Test。

验证：真实冻结 π0.5 + AEGIS 闭环中，是否存在「当前修正更小 → 未来负担更大」的
排序反转。

协议（裁决）：
- Gate 1：构造两个当前安全候选 u_A（AEGIS QP 解）与 u_B（切向绕行，当前修正更大）；
- Gate 2：两条分支从同一父状态分叉，只第一步不同，后续都恢复原始 AEGIS；
- Gate 3：分别衡量 D_H（未来累积修正幅度）、P_H（任务进度）、S_H（物理安全）。

规模：2 状态 × 2 检查点 × 2 stream × 2 候选 = 8 对（16 短 rollout），H=2 优先。
"""

from __future__ import annotations

import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

ROOT = "/data/safeVLA"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "dataset", "SafeLIBERO", "safelibero"))
sys.path.insert(0, "/data/2608.SDL/MojocoProj/src/pi05")
sys.path.insert(0, os.path.join(ROOT, "third_party", "aegis"))

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
import cvxpy as cp  # noqa: E402
import mujoco  # noqa: E402

from src.policies import Pi05Policy  # noqa: E402
from src.safety.aegis_adapted import AegisSafetyLayer  # noqa: E402
from src.safety.contract_objects import monitor_contacts  # noqa: E402
from src.safety.cca_asf.intention import chunk_direction  # noqa: E402
from src.safety.cca_asf.metric import map_dz_to_dy  # noqa: E402
from tools.g1a_soup_trace import make_env, body_state, eef_pos  # noqa: E402
from tools.m1_p2_p1a_pilot import save_full, restore_full  # noqa: E402

SUITE = "safelibero_long"
LEVEL = "I"
TASK_ID = 0
REPLAN = 5
IMAGE_SIZE = 256
H = 2  # 未来重规划周期（每个周期 replan_steps=5 步）
GAMMA = 0.9

W = np.diag([1.0 / 25] * 6 + [1.0, 1.0, 1.0])
PROTECTED = "red_coffee_mug_obstacle_1_main"
SOUP_BODY = "alphabet_soup_1_main"
CAR_EPS = 0.001

# 预注册：2 状态 × 检查点（干预步，从 Gate1 探测的干预步里取，固定）
STATES = [1, 9]
CHECKPOINT_OFFSETS = [0, 5]  # 从首次干预起算的第 0、5 个干预步
STREAMS = [3, 4]


def solve_u(Wm, u_ref, a_full, r):
    u = cp.Variable(9)
    obj = cp.Minimize(cp.quad_form(u - u_ref, Wm))
    con = [a_full @ u >= r]
    prob = cp.Problem(obj, con)
    try:
        prob.solve(solver=cp.OSQP)
    except Exception:
        return None
    if u.value is None:
        return None
    return np.asarray(u.value, dtype=float)


def u_to_action(u_9, action7, R1):
    """9D u 空间 → 7D action 空间（AEGIS 输出反映射）。"""
    out = np.zeros(7)
    out[:3] = 0.2 * R1 @ u_9[:3]
    out[3:6] = 0.2 * u_9[3:6]
    out[6] = action7[6]
    return out


def task_progress_fn(env, s0_soup_dist):
    """任务进度：soup 距离相对初始的减少量（正=靠近）。"""
    return float(np.linalg.norm(body_state(env, SOUP_BODY)["pos"] - eef_pos(env)))


def run_branch(ts, task, prompt, pol, env, layer, parent_state, u_first_9, action7,
               R1, p0, soup_dist0, n_steps, use_u_first):
    """从父状态分叉，跑 n_steps 步。第一步用 u_first_9（若 use_u_first），后续 AEGIS。

    返回 (D_H, P_H, S_H, truncated)。
    """
    restore_full(env, layer, parent_state)
    obs = env.env._get_observations(force_update=True)
    pol.reset()

    D_H = 0.0
    n_forbidden = 0
    Dmax = 0.0
    truncated = False
    success = False

    for step in range(n_steps):
        action = pol.act(obs, prompt)
        if step == 0 and use_u_first:
            exec_a = u_to_action(u_first_9, action, R1)
        else:
            exec_a, info = layer.filter(action, obs)

        # 未来负担：||u_safe - u_nom||_W^2（后续 AEGIS 步）
        if step > 0 or not use_u_first:
            # 用 AEGIS 的 u 空间偏差（需要重算 u_ref 和 u_safe）
            u_nom = _u_nom(action, obs, layer)
            a_full, r = _constraint(action, obs, layer)
            u_safe = solve_u(W, u_nom, a_full, r) if a_full is not None else u_nom
            if u_safe is not None:
                dev = u_safe - u_nom
                D_H += (GAMMA ** (step - 1)) * float(dev @ W @ dev)

        obs, _r, _d, _i = env.step(exec_a)

        # 物理安全
        mon = monitor_contacts(env)
        n_forbidden += len(mon["forbidden"]) + len(mon["unclassified"])
        Dmax = max(Dmax, float(np.sum(np.abs(body_state(env, PROTECTED)["pos"] - p0))))

        if env.check_success():
            success = True
            break

    soup_dist_final = float(np.linalg.norm(body_state(env, SOUP_BODY)["pos"] - eef_pos(env)))
    P_H = soup_dist0 - soup_dist_final  # 正 = soup 距离减小（接近）
    S_H = (Dmax < CAR_EPS) and (n_forbidden == 0)
    return D_H, P_H, S_H, truncated, success


def _u_nom(action, obs, layer):
    from src.safety.aegis_adapted import V_REF_GAIN, OMEGA_REF_GAIN, U_Z_NOM_GAIN
    from src.safety.aegis_adapted import _quat_wxyz_to_rot, compute_h_coeffs_3d
    R1 = _quat_wxyz_to_rot(np.asarray(obs["robot0_eef_quat"], dtype=np.float64))
    v_ref = R1.T @ action[:3]
    u_v_ref = V_REF_GAIN * v_ref
    omega_ref = action[3:6]
    u_omega_ref = OMEGA_REF_GAIN * omega_ref
    p1 = layer._eef_center(obs)
    _, _, _, _, mu_row = compute_h_coeffs_3d(p1, layer.Q1_diag, R1,
                                             layer.p2, layer.Q2_diag, layer.R2,
                                             layer.z_fixed)
    u_z_nom = U_Z_NOM_GAIN * mu_row
    return np.hstack([u_v_ref, u_omega_ref, u_z_nom])


def _constraint(action, obs, layer):
    from src.safety.aegis_adapted import (V_REF_GAIN, OMEGA_REF_GAIN, QP_ACTION_SCALE,
                                          QP_H_GAIN, _quat_wxyz_to_rot, compute_h_coeffs_3d)
    R1 = _quat_wxyz_to_rot(np.asarray(obs["robot0_eef_quat"], dtype=np.float64))
    p1 = layer._eef_center(obs)
    a_v, a_omega, a_uz, h, mu_row = compute_h_coeffs_3d(p1, layer.Q1_diag, R1,
                                                        layer.p2, layer.Q2_diag, layer.R2,
                                                        layer.z_fixed)
    a_u_v = QP_ACTION_SCALE * a_v
    a_u_omega = QP_ACTION_SCALE * a_omega
    a_full = np.hstack([a_u_v, a_u_omega, a_uz])
    r = -QP_H_GAIN * h
    return a_full, r


def main():
    from libero.libero import benchmark
    ts = benchmark.get_benchmark_dict()[SUITE](safety_level=LEVEL)
    task = ts.get_task(TASK_ID)
    prompt = task.language

    all_pairs = []

    for state_id in STATES:
        for stream in STREAMS:
            pol = Pi05Policy(verbose=False, replan_steps=REPLAN, image_size=IMAGE_SIZE)
            env = make_env(ts, task, state_id)
            layer = AegisSafetyLayer(env, filter_mode="euclidean")
            layer.reset(env.env._get_observations(), prompt)
            pol.reset()

            p0 = body_state(env, PROTECTED)["pos"].copy()
            soup_dist0 = float(np.linalg.norm(body_state(env, SOUP_BODY)["pos"] - eef_pos(env)))

            # 跑 AEGIS 轨迹，收集干预步的父状态
            intervention_states = []
            for step in range(300):
                obs = env.env._get_observations()
                action = pol.act(obs, prompt)
                exec_a, info = layer.filter(action, obs)
                if info.get("intervened") is True:
                    # 记录父状态 + 构造候选所需的量
                    from src.safety.aegis_adapted import _quat_wxyz_to_rot
                    R1 = _quat_wxyz_to_rot(np.asarray(obs["robot0_eef_quat"], dtype=np.float64))
                    u_nom = _u_nom(action, obs, layer)
                    a_full, r = _constraint(action, obs, layer)
                    u_A = solve_u(W, u_nom, a_full, r)
                    if u_A is not None:
                        # 构造 u_B（切向绕行）
                        d_z = chunk_direction(pol.get_current_chunk()[max(0, pol.buffer_index-1):]) \
                            if pol.get_current_chunk() is not None else None
                        if d_z is not None:
                            d_y = map_dz_to_dy(d_z, R1)
                            if d_y is not None:
                                a_norm_sq = float(a_full @ a_full)
                                v_null = d_y - (d_y @ a_full) / max(a_norm_sq, 1e-12) * a_full
                                vn = float(np.linalg.norm(v_null))
                                if vn > 1e-6:
                                    v_null = v_null / vn
                                    dev_A = float(np.linalg.norm(u_A - u_nom))
                                    u_B = None
                                    for beta in [0.5, 1.0, 1.5, 2.0]:
                                        cand = u_A + beta * v_null
                                        if a_full @ cand >= r - 1e-9:
                                            if float(np.linalg.norm(cand - u_nom)) > dev_A + 1e-6:
                                                u_B = cand
                                                break
                                    if u_B is not None:
                                        intervention_states.append({
                                            "step": step,
                                            "parent": save_full(env, layer),
                                            "u_A": u_A, "u_B": u_B,
                                            "action": action.copy(), "R1": R1.copy(),
                                        })
                env.step(exec_a)

            # 从干预步里选 2 个检查点（固定偏移）
            n_int = len(intervention_states)
            if n_int == 0:
                print(f"S{state_id:02d}/st{stream}: 无干预步，跳过", flush=True)
                pol.close(); env.close()
                continue

            for off in CHECKPOINT_OFFSETS:
                if off >= n_int:
                    continue
                cp_state = intervention_states[off]

                # 两条分支
                res_A = run_branch(ts, task, prompt, pol, env, layer, cp_state["parent"],
                                   cp_state["u_A"], cp_state["action"], cp_state["R1"],
                                   p0, soup_dist0, H * REPLAN, use_u_first=True)
                res_B = run_branch(ts, task, prompt, pol, env, layer, cp_state["parent"],
                                   cp_state["u_B"], cp_state["action"], cp_state["R1"],
                                   p0, soup_dist0, H * REPLAN, use_u_first=True)

                D_A, P_A, S_A, trunc_A, succ_A = res_A
                D_B, P_B, S_B, trunc_B, succ_B = res_B

                # 排序反转判定（只在两分支都安全、未截断时有效）
                reversal = None
                if S_A and S_B and not trunc_A and not trunc_B:
                    dev_A = float(np.linalg.norm(cp_state["u_A"] - _u_nom(cp_state["action"], env.env._get_observations(), layer)))
                    reversal = (D_A > D_B + 1e-9) and (P_B >= P_A - 0.01)

                pair = {
                    "state_id": state_id, "stream": stream,
                    "checkpoint_step": cp_state["step"], "off": off,
                    "D_A": D_A, "D_B": D_B,
                    "P_A": P_A, "P_B": P_B,
                    "S_A": S_A, "S_B": S_B,
                    "trunc_A": trunc_A, "trunc_B": trunc_B,
                    "succ_A": succ_A, "succ_B": succ_B,
                    "reversal": reversal,
                }
                all_pairs.append(pair)
                print(f"S{state_id:02d}/st{stream} 检查点@{cp_state['step']}: "
                      f"D_A={D_A:.4f} D_B={D_B:.4f} P_A={P_A:.4f} P_B={P_B:.4f} "
                      f"S_A={S_A} S_B={S_B} 反转={reversal}", flush=True)

            pol.close()
            env.close()

    # 汇总
    valid = [p for p in all_pairs if p["reversal"] is not None]
    n_reversal = sum(1 for p in valid if p["reversal"])
    print(f"\n=== M3-P1a 汇总 ===")
    print(f"总配对: {len(all_pairs)}")
    print(f"有效配对（双安全+未截断）: {len(valid)}")
    print(f"排序反转: {n_reversal}/{len(valid)}")

    out = {"pairs": all_pairs, "n_valid": len(valid), "n_reversal": n_reversal}
    out_path = os.path.join(ROOT, "data", "audit", "m3_p1a_burden_transfer.json")
    with open(out_path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果: {out_path}")


if __name__ == "__main__":
    main()
