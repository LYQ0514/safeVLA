"""R3-G4-①：控制执行链审计（25 substep）。

对 state02/stream0 step15，三种几何 G-L/G-S/G-P，区分三层：
- γ_QP：QP 输出动作的物理位移投影（out[:3] × 0.05）
- γ_cmd：OSC set_goal 的目标位移投影（scale_action(out)）
- γ_real：真实 EEF 位移投影（n_t · Δp_real）

并分解 Δp_real = γ_real·n + Δp_⊥，验证 d² 公式，判断 M1/M2/M3。

门禁：同一快照恢复，不额外调 filter。
"""

from __future__ import annotations

import os
import pathlib
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
import pi05_libero_runner as R  # noqa: E402

from src.pnsg.crn import nominal_seed, sample_z  # noqa: E402
from src.pnsg.policy_adapter import PnsgPolicy  # noqa: E402
from src.safety.aegis_adapted import (  # noqa: E402
    AegisSafetyLayer, _quat_wxyz_to_rot, EEF_OFFSET_LOCAL,
    QP_ACTION_SCALE, V_REF_GAIN, OMEGA_REF_GAIN, QP_H_GAIN, U_Z_NOM_GAIN,
)
from tools.g1a_soup_trace import make_env, body_state, eef_pos  # noqa: E402

SUITE = "safelibero_long"
LEVEL = "I"
TASK_ID = 0
REPLAN_STEPS = 5
IMAGE_SIZE = 224
TRANS_SCALE = 0.05

GEO = {"G-L": np.array([0.06, 0.12, 0.20]),
       "G-S": np.array([0.06, 0.12, 0.11]),
       "G-P": np.array([0.14, 0.14, 0.14])}


def filter_with_q(nom_a, layer, R1, Q1_diag):
    from utils import compute_h_coeffs_3d
    p1 = layer._eef_center(None) if False else layer._eef_center
    # 直接用 layer 的当前 obs 算
    # 这里需要 obs，但 layer 内部有 env，从 env 读
    obs = layer.env.env._get_observations()
    p1 = layer._eef_center(obs)
    a_v, a_omega, a_uz, h, mu_row = compute_h_coeffs_3d(
        p1, Q1_diag, R1, layer.p2, layer.Q2_diag, layer.R2, layer.z_fixed)
    a_u_v = QP_ACTION_SCALE * a_v
    a_u_omega = QP_ACTION_SCALE * a_omega
    u_z_nom = U_Z_NOM_GAIN * mu_row
    v_ref = R1.T @ nom_a[:3]
    u_v_ref = V_REF_GAIN * v_ref
    u_omega_ref = OMEGA_REF_GAIN * nom_a[3:6]
    u_ref = np.hstack([u_v_ref, u_omega_ref, u_z_nom])
    u = cp.Variable(9)
    W = np.diag([1.0/25]*6 + [1.0, 1.0, 1.0])
    obj = cp.Minimize(cp.quad_form(u - u_ref, W))
    cons = [a_u_v @ u[:3] + a_u_omega @ u[3:6] + a_uz @ u[6:] + QP_H_GAIN * h >= 0]
    prob = cp.Problem(obj, cons)
    try:
        prob.solve(solver=cp.OSQP)
        if u.value is None:
            return None, h
        u_opt = np.asarray(u.value).flatten()
        out = np.zeros(7)
        out[:3] = QP_ACTION_SCALE * (R1 @ u_opt[:3])
        out[3:6] = QP_ACTION_SCALE * u_opt[3:6]
        out[6] = nom_a[6]
        return out, h
    except Exception:
        return None, h


def main():
    from libero.libero import benchmark
    ts = benchmark.get_benchmark_dict()[SUITE](safety_level=LEVEL)
    task = ts.get_task(TASK_ID)
    prompt = task.language
    state_id, stream = 2, 0

    env = make_env(ts, task, state_id)
    layer = AegisSafetyLayer(env)
    obs = env.env._get_observations()
    layer.reset(obs, prompt)
    pol = PnsgPolicy()

    target_step = 15
    print(f"=== R3-G4-①：控制执行链审计（step {target_step}）===")

    for step in range(550):
        qi = step // REPLAN_STEPS
        if step % REPLAN_STEPS == 0:
            z0, _ = sample_z(nominal_seed(SUITE, TASK_ID, state_id, stream, qi))
            po = R.obs_to_policy_obs(obs, prompt, image_size=IMAGE_SIZE)
            nominal = np.clip(pol.generate(z0, po), -1, 1)
        nom_a = nominal[step % REPLAN_STEPS]

        if step == target_step:
            R1 = _quat_wxyz_to_rot(np.asarray(obs["robot0_eef_quat"], dtype=np.float64))
            soup = body_state(env, "alphabet_soup_1_main")["pos"]
            eef = eef_pos(env)
            d_vec = soup - eef
            n_t = d_vec / (np.linalg.norm(d_vec) + 1e-9)
            d_t = np.linalg.norm(d_vec)

            snapshot = env.sim.get_state()
            print(f"\nstep {step}: d_t={d_t:.4f}m")

            for gname, Q1 in GEO.items():
                env.sim.set_state(snapshot)
                env.sim.forward()
                out, h = filter_with_q(nom_a, layer, R1, Q1)
                if out is None:
                    print(f"  {gname}: 不可行")
                    continue

                # γ_QP（QP 输出的归一化动作 × 缩放 = 物理位移投影）
                gamma_qp = float(n_t @ (out[:3] * TRANS_SCALE))
                # γ_cmd（OSC scale_action 后的目标位移）
                from robosuite.controllers.base_controller import Controller
                cmd = layer.env.robots[0].controller.scale_action(out[:6])
                gamma_cmd = float(n_t @ cmd[:3])

                # 执行，记录 25 substep EEF 轨迹
                eef_before = eef_pos(layer.env)
                act7 = np.zeros(7); act7[:6] = out[:6]; act7[6] = out[6]
                eef_traj = []
                for i in range(int(env.env.control_timestep / env.env.model_timestep)):
                    env.sim.forward()
                    env.env._pre_action(act7, (i == 0))
                    env.sim.step()
                    eef_traj.append(eef_pos(env).copy())
                eef_after = eef_pos(env)
                delta_p_real = eef_after - eef_before
                gamma_real = float(n_t @ delta_p_real)
                delta_p_perp = delta_p_real - gamma_real * n_t
                perp_norm = float(np.linalg.norm(delta_p_perp))
                d_after = float(np.linalg.norm(body_state(env, "alphabet_soup_1_main")["pos"] - eef_after))
                delta_d = d_after - d_t

                # 验证 d² 公式
                lhs = d_after**2 - d_t**2
                rhs = -2*d_t*gamma_real + np.linalg.norm(delta_p_real)**2
                formula_ok = abs(lhs - rhs) < 1e-9

                # OSC 目标 vs 实际（第一个 substep 的目标误差）
                controller = env.robots[0].controller
                goal_pos = np.asarray(controller.goal_pos)
                tracking_err = float(np.linalg.norm(goal_pos - eef_after))

                print(f"  {gname}: γ_QP={gamma_qp*1000:.2f}mm γ_cmd={gamma_cmd*1000:.2f}mm "
                      f"γ_real={gamma_real*1000:.2f}mm |Δp_⊥|={perp_norm*1000:.2f}mm "
                      f"Δd={delta_d*1000:.2f}mm 公式{'✓' if formula_ok else '✗'}")
                print(f"         OSC目标误差={tracking_err*1000:.2f}mm goal_pos={np.round(goal_pos,3)}")

        # F 轨迹推进
        safe_a, _ = layer.filter(nom_a, obs)
        obs, _r, done, _i = env.step(safe_a)
        if done or env.check_success():
            break

    pol.close()
    env.close()


if __name__ == "__main__":
    main()
