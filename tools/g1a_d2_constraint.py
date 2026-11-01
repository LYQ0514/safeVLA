"""D2：AEGIS 活跃约束审计（state02 step15/20）—— 量纲修正版。

统一在「动作空间」计算：
- 物理位移 = 动作平移分量 a[:3] × 0.05 m
- γ = d_soup · (a[:3] × 0.05)，单位 m
- AEGIS 约束在 u 空间：a_u_v·u_v + a_u_omega·u_omega + a_uz·u_z + 10h >= 0
  u_v = 5·R1ᵀ·a_v, u_omega = 5·a_omega, u_z 自由

γ_max_safe：在动作空间 a 里最大化 d_soup·(a[:3]×0.05)，
s.t. u 空间 CBF 约束 + a 在动作盒 [-1,1] 内 + u_z 自由。
"""

from __future__ import annotations

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
TRANS_SCALE = 0.05  # 动作 [-1,1] -> 物理位移 [-0.05,0.05]m


def main():
    from libero.libero import benchmark
    ts = benchmark.get_benchmark_dict()[SUITE](safety_level=LEVEL)
    task = ts.get_task(TASK_ID)
    prompt = task.language

    state_id, stream = 2, 0
    target_steps = [15, 20]

    env = make_env(ts, task, state_id)
    layer = AegisSafetyLayer(env)
    obs = env.env._get_observations()
    layer.reset(obs, prompt)
    pol = PnsgPolicy()

    print(f"=== D2：state02/stream0 活跃约束审计（量纲修正）===")

    for step in range(550):
        qi = step // REPLAN_STEPS
        if step % REPLAN_STEPS == 0:
            z0, _ = sample_z(nominal_seed(SUITE, TASK_ID, state_id, stream, qi))
            po = R.obs_to_policy_obs(obs, prompt, image_size=IMAGE_SIZE)
            nominal = np.clip(pol.generate(z0, po), -1, 1)

        nom_a = nominal[step % REPLAN_STEPS]

        if step in target_steps:
            p1 = layer._eef_center(obs)
            R1 = _quat_wxyz_to_rot(np.asarray(obs["robot0_eef_quat"], dtype=np.float64))

            from utils import compute_h_coeffs_3d
            a_v, a_omega, a_uz, h, mu_row = compute_h_coeffs_3d(
                p1, layer.Q1_diag, R1, layer.p2, layer.Q2_diag, layer.R2, layer.z_fixed)
            a_u_v = QP_ACTION_SCALE * a_v
            a_u_omega = QP_ACTION_SCALE * a_omega

            soup = body_state(env, "alphabet_soup_1_main")["pos"]
            eef = eef_pos(env)
            d_soup = soup - eef
            d_soup = d_soup / (np.linalg.norm(d_soup) + 1e-9)

            # γ（动作空间，物理量纲）
            def gamma_of_action(a):
                return float(d_soup @ (np.asarray(a)[:3] * TRANS_SCALE))

            # 名义 γ 和 实际过滤 γ（通过正式 filter 得到，但这里先算名义，实际在下方 filter）
            gamma_nom = gamma_of_action(nom_a)

            # γ_max_safe：动作空间优化
            # 优化变量：a (6 维动作) + u_z (3 维自由)
            a_var = cp.Variable(6)
            u_z = cp.Variable(3)
            # 映射到 u 空间
            u_v = V_REF_GAIN * (R1.T @ a_var[:3])
            u_omega = OMEGA_REF_GAIN * a_var[3:6]
            # CBF 约束
            cons = [a_u_v @ u_v + a_u_omega @ u_omega + a_uz @ u_z + QP_H_GAIN * h >= 0,
                    a_var >= -1, a_var <= 1]
            obj = cp.Maximize(d_soup @ (a_var[:3] * TRANS_SCALE))
            prob = cp.Problem(obj, cons)
            try:
                prob.solve(solver=cp.OSQP)
                gamma_max = prob.value if prob.status in ("optimal", "optimal_inaccurate") else None
                a_max = np.asarray(a_var.value).flatten() if a_var.value is not None else None
            except Exception:
                gamma_max = None
                a_max = None

            print(f"\n--- step {step} ---")
            print(f"  h={h:.4f}  a_uz范数={np.linalg.norm(a_uz):.4f}")
            print(f"  γ_nom={gamma_nom*1000:.2f}mm")
            print(f"  γ_max_safe={gamma_max*1000:.2f}mm" if gamma_max is not None else "  γ_max: 无解")

            # ΔJ_η：最小代价正向接近
            # J_AEGIS(u) = (u-u_ref)^T W (u-u_ref)，u 空间
            # u_ref 由名义动作决定：u_v_ref=5*R1ᵀ*a_v, u_omega_ref=5*a_omega, u_z_ref=10*mu
            W = np.diag([1.0/25]*6 + [1.0, 1.0, 1.0])
            u_v_ref = V_REF_GAIN * (R1.T @ nom_a[:3])
            u_omega_ref = OMEGA_REF_GAIN * nom_a[3:6]
            u_z_ref = U_Z_NOM_GAIN * mu_row
            u_ref = np.hstack([u_v_ref, u_omega_ref, u_z_ref])

            # 原 QP 最优（无方向约束）在 u 空间的 J
            u_orig = cp.Variable(9)
            obj_orig = cp.Minimize(cp.quad_form(u_orig - u_ref, W))
            cons_orig = [a_u_v @ u_orig[:3] + a_u_omega @ u_orig[3:6] + a_uz @ u_orig[6:] + QP_H_GAIN * h >= 0]
            cp.Problem(obj_orig, cons_orig).solve(solver=cp.OSQP)
            J_orig = float(obj_orig.value) if u_orig.value is not None else None

            # 对 η ∈ {0, γ_nom}：min J s.t. γ(a) >= η
            for eta_name, eta_val in [("0", 0.0), ("γ_nom", gamma_nom)]:
                a_eta = cp.Variable(6)
                u_z_eta = cp.Variable(3)
                u_v_eta = V_REF_GAIN * (R1.T @ a_eta[:3])
                u_omega_eta = OMEGA_REF_GAIN * a_eta[3:6]
                u_eta_full = cp.hstack([u_v_eta, u_omega_eta, u_z_eta])
                obj_eta = cp.Minimize(cp.quad_form(u_eta_full - u_ref, W))
                cons_eta = [
                    a_u_v @ u_v_eta + a_u_omega @ u_omega_eta + a_uz @ u_z_eta + QP_H_GAIN * h >= 0,
                    a_eta >= -1, a_eta <= 1,
                    d_soup @ (a_eta[:3] * TRANS_SCALE) >= eta_val,
                ]
                try:
                    cp.Problem(obj_eta, cons_eta).solve(solver=cp.OSQP)
                    if a_eta.value is not None:
                        J_eta = float(obj_eta.value)
                        dJ = J_eta - J_orig if J_orig is not None else None
                        print(f"  ΔJ(γ≥{eta_name}) = {dJ:.4f}" if dJ is not None else f"  η={eta_name} 无解")
                    else:
                        print(f"  η={eta_name}: 不可行（约束不允许保留该接近分量）")
                except Exception:
                    print(f"  η={eta_name}: 求解失败")


        # 正式 filter（唯一一次调用）
        safe_a, info = layer.filter(nom_a, obs)
        if step in target_steps:
            gamma_aegis = gamma_of_action(safe_a)
            print(f"  γ_AEGIS(实际过滤)={gamma_aegis*1000:.2f}mm")
            if gamma_max is not None:
                if gamma_max > 0 and gamma_aegis < 0:
                    print(f"  判决: A：可接近(γ_max={gamma_max*1000:.1f}mm>0) 但实际选择了远离(γ_AEGIS={gamma_aegis*1000:.1f}mm)")
                elif gamma_max <= 0:
                    print(f"  判决: B：代理约束禁止接近(γ_max={gamma_max*1000:.1f}mm<=0)")
                else:
                    print(f"  判决: C：可接近但需付出代价")
            if a_max is not None:
                print(f"  最优接近动作 a_max[:3]={np.round(a_max[:3],3)} vs 名义 a[:3]={np.round(nom_a[:3],3)}")

        obs, _r, done, _i = env.step(safe_a)
        if done or env.check_success():
            break

    pol.close()
    env.close()


if __name__ == "__main__":
    main()
