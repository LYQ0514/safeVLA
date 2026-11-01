"""C2-P0/P1：任务进度约束 QP 候选 + 单步物理反事实。

P0：在 AEGIS 原 QP 上增加 γ(u)≥η（η∈{0, 0.5γ_nom, γ_nom}），构造候选。
P1：state02/stream0 step15/20，4 候选 × 2 状态 = 8 次单步物理执行，
    记录真实进度 Δd_soup、物理安全（25 substep 接触）、参考vs物理进度、ΔJ、执行器响应。

状态身份：在正式过滤点截取 QP 数据，副本上解候选，恢复同一 MuJoCo 状态执行每个候选。
"""

from __future__ import annotations

import json
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


def build_geom_sets(env):
    m = env.sim.model._model
    robot_geoms = {g for g in range(m.ngeom) if m.geom_contype[g] > 0 and
                   (env.sim.model.body_names[m.geom_bodyid[g]].startswith("robot0") or
                    env.sim.model.body_names[m.geom_bodyid[g]].startswith("gripper0"))}
    obstacle_names = set(n.replace("_joint0", "") for n in env.sim.model.joint_names if "obstacle" in n)
    obstacle_geoms = {g for g in range(m.ngeom) if m.geom_contype[g] > 0 and
                      any(env.sim.model.body_names[m.geom_bodyid[g]].startswith(o) for o in obstacle_names)}
    return robot_geoms, obstacle_geoms


def solve_candidates(nom_a, R1, a_u_v, a_u_omega, a_uz, h, mu_row, d_soup):
    """构造 4 个候选动作：原始 AEGIS + η∈{0, 0.5γ_nom, γ_nom}。返回 dict。"""
    W = np.diag([1.0/25]*6 + [1.0, 1.0, 1.0])
    u_v_ref = V_REF_GAIN * (R1.T @ nom_a[:3])
    u_omega_ref = OMEGA_REF_GAIN * nom_a[3:6]
    u_z_ref = U_Z_NOM_GAIN * mu_row
    u_ref = np.hstack([u_v_ref, u_omega_ref, u_z_ref])
    gamma_nom = float(d_soup @ (nom_a[:3] * TRANS_SCALE))

    def solve(eta):
        a_var = cp.Variable(6)
        u_z = cp.Variable(3)
        u_v = V_REF_GAIN * (R1.T @ a_var[:3])
        u_omega = OMEGA_REF_GAIN * a_var[3:6]
        u_full = cp.hstack([u_v, u_omega, u_z])
        obj = cp.Minimize(cp.quad_form(u_full - u_ref, W))
        cons = [
            a_u_v @ u_v + a_u_omega @ u_omega + a_uz @ u_z + QP_H_GAIN * h >= 0,
            a_var >= -1, a_var <= 1,
        ]
        if eta is not None:
            cons.append(d_soup @ (a_var[:3] * TRANS_SCALE) >= eta)
        prob = cp.Problem(obj, cons)
        try:
            prob.solve(solver=cp.OSQP)
            if a_var.value is None:
                return None
            a = np.asarray(a_var.value).flatten()
            u_full_val = np.hstack([V_REF_GAIN*(R1.T@a[:3]), OMEGA_REF_GAIN*a[3:6],
                                    np.asarray(u_z.value).flatten()])
            J = float((u_full_val - u_ref) @ W @ (u_full_val - u_ref))
            gamma = float(d_soup @ (a[:3] * TRANS_SCALE))
            return {"a": a, "J": J, "gamma": gamma}
        except Exception:
            return None

    cands = {}
    cands["AEGIS"] = solve(None)  # 原 QP（无方向约束）
    cands["C0"] = solve(0.0)
    cands["C1"] = solve(0.5 * gamma_nom)
    cands["C2"] = solve(gamma_nom)
    return cands, gamma_nom, u_ref, W


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

    robot_geoms, obstacle_geoms = build_geom_sets(env)
    results = []

    print(f"=== C2-P0/P1：任务进度约束候选 + 单步物理反事实 ===")

    for step in range(550):
        qi = step // REPLAN_STEPS
        if step % REPLAN_STEPS == 0:
            z0, _ = sample_z(nominal_seed(SUITE, TASK_ID, state_id, stream, qi))
            po = R.obs_to_policy_obs(obs, prompt, image_size=IMAGE_SIZE)
            nominal = np.clip(pol.generate(z0, po), -1, 1)
        nom_a = nominal[step % REPLAN_STEPS]

        if step in target_steps:
            # 截取 QP 数据（复刻 filter 约束计算）
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

            cands, gamma_nom, u_ref, W = solve_candidates(
                nom_a, R1, a_u_v, a_u_omega, a_uz, h, mu_row, d_soup)

            print(f"\n--- step {step} (γ_nom={gamma_nom*1000:.2f}mm) ---")
            for cname, c in cands.items():
                if c is None:
                    print(f"  {cname}: 不可行")
                else:
                    print(f"  {cname}: γ={c['gamma']*1000:.2f}mm  J={c['J']:.4f}  a[:3]={np.round(c['a'][:3],2)}")

            # P1：每个候选单步物理执行
            # 保存当前完整状态，逐个候选恢复执行
            snapshot = env.sim.get_state()
            for cname, c in cands.items():
                if c is None:
                    continue
                # 恢复快照
                env.sim.set_state(snapshot)
                env.sim.forward()
                d_before = float(np.linalg.norm(eef_pos(env) - body_state(env, "alphabet_soup_1_main")["pos"]))

                # 执行候选动作（7维，夹爪用名义）
                act = np.zeros(7)
                act[:6] = c["a"]
                act[6] = nom_a[6]
                # 单步执行，记录 25 substep 接触
                contact = False
                for i in range(int(env.env.control_timestep / env.env.model_timestep)):
                    env.sim.forward()
                    env.env._pre_action(act, (i == 0))
                    env.sim.step()
                    dd = env.sim.data._data
                    for cc in range(dd.ncon):
                        g1, g2 = dd.contact[cc].geom1, dd.contact[cc].geom2
                        if (g1 in robot_geoms and g2 in obstacle_geoms) or (g2 in robot_geoms and g1 in obstacle_geoms):
                            contact = True

                d_after = float(np.linalg.norm(eef_pos(env) - body_state(env, "alphabet_soup_1_main")["pos"]))
                delta_d = d_after - d_before  # 负 = 接近

                results.append({
                    "step": step, "candidate": cname,
                    "gamma": c["gamma"], "J": c["J"],
                    "d_before": d_before, "d_after": d_after, "delta_d": delta_d,
                    "contact": contact,
                })
                print(f"    [{cname}] Δd_soup={delta_d*1000:.2f}mm 接触={contact}")

        # 正式 filter 推进（保持闭环）
        safe_a, _ = layer.filter(nom_a, obs)
        obs, _r, done, _i = env.step(safe_a)
        if done or env.check_success():
            break

    pol.close()
    env.close()

    out = pathlib.Path(ROOT, "data", "audit", "g1a_c2_candidates.json")
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str))
    print(f"\n结果: {out}")


if __name__ == "__main__":
    main()
