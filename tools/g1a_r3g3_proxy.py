"""R3-G3：真实碰撞几何驱动的代理约束反事实审计。

三种几何（同 state02 step15/20）：
- G-L: AEGIS Long 大椭球 [0.06,0.12,0.20]
- G-S: 小椭球 [0.06,0.12,0.11]
- G-P: 球代理（诊断）半径 0.14m = gripper末端最大rbound(0.12) + 裕度0.02

输出：h、γ_safe、约束可行性、真实 Δd_soup、接触。

注意：G-P 是诊断几何，用真实 gripper 末端 rbound(0.012~0.119m) 构造，
不声称可部署安全模型；球代理是简化，真正紧致代理需多体包围并集。
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

GEO = {
    "G-L": np.array([0.06, 0.12, 0.20]),
    "G-S": np.array([0.06, 0.12, 0.11]),
    "G-P": np.array([0.14, 0.14, 0.14]),  # 球诊断
}


def filter_with_q(policy_obs_state, nom_a, layer, R1, Q1_diag):
    """用指定 Q1 几何重新解 QP，返回 (动作, h)。复刻 filter 的 QP，只改 Q1。"""
    from utils import compute_h_coeffs_3d
    p1 = layer._eef_center(policy_obs_state["obs"])
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
        act = np.zeros(7)
        act[:3] = QP_ACTION_SCALE * (R1 @ u_opt[:3])
        act[3:6] = QP_ACTION_SCALE * u_opt[3:6]
        act[6] = nom_a[6]
        return act, h
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

    robot_geoms = {g for g in range(env.sim.model._model.ngeom)
                   if env.sim.model._model.geom_contype[g] > 0 and
                   (env.sim.model.body_names[env.sim.model._model.geom_bodyid[g]].startswith("robot0") or
                    env.sim.model.body_names[env.sim.model._model.geom_bodyid[g]].startswith("gripper0"))}
    obstacle_names = set(n.replace("_joint0", "") for n in env.sim.model.joint_names if "obstacle" in n)
    obstacle_geoms = {g for g in range(env.sim.model._model.ngeom)
                      if env.sim.model._model.geom_contype[g] > 0 and
                      any(env.sim.model.body_names[env.sim.model._model.geom_bodyid[g]].startswith(o) for o in obstacle_names)}

    target_steps = [15, 20]
    print(f"=== R3-G3：几何反事实（step {target_steps}）===")

    for step in range(550):
        qi = step // REPLAN_STEPS
        if step % REPLAN_STEPS == 0:
            z0, _ = sample_z(nominal_seed(SUITE, TASK_ID, state_id, stream, qi))
            po = R.obs_to_policy_obs(obs, prompt, image_size=IMAGE_SIZE)
            nominal = np.clip(pol.generate(z0, po), -1, 1)
        nom_a = nominal[step % REPLAN_STEPS]

        if step in target_steps:
            R1 = _quat_wxyz_to_rot(np.asarray(obs["robot0_eef_quat"], dtype=np.float64))
            soup = body_state(env, "alphabet_soup_1_main")["pos"]
            eef = eef_pos(env)
            d_vec = soup - eef
            d_soup = d_vec / (np.linalg.norm(d_vec) + 1e-9)
            gamma_nom = float(d_soup @ (nom_a[:3] * TRANS_SCALE))

            print(f"\n--- step {step} (γ_nom={gamma_nom*1000:.2f}mm) ---")
            results = {}
            for gname, Q1 in GEO.items():
                act, h = filter_with_q({"obs": obs}, nom_a, layer, R1, Q1)
                if act is None:
                    print(f"  {gname}: 不可行")
                    results[gname] = None
                    continue
                gamma_safe = float(d_soup @ (act[:3] * TRANS_SCALE))
                results[gname] = {"act": act, "h": h, "gamma_safe": gamma_safe}
                print(f"  {gname}: h={h:.4f} γ_safe={gamma_safe*1000:.2f}mm")

            # 单步物理执行（每个候选）
            snapshot = env.sim.get_state()
            for gname, r in results.items():
                if r is None:
                    continue
                env.sim.set_state(snapshot)
                env.sim.forward()
                d_before = float(np.linalg.norm(eef_pos(env) - body_state(env, "alphabet_soup_1_main")["pos"]))
                contact = False
                act7 = np.asarray(r["act"])
                for i in range(int(env.env.control_timestep / env.env.model_timestep)):
                    env.sim.forward()
                    env.env._pre_action(act7, (i == 0))
                    env.sim.step()
                    dd = env.sim.data._data
                    for cc in range(dd.ncon):
                        g1, g2 = dd.contact[cc].geom1, dd.contact[cc].geom2
                        if (g1 in robot_geoms and g2 in obstacle_geoms) or (g2 in robot_geoms and g1 in obstacle_geoms):
                            contact = True
                d_after = float(np.linalg.norm(eef_pos(env) - body_state(env, "alphabet_soup_1_main")["pos"]))
                delta_d = d_after - d_before
                print(f"    [{gname}] Δd_soup={delta_d*1000:.2f}mm 接触={contact}")

        # 正式 filter 推进（F 轨迹）
        safe_a, _ = layer.filter(nom_a, obs)
        obs, _r, done, _i = env.step(safe_a)
        if done or env.check_success():
            break

    pol.close()
    env.close()


if __name__ == "__main__":
    main()
