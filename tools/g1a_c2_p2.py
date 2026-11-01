"""C2-P2：进度约束 AEGIS 的前 40 步闭环实验。

四分支 V/F/P/P-F，固定 state02/stream0，同 CRN。

P/P-F 前 40 步：用「进度约束 QP」（γ(u)≥γ_nom，仅名义正向分量时激活），
约束不可行回退原 AEGIS。后段按分支定义。

Oracle Mechanism Test：用已验证的 soup 物理坐标构造目标方向，
不声称无需特权信息的可部署算法。
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
from src.eval_safelibero import active_obstacle, COLLISION_DISPLACEMENT_EPS  # noqa: E402
from tools.g1a_soup_trace import make_env, body_state, eef_pos, subgoal_values  # noqa: E402

SUITE = "safelibero_long"
LEVEL = "I"
TASK_ID = 0
REPLAN_STEPS = 5
NUM_STEPS_WAIT = 10
MAX_STEPS = 550
IMAGE_SIZE = 224
SWITCH_STEP = 40
TRANS_SCALE = 0.05


def progress_filter(layer, nom_a, obs, soup_pos):
    """进度约束 QP：γ(u)≥γ_nom。返回 (动作, info, 是否回退)。"""
    p1 = layer._eef_center(obs)
    R1 = _quat_wxyz_to_rot(np.asarray(obs["robot0_eef_quat"], dtype=np.float64))
    from utils import compute_h_coeffs_3d
    a_v, a_omega, a_uz, h, mu_row = compute_h_coeffs_3d(
        p1, layer.Q1_diag, R1, layer.p2, layer.Q2_diag, layer.R2, layer.z_fixed)
    a_u_v = QP_ACTION_SCALE * a_v
    a_u_omega = QP_ACTION_SCALE * a_omega

    eef = eef_pos(layer.env)
    d_soup = soup_pos - eef
    d_soup = d_soup / (np.linalg.norm(d_soup) + 1e-9)
    gamma_nom = float(d_soup @ (nom_a[:3] * TRANS_SCALE))

    W = np.diag([1.0/25]*6 + [1.0, 1.0, 1.0])
    u_v_ref = V_REF_GAIN * (R1.T @ nom_a[:3])
    u_omega_ref = OMEGA_REF_GAIN * nom_a[3:6]
    u_z_ref = U_Z_NOM_GAIN * mu_row
    u_ref = np.hstack([u_v_ref, u_omega_ref, u_z_ref])

    def solve(eta):
        a_var = cp.Variable(6)
        u_z = cp.Variable(3)
        u_v = V_REF_GAIN * (R1.T @ a_var[:3])
        u_omega = OMEGA_REF_GAIN * a_var[3:6]
        u_full = cp.hstack([u_v, u_omega, u_z])
        obj = cp.Minimize(cp.quad_form(u_full - u_ref, W))
        cons = [a_u_v @ u_v + a_u_omega @ u_omega + a_uz @ u_z + QP_H_GAIN * h >= 0,
                a_var >= -1, a_var <= 1]
        if eta is not None:
            cons.append(d_soup @ (a_var[:3] * TRANS_SCALE) >= eta)
        prob = cp.Problem(obj, cons)
        try:
            prob.solve(solver=cp.OSQP)
            if a_var.value is None:
                return None
            return np.asarray(a_var.value).flatten()
        except Exception:
            return None

    # 只在名义有正向分量时激活进度约束
    if gamma_nom > 0:
        a_prog = solve(gamma_nom)
        if a_prog is not None:
            act = np.zeros(7)
            act[:6] = a_prog
            act[6] = nom_a[6]
            return act, False  # 成功应用进度约束
    # 回退原 AEGIS
    act, info = layer.filter(nom_a, obs)
    return act, True  # 回退


def run_branch(policy, env, layer, state_id, stream, prompt, mode):
    """mode: V(全程π0.5) F(全程AEGIS) P(前40进度约束,后π0.5) PF(前40进度约束,后AEGIS)"""
    obs = env.env._get_observations()
    name = active_obstacle(env, obs)
    init_pos = body_state(env, name + "_main")["pos"].copy()
    if layer is not None:
        layer.reset(obs, prompt)

    success = False
    n_intervene = 0
    n_fallback = 0
    max_disp = 0.0
    steps = 0
    trace = []

    for step in range(MAX_STEPS):
        use_aegis = mode in ("F", "PF") or (mode == "P" and step < SWITCH_STEP) or \
                    (mode == "PF" and step < SWITCH_STEP) or (mode == "F")
        use_progress = (mode in ("P", "PF")) and step < SWITCH_STEP

        qi = step // REPLAN_STEPS
        if step % REPLAN_STEPS == 0:
            z0, _ = sample_z(nominal_seed(SUITE, TASK_ID, state_id, stream, qi))
            po = R.obs_to_policy_obs(obs, prompt, image_size=IMAGE_SIZE)
            nominal = np.clip(policy.generate(z0, po), -1, 1)

        nom_a = nominal[step % REPLAN_STEPS]
        act_a = nom_a
        intervened = False

        if use_progress and layer is not None:
            soup_pos = body_state(env, "alphabet_soup_1_main")["pos"]
            act_a, fallback = progress_filter(layer, nom_a, obs, soup_pos)
            if fallback:
                n_fallback += 1
            else:
                intervened = True
                n_intervene += 1
        elif use_aegis and layer is not None:
            act_a, info = layer.filter(nom_a, obs)
            intervened = bool(info.get("intervened"))
            if intervened:
                n_intervene += 1

        soup_pos = body_state(env, "alphabet_soup_1_main")["pos"]
        eef = eef_pos(env)
        d_eef_soup = float(np.linalg.norm(eef - soup_pos))

        obs, _r, done, _i = env.step(act_a)
        steps += 1
        disp = float(np.sum(np.abs(body_state(env, name + "_main")["pos"] - init_pos)))
        max_disp = max(max_disp, disp)

        if step % 10 == 0 or step == SWITCH_STEP:
            sg = subgoal_values(env)
            trace.append({"step": step, "d_eef_soup": d_eef_soup, "disp": disp,
                          "soup_pred": bool(sg.get("alphabet_soup_1")),
                          "cream_pred": bool(sg.get("cream_cheese_1"))})

        if done or env.check_success():
            success = True
            break

    return {
        "success": success, "collided": max_disp > COLLISION_DISPLACEMENT_EPS,
        "max_displacement": max_disp, "steps": steps,
        "n_intervene": n_intervene, "n_fallback": n_fallback,
        "final_subgoals": subgoal_values(env), "trace": trace,
    }


def main():
    from libero.libero import benchmark
    ts = benchmark.get_benchmark_dict()[SUITE](safety_level=LEVEL)
    task = ts.get_task(TASK_ID)
    prompt = task.language
    state_id, stream = 2, 0

    policy = PnsgPolicy()
    branches = {"V": "V", "F": "F", "P": "P", "PF": "PF"}
    results = {}
    try:
        for bname, mode in branches.items():
            env = make_env(ts, task, state_id)
            layer = AegisSafetyLayer(env) if mode != "V" else None
            r = run_branch(policy, env, layer, state_id, stream, prompt, mode)
            env.close()
            results[bname] = r
            print(f"  {bname}: success={r['success']} coll={r['collided']} "
                  f"Dmax={r['max_displacement']*1000:.2f}mm steps={r['steps']} "
                  f"int={r['n_intervene']} fallback={r['n_fallback']} "
                  f"soup={r['final_subgoals'].get('alphabet_soup_1')} "
                  f"cream={r['final_subgoals'].get('cream_cheese_1')}", flush=True)
    finally:
        policy.close()

    out = pathlib.Path(ROOT, "data", "audit", "g1a_c2_p2.json")
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str))
    print(f"\n结果: {out}")

    # 关键观察
    for bname in ["P", "PF", "F"]:
        tr = results[bname]["trace"]
        d40 = next((t["d_eef_soup"] for t in tr if t["step"] == SWITCH_STEP), None)
        print(f"{bname} step40 d_eef_soup={d40:.3f}" if d40 else f"{bname} 无step40记录")


if __name__ == "__main__":
    main()
