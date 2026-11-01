"""R3-G4-③：40 步闭环控制执行失配审计。

比较 V（π0.5）/ F-L（大椭球AEGIS）/ F-S（小椭球AEGIS）前 40 步：
- G_cmd = Σ γ_cmd（累计参考接近）
- G_real = Σ γ_rel（累计真实相对接近，含 soup 运动）
- E_exec = G_cmd - G_real（累计执行差）
分四段 0-9/10-19/20-29/30-39。

判断 M2 是偶发短时还是持续闭环现象。
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
from tools.g1a_soup_trace import make_env, body_state, eef_pos  # noqa: E402

SUITE = "safelibero_long"
LEVEL = "I"
TASK_ID = 0
REPLAN_STEPS = 5
IMAGE_SIZE = 224
TRANS_SCALE = 0.05
N_STEPS = 40

EEF_Q_LONG = np.array([0.06, 0.12, 0.20])
EEF_Q_SMALL = np.array([0.06, 0.12, 0.11])


def progress_filter(nom_a, layer, R1, Q1_diag):
    """用指定 Q1 解 QP，返回 (out动作, h)。复刻 AEGIS filter 只改 Q1。"""
    from utils import compute_h_coeffs_3d
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


def run_40(mode):
    """mode: V / F-L / F-S。返回前40步的 γ 统计 + 距离。"""
    from libero.libero import benchmark
    ts = benchmark.get_benchmark_dict()[SUITE](safety_level=LEVEL)
    task = ts.get_task(TASK_ID)
    prompt = task.language
    state_id, stream = 2, 0

    env = make_env(ts, task, state_id)
    layer = AegisSafetyLayer(env) if mode != "V" else None
    obs = env.env._get_observations()
    if layer is not None:
        layer.reset(obs, prompt)
    name = active_obstacle(env, obs)
    init_pos = body_state(env, name + "_main")["pos"].copy()

    pol = PnsgPolicy()
    steps_data = []

    for step in range(N_STEPS):
        qi = step // REPLAN_STEPS
        if step % REPLAN_STEPS == 0:
            z0, _ = sample_z(nominal_seed(SUITE, TASK_ID, state_id, stream, qi))
            po = R.obs_to_policy_obs(obs, prompt, image_size=IMAGE_SIZE)
            nominal = np.clip(pol.generate(z0, po), -1, 1)
        nom_a = nominal[step % REPLAN_STEPS]

        # 记录 soup/eef 位置（执行前）
        soup_before = body_state(env, "alphabet_soup_1_main")["pos"].copy()
        eef_before = eef_pos(env).copy()
        n_t = soup_before - eef_before
        d_t = np.linalg.norm(n_t)
        n_t = n_t / (d_t + 1e-9)

        # 选择动作
        act_a = nom_a
        if mode == "V":
            pass
        elif mode == "F-L":
            act_a, _ = layer.filter(nom_a, obs)
        elif mode == "F-S":
            R1 = _quat_wxyz_to_rot(np.asarray(obs["robot0_eef_quat"], dtype=np.float64))
            act_a, h = progress_filter(nom_a, layer, R1, EEF_Q_SMALL)
            if act_a is None:
                act_a, _ = layer.filter(nom_a, obs)  # 回退

        # γ_cmd = 指令物理位移投影；γ_nom = 名义动作接近分量
        gamma_nom = float(n_t @ (nom_a[:3] * TRANS_SCALE))
        gamma_cmd = float(n_t @ (act_a[:3] * TRANS_SCALE))

        obs, _r, done, _i = env.step(act_a)

        soup_after = body_state(env, "alphabet_soup_1_main")["pos"].copy()
        eef_after = eef_pos(env).copy()
        gamma_rel = float(n_t @ ((eef_after - eef_before) - (soup_after - soup_before)))

        steps_data.append({"step": step, "gamma_nom": gamma_nom,
                           "gamma_cmd": gamma_cmd, "gamma_rel": gamma_rel})

        if done or env.check_success():
            break

    d_final = float(np.linalg.norm(body_state(env, "alphabet_soup_1_main")["pos"] - eef_pos(env)))
    disp = float(np.sum(np.abs(body_state(env, name + "_main")["pos"] - init_pos)))
    pol.close()
    env.close()

    return steps_data, d_final, disp


def main():
    print(f"=== R3-G4-③：40 步闭环执行失配审计 ===")
    results = {}
    for mode in ["V", "F-L", "F-S"]:
        data, d_final, disp = run_40(mode)
        results[mode] = {"data": data, "d_final": d_final, "disp": disp}
        G_cmd = sum(s["gamma_cmd"] for s in data)
        G_real = sum(s["gamma_rel"] for s in data)
        G_nom = sum(s["gamma_nom"] for s in data)
        G_filter = G_cmd - G_nom
        E_exec = G_cmd - G_real
        n_flip = sum(1 for s in data if s["gamma_cmd"] > 0 and s["gamma_rel"] < 0)
        print(f"\n{mode}:")
        print(f"  G_nom={G_nom*1000:.2f}mm G_cmd={G_cmd*1000:.2f}mm G_filter={G_filter*1000:.2f}mm")
        print(f"  G_real={G_real*1000:.2f}mm E_exec={E_exec*1000:.2f}mm")
        print(f"  指令正向实际反向步数: {n_flip}/{len(data)}")
        print(f"  d_final(soup)={d_final:.3f}m  disp(mug)={disp*1000:.2f}mm")
        # 分四段
        for seg, (a, b) in enumerate([(0,10),(10,20),(20,30),(30,40)]):
            seg_nom = sum(s["gamma_nom"] for s in data if a <= s["step"] < b)
            seg_cmd = sum(s["gamma_cmd"] for s in data if a <= s["step"] < b)
            seg_filt = seg_cmd - seg_nom
            print(f"    段{seg}({a}-{b-1}): G_nom={seg_nom*1000:.2f}mm G_cmd={seg_cmd*1000:.2f}mm G_filter={seg_filt*1000:.2f}mm")

    out = pathlib.Path(ROOT, "data", "audit", "g1a_r3g4_40step.json")
    out.write_text(json.dumps({m: {"data": r["data"], "d_final": r["d_final"], "disp": r["disp"]}
                               for m, r in results.items()}, ensure_ascii=False, indent=2, default=str))
    print(f"\n结果: {out}")


if __name__ == "__main__":
    main()
