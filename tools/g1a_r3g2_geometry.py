"""R3-G2：几何反事实审计（固定状态）。

对 step0/15/20/40/55，比较：
- G-L：原始 Long 大椭球 [0.06,0.12,0.20]
- G-S：小椭球 [0.06,0.12,0.11]
- G-O：审计大椭球长轴方向（是否沿接近 soup 方向）

输出每个状态：h、γ、约束可行性，以及椭球长轴方向 vs soup 方向的角度。
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
import pi05_libero_runner as R  # noqa: E402

from src.pnsg.crn import nominal_seed, sample_z  # noqa: E402
from src.pnsg.policy_adapter import PnsgPolicy  # noqa: E402
from src.safety.aegis_adapted import (  # noqa: E402
    AegisSafetyLayer, _quat_wxyz_to_rot, EEF_OFFSET_LOCAL,
    QP_ACTION_SCALE, V_REF_GAIN, OMEGA_REF_GAIN, QP_H_GAIN,
)
from tools.g1a_soup_trace import make_env, body_state, eef_pos  # noqa: E402

SUITE = "safelibero_long"
LEVEL = "I"
TASK_ID = 0
REPLAN_STEPS = 5
IMAGE_SIZE = 224
TRANS_SCALE = 0.05

EEF_Q_LONG = np.array([0.06, 0.12, 0.20])
EEF_Q_SMALL = np.array([0.06, 0.12, 0.11])


def compute_h_for(p1, R1, Q1, p2, R2, Q2, z_fixed):
    from utils import compute_h_coeffs_3d
    a_v, a_omega, a_uz, h, mu = compute_h_coeffs_3d(p1, Q1, R1, p2, Q2, R2, z_fixed)
    return h


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

    target_steps = [0, 15, 20, 40, 55]
    print(f"=== R3-G2：几何反事实（step {target_steps}）===")

    for step in range(MAX_STEPS := 550):
        qi = step // REPLAN_STEPS
        if step % REPLAN_STEPS == 0:
            z0, _ = sample_z(nominal_seed(SUITE, TASK_ID, state_id, stream, qi))
            po = R.obs_to_policy_obs(obs, prompt, image_size=IMAGE_SIZE)
            nominal = np.clip(pol.generate(z0, po), -1, 1)
        nom_a = nominal[step % REPLAN_STEPS]

        if step in target_steps:
            p1 = layer._eef_center(obs)
            R1 = _quat_wxyz_to_rot(np.asarray(obs["robot0_eef_quat"], dtype=np.float64))
            soup = body_state(env, "alphabet_soup_1_main")["pos"]
            eef = eef_pos(env)
            d_vec = soup - eef
            d_soup = d_vec / (np.linalg.norm(d_vec) + 1e-9)

            # 椭球长轴方向（世界坐标）：Q1 的最大半轴对应方向 = R1 的第 k 列
            k = int(np.argmax(layer.Q1_diag))  # 最大半轴的局部轴
            long_axis = R1[:, k]  # 世界坐标方向
            # 长轴方向 vs soup 方向夹角
            cos_angle = abs(float(long_axis @ d_soup))
            angle_deg = float(np.degrees(np.arccos(np.clip(cos_angle, -1, 1))))

            # 三种几何配置的 h
            h_long = compute_h_for(p1, R1, EEF_Q_LONG, layer.p2, layer.R2, layer.Q2_diag, layer.z_fixed)
            h_small = compute_h_for(p1, R1, EEF_Q_SMALL, layer.p2, layer.R2, layer.Q2_diag, layer.z_fixed)

            gamma_nom = float(d_soup @ (nom_a[:3] * TRANS_SCALE))

            print(f"\nstep {step}:")
            print(f"  Q1长轴方向(局部轴{k}) vs soup方向 夹角={angle_deg:.1f}°")
            print(f"  h(大椭球)={h_long:.4f}  h(小椭球)={h_small:.4f}")
            print(f"  γ_nom={gamma_nom*1000:.2f}mm")

        # 执行（V 用 nominal）
        obs, _r, done, _i = env.step(nom_a)
        if done or env.check_success():
            break

    pol.close()
    env.close()


if __name__ == "__main__":
    main()
