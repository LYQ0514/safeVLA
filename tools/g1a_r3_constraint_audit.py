"""R3 第②部分：原始成功路径（V）的 AEGIS 代理约束离线审计。

沿 V 的真实轨迹（π0.5 前 40 步），在每一步物理状态上：
- 计算 AEGIS CBF 约束系数（h, a_u_v, a_u_omega, a_uz）
- 检查 V 的真实执行动作是否满足该约束（h >= 0 或 CBF 不等式成立）
- 重建 shadow 状态（z_fixed 演化），不改变 V 的执行

回答：V 的关键接近动作是否满足 AEGIS 代理约束？
- 满足 → 存在既代理可行又快速接近的动作，重点在动作选择
- 违反 → 代理几何与真实物理安全契约存在保守性差距
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
from src.eval_safelibero import active_obstacle  # noqa: E402
from tools.g1a_soup_trace import make_env, body_state, eef_pos  # noqa: E402

SUITE = "safelibero_long"
LEVEL = "I"
TASK_ID = 0
REPLAN_STEPS = 5
NUM_STEPS_WAIT = 10
MAX_STEPS = 550
IMAGE_SIZE = 224
TRANS_SCALE = 0.05


def main():
    from libero.libero import benchmark
    ts = benchmark.get_benchmark_dict()[SUITE](safety_level=LEVEL)
    task = ts.get_task(TASK_ID)
    prompt = task.language
    state_id, stream = 2, 0

    env = make_env(ts, task, state_id)
    # shadow AEGIS layer（用于约束计算，不影响执行）
    shadow = AegisSafetyLayer(env)
    obs = env.env._get_observations()
    shadow.reset(obs, prompt)

    pol = PnsgPolicy()

    print(f"=== R3-②：V 成功路径的 AEGIS 代理约束审计（前 60 步）===")

    audit = []
    for step in range(MAX_STEPS):
        qi = step // REPLAN_STEPS
        if step % REPLAN_STEPS == 0:
            z0, _ = sample_z(nominal_seed(SUITE, TASK_ID, state_id, stream, qi))
            po = R.obs_to_policy_obs(obs, prompt, image_size=IMAGE_SIZE)
            nominal = np.clip(pol.generate(z0, po), -1, 1)

        nom_a = nominal[step % REPLAN_STEPS]

        # 计算当前状态下的 AEGIS 约束（不调用有副作用的 filter，只算系数）
        if step < 60:
            p1 = shadow._eef_center(obs)
            R1 = _quat_wxyz_to_rot(np.asarray(obs["robot0_eef_quat"], dtype=np.float64))
            from utils import compute_h_coeffs_3d
            a_v, a_omega, a_uz, h, mu_row = compute_h_coeffs_3d(
                p1, shadow.Q1_diag, R1, shadow.p2, shadow.Q2_diag, shadow.R2, shadow.z_fixed)
            a_u_v = QP_ACTION_SCALE * a_v
            a_u_omega = QP_ACTION_SCALE * a_omega
            # V 的真实动作（nominal）映射到 u 空间
            v_ref = R1.T @ nom_a[:3]
            u_v = V_REF_GAIN * v_ref
            u_omega = OMEGA_REF_GAIN * nom_a[3:6]
            u_z_nom = U_Z_NOM_GAIN if False else 0.0  # 占位
            # CBF 约束值（对 nominal 动作）
            from src.safety.aegis_adapted import U_Z_NOM_GAIN as UZN
            u_z_nom = UZN * mu_row
            cbf_value = float(a_u_v @ u_v + a_u_omega @ u_omega + a_uz @ u_z_nom + QP_H_GAIN * h)
            audit.append({"step": step, "h": float(h), "cbf_value": cbf_value,
                          "cbf_satisfied": cbf_value >= 0})

        # V 执行 nominal（无 AEGIS）
        obs, _r, done, _i = env.step(nom_a)

        # shadow z_fixed 演化（模拟 AEGIS 会如何更新）
        # 简化：这里不演化 shadow 状态，因为 V 不调 filter，z_fixed 应保持 reset 后的初始值
        if done or env.check_success():
            break

    pol.close()
    env.close()

    # 统计
    h_neg = [a for a in audit if a["h"] < 0]
    cbf_viol = [a for a in audit if not a["cbf_satisfied"]]
    print(f"\n前 60 步审计结果:")
    print(f"  h < 0 的步数（椭球相交）: {len(h_neg)}/60")
    print(f"  CBF 约束不满足的步数: {len(cbf_viol)}/60")
    print(f"\n  h 演化（每5步）:")
    for a in audit[::5]:
        print(f"    step{a['step']:>3}: h={a['h']:.4f} cbf={a['cbf_satisfied']}")


if __name__ == "__main__":
    main()
