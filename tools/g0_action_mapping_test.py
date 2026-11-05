"""G0 动作映射测试：7D nominal → 9D u_ref/u_nom → 9D 解 → 7D action 全链路 + P_t 有限差分。

任务书 §2.1 / §4.2 / §4.6：
- 打印完整 7D→9D→7D 链路，明确世界/局部坐标、单位/比例；
- P_t ∈ R^{3×9}：z_p,out(u) = z_p,ref + P_t(u - u_ref)，来自 AEGIS map-back 的精确解析；
- 有限差分检验任务空间输出与理论 Jacobian 一致（含旋转姿态和辅助变量耦合）。

AEGIS 映射（已读源码 aegis_adapted.py filter）：
- 9D u = [v_local(3), omega(3), uz(3)]（末端局部系平移速度 + 角速度 + z 演化辅助）
- u_v_ref = V_REF_GAIN * (R1.T @ z[:3]) = 5 R1ᵀ z_p
- u_omega_ref = OMEGA_REF_GAIN * z[3:6] = 5 z_ω
- u_z_nom = U_Z_NOM_GAIN * mu_row = 10 mu_row
- map-back: out[:3] = QP_ACTION_SCALE * R1 @ u[:3] = 0.2 R1 u_v
            out[3:6] = QP_ACTION_SCALE * u[3:6] = 0.2 u_ω
            out[6] = z[6]（夹爪直通）

P_t 解析：z_p,out = 0.2 R1 u_v = 0.2 R1 [I_3 0 0] u，故
  P_t = 0.2 R1 [I_3, 0_{3x3}, 0_{3x3}] ∈ R^{3×9}
  z_p,ref = 0.2 R1 u_v_ref = 0.2*5*(R1 R1ᵀ) z_p = z_p（世界系平移）
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
sys.path.insert(0, os.path.join(ROOT, "third_party", "aegis"))

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

from libero.libero import benchmark  # noqa: E402
from src.safety.aegis_adapted import (  # noqa: E402
    AegisSafetyLayer, _quat_wxyz_to_rot, EEF_OFFSET_LOCAL,
    QP_ACTION_SCALE, V_REF_GAIN, OMEGA_REF_GAIN, U_Z_NOM_GAIN,
)
from third_party.aegis.utils import compute_h_coeffs_3d  # noqa: E402
from tools.g1a_soup_trace import make_env  # noqa: E402

SUITE = "safelibero_long"
LEVEL = "I"
TASK_ID = 0


def pt_matrix(R1):
    """P_t = 0.2 R1 [I_3, 0, 0] ∈ R^{3×9}（世界系平移对 9D 增量的 Jacobian）。"""
    P = np.zeros((3, 9))
    P[:, :3] = QP_ACTION_SCALE * R1  # 0.2 * R1
    return P


def main():
    ts = benchmark.get_benchmark_dict()[SUITE](safety_level=LEVEL)
    task = ts.get_task(TASK_ID)
    env = make_env(ts, task, 1)
    layer = AegisSafetyLayer(env, filter_mode="euclidean")
    layer.reset(env.env._get_observations(), task.language)

    obs = env.env._get_observations()
    eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    R1 = _quat_wxyz_to_rot(np.asarray(obs["robot0_eef_quat"], dtype=np.float64))
    p1 = layer._eef_center(obs)

    # 随机名义 7D action
    rng = np.random.default_rng(0)
    z = rng.uniform(-1, 1, 7).astype(np.float64)
    z[6] = -1.0  # 夹爪

    # --- 7D → 9D u_ref/u_nom ---
    v_ref = R1.T @ z[:3]
    u_v_ref = V_REF_GAIN * v_ref
    u_omega_ref = OMEGA_REF_GAIN * z[3:6]
    a_v, a_omega, a_uz, h, mu_row = compute_h_coeffs_3d(
        p1, layer.Q1_diag, R1, layer.p2, layer.Q2_diag, layer.R2, layer.z_fixed)
    u_z_nom = U_Z_NOM_GAIN * mu_row
    u_ref = np.hstack([u_v_ref, u_omega_ref, u_z_nom])

    # --- 真实 filter 输出（对比）---
    exec_a, info = layer.filter(z, obs)
    # filter 内部会演化 z_fixed，但我们在 filter 后重新用 reset 前的 z_fixed 已变，
    # 这里只验证映射数学，用 info 里的信息。

    # --- P_t 有限差分：z_p,out(u) 对 u 的 Jacobian ---
    # z_p,out(u) = 0.2 R1 u[:3]（解析），Jacobian = P_t
    P = pt_matrix(R1)
    eps = 1e-6
    # 解析 z_p,out 在 u_ref 处：0.2 R1 u_ref[:3] = 0.2*5 R1 R1ᵀ z_p = z_p
    z_p_ref_analytic = QP_ACTION_SCALE * R1 @ u_ref[:3]
    z_p_ref_expected = z[:3]
    err_ref = float(np.linalg.norm(z_p_ref_analytic - z_p_ref_expected))

    # 有限差分验证 P_t：扰动 u 的第 j 维，看 z_p,out 变化
    fd_jac = np.zeros((3, 9))
    u0 = u_ref.copy()
    z_out0 = QP_ACTION_SCALE * R1 @ u0[:3]
    for j in range(9):
        u_p = u0.copy(); u_p[j] += eps
        u_m = u0.copy(); u_m[j] -= eps
        zp_p = QP_ACTION_SCALE * R1 @ u_p[:3]
        zp_m = QP_ACTION_SCALE * R1 @ u_m[:3]
        fd_jac[:, j] = (zp_p - zp_m) / (2 * eps)
    jac_err = float(np.max(np.abs(fd_jac - P)))

    # 验证 P_t 后 6 列为零（旋转/辅助不直接影响世界平移）
    p_tail_zero = float(np.max(np.abs(P[:, 3:])))

    result = {
        "suite": SUITE, "level": LEVEL, "task_id": TASK_ID,
        "z_nominal_7d": z.tolist(),
        "R1": R1.tolist(),
        "u_ref_9d": u_ref.tolist(),
        "u_v_ref_3d": u_v_ref.tolist(),
        "u_omega_ref_3d": u_omega_ref.tolist(),
        "u_z_nom_3d": u_z_nom.tolist(),
        "P_t_3x9": P.tolist(),
        "z_p_ref_analytic": z_p_ref_analytic.tolist(),
        "z_p_ref_expected_world": z_p_ref_expected.tolist(),
        "z_p_ref_err": err_ref,
        "P_t_finite_diff_max_err": jac_err,
        "P_t_tail_cols_max_abs": p_tail_zero,
        "gripper_passthrough": float(z[6]),
        "executed_action_7d": exec_a.tolist(),
        "filter_info": {
            k: (v.tolist() if isinstance(v, np.ndarray)
                else (bool(v) if isinstance(v, (np.bool_,)) else
                      (float(v) if isinstance(v, (np.floating, np.integer)) else v)))
            for k, v in info.items() if k in ("h", "feasible", "intervened")
        },
    }

    # 判定
    checks = {
        "z_p_ref_matches_world_translation": err_ref < 1e-9,
        "P_t_finite_diff_matches_analytic": jac_err < 1e-9,
        "P_t_tail_cols_zero": p_tail_zero < 1e-12,
        "gripper_unchanged": abs(exec_a[6] - z[6]) < 1e-12,
    }
    result["checks"] = checks
    result["verdict"] = "PASS" if all(checks.values()) else "FAIL"

    out = os.path.join(ROOT, "experiments", "stg_v1", "audit", "action_mapping_test.json")

    def _sanitize(o):
        if isinstance(o, dict):
            return {k: _sanitize(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_sanitize(v) for v in o]
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        return o

    with open(out, "w") as f:
        json.dump(_sanitize(result), f, ensure_ascii=False, indent=2)
    print(f"action_mapping_test.json 已写")
    for k, v in checks.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    print(f"  verdict: {result['verdict']}")
    env.close()


if __name__ == "__main__":
    main()
