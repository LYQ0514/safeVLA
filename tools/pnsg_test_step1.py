"""PNSG 第一步验证：核心 proposal 的端到端闭环。

验证：J, C, b, A=CJ, Δz*, r_crit, χ_lin，并用真实状态检查
「源空间最小修正确实改善净空」。

对应 docs/设计.md 第 19 节「第一步」。
"""

from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "dataset", "SafeLIBERO", "safelibero"))
sys.path.insert(0, "/data/2608.SDL/MojocoProj/src/pi05")

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
import pathlib  # noqa: E402

import pi05_libero_runner as R  # noqa: E402


def main():
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    from src.pnsg.geometry_proxy import (
        build_control_points,
        build_obstacles,
        selected_constraints,
        compute_clearance,
    )
    from src.pnsg.policy_adapter import PnsgPolicy
    from src.pnsg.source_solver import minimum_norm_halfspaces, farkas_certificate, compute_chi_lin

    # 1. 环境
    ts = benchmark.get_benchmark_dict()["safelibero_spatial"](safety_level="I")
    task = ts.get_task(0)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256,
                             controller="OSC_POSE")
    env.seed(7)
    obs = env.reset()
    obs = env.set_init_state(ts.get_task_init_states(0)[0])
    dummy = np.array([0.0] * 6 + [-1.0])
    for _ in range(10):
        obs, _r, _d, _i = env.step(dummy)

    prompt = task.language
    policy_obs = R.obs_to_policy_obs(obs, prompt, image_size=224)
    print(f"任务: {prompt}")

    # 2. 源空间策略 + Jacobian
    print("[1] 加载 PNSG policy (openpi worker)…")
    policy = PnsgPolicy()
    z0 = policy.sample_noise(seed=42)
    raw, raw_J, ext_actions, ext_J = policy.generate_with_jacobian(z0, policy_obs)
    print(f"    external action shape: {ext_actions.shape} (应 10×7)")
    print(f"    external J shape: {ext_J.shape} (应 60×320)")

    # 3. 几何代理
    cps = build_control_points(env, obs)
    obstacles = build_obstacles(env, obs)
    print(f"[2] 控制点 {len(cps)} 个，障碍物球 {len(obstacles)} 个")
    for ob in obstacles:
        print(f"    障碍球: center={np.round(ob.center,3)} radius={ob.radius:.3f}")

    # 4. 名义动作的净空
    nominal_pose = ext_actions[:, :6]  # (10,6)
    nominal_pose = np.clip(nominal_pose, -1, 1)
    clear_nominal = compute_clearance(cps, obstacles, nominal_pose)
    min_clear = float(clear_nominal.min())
    print(f"[3] 名义动作最小净空: {min_clear*1000:.2f} mm (目标 5mm)")

    # 5. 线性约束
    rows = selected_constraints(cps, obstacles, nominal_pose, h_floor=0.005,
                                max_constraints=3, margin=0.01)
    print(f"[4] 选中约束 {len(rows)} 条")
    C = np.vstack([c for c, _ in rows]) if rows else np.zeros((0, 60))
    b = np.array([bb for _, bb in rows])
    A = C @ ext_J  # (m, 320)
    print(f"    A shape: {A.shape}, b: {np.round(b*1000,2)} mm")

    # 6. 最小源空间修正
    dz, rcrit = minimum_norm_halfspaces(A, b)
    chi = compute_chi_lin(dz, r_budget=12.0)
    print(f"[5] Δz 范数 (r_crit) = {rcrit:.4f}, χ_lin = {chi:.4f}")

    if dz is None:
        fk = farkas_certificate(A, b)
        print(f"    线性不可行！Farkas witness: {fk}")
        policy.close()
        env.close()
        return

    # 7. 候选重新前向，验证净空改善
    print(f"[6] 候选验证（α ∈ 0.25/0.5/0.75/1.0）…")
    dz_2d = dz.reshape(policy.H, policy.D)  # (320,) -> (10,32)
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        z_cand = z0 + alpha * dz_2d
        cand_actions = policy.generate(z_cand, policy_obs)  # (10,7)
        cand_pose = np.clip(cand_actions[:, :6], -1, 1)
        clear_cand = compute_clearance(cps, obstacles, cand_pose)
        mc = float(clear_cand.min())
        # 源代价
        c_src = 0.5 * float(np.linalg.norm(alpha * dz) ** 2)
        print(f"    α={alpha:.2f}: 最小净空 {mc*1000:6.2f} mm, C_src={c_src:.4f}")

    policy.close()
    env.close()
    print("\n✅ 第一步闭环跑通：源空间 Δz 应随 α 增大而改善最小净空")


if __name__ == "__main__":
    main()
