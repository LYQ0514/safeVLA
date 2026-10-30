"""PNSG 第二步验证：源空间 Δz 是否真实改善净空（非线性）。

核心问题：线性约束 A=CJ 预测的 Δz，在真实前向（非线性）下是否确实
让动作避开障碍？

方法：
1. 固定真实状态，求名义动作 + Jacobian
2. 人工放大障碍半径（逼出名义动作违反安全）
3. 几何代理选约束 C, b，A = C J，解 Δz
4. 候选 z0+αΔz 真实前向，验证真实净空是否随 α 改善
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
        build_control_points, build_obstacles, selected_constraints, compute_clearance,
    )
    from src.pnsg.policy_adapter import PnsgPolicy
    from src.pnsg.source_solver import minimum_norm_halfspaces

    # 环境
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

    policy = PnsgPolicy()
    z0 = policy.sample_noise(seed=42)
    _, _, ext_actions, ext_J = policy.generate_with_jacobian(z0, policy_obs)

    cps = build_control_points(env, obs)
    obstacles = build_obstacles(env, obs)
    # 人工放大障碍半径，逼出危险场景
    for ob in obstacles:
        ob.radius *= 1.6
    print(f"障碍球半径放大 1.6x: {[round(o.radius,3) for o in obstacles]}")

    nominal_pose = np.clip(ext_actions[:, :6], -1, 1)
    clear_nom = compute_clearance(cps, obstacles, nominal_pose)
    print(f"名义动作最小净空: {clear_nom.min()*1000:.2f} mm")

    # 选约束 + 解 Δz
    rows = selected_constraints(cps, obstacles, nominal_pose, h_floor=0.005,
                                max_constraints=3, margin=0.02)
    print(f"选中约束 {len(rows)} 条")
    if not rows:
        print("无约束（名义动作安全），退出")
        policy.close(); env.close(); return

    C = np.vstack([c for c, _ in rows])
    b = np.array([bb for _, bb in rows])
    A = C @ ext_J
    dz, rcrit = minimum_norm_halfspaces(A, b)
    print(f"r_crit = {rcrit:.4f}, 线性预测满足 A dz ≥ b: {np.all(A @ dz >= b - 1e-9)}")

    # 候选真实前向验证
    dz_2d = dz.reshape(policy.H, policy.D)
    print(f"\n{'α':>6} {'真实最小净空(mm)':>16} {'源代价':>10}")
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        z_cand = z0 + alpha * dz_2d
        cand_actions = policy.generate(z_cand, policy_obs)
        cand_pose = np.clip(cand_actions[:, :6], -1, 1)
        clear = compute_clearance(cps, obstacles, cand_pose)
        c_src = 0.5 * float(np.linalg.norm(alpha * dz) ** 2)
        print(f"{alpha:>6.2f} {clear.min()*1000:>16.2f} {c_src:>10.4f}")

    policy.close()
    env.close()
    print("\n✅ 若真实净空随 α 单调上升，则源空间 steering 有效（非线性）")


if __name__ == "__main__":
    main()
