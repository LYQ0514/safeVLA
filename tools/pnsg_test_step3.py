"""PNSG 第三步验证：接非线性认证 C_2。

验证：
1. save/restore 后每个候选从同一状态开始
2. C_2 能区分「代理通过但真实碰撞」的候选（设计文档 17.5 节四格表）
3. 候选池（nominal + 4 个 scaled）的真实认证结果
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
    from src.pnsg.full_state import save_full_state, restore_full_state
    from src.pnsg.certifier import Certifier

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
    for ob in obstacles:
        ob.radius *= 1.6  # 逼出危险场景

    nominal_pose = np.clip(ext_actions[:, :6], -1, 1)
    rows = selected_constraints(cps, obstacles, nominal_pose, h_floor=0.005,
                                max_constraints=3, margin=0.02)
    C = np.vstack([c for c, _ in rows])
    b = np.array([bb for _, bb in rows])
    A = C @ ext_J
    dz, rcrit = minimum_norm_halfspaces(A, b)
    dz_2d = dz.reshape(policy.H, policy.D)

    # 快照 + certifier
    snapshot = save_full_state(env)
    certifier = Certifier(env)

    print(f"名义净空 {compute_clearance(cps, obstacles, nominal_pose).min()*1000:.1f}mm, "
          f"r_crit={rcrit:.2f}")
    print(f"\n{'候选':<10} {'S_wide':>8} {'真实净空mm':>10} {'C_2':>5} {'四格表分类':>12}")
    print("-" * 55)

    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        z_cand = z0 + alpha * dz_2d
        cand_actions = policy.generate(z_cand, policy_obs)  # (10,7) external
        cand_full = np.clip(cand_actions, -1, 1)  # 7 维都 clip

        # S_wide 代理
        cand_pose = cand_full[:, :6]
        clear = compute_clearance(cps, obstacles, cand_pose)
        mc = float(clear.min())
        S_wide = mc >= 0.005

        # C_2 真实认证（从同一快照恢复）
        restore_full_state(env, snapshot)
        c2 = certifier.certify_prefix(cand_full[:2])

        # 四格表分类
        if S_wide and c2.pass_:
            cat = "S=1,C2=1"
        elif S_wide and not c2.pass_:
            cat = "S=1,C2=0 ⚠️"
        elif not S_wide and c2.pass_:
            cat = "S=0,C2=1"
        else:
            cat = "S=0,C2=0"

        label = "nominal" if alpha == 0 else f"α={alpha:.2f}"
        print(f"{label:<10} {str(S_wide):>8} {mc*1000:>10.1f} {str(c2.pass_):>5} {cat:>12}"
              + (f"  [{c2.reason}]" if not c2.pass_ else ""))

    policy.close()
    env.close()
    print("\n✅ 若出现 S=1,C2=0 的候选，说明代理通过但真实碰撞，C_2 有区分力")


if __name__ == "__main__":
    main()
