"""PNSG 完整单 episode 评测：真实危险场景下的源空间 steering + C_2 认证。

用 baseline 里真实碰撞最狠的 episode（spatial task3 ep44，障碍被撞飞 1.2m），
走完整 PNSG 流程：
1. 从初始状态采样 z0，求名义动作 + Jacobian
2. 几何代理选约束，解 Δz
3. 候选池（nominal + 4 scaled）真实前向 + C_2 认证
4. 对比：执行名义 vs 执行 PNSG 选出的候选，谁更安全/更成功

这是 PNSG 是否真正有用的关键验证。
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


def make_env(suite, level, task_id):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    ts = benchmark.get_benchmark_dict()[suite](safety_level=level)
    task = ts.get_task(task_id)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256,
                             controller="OSC_POSE")
    env.seed(7)
    return env, ts, task


def run_chunk(env, obs, actions, n_exec=2):
    """执行 action chunk 的前 n_exec 步，返回是否碰撞、是否成功。"""
    from src.pnsg.certifier import Certifier
    cert = Certifier(env)
    c2 = cert.certify_prefix(actions[:n_exec])
    return c2


def main():
    from src.pnsg.geometry_proxy import build_control_points, build_obstacles, selected_constraints, compute_clearance
    from src.pnsg.policy_adapter import PnsgPolicy
    from src.pnsg.source_solver import minimum_norm_halfspaces
    from src.pnsg.full_state import save_full_state, restore_full_state
    from src.eval_safelibero import active_obstacle

    # 危险 episode：spatial task3 ep44（障碍被撞飞 1.2m）
    suite, level, task_id, ep = "safelibero_spatial", "I", 3, 44
    env, ts, task = make_env(suite, level, task_id)
    obs = env.reset()
    obs = env.set_init_state(ts.get_task_init_states(task_id)[ep])
    dummy = np.array([0.0] * 6 + [-1.0])
    for _ in range(10):
        obs, _r, _d, _i = env.step(dummy)

    prompt = task.language
    obst = active_obstacle(env, obs)
    p0 = np.asarray(obs[obst + "_pos"]).copy()
    print(f"危险场景: spatial task{task_id} ep{ep}, 障碍 {obst} @ {np.round(p0,3)}")
    print(f"任务: {prompt}")

    policy_obs = R.obs_to_policy_obs(obs, prompt, image_size=224)
    policy = PnsgPolicy()
    z0 = policy.sample_noise(seed=42)
    _, _, ext_actions, ext_J = policy.generate_with_jacobian(z0, policy_obs)
    nominal = np.clip(ext_actions, -1, 1)

    cps = build_control_points(env, obs)
    obstacles = build_obstacles(env, obs)
    clear_nom = compute_clearance(cps, obstacles, nominal[:, :6])
    print(f"\n名义动作最小净空: {clear_nom.min()*1000:.1f} mm")

    rows = selected_constraints(cps, obstacles, nominal[:, :6], h_floor=0.005,
                                max_constraints=3, margin=0.01)
    print(f"选中约束 {len(rows)} 条")

    if rows:
        C = np.vstack([c for c, _ in rows])
        b = np.array([bb for _, bb in rows])
        A = C @ ext_J
        dz, rcrit = minimum_norm_halfspaces(A, b)
        dz_2d = dz.reshape(policy.H, policy.D) if dz is not None else None
        print(f"r_crit = {rcrit:.2f}")
    else:
        dz_2d = None
        print("名义动作本就安全，无约束")

    # 候选池 + 真实执行对比
    snapshot = save_full_state(env)
    cands = [("nominal", z0, nominal)]
    if dz_2d is not None:
        for alpha in [0.25, 0.5, 0.75, 1.0]:
            z_c = z0 + alpha * dz_2d
            a_c = np.clip(policy.generate(z_c, policy_obs), -1, 1)
            cands.append((f"α={alpha:.2f}", z_c, a_c))

    print(f"\n{'候选':<10} {'代理净空mm':>10} {'C_2前2步':>10} {'真实执行结果'}")
    print("-" * 60)
    for label, zc, ac in cands:
        # 代理净空
        mc = compute_clearance(cps, obstacles, ac[:, :6]).min() * 1000
        # C_2 前 2 步
        restore_full_state(env, snapshot)
        c2 = run_chunk(env, obs, ac, n_exec=2)
        # 真实执行整个 chunk（不恢复，看最终结果）—— 但这会污染状态，所以重新恢复
        restore_full_state(env, snapshot)
        # 只执行前 2 步，看障碍位移
        obs2 = obs
        for i in range(2):
            obs2, _, _, _ = env.step(ac[i])
        disp = float(np.sum(np.abs(np.asarray(obs2[obst + "_pos"]) - p0)))
        print(f"{label:<10} {mc:>10.1f} {str(c2.pass_):>10} 障碍位移 {disp*1000:.1f}mm")

    policy.close()
    env.close()
    print("\n✅ 关键指标：PNSG 候选（α>0）的障碍位移应显著小于 nominal")


if __name__ == "__main__":
    main()
