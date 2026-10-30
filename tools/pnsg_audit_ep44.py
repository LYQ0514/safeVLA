"""PNSG ep44 单状态审计：定位 first contact pair + 失效位置。

按 docs/想法v1.md 第 110 节，重跑 task3 ep44，碰撞前后记录：
- 每次决策的 nominal 净空、nearest pair
- C_2 认证结果
- first native contact 的 geom/body pair
- target grasp state（是否抓到物体）

重点检查决策点 q≈8,9,10（对应真实 step 16-20）。
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

    from src.pnsg.geometry_proxy import build_control_points, build_obstacles, compute_clearance
    from src.pnsg.policy_adapter import PnsgPolicy
    from src.pnsg.certifier import Certifier
    from src.pnsg.full_state import save_full_state, restore_full_state
    from src.eval_safelibero import active_obstacle

    ts = benchmark.get_benchmark_dict()["safelibero_spatial"](safety_level="I")
    task = ts.get_task(3)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256,
                             controller="OSC_POSE")
    env.seed(7)
    obs = env.reset()
    obs = env.set_init_state(ts.get_task_init_states(3)[44])
    dummy = np.array([0.0] * 6 + [-1.0])
    for _ in range(10):
        obs, _r, _d, _i = env.step(dummy)

    obst = active_obstacle(env, obs)
    p0 = np.asarray(obs[obst + "_pos"]).copy()
    print(f"障碍 {obst} @ {np.round(p0,3)}")
    # 目标物体
    m = env.sim.model._model
    d = env.sim.data._data

    policy = PnsgPolicy()
    certifier = Certifier(env)
    po = R.obs_to_policy_obs(obs, task.language, image_size=224)

    # 执行完整 episode，逐 chunk 决策，记录碰撞前信息
    z_prev = None
    for decision in range(20):  # 最多 20 次决策
        z0 = policy.sample_noise(seed=decision)
        _, _, ext, ext_J = policy.generate_with_jacobian(z0, po)
        nominal = np.clip(ext, -1, 1)

        cps = build_control_points(env, obs)
        obstacles = build_obstacles(env, obs)
        clear = compute_clearance(cps, obstacles, nominal[:, :6])
        mc = float(clear.min())

        # 目标物体（碗）位置 vs 障碍
        bowl = obs.get("akita_black_bowl_1_pos", None)
        bowl_dist = None
        if bowl is not None:
            bowl_dist = float(np.linalg.norm(bowl - p0))

        # gripper 抓取状态
        gq = obs["robot0_gripper_qpos"]
        gripper_w = abs(gq[0] - gq[1])

        # C_2 认证名义动作前 2 步
        snapshot = save_full_state(env)
        c2 = certifier.certify_prefix(nominal[:2])

        # 真实执行 2 步，看碰撞
        restore_full_state(env, snapshot)
        disp_before = float(np.sum(np.abs(np.asarray(obs[obst + "_pos"]) - p0)))
        for i in range(2):
            obs, _, _, _ = env.step(nominal[i])
        disp_after = float(np.sum(np.abs(np.asarray(obs[obst + "_pos"]) - p0)))

        collided_this = disp_after > 0.001
        step_real = decision * 2

        print(f"\n决策{decision} (step {step_real}-{step_real+1}): "
              f"净空={mc*1000:.1f}mm 碗距障碍={bowl_dist if bowl_dist is None else round(bowl_dist*1000,1)}mm "
              f"夹爪宽={gripper_w:.3f} C2={c2.pass_} 执行后disp={disp_after*1000:.1f}mm")

        if collided_this:
            print(f"  ⚠️ 碰撞发生在决策 {decision}！")
            # 记录 first contact pair
            print(f"  当前 contact 里 robot/obstacle/target 相关:")
            for c in range(d.ncon):
                g1, g2 = d.contact[c].geom1, d.contact[c].geom2
                b1 = env.sim.model.body_names[m.geom_bodyid[g1]]
                b2 = env.sim.model.body_names[m.geom_bodyid[g2]]
                if "obstacle" in b1 or "obstacle" in b2:
                    print(f"    contact{c}: {b1} <-> {b2} (geom {g1},{g2}) dist={d.contact[c].dist:.4f}")
            break

    policy.close()
    env.close()


if __name__ == "__main__":
    main()
