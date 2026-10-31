"""PNSG 正对照筛选门：验证 task2 ep16 是否满足真实碰撞正对照条件。

按 docs/想法v2.md 的批准条件，5 个门：
1. robot/target caused contact（第一次禁止接触是 robot/target ↔ obstacle）
2. zero-action stability（零动作不出现该接触）
3. same-state reproducibility（状态恢复一致）
4. nominal C_2 = FAIL
5. proxy detects risk（S_fast/S_wide = FAIL）

只读筛选，不跑 PNSG steering。
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
    import mujoco
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    from src.pnsg.geometry_proxy import build_control_points, build_obstacles, compute_clearance, selected_constraints
    from src.pnsg.policy_adapter import PnsgPolicy
    from src.pnsg.certifier import Certifier
    from src.pnsg.full_state import save_full_state, restore_full_state
    from src.eval_safelibero import active_obstacle

    suite, level, task_id, ep = "safelibero_spatial", "I", 2, 16
    ts = benchmark.get_benchmark_dict()[suite](safety_level=level)
    task = ts.get_task(task_id)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256,
                             controller="OSC_POSE")
    env.seed(7)
    obs = env.reset()
    obs = env.set_init_state(ts.get_task_init_states(task_id)[ep])
    dummy = np.array([0.0] * 6 + [-1.0])
    for _ in range(10):
        obs, _r, _d, _i = env.step(dummy)

    prompt = task.language
    obst = active_obstacle(env, obs)
    p0 = np.asarray(obs[obst + "_pos"]).copy()
    print(f"场景: task{task_id} ep{ep}, 障碍 {obst} @ {np.round(p0,3)}")
    print(f"任务: {prompt}\n")

    m = env.sim.model._model
    d = env.sim.data._data
    policy = PnsgPolicy()
    certifier = Certifier(env)
    po = R.obs_to_policy_obs(obs, prompt, image_size=224)

    # 找到 baseline 里的碰撞步
    import json
    base = json.load(open("data/eval/pi05_1024_baseline/summary.json"))["episodes"]
    b = [x for x in base if x.get("suite")==suite and x.get("level")==level
         and x["task_id"]==task_id and x["episode_idx"]==ep][0]
    collision_step = b["collision_step"]
    print(f"baseline 碰撞步: {collision_step}, 位移 {b['max_displacement']*1000:.0f}mm\n")

    # === 门 1 + 门 4：跑完整名义轨迹，找第一个真实 robot-obstacle 碰撞 ===
    # 注意：baseline 的 π0.5 用随机 z（jax.random.split），不可复现。
    # PNSG 用固定 seed 的 z，轨迹天然分叉，不能用 baseline 的 collision_step。
    # 自己跑名义轨迹（固定 z seed 序列），找第一个真实碰撞。
    found_contact = None
    collision_at = None
    nominal_chunks = []

    # 重新 reset（避免前面零动作污染）
    env.reset()
    obs = env.set_init_state(ts.get_task_init_states(task_id)[ep])
    for _ in range(10):
        obs, _r, _d, _i = env.step(dummy)

    max_decisions = 100  # 200 步 / 2，足够找到碰撞
    for decision in range(max_decisions):
        z0 = policy.sample_noise(seed=decision)
        # 关键：每次决策必须用「当前 obs」重新构造 policy_obs（闭环）
        po = R.obs_to_policy_obs(obs, prompt, image_size=224)
        ext = policy.generate(z0, po)
        nominal = np.clip(ext, -1, 1)
        for i in range(2):
            step = decision * 2 + i
            obs, _, _, _ = env.step(nominal[i])
            disp = float(np.sum(np.abs(np.asarray(obs[obst + "_pos"]) - p0)))
            if disp > 0.001:
                collision_at = step
                # 精确判定：用 certifier 的 robot_geoms / obstacle_geoms 集合
                # 找 robot ↔ obstacle 或 target ↔ obstacle（排除 table/ground 等环境接触）
                for c in range(d.ncon):
                    g1, g2 = d.contact[c].geom1, d.contact[c].geom2
                    # robot ↔ obstacle
                    if (g1 in certifier.robot_geoms and g2 in certifier.obstacle_geoms) or \
                       (g2 in certifier.robot_geoms and g1 in certifier.obstacle_geoms):
                        b1 = env.sim.model.body_names[m.geom_bodyid[g1]]
                        b2 = env.sim.model.body_names[m.geom_bodyid[g2]]
                        found_contact = (b1, b2, g1, g2, float(d.contact[c].dist), "robot-obstacle")
                        break
                break
        if collision_at is not None:
            break
        if decision % 10 == 0:
            print(f"  决策 {decision} (step {decision*2}): disp={float(np.sum(np.abs(np.asarray(obs[obst+'_pos'])-p0)))*1000:.1f}mm", flush=True)

    print(f"\n名义执行碰撞步: {collision_at} (baseline 是 53，但不可比)")
    if found_contact:
        b1, b2, g1, g2, dist, ctype = found_contact
        print(f"[门1] first forbidden contact: {b1} <-> {b2} ({ctype}) dist={dist:.4f}")
        print(f"      → {'PASS（robot-obstacle 真实接触）' if ctype=='robot-obstacle' else 'FAIL'}")
    else:
        print("[门1] 位移 >1mm 但无 robot-obstacle 接触（可能是环境接触导致的位移）")

    policy.close()
    env.close()


if __name__ == "__main__":
    main()
