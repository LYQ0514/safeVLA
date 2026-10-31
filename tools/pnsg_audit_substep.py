"""substep vs step 级接触检测对比：验证瞬时接触是否会被漏掉。

这是 v3 第 6 条「必须重跑」判断的前提：如果 substep 能捕获 step 后看不到的
接触，说明旧数据（step 后看 data.contact）漏掉了瞬时接触，必须重跑。
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

    from src.pnsg.contact_audit import ContactEventRecorder
    from src.pnsg.policy_adapter import PnsgPolicy
    from src.eval_safelibero import active_obstacle

    suite, level, task_id, ep = "safelibero_spatial", "I", 0, 0  # task0 是必碰撞任务
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

    policy = PnsgPolicy()
    rec = ContactEventRecorder(env)
    m = env.sim.model._model
    d = env.sim.data._data

    # 跑名义轨迹，逐步对比 substep 接触 vs step 后接触
    print(f"场景 task{task_id} ep{ep}, 障碍 {obst}\n")
    print(f"{'step':>4} {'substep接触数':>12} {'step后接触数':>12} {'障碍位移L2':>12}")
    substep_total = 0
    step_total = 0
    for decision in range(30):
        po = R.obs_to_policy_obs(obs, prompt, image_size=224)
        z0 = policy.sample_noise(seed=decision)
        ext = policy.generate(z0, po)
        nominal = np.clip(ext, -1, 1)
        for i in range(2):
            step = decision * 2 + i
            # substep 级
            obs, events = rec.substep_step(nominal[i])
            n_sub = len(events)
            # step 后级（env.step 已经跑完，看 data.contact）
            n_step = 0
            for c in range(d.ncon):
                g1, g2 = d.contact[c].geom1, d.contact[c].geom2
                if d.contact[c].dist <= 0:
                    if (g1 in rec.robot_geoms and g2 in rec.obstacle_geoms) or \
                       (g2 in rec.robot_geoms and g1 in rec.obstacle_geoms):
                        n_step += 1
            disp = float(np.linalg.norm(np.asarray(obs[obst + "_pos"]) - p0))
            substep_total += n_sub
            step_total += n_step
            if n_sub > 0 or n_step > 0:
                print(f"{step:>4} {n_sub:>12} {n_step:>12} {disp*1000:>11.2f}mm")

    print(f"\n累计：substep 接触 {substep_total} 次，step 后接触 {step_total} 次")
    print(f"结论：{'⚠️ substep 能捕获更多瞬时接触，旧数据漏检' if substep_total > step_total else '两者一致（接触持续到 step 结束）'}")
    policy.close()
    env.close()


if __name__ == "__main__":
    main()
