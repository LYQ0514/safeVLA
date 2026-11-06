"""诊断：π0.5 在 long task0 state2 上的抓取行为（soup 抬升 / gripper 闭合 / 接触）。

目的：排查 positive control「无 CONTROLLED 见证」是工程 bug 还是 π0.5 真实不抓取。
"""

from __future__ import annotations
import os
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
from src.pnsg.policy_adapter import PnsgPolicy  # noqa: E402
from src.pnsg.crn import nominal_seed, sample_z  # noqa: E402
from src.safety.contract_objects import monitor_contacts  # noqa: E402
from tools.g1a_soup_trace import body_state  # noqa: E402

SUITE = "safelibero_long"
LEVEL = "I"
TASK_ID = 0
IMAGE_SIZE = 224
REPLAN = 5
NUM_STEPS_WAIT = 20
STATE_ID = 2


def make_env_local(ts, task, state_id):
    import pathlib
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256,
                             controller="OSC_POSE")
    env.seed(7); env.reset()
    env.set_init_state(ts.get_task_init_states(TASK_ID)[state_id])
    dummy = np.zeros(7, dtype=np.float32); dummy[-1] = -1.0
    for _ in range(NUM_STEPS_WAIT):
        env.step(dummy)
    return env


def main():
    from libero.libero import benchmark
    ts = benchmark.get_benchmark_dict()[SUITE](safety_level=LEVEL)
    task = ts.get_task(TASK_ID)
    prompt = task.language
    env = make_env_local(ts, task, STATE_ID)
    policy = PnsgPolicy()
    obs = env.env._get_observations()

    soup0 = body_state(env, "alphabet_soup_1_main")["pos"].copy()
    cream0 = body_state(env, "cream_cheese_1_main")["pos"].copy()
    print(f"soup 初始 z={soup0[2]:.3f} cream 初始 z={cream0[2]:.3f}")

    chunk = None
    query_idx = 0
    buf_idx = 0
    for step in range(200):
        if chunk is None or buf_idx >= len(chunk):
            z0, _ = sample_z(nominal_seed(SUITE, TASK_ID, STATE_ID, 0, query_idx))
            po = R.obs_to_policy_obs(obs, prompt, image_size=IMAGE_SIZE)
            chunk = np.clip(policy.generate(z0, po), -1, 1)[:REPLAN]
            query_idx += 1; buf_idx = 0
        action = chunk[buf_idx]
        obs, _r, _d, _i = env.step(action)

        soup = body_state(env, "alphabet_soup_1_main")["pos"]
        cream = body_state(env, "cream_cheese_1_main")["pos"]
        gq = np.asarray(obs["robot0_gripper_qpos"])
        mon = monitor_contacts(env)
        task_bodies = [(ev["body1"], ev["body2"]) for ev in mon["task"]]
        # soup 抬升
        soup_lift = (soup[2] - soup0[2]) * 1000
        cream_lift = (cream[2] - cream0[2]) * 1000

        # 每 10 步打印一次 + 每次 soup 抬升或接触变化时打印
        interesting = (soup_lift > 2 or cream_lift > 2 or task_bodies)
        if step % 10 == 0 or interesting:
            print(f"step={step:3d} soup_z={soup[2]:.3f}(lift {soup_lift:+.1f}mm) "
                  f"cream_z={cream[2]:.3f}(lift {cream_lift:+.1f}mm) "
                  f"gripper={np.round(gq,3)} task_contacts={task_bodies[:2]}")

        buf_idx += 1
        if env.check_success():
            print(f"step={step} native success!")
            break

    policy.close()
    env.close()


if __name__ == "__main__":
    main()
