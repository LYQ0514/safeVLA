"""最小对照：find_active_obstacle 是否改变轨迹。

同一 rollout，对比「有 find_active_obstacle（bridge 流程）」vs「无」。
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
from tools.g1a_soup_trace import body_state  # noqa: E402
from src.safety.aegis_adapted import find_active_obstacle  # noqa: E402

SUITE = "safelibero_spatial"
TASK_ID = 1
STATE_ID = 1
IMAGE_SIZE = 224
REPLAN = 5
NUM_STEPS_WAIT = 20
N_STEPS = 40
TARGET = "akita_black_bowl_1_main"


def run(use_obstacle):
    from libero.libero import benchmark
    import pathlib
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    ts = benchmark.get_benchmark_dict()[SUITE](safety_level="I")
    task = ts.get_task(TASK_ID)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256,
                             controller="OSC_POSE")
    env.seed(7); env.reset()
    env.set_init_state(ts.get_task_init_states(TASK_ID)[STATE_ID])
    dummy = np.zeros(7, dtype=np.float32); dummy[-1] = -1.0
    for _ in range(NUM_STEPS_WAIT):
        env.step(dummy)
    obs = env.env._get_observations()
    if use_obstacle:
        obstacle = find_active_obstacle(env, obs)
        print(f"  find_active_obstacle 返回: {obstacle}")
    policy = PnsgPolicy()
    settled = body_state(env, TARGET)["pos"].copy()
    chunk = None; q = 0; b = 0
    disp37 = None
    try:
        for step in range(N_STEPS):
            if chunk is None or b >= len(chunk):
                z0, _ = sample_z(nominal_seed(SUITE, TASK_ID, STATE_ID, 0, q))
                po = R.obs_to_policy_obs(obs, task.language, image_size=IMAGE_SIZE)
                chunk = np.clip(policy.generate(z0, po), -1, 1)[:REPLAN]
                q += 1; b = 0
            act = chunk[b]
            obs, _r, _d, _i = env.step(act)
            b += 1
            if step == 37:
                disp37 = float(np.sum(np.abs(body_state(env, TARGET)["pos"] - settled))) * 1000
    finally:
        policy.close()
        env.close()
    return disp37


def main():
    for use_ob in [False, True, False, True]:
        d = run(use_ob)
        print(f"use_obstacle={use_ob}: step37 disp={d:.4f}")


if __name__ == "__main__":
    main()
