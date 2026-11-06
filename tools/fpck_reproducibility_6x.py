"""6 次纯原生 rollout，统计 step37 disp 分布（检测进程级非确定性）。

每次独立进程内 make_env + rollout，只读 env.step 后 xpos。
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

SUITE = "safelibero_spatial"; TASK_ID = 1; STATE_ID = 1
IMAGE_SIZE = 224; REPLAN = 5; NUM_STEPS_WAIT = 20; N_STEPS = 40
TARGET = "akita_black_bowl_1_main"


def run_once(ts, task):
    import pathlib
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256,
                             controller="OSC_POSE")
    env.seed(7); env.reset()
    env.set_init_state(ts.get_task_init_states(TASK_ID)[STATE_ID])
    dummy = np.zeros(7, dtype=np.float32); dummy[-1] = -1.0
    for _ in range(NUM_STEPS_WAIT):
        env.step(dummy)
    obs = env.env._get_observations()
    policy = PnsgPolicy()
    settled = body_state(env, TARGET)["pos"].copy()
    chunk = None; q = 0; b = 0
    d36 = d37 = d38 = None
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
            if step == 36: d36 = float(np.sum(np.abs(body_state(env, TARGET)["pos"]-settled)))*1000
            if step == 37: d37 = float(np.sum(np.abs(body_state(env, TARGET)["pos"]-settled)))*1000
            if step == 38: d38 = float(np.sum(np.abs(body_state(env, TARGET)["pos"]-settled)))*1000
    finally:
        policy.close(); env.close()
    return d36, d37, d38


def main():
    from libero.libero import benchmark
    ts = benchmark.get_benchmark_dict()[SUITE](safety_level="I")
    task = ts.get_task(TASK_ID)
    vals = []
    for i in range(6):
        d = run_once(ts, task)
        vals.append(d)
        print(f"run#{i}: step36={d[0]:.4f} step37={d[1]:.4f} step38={d[2]:.4f}")
    print()
    # 分布
    d37s = [v[1] for v in vals]
    print(f"step37 值集合: {sorted(set(f'{x:.4f}' for x in d37s))}")
    print(f"step37 分布: {d37s}")


if __name__ == "__main__":
    main()
