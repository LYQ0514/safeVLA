"""定位 xpos 在 env.step 后的状态：是否需 forward 才更新到最新 qpos。

MuJoCo 语义：sim.step() 内部 = forward(qpos_t) + 积分(qpos_t→qpos_{t+1})，
返回后 xpos 是 f(qpos_t)（旧），qpos 是 qpos_{t+1}（新）。
若 eval_goal_clauses/check_success 触发 sim.forward()，会把 xpos 更新到 f(qpos_{t+1})。

实验：step37 后分别记录：
  A. env.step 后立即读 bowl xpos（无 forward）
  B. sim.forward() 后读
  C. check_success() 后读
看哪个对应 7.809 / 10.194。
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

SUITE = "safelibero_spatial"
TASK_ID = 1
STATE_ID = 1
IMAGE_SIZE = 224
REPLAN = 5
NUM_STEPS_WAIT = 20
N_STEPS = 40
TARGET = "akita_black_bowl_1_main"


def main():
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
    policy = PnsgPolicy()
    settled = body_state(env, TARGET)["pos"].copy()
    chunk = None; q = 0; b = 0

    def disp():
        return float(np.sum(np.abs(body_state(env, TARGET)["pos"] - settled))) * 1000

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
                dA = disp()  # 立即读（无 forward）
                env.env.sim.forward()
                dB = disp()  # forward 后读
                env.env.sim.forward()
                _ = env.check_success()  # check_success 可能 forward
                dC = disp()  # check_success 后读
                print(f"step37: A(直接读)={dA:.4f}  B(forward后)={dB:.4f}  C(check_success后)={dC:.4f}")
    finally:
        policy.close()
        env.close()


if __name__ == "__main__":
    main()
