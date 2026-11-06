"""精确验证 xpos 在 env.step 后的时序语义（MuJoCo forward/积分顺序）。

MuJoCo mj_step = forward(qpos_t) + Euler 积分到 qpos_{t+1}。
返回后：qpos = qpos_{t+1}（新），xpos = f(qpos_t)（旧，forward 缓存）。
再调 forward() 会把 xpos 更新到 f(qpos_{t+1})。

因此「直接读 xpos」读到的是**上一步 forward 的缓存**，可能滞后一个 substep。
bridge 用 body_state(xpos) 读 target disp，所以 bridge 读到的是滞后值。

但注意：robosuite env.step 的循环里，每次 sim.step() 前都 sim.forward()，
最后一次 sim.step() 后不再 forward，所以 env.step 返回时 xpos 是
f(qpos_{24})（第24 substep 积分前），qpos 是 qpos_{25}（积分后）。
"""

from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = "/data/safeVLA"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "dataset", "SafeLIBERO", "safelibero"))
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

from libero.libero import benchmark  # noqa: E402
from tools.g1a_soup_trace import body_state  # noqa: E402

SUITE = "safelibero_spatial"
TASK_ID = 1
STATE_ID = 1
NUM_STEPS_WAIT = 20
TARGET = "akita_black_bowl_1_main"


def main():
    from libero.libero import benchmark as bm
    import pathlib
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    ts = bm.get_benchmark_dict()[SUITE](safety_level="I")
    task = ts.get_task(TASK_ID)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256,
                             controller="OSC_POSE")
    env.seed(7); env.reset()
    env.set_init_state(ts.get_task_init_states(TASK_ID)[STATE_ID])
    dummy = np.zeros(7, dtype=np.float32); dummy[-1] = -1.0
    for _ in range(NUM_STEPS_WAIT):
        env.step(dummy)

    d = env.sim.data._data
    m = env.sim.model._model
    bid = env.sim.model.body_name2id(TARGET)

    # 单步 step，看 xpos 与 qpos 的关系
    act = np.zeros(7, dtype=np.float64); act[-1] = 1.0  # close gripper
    for i in range(3):
        xpos_before = np.asarray(d.xpos[bid]).copy()
        qpos_before = np.asarray(d.qpos).copy()
        env.step(act)
        xpos_after_step = np.asarray(d.xpos[bid]).copy()
        qpos_after_step = np.asarray(d.qpos).copy()
        # forward 后
        env.env.sim.forward()
        xpos_after_forward = np.asarray(d.xpos[bid]).copy()
        qpos_after_forward = np.asarray(d.qpos).copy()
        print(f"step {i}: xpos 变化={np.sum(np.abs(xpos_after_step-xpos_before))*1000:.6f}mm  "
              f"forward后 xpos 变化={np.sum(np.abs(xpos_after_forward-xpos_after_step))*1000:.6f}mm  "
              f"forward后 qpos 变化={np.sum(np.abs(qpos_after_forward-qpos_after_step)):.3e}")
    env.close()


if __name__ == "__main__":
    main()
