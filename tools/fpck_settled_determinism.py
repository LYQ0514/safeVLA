"""定位 7.81 vs 10.19 的根因：settled baseline 是否确定。

关键问题：多次 make_env（相同 seed）是否产生相同 settled bowl 位置。
若确定 → 差异来自代码路径（find_active_obstacle 等）；
若非确定 → 差异来自 MuJoCo reset/step 浮点非确定性。
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


def make_env(ts, task):
    import pathlib
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256,
                             controller="OSC_POSE")
    env.seed(7)
    env.reset()
    env.set_init_state(ts.get_task_init_states(TASK_ID)[STATE_ID])
    dummy = np.zeros(7, dtype=np.float32); dummy[-1] = -1.0
    for _ in range(NUM_STEPS_WAIT):
        env.step(dummy)
    return env


def main():
    ts = benchmark.get_benchmark_dict()[SUITE](safety_level="I")
    task = ts.get_task(TASK_ID)

    print("连续 4 次 make_env，对比 settled bowl_pos（检测非确定性）：")
    positions = []
    for i in range(4):
        env = make_env(ts, task)
        p = body_state(env, TARGET)["pos"].copy()
        positions.append(p)
        env.close()
        print(f"  make_env#{i}: bowl_pos = {p.tolist()}")

    # 两两最大差
    arr = np.array(positions)
    print()
    print("两两最大 L1 差 (mm):")
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            d = float(np.sum(np.abs(arr[i] - arr[j]))) * 1000
            print(f"  #{i} vs #{j}: {d:.6f} mm")


if __name__ == "__main__":
    main()
