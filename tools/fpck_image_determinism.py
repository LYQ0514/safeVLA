"""图像渲染确定性：多次 make_env + 采集 obs，对比 RGB 图像 hash。

若 image hash 跨进程不同 → 根因是 EGL 渲染非确定（输入张量差异），非 VLA 推理；
若 image hash 一致 → 根因回到 VLA 推理。
"""
from __future__ import annotations

import os
import sys
import warnings
import hashlib

warnings.filterwarnings("ignore")
ROOT = "/data/safeVLA"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "dataset", "SafeLIBERO", "safelibero"))
sys.path.insert(0, "/data/2608.SDL/MojocoProj/src/pi05")
sys.path.insert(0, os.path.join(ROOT, "third_party", "aegis"))
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
import pi05_libero_runner as R  # noqa: E402

SUITE = "safelibero_spatial"; TASK_ID = 1; STATE_ID = 1
IMAGE_SIZE = 224; NUM_STEPS_WAIT = 20


def hash_arr(arr):
    a = np.ascontiguousarray(np.asarray(arr))
    return hashlib.sha256(a.tobytes(order="C")).hexdigest()


def collect(ts, task):
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
    po = R.obs_to_policy_obs(obs, task.language, image_size=IMAGE_SIZE)
    env.close()
    # hash 每个图像
    hashes = {}
    for k, v in po.items():
        if "image" in k.lower() and v is not None:
            hashes[k] = hash_arr(v)
    return hashes


def main():
    from libero.libero import benchmark
    ts = benchmark.get_benchmark_dict()[SUITE](safety_level="I")
    task = ts.get_task(TASK_ID)
    all_h = []
    for i in range(6):
        h = collect(ts, task)
        all_h.append(h)
        print(f"run#{i}: " + " ".join(f"{k.split('_')[0]}:{v[:8]}" for k, v in sorted(h.items())))
    # 对比每个图像键
    print()
    print("=== 图像 hash 跨进程一致性 ===")
    keys = sorted(all_h[0].keys())
    for k in keys:
        vals = [h[k] for h in all_h]
        same = all(v == vals[0] for v in vals)
        print(f"  {k}: {'一致' if same else '不一致! ' + str([v[:8] for v in vals])}")


if __name__ == "__main__":
    main()
