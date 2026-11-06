"""推理延迟对照（Commit 1，裁决 §一/§十一）。

对比默认配置（组 A）vs 确定性配置（autotune_level=0 + exclude_nondeterministic_ops）
的单次 generate 推理延迟，确认确定性配置的性能代价。

裁决 §十一：确定性配置可能降低/改变推理速度，H0 必须重新测量 p50/p95/p99。
"""
from __future__ import annotations

import os
import sys
import warnings
import time

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

SUITE = "safelibero_spatial"; TASK_ID = 1; STATE_ID = 1
IMAGE_SIZE = 224; NUM_STEPS_WAIT = 20
N_REPS = 30  # 热身后计时


def collect_obs(ts, task):
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
    return po


def measure(xla_flags, label):
    from libero.libero import benchmark
    ts = benchmark.get_benchmark_dict()[SUITE](safety_level="I")
    task = ts.get_task(TASK_ID)
    po = collect_obs(ts, task)
    z, _ = sample_z(nominal_seed(SUITE, TASK_ID, STATE_ID, 0, 0))
    policy = PnsgPolicy(xla_flags=xla_flags)
    # 热身（编译 + autotune 缓存）
    for _ in range(5):
        policy.generate(z, po)
    lats = []
    for _ in range(N_REPS):
        t0 = time.perf_counter()
        policy.generate(z, po)
        lats.append((time.perf_counter() - t0) * 1000)
    policy.close()
    lats = np.array(lats)
    p50, p95, p99 = np.percentile(lats, [50, 95, 99])
    print(f"{label}: p50={p50:.2f}ms p95={p95:.2f}ms p99={p99:.2f}ms mean={lats.mean():.2f}ms")
    return p50, p95, p99, lats.mean()


if __name__ == "__main__":
    a = measure(None, "组A 默认")
    b = measure("--xla_gpu_autotune_level=0 --xla_gpu_exclude_nondeterministic_ops=true",
                "组C 确定性")
    print()
    print(f"确定性配置延迟代价: p50 {b[0]/a[0]:.2f}x  p95 {b[1]/a[1]:.2f}x  mean {b[3]/a[3]:.2f}x")
