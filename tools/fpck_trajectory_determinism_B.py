"""轨迹级确定性验证（组 B：--xla_gpu_autotune_level=0）。

裁决 §五轨迹级补充验证：3 个冷启动进程，step37 disp + 首次接触步 + 语义 trace 是否一致。
"""
from __future__ import annotations

import hashlib
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
XLA_FLAGS = "--xla_gpu_autotune_level=0"


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
    policy = PnsgPolicy(xla_flags=XLA_FLAGS)
    settled = body_state(env, TARGET)["pos"].copy()
    chunk = None; q = 0; b = 0
    d36 = d37 = d38 = None
    chunk_hashes = []
    try:
        for step in range(N_STEPS):
            if chunk is None or b >= len(chunk):
                z0, _ = sample_z(nominal_seed(SUITE, TASK_ID, STATE_ID, 0, q))
                po = R.obs_to_policy_obs(obs, task.language, image_size=IMAGE_SIZE)
                chunk = np.clip(policy.generate(z0, po), -1, 1)[:REPLAN]
                chunk_hashes.append(hashlib.sha256(np.asarray(chunk).tobytes()).hexdigest()[:12])
                q += 1; b = 0
            act = chunk[b]
            obs, _r, _d, _i = env.step(act)
            b += 1
            if step == 36: d36 = float(np.sum(np.abs(body_state(env, TARGET)["pos"]-settled)))*1000
            if step == 37: d37 = float(np.sum(np.abs(body_state(env, TARGET)["pos"]-settled)))*1000
            if step == 38: d38 = float(np.sum(np.abs(body_state(env, TARGET)["pos"]-settled)))*1000
    finally:
        policy.close(); env.close()
    return d36, d37, d38, chunk_hashes


def main():
    from libero.libero import benchmark
    ts = benchmark.get_benchmark_dict()[SUITE](safety_level="I")
    task = ts.get_task(TASK_ID)
    vals = []
    for i in range(3):
        d = run_once(ts, task)
        vals.append(d)
        print(f"run#{i}: step36={d[0]:.4f} step37={d[1]:.4f} step38={d[2]:.4f} "
              f"chunk0={d[3][0]}")
    print()
    d37s = [v[1] for v in vals]
    print(f"step37 值集合: {sorted(set(f'{x:.4f}' for x in d37s))}")
    print(f"step37 唯一: {len(set(d37s))==1}  ({'PASS' if len(set(d37s))==1 else 'FAIL'})")
    # chunk 一致性
    for pos in range(8):
        ch = [v[3][pos] for v in vals]
        same = all(c == ch[0] for c in ch)
        if not same:
            print(f"chunk[{pos}] 不一致: {ch}")


if __name__ == "__main__":
    main()
