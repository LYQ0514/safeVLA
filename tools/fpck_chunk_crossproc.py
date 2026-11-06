"""验证：不同进程的 policy chunk 是否 bit 级一致（检测 GPU 非确定性）。

若 chunk hash 跨进程一致但轨迹分歧 → MuJoCo 非确定；
若 chunk hash 跨进程不一致 → openpi GPU 推理非确定。
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
from src.pnsg.policy_adapter import PnsgPolicy  # noqa: E402
from src.pnsg.crn import nominal_seed, sample_z  # noqa: E402

SUITE = "safelibero_spatial"; TASK_ID = 1; STATE_ID = 1
IMAGE_SIZE = 224; REPLAN = 5; NUM_STEPS_WAIT = 20


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
    chunk_hashes = []
    try:
        for q in range(8):  # 8 个 query
            z0, _ = sample_z(nominal_seed(SUITE, TASK_ID, STATE_ID, 0, q))
            po = R.obs_to_policy_obs(obs, task.language, image_size=IMAGE_SIZE)
            chunk = np.clip(policy.generate(z0, po), -1, 1)[:REPLAN]
            chunk_hashes.append(hashlib.sha256(np.asarray(chunk).tobytes()).hexdigest()[:16])
    finally:
        policy.close(); env.close()
    return chunk_hashes


def main():
    from libero.libero import benchmark
    ts = benchmark.get_benchmark_dict()[SUITE](safety_level="I")
    task = ts.get_task(TASK_ID)
    all_hashes = []
    for i in range(3):
        h = run_once(ts, task)
        all_hashes.append(h)
        print(f"run#{i}: chunk0={h[0]} ... chunk7={h[7]}")
    print()
    for pos in range(8):
        vals = [h[pos] for h in all_hashes]
        same = all(v == vals[0] for v in vals)
        print(f"chunk[{pos}]: {'一致' if same else '不一致! ' + str(vals)}")


if __name__ == "__main__":
    main()
