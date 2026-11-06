"""聚焦：rollout 内 8 次连续 chunk 查询（不同 z）的跨进程一致性。

reproducibility_6x 发现 step36（第 8 个 chunk q=7 的第 2 步）首次分歧，
而 determinism_matrix 证明单次推理（q=0）确定。本实验测：同一 settled obs 下，
连续 8 次查询（q=0..7）的 raw/deployed hash 是否跨进程唯一。

若 8 次查询跨进程唯一 → rollout 分歧另有原因（MuJoCo 接触非确定？）；
若某次查询（尤其 q=7）跨进程不一致 → 多次查询累积非确定。
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

SUITE = "safelibero_spatial"; TASK_ID = 1; STATE_ID = 1
IMAGE_SIZE = 224; NUM_STEPS_WAIT = 20
N_QUERY = 8
N_PROC = 4
XLA_FLAGS = os.environ.get("DET_XLA_FLAGS")  # 可选，组 B/C 用


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


def main():
    from libero.libero import benchmark
    ts = benchmark.get_benchmark_dict()[SUITE](safety_level="I")
    task = ts.get_task(TASK_ID)
    po = collect_obs(ts, task)  # 固定 obs，所有进程复用

    all_hashes = []
    for pi in range(N_PROC):
        policy = PnsgPolicy(xla_flags=XLA_FLAGS)
        proc_hashes = []
        for q in range(N_QUERY):
            z, _ = sample_z(nominal_seed(SUITE, TASK_ID, STATE_ID, 0, q))
            actions, h = policy.generate_with_hashes(z, po)
            proc_hashes.append(h["deployed_chunk_sha256"])
        all_hashes.append(proc_hashes)
        policy.close()
        print(f"proc{pi}: " + " ".join(f"q{q}:{h[:8]}" for q, h in enumerate(proc_hashes)))

    print()
    print("=== 逐 query 跨进程一致性（deployed chunk hash）===")
    for q in range(N_QUERY):
        vals = [h[q] for h in all_hashes]
        same = all(v == vals[0] for v in vals)
        print(f"  q{q}: {'一致' if same else '不一致! ' + str([v[:8] for v in vals])}")


if __name__ == "__main__":
    main()
