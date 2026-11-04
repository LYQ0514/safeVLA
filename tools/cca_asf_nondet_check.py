"""CCA-ASF 非确定性量化：连续跑 N 次同一 (state_id, stream, mode)，测 mug 位移跨 run 方差。

背景：requick 与 pilot 的同条件（S01/st0 euclidean）结果不同（25.6 vs 7.8mm），
怀疑 PnsgPolicy 的「确定性 CRN」被 π0.5 GPU 非确定性破坏。本脚本量化这个方差。
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
from src.safety.aegis_adapted import AegisSafetyLayer, find_active_obstacle  # noqa: E402
from tools.g1a_soup_trace import make_env, body_state  # noqa: E402

SUITE = "safelibero_long"
LEVEL = "I"
TASK_ID = 0
IMAGE_SIZE = 224
REPLAN = 5
MAX_STEPS = 550
STATE_ID = 1
STREAM = 0
N_RUNS = 3


def run_once(policy, ts, task):
    env = make_env(ts, task, STATE_ID)
    layer = AegisSafetyLayer(env, filter_mode="euclidean")
    layer.reset(env.env._get_observations(), task.language)
    obs0 = env.env._get_observations()
    name = find_active_obstacle(env, obs0)
    protect = name + "_main"
    p0 = body_state(env, protect)["pos"].copy()
    max_disp = 0.0
    success = False
    first_chunk_hash = None
    for step in range(MAX_STEPS):
        obs = env.env._get_observations()
        if step % REPLAN == 0:
            q = step // REPLAN
            z0, _ = sample_z(nominal_seed(SUITE, TASK_ID, STATE_ID, STREAM, q))
            po = R.obs_to_policy_obs(obs, task.language, image_size=IMAGE_SIZE)
            chunk = np.clip(policy.generate(z0, po), -1, 1)
            if first_chunk_hash is None:
                first_chunk_hash = hash(chunk.tobytes())
        action = chunk[step % REPLAN]
        exec_a, info = layer.filter(action, obs)
        env.step(exec_a)
        disp = float(np.sum(np.abs(body_state(env, protect)["pos"] - p0)))
        max_disp = max(max_disp, disp)
        if env.check_success():
            success = True
            break
    env.close()
    return {"max_disp_mm": max_disp * 1000, "success": success,
            "steps": step + 1, "first_chunk_hash": first_chunk_hash}


def main():
    from libero.libero import benchmark
    ts = benchmark.get_benchmark_dict()[SUITE](safety_level=LEVEL)
    task = ts.get_task(TASK_ID)
    policy = PnsgPolicy()
    results = []
    for i in range(N_RUNS):
        r = run_once(policy, ts, task)
        results.append(r)
        print(f"run{i}: mug={r['max_disp_mm']:.1f}mm success={r['success']} "
              f"steps={r['steps']} first_chunk_hash={r['first_chunk_hash']}",
              flush=True)
    policy.close()
    disp = [r["max_disp_mm"] for r in results]
    print(f"\n=== 跨 run 方差（S{STATE_ID:02d}/st{STREAM} euclidean × {N_RUNS}）===")
    print(f"mug 位移: {[round(x,1) for x in disp]}")
    print(f"均值={np.mean(disp):.1f}mm  标准差={np.std(disp):.2f}mm  "
          f"极差={max(disp)-min(disp):.1f}mm")
    hashes = set(r["first_chunk_hash"] for r in results)
    print(f"first_chunk_hash 是否一致: {len(hashes)==1} ({hashes})")


if __name__ == "__main__":
    main()
