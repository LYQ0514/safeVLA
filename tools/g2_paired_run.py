"""G2 机制鉴别配对执行：冻结病例 × B0/B1/B2/B3。

对每个冻结病例 (task, init, stream)，在同一进程内依次跑 B0/B1/B2/B3：
- 相同 CRN seed（PnsgPolicy 确定性 + 同 seed → 首 chunk 一致）；
- 后续各方法独立闭环重规划（filter 不同 → obs 分叉）；
- 记录每步机制字段（含 STG 状态 reference_r / tangent / lambda）。

输出：experiments/stg_v1/outputs/g2_paired_episodes.jsonl
"""

from __future__ import annotations
import csv
import json
import os
import sys
import time
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
from src.safety.contract_objects import has_forbidden_contact  # noqa: E402
from tools.g1a_soup_trace import body_state, eef_pos  # noqa: E402

SUITE = "safelibero_long"
LEVEL = "I"
IMAGE_SIZE = 224
REPLAN = 5
MAX_STEPS = 550
NUM_STEPS_WAIT = 20
METHODS = ["B0", "B1", "B2", "B3"]

CASES = os.path.join(ROOT, "experiments", "stg_v1", "outputs", "G2_FROZEN_CASES.csv")
OUT = os.path.join(ROOT, "experiments", "stg_v1", "outputs", "g2_paired_episodes.jsonl")

TASK_BODIES = {0: ("alphabet_soup_1_main", "cream_cheese_1_main"),
               1: ("alphabet_soup_1_main", "tomato_sauce_1_main")}


def make_env_local(ts, task, task_id, state_id):
    import pathlib
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256,
                             controller="OSC_POSE")
    env.seed(7)
    env.reset()
    env.set_init_state(ts.get_task_init_states(task_id)[state_id])
    dummy = np.zeros(7, dtype=np.float32); dummy[-1] = -1.0
    for _ in range(NUM_STEPS_WAIT):
        env.step(dummy)
    return env


def run_one(policy, ts, task, task_id, state_id, stream, method):
    env = make_env_local(ts, task, task_id, state_id)
    layer = AegisSafetyLayer(env, filter_mode="euclidean", stg_mode=method)
    layer.reset(env.env._get_observations(), task.language)

    obs0 = env.env._get_observations()
    obstacle_name = find_active_obstacle(env, obs0)
    protect = obstacle_name + "_main"
    p0 = body_state(env, protect)["pos"].copy()

    steps_log = []
    success = False
    chunk = None
    chunk_hash = None
    max_mug_disp = 0.0
    forbidden_steps = 0
    n_intervene = 0
    for step in range(MAX_STEPS):
        obs = env.env._get_observations()
        if step % REPLAN == 0:
            q = step // REPLAN
            z0, _ = sample_z(nominal_seed(SUITE, task_id, state_id, stream, q))
            po = R.obs_to_policy_obs(obs, task.language, image_size=IMAGE_SIZE)
            chunk = np.clip(policy.generate(z0, po), -1, 1)[:REPLAN]
            chunk_hash = hash(chunk.tobytes())
        action = chunk[step % REPLAN]
        t0 = time.perf_counter()
        exec_a, info = layer.filter(action, obs)
        qp_ms = (time.perf_counter() - t0) * 1e3
        env.step(exec_a)

        mug_disp = float(np.sum(np.abs(body_state(env, protect)["pos"] - p0)))
        max_mug_disp = max(max_mug_disp, mug_disp)
        if info.get("intervened"):
            n_intervene += 1
        if has_forbidden_contact(env):
            forbidden_steps += 1

        steps_log.append({
            "step": step,
            "chunk_hash": chunk_hash,
            "eef_pos": eef_pos(env).tolist(),
            "mug_disp_mm": round(mug_disp * 1000, 4),
            "h": round(float(info.get("h", 0.0)), 6),
            "intervened": bool(info.get("intervened", False)),
            "feasible": bool(info.get("feasible", False)),
            "nominal_7d": action.tolist(),
            "exec_7d": exec_a.tolist(),
            "qp_ms": round(qp_ms, 3),
            "stg_lambda": float(info.get("stg_lambda", 0.0)),
            "stg_in_risk": bool(info.get("stg_in_risk", False)),
        })
        if env.check_success():
            success = True
            break

    env.close()
    return {
        "task_id": task_id, "state_id": state_id, "stream": stream, "method": method,
        "success": success, "steps": step + 1,
        "max_mug_disp_mm": round(max_mug_disp * 1000, 2),
        "forbidden_steps": forbidden_steps,
        "n_intervene": n_intervene,
        "obstacle": obstacle_name,
        "steps_log": steps_log,
    }


def main():
    from libero.libero import benchmark
    ts = benchmark.get_benchmark_dict()[SUITE](safety_level=LEVEL)
    policy = PnsgPolicy()

    cases = list(csv.DictReader(open(CASES)))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        for c in cases:
            task_id = int(c["task_id"]); state_id = int(c["state_id"]); stream = int(c["stream"])
            task = ts.get_task(task_id)
            for method in METHODS:
                r = run_one(policy, ts, task, task_id, state_id, stream, method)
                f.write(json.dumps(r) + "\n")
                f.flush()
                print(f"case task{task_id}/s{state_id}/st{stream} {method}: "
                      f"success={r['success']} steps={r['steps']} "
                      f"mug={r['max_mug_disp_mm']}mm interv={r['n_intervene']}",
                      flush=True)
    policy.close()
    print(f"\n结果: {OUT}")


if __name__ == "__main__":
    main()
