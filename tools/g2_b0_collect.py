"""G2 B0 基线病例收集：long task0/1 × 8 init × 2 stream = 32 episode。

记录每步机制诊断字段（任务书 §6），供离线冻结病例选择（S1-S4）。

输出：experiments/stg_v1/outputs/g2_b0_episodes.jsonl（每行一个 episode，含 steps 列表）
"""

from __future__ import annotations
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
TASKS = [0, 1]
INIT_STATES = list(range(8))
STREAMS = [0, 1]
IMAGE_SIZE = 224
REPLAN = 5
MAX_STEPS = 550
MODE = "B0"
NUM_STEPS_WAIT = 20

TASK_BODIES = {0: ("alphabet_soup_1_main", "cream_cheese_1_main"),
               1: ("alphabet_soup_1_main", "tomato_sauce_1_main")}


def make_env_local(ts, task, task_id, state_id):
    """正确的 env 构造：用传入 task 的 init state（不依赖 g1a_soup_trace 的全局 TASK_ID）。"""
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


def run_episode(policy, ts, task, task_id, state_id, stream):
    env = make_env_local(ts, task, task_id, state_id)
    layer = AegisSafetyLayer(env, filter_mode="euclidean", stg_mode=MODE)
    layer.reset(env.env._get_observations(), task.language)

    obs0 = env.env._get_observations()
    obstacle_name = find_active_obstacle(env, obs0)
    protect = obstacle_name + "_main"
    p0 = body_state(env, protect)["pos"].copy()
    task_bodies = TASK_BODIES.get(task_id, ("alphabet_soup_1_main",))
    task_p0 = {b: body_state(env, b)["pos"].copy() for b in task_bodies if b in env.sim.model.body_names}

    steps_log = []
    success = False
    chunk = None
    chunk_hash = None
    max_mug_disp = 0.0
    forbidden_steps = 0
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
        if has_forbidden_contact(env):
            forbidden_steps += 1

        steps_log.append({
            "step": step,
            "chunk_index": step % REPLAN,
            "chunk_hash": chunk_hash,
            "eef_pos": eef_pos(env).tolist(),
            "mug_pos": body_state(env, protect)["pos"].tolist(),
            "mug_disp_mm": round(mug_disp * 1000, 4),
            "h": round(float(info.get("h", 0.0)), 6),
            "intervened": bool(info.get("intervened", False)),
            "feasible": bool(info.get("feasible", False)),
            "nominal_7d": action.tolist(),
            "exec_7d": exec_a.tolist(),
            "qp_ms": round(qp_ms, 3),
            "gripper_qpos": [round(float(x), 4) for x in obs["robot0_gripper_qpos"]],
        })
        if env.check_success():
            success = True
            break

    # 任务物体最终位移
    task_final = {b: round(float(np.sum(np.abs(body_state(env, b)["pos"] - task_p0[b])))*1000, 2)
                  for b in task_p0}

    env.close()
    return {
        "task_id": task_id, "state_id": state_id, "stream": stream, "mode": MODE,
        "success": success, "steps": step + 1,
        "max_mug_disp_mm": round(max_mug_disp * 1000, 2),
        "forbidden_steps": forbidden_steps,
        "obstacle": obstacle_name,
        "task_final_disp_mm": task_final,
        "steps_log": steps_log,
    }


def main():
    from libero.libero import benchmark
    ts = benchmark.get_benchmark_dict()[SUITE](safety_level=LEVEL)
    policy = PnsgPolicy()

    out = os.path.join(ROOT, "experiments", "stg_v1", "outputs", "g2_b0_episodes.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        for task_id in TASKS:
            task = ts.get_task(task_id)
            for state_id in INIT_STATES:
                for stream in STREAMS:
                    r = run_episode(policy, ts, task, task_id, state_id, stream)
                    f.write(json.dumps(r) + "\n")
                    f.flush()
                    print(f"task{task_id}/s{state_id}/st{stream}: "
                          f"success={r['success']} steps={r['steps']} "
                          f"mug={r['max_mug_disp_mm']}mm forbidden={r['forbidden_steps']}",
                          flush=True)
    policy.close()
    print(f"\n结果: {out}")


if __name__ == "__main__":
    main()
