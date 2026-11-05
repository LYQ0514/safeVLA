"""G2.5 λ 消融：B1/B2 在 λ=0.1/0.5 下的表现，与 λ=1.0 及同进程 B0 对照。

关键（因果/复现纪律）：
- 同一进程、同 worker、同 seed 跑 B0（重跑）+ B1(λ=0.1/0.5) + B2(λ=0.1/0.5)；
- 病例 = 9 S1 停滞 + 3 成功负对照（12 例）；
- 方法序列：B0, B1_lam0.1, B1_lam0.5, B2_lam0.1, B2_lam0.5（5 方法 × 12 = 60 ep）；
- λ=1.0 的既有结果来自 g2_paired_episodes.jsonl（不同进程），单独记录可比性。

输出：experiments/stg_v1/outputs/g2_5_lambda_ablation.jsonl
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

# 方法配置：(method_key, stg_mode, lam_max)
METHODS = [
    ("B0", "B0", 1.0),
    ("B1_lam0.1", "B1", 0.1),
    ("B1_lam0.5", "B1", 0.5),
    ("B2_lam0.1", "B2", 0.1),
    ("B2_lam0.5", "B2", 0.5),
]

CASES = os.path.join(ROOT, "experiments", "stg_v1", "outputs", "G2_FROZEN_CASES.csv")
OUT = os.path.join(ROOT, "experiments", "stg_v1", "outputs", "g2_5_lambda_ablation.jsonl")


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


def run_one(policy, ts, task, task_id, state_id, stream, method_key, stg_mode, lam):
    env = make_env_local(ts, task, task_id, state_id)
    layer = AegisSafetyLayer(env, filter_mode="euclidean", stg_mode=stg_mode,
                             stg_lam_max=lam)
    layer.reset(env.env._get_observations(), task.language)

    obs0 = env.env._get_observations()
    obstacle_name = find_active_obstacle(env, obs0)
    protect = obstacle_name + "_main"
    p0 = body_state(env, protect)["pos"].copy()

    success = False
    chunk = None
    max_mug_disp = 0.0
    forbidden_steps = 0
    n_intervene = 0
    risk_steps = 0
    n_stg_lam_pos = 0
    hs = []
    exec_norms = []
    nom_norms = []
    for step in range(MAX_STEPS):
        obs = env.env._get_observations()
        if step % REPLAN == 0:
            q = step // REPLAN
            z0, _ = sample_z(nominal_seed(SUITE, task_id, state_id, stream, q))
            po = R.obs_to_policy_obs(obs, task.language, image_size=IMAGE_SIZE)
            chunk = np.clip(policy.generate(z0, po), -1, 1)[:REPLAN]
        action = chunk[step % REPLAN]
        exec_a, info = layer.filter(action, obs)
        env.step(exec_a)

        mug_disp = float(np.sum(np.abs(body_state(env, protect)["pos"] - p0)))
        max_mug_disp = max(max_mug_disp, mug_disp)
        if info.get("intervened"):
            n_intervene += 1
        if info.get("stg_in_risk"):
            risk_steps += 1
        if info.get("stg_lambda", 0) > 0:
            n_stg_lam_pos += 1
        if has_forbidden_contact(env):
            forbidden_steps += 1
        hs.append(float(info.get("h", 0.0)))
        exec_norms.append(float(np.linalg.norm(np.asarray(exec_a)[:3])))
        nom_norms.append(float(np.linalg.norm(np.asarray(action)[:3])))
        if env.check_success():
            success = True
            break

    env.close()
    return {
        "task_id": task_id, "state_id": state_id, "stream": stream,
        "method": method_key, "stg_mode": stg_mode, "lam": lam,
        "success": success, "steps": step + 1,
        "max_mug_disp_mm": round(max_mug_disp * 1000, 2),
        "forbidden_steps": forbidden_steps,
        "n_intervene": n_intervene,
        "risk_steps": risk_steps,
        "stg_lam_pos_steps": n_stg_lam_pos,
        "h_mean": round(float(np.mean(hs)), 6),
        "h_lt_0.01_ratio": round(float(np.mean(np.array(hs) < 0.01)), 4),
        "exec_action_mean_norm": round(float(np.mean(exec_norms)), 4),
        "nom_action_mean_norm": round(float(np.mean(nom_norms)), 4),
        "obstacle": obstacle_name,
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
            for method_key, stg_mode, lam in METHODS:
                r = run_one(policy, ts, task, task_id, state_id, stream,
                            method_key, stg_mode, lam)
                f.write(json.dumps(r) + "\n")
                f.flush()
                print(f"case task{task_id}/s{state_id}/st{stream} {method_key}: "
                      f"succ={int(r['success'])} steps={r['steps']} mug={r['max_mug_disp_mm']}mm "
                      f"risk={r['risk_steps']} exec_norm={r['exec_action_mean_norm']}",
                      flush=True)
    policy.close()
    print(f"\n结果: {OUT}")


if __name__ == "__main__":
    main()
