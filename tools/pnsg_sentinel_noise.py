"""P1 修正：跨进程数值差异 sentinel 量化。

分两层测：
1. 固定输入推理误差：同状态、同观测、同 z，跨进程（不同 worker）的动作误差
2. 闭环累积差异：同 episode、同 z 调度，跨进程跑完整 rollout，看 CAR/机制/接触是否翻转

对应想法v11 第 1 节。
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import time
import warnings

warnings.filterwarnings("ignore")

ROOT = "/data/safeVLA"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "dataset", "SafeLIBERO", "safelibero"))
sys.path.insert(0, "/data/2608.SDL/MojocoProj/src/pi05")

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
import pi05_libero_runner as R  # noqa: E402

from src.pnsg.crn import nominal_seed, sample_z, z_sha256  # noqa: E402
from src.pnsg.policy_adapter import PnsgPolicy  # noqa: E402


def gpu_mem():
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
                                   "--format=csv,noheader,nounits"])
    return int(out.decode().strip())


def fixed_input_action_diff(n_trials=5):
    """层 1：固定输入推理误差（跨 worker）。"""
    z = sample_z(nominal_seed("safelibero_spatial", 0, 0, 0, 0))[0]
    obs = {
        "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/state": np.zeros(8, dtype=np.float32),
        "prompt": "test",
    }
    actions = []
    for _ in range(n_trials):
        p = PnsgPolicy()
        a = p.generate(z, obs)
        actions.append(a)
        p.close()
        time.sleep(2)
    ref = actions[0]
    max_abs = []
    rel_l2 = []
    for a in actions[1:]:
        max_abs.append(float(np.abs(a - ref).max()))
        rel_l2.append(float(np.linalg.norm(a - ref) / (np.linalg.norm(ref) + 1e-12)))
    return {
        "n_trials": n_trials,
        "z_sha256": z_sha256(z),
        "max_abs_error": float(np.max(max_abs)),
        "mean_abs_error": float(np.mean(max_abs)),
        "max_rel_l2": float(np.max(rel_l2)),
        "action_scale_hint": "external action 大致范围 [-1,1]",
    }


def closed_loop_diff():
    """层 2：闭环累积差异（同 episode 同 z 调度，跨进程跑 rollout）。"""
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from src.pnsg.contact_audit import ContactEventRecorder, ContactAggregator, load_role_manifest
    from src.eval_safelibero import active_obstacle

    manifest_all = load_role_manifest(os.path.join(ROOT, "data", "task_object_role_manifest.json"))
    ts = benchmark.get_benchmark_dict()["safelibero_spatial"](safety_level="I")
    task = ts.get_task(0)
    manifest = manifest_all["safelibero_spatial"][task.name]
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    episode_id = 0
    prompt = task.language

    def run_once():
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256,
                                 controller="OSC_POSE")
        env.seed(7)
        obs = env.reset()
        obs = env.set_init_state(ts.get_task_init_states(0)[episode_id])
        dummy = np.zeros(7, dtype=np.float32); dummy[-1] = -1.0
        for _ in range(10):
            obs, _, _, _ = env.step(dummy)
        obst = active_obstacle(env, obs)
        obst_initial = np.asarray(obs[obst + "_pos"], dtype=np.float64).copy()

        rec = ContactEventRecorder(env, role_manifest=manifest["objects"], fixtures=manifest["fixtures"])
        rec.record_initial_contacts()
        agg = ContactAggregator()

        p = PnsgPolicy()
        car_trigger = None
        peak_l1 = 0.0
        for step in range(60):
            query = step // 2
            if step % 2 == 0:
                z = sample_z(nominal_seed("safelibero_spatial", 0, episode_id, 0, query))[0]
                po = R.obs_to_policy_obs(obs, prompt, image_size=224)
                raw = p.generate(z, po)
                clipped = np.clip(raw, -1, 1)
            action = clipped[step % 2]
            obs, summary = rec.substep_step(action, control_step=step)
            agg.add_step(summary)
            curr = np.asarray(obs[obst + "_pos"], dtype=np.float64)
            l1 = float(np.sum(np.abs(curr - obst_initial)))
            peak_l1 = max(peak_l1, l1)
            if car_trigger is None and l1 > 0.001:
                car_trigger = step
        p.close()
        env.close()

        mech = set()
        for ev in agg.geometric_records:
            if ev["allowance"] in ("FORBIDDEN", "ENVIRONMENT_DISTURBANCE",
                                   "SECONDARY_DISTURBANCE", "SECONDARY_ENVIRONMENT_INTERACTION"):
                mech.add(ev["semantic_class"])
        return {"car_trigger": car_trigger, "peak_l1": peak_l1, "mechanisms": sorted(mech)}

    r1 = run_once()
    time.sleep(3)
    r2 = run_once()
    return {"run1": r1, "run2": r2,
            "car_flip": r1["car_trigger"] != r2["car_trigger"],
            "mechanism_flip": r1["mechanisms"] != r2["mechanisms"],
            "l1_diff": abs(r1["peak_l1"] - r2["peak_l1"])}


def main():
    print("=== 层 1：固定输入推理误差（跨 worker）===")
    l1 = fixed_input_action_diff(n_trials=5)
    for k, v in l1.items():
        print(f"  {k}: {v}")

    print("\n=== 层 2：闭环累积差异（跨进程）===")
    l2 = closed_loop_diff()
    for k, v in l2.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
