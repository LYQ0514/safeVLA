"""审计 B 重采集：spatial/task1/state1 的 physics-substep 原始真值。

用 SubstepCollector 重跑，在首个 5mm 越界（step 36-39）前后，每个 physics substep
记录：原始 contact pair、bowl pose/vel、夹爪关节 qpos/qvel。

输出：data/audit/fpck_substep_reaudit_spatial_t1_s1.jsonl（每 substep 一行）

重点回答：step 36-39 的 7.81mm 位移，是「夹爪闭合顺应」还是「手臂推动」。
"""

from __future__ import annotations
import json
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
from src.fpck.substep_collector import SubstepCollector  # noqa: E402
from tools.g1a_soup_trace import body_state, eef_pos  # noqa: E402

SUITE = "safelibero_spatial"
LEVEL = "I"
TASK_ID = 1
STATE_ID = 1
IMAGE_SIZE = 224
REPLAN = 5
NUM_STEPS_WAIT = 20
N_STEPS = 45
TARGET = "akita_black_bowl_1_main"


def make_env_local(ts, task):
    import pathlib
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256,
                             controller="OSC_POSE")
    env.seed(7)
    env.reset()
    env.set_init_state(ts.get_task_init_states(TASK_ID)[STATE_ID])
    dummy = np.zeros(7, dtype=np.float32); dummy[-1] = -1.0
    for _ in range(NUM_STEPS_WAIT):
        env.step(dummy)
    return env


def main():
    from libero.libero import benchmark
    ts = benchmark.get_benchmark_dict()[SUITE](safety_level=LEVEL)
    task = ts.get_task(TASK_ID)

    out_path = os.path.join(ROOT, "data", "audit",
                            "fpck_substep_reaudit_spatial_t1_s1.jsonl")
    out_f = open(out_path, "w", encoding="utf-8")

    env = make_env_local(ts, task)
    collector = SubstepCollector(env)
    obs = env.env._get_observations()
    policy = PnsgPolicy()

    settled_bowl = body_state(env, TARGET)["pos"].copy()
    chunk = None; q = 0; b = 0

    def hook(substep, snap):
        bowl = body_state(env, TARGET)
        ee = eef_pos(env)
        row = {
            "control_step": step,
            "physics_substep": substep,
            "bowl_pos": bowl["pos"].tolist(),
            "bowl_vel": bowl["vel"].tolist(),
            "bowl_disp_from_settled": float(np.sum(np.abs(bowl["pos"] - settled_bowl))) * 1000,
            "eef_pos": ee.tolist(),
            "gripper_qpos": snap["gripper"]["qpos"].tolist() if snap["gripper"]["qpos"] is not None else None,
            "gripper_qvel": snap["gripper"]["qvel"].tolist() if snap["gripper"]["qvel"] is not None else None,
            "contacts": snap["contacts"],
        }
        out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

    try:
        for step in range(N_STEPS):
            if chunk is None or b >= len(chunk):
                z0, _ = sample_z(nominal_seed(SUITE, TASK_ID, STATE_ID, 0, q))
                po = R.obs_to_policy_obs(obs, task.language, image_size=IMAGE_SIZE)
                chunk = np.clip(policy.generate(z0, po), -1, 1)[:REPLAN]
                q += 1; b = 0
            act = chunk[b]
            obs, _r, _d, _i = collector.step_with_substep(act, hook)
            b += 1
            if env.check_success():
                break
    finally:
        policy.close()
        out_f.close()
        env.close()

    print(f"结果: {out_path}")
    n = sum(1 for _ in open(out_path))
    print(f"总 substep 数: {n}")


if __name__ == "__main__":
    main()
