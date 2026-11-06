"""定位 substep 采集器 vs 原生 env.step 的轨迹分歧根因。

对比三种执行路径在 step36-37（首次接触附近）的 bowl 位移：
A. 原生 env.step
B. SubstepCollector.step_with_substep（含 _get_observations + snapshot_contacts + hook）
C. 纯复制循环（forward/_pre_action/step/_update_observables，无任何额外读取）

若 A==C 且 A!=B，则分歧来自 _get_observations/snapshot/hook 的读取；
若 A!=C 且 A!=B，则分歧来自循环结构本身（不太可能，结构一致）。
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
from tools.g1a_soup_trace import body_state  # noqa: E402

SUITE = "safelibero_spatial"
TASK_ID = 1
STATE_ID = 1
IMAGE_SIZE = 224
REPLAN = 5
NUM_STEPS_WAIT = 20
N_STEPS = 40
TARGET = "akita_black_bowl_1_main"


def make_env(ts, task):
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


def run(mode):
    """mode: 'native' | 'full' | 'pure'"""
    from libero.libero import benchmark
    ts = benchmark.get_benchmark_dict()[SUITE](safety_level="I")
    task = ts.get_task(TASK_ID)
    env = make_env(ts, task)
    collector = SubstepCollector(env)
    obs = env.env._get_observations()
    policy = PnsgPolicy()
    settled = body_state(env, TARGET)["pos"].copy()
    chunk = None; q = 0; b = 0
    disp = {}
    try:
        for step in range(N_STEPS):
            if chunk is None or b >= len(chunk):
                z0, _ = sample_z(nominal_seed(SUITE, TASK_ID, STATE_ID, 0, q))
                po = R.obs_to_policy_obs(obs, task.language, image_size=IMAGE_SIZE)
                chunk = np.clip(policy.generate(z0, po), -1, 1)[:REPLAN]
                q += 1; b = 0
            act = chunk[b]
            if mode == "native":
                obs, _r, _d, _i = env.step(act)
            elif mode == "full":
                obs, _r, _d, _i = collector.step_with_substep(act, lambda i, s: None)
            elif mode == "pure":
                # 纯复制循环，无 _get_observations / snapshot / hook
                e = env.env
                if e.done:
                    raise ValueError("terminated")
                e.timestep += 1
                ps = True
                for _ in range(collector.n_substeps):
                    e.sim.forward()
                    e._pre_action(act, ps)
                    e.sim.step()
                    e._update_observables()
                    ps = False
                e.cur_time += collector.control_timestep
                _r, _d, _i = e._post_action(act)
                obs = e._get_observations()
            b += 1
            disp[step] = float(np.sum(np.abs(body_state(env, TARGET)["pos"] - settled))) * 1000
    finally:
        policy.close()
        env.close()
    return disp


if __name__ == "__main__":
    results = {}
    for mode in ["native", "full", "pure"]:
        d = run(mode)
        results[mode] = d
        print(f"{mode}: step36={d[36]:.4f} step37={d[37]:.4f} step38={d[38]:.4f}")
    # 对比
    print()
    print(f"{'step':>4} {'native':>10} {'full':>10} {'pure':>10}")
    for s in [35, 36, 37, 38, 39]:
        print(f"{s:>4} {results['native'][s]:>10.4f} {results['full'][s]:>10.4f} {results['pure'][s]:>10.4f}")
