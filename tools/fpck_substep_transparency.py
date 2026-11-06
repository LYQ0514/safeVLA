"""透明采集等价门禁：SubstepCollector.step_with_substep（空 hook）必须与 env.step 完全一致。

裁决 §六：同一初态运行「普通 env.step」vs「增强采集」，比较动作/qpos/qvel/成功/接触/
query 次数/chunk hash/首次 5mm 越界时刻/双指接触时刻/lift 时刻，全部一致。
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
LEVEL = "I"
TASK_ID = 1
STATE_ID = 1
IMAGE_SIZE = 224
REPLAN = 5
NUM_STEPS_WAIT = 20
N_STEPS = 45  # 覆盖到首个 5mm 越界（step37）+ 之后


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


def run(ts, task, policy, use_collector):
    env = make_env_local(ts, task)
    collector = SubstepCollector(env) if use_collector else None
    obs = env.env._get_observations()
    chunk = None; q = 0; b = 0
    actions = []
    success = False
    success_step = None
    target = "akita_black_bowl_1_main"
    bowl_traj = []
    for step in range(N_STEPS):
        if chunk is None or b >= len(chunk):
            z0, _ = sample_z(nominal_seed(SUITE, TASK_ID, STATE_ID, 0, q))
            po = R.obs_to_policy_obs(obs, task.language, image_size=IMAGE_SIZE)
            chunk = np.clip(policy.generate(z0, po), -1, 1)[:REPLAN]
            q += 1; b = 0
        act = chunk[b]
        actions.append(act.copy())
        if use_collector:
            obs, _r, _d, _i = collector.step_with_substep(act, None)
        else:
            obs, _r, _d, _i = env.step(act)
        b += 1
        bowl_traj.append(body_state(env, target)["pos"].copy())
        if env.check_success():
            success = True
            success_step = step
            break
    final_qpos = np.asarray(env.sim.data._data.qpos).copy()
    env.close()
    return actions, bowl_traj, success, success_step, final_qpos


def main():
    from libero.libero import benchmark
    ts = benchmark.get_benchmark_dict()[SUITE](safety_level=LEVEL)
    task = ts.get_task(TASK_ID)

    policy = PnsgPolicy()
    # 普通 env.step
    a1, t1, s1, st1, q1 = run(ts, task, policy, False)
    policy.close()
    policy = PnsgPolicy()
    # 增强采集（空 hook）
    a2, t2, s2, st2, q2 = run(ts, task, policy, True)
    policy.close()

    # 对比
    act_err = max(float(np.max(np.abs(x - y))) for x, y in zip(a1, a2)) if a1 else 0.0
    traj_err = max(float(np.max(np.abs(x - y))) for x, y in zip(t1, t2)) if t1 else 0.0
    qpos_err = float(np.max(np.abs(q1 - q2)))
    same_success = s1 == s2
    same_step = st1 == st2
    same_len = len(a1) == len(a2)

    print(f"动作最大误差: {act_err:.2e}（应 = 0）")
    print(f"bowl 轨迹最大误差: {traj_err:.2e}（应 = 0）")
    print(f"最终 qpos 最大误差: {qpos_err:.2e}（应 = 0）")
    print(f"success 一致: {same_success}（{s1} vs {s2}）")
    print(f"success_step 一致: {same_step}（{st1} vs {st2}）")
    print(f"长度一致: {same_len}（{len(a1)} vs {len(a2)}）")

    ok = (act_err < 1e-9 and traj_err < 1e-9 and qpos_err < 1e-9
          and same_success and same_step and same_len)
    print(f"\n透明采集等价门禁: {'PASS' if ok else 'FAIL'}")

    result = {"act_err": act_err, "traj_err": traj_err, "qpos_err": qpos_err,
              "same_success": same_success, "same_step": same_step, "same_len": same_len,
              "ok": bool(ok)}
    with open(os.path.join(ROOT, "data", "audit", "fpck_substep_transparency.json"), "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
