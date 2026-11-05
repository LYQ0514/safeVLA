"""G2.5 几何因果审计：任务物体豁免的受控诊断测试（独立分支，不进入部署算法）。

任务书 B.4/B.5/B.6：
- 在独立诊断分支测试仅针对可疑任务几何的受控修正（末端椭球「任务物体豁免」）；
- 检查真实抓取是否恢复，同时严格记录 forbidden contact / CAR / task integrity；
- 不得把去掉约束后产生的 unsafe success 当成有效收益。

诊断方法（受控、离线诊断）：
- grasp_aware 模式（AEGIS 已有）：接近阶段（夹爪开）用小椭球 [0.06,0.04,0.20]，
  抓住后（夹爪闭）用大椭球。小椭球 y 半轴 0.04 < 大椭球 0.12，接近 soup 时
  末端椭球不覆盖 soup，理论上允许接近。

这不是 STG 的方法，是「任务物体豁免」的雏形诊断，用于反事实验证：
如果小椭球能让 π0.5 抓住 soup，则证明「几何是根因」；如果仍不能，则几何非唯一根因。
"""

from __future__ import annotations
import csv
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
from src.safety.aegis_adapted import AegisSafetyLayer, find_active_obstacle  # noqa: E402
from src.safety.contract_objects import has_forbidden_contact  # noqa: E402
from tools.g1a_soup_trace import body_state  # noqa: E402

SUITE = "safelibero_long"
LEVEL = "I"
IMAGE_SIZE = 224
REPLAN = 5
MAX_STEPS = 550
NUM_STEPS_WAIT = 20

# 诊断病例：全部 S1 停滞（task1，soup 挡 mug）
CASES = os.path.join(ROOT, "experiments", "stg_v1", "outputs", "G2_FROZEN_CASES.csv")
OUT = os.path.join(ROOT, "experiments", "stg_v1", "outputs", "g2_5_geometry_diag.jsonl")


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


def run_one(policy, ts, task, task_id, state_id, stream, use_grasp_aware):
    env = make_env_local(ts, task, task_id, state_id)
    layer = AegisSafetyLayer(env, filter_mode="euclidean", stg_mode="B0",
                             grasp_aware=use_grasp_aware)
    layer.reset(env.env._get_observations(), task.language)

    obs0 = env.env._get_observations()
    obstacle_name = find_active_obstacle(env, obs0)
    protect = obstacle_name + "_main"
    p0 = body_state(env, protect)["pos"].copy()

    success = False
    chunk = None
    max_mug_disp = 0.0
    forbidden_steps = 0
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
        if has_forbidden_contact(env):
            forbidden_steps += 1
        if env.check_success():
            success = True
            break

    env.close()
    return {
        "task_id": task_id, "state_id": state_id, "stream": stream,
        "use_grasp_aware": use_grasp_aware,
        "success": success, "steps": step + 1,
        "max_mug_disp_mm": round(max_mug_disp * 1000, 2),
        "forbidden_steps": forbidden_steps,
        "obstacle": obstacle_name,
    }


def main():
    from libero.libero import benchmark
    ts = benchmark.get_benchmark_dict()[SUITE](safety_level=LEVEL)
    policy = PnsgPolicy()

    cases = list(csv.DictReader(open(CASES)))
    # 只取 S1 停滞病例（expected_mechanism == deadlock）
    s1_cases = [c for c in cases if c["expected_mechanism"] == "deadlock"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        for c in s1_cases:
            task_id = int(c["task_id"]); state_id = int(c["state_id"]); stream = int(c["stream"])
            task = ts.get_task(task_id)
            for use_grasp_aware in [False, True]:
                r = run_one(policy, ts, task, task_id, state_id, stream, use_grasp_aware)
                f.write(json.dumps(r) + "\n")
                f.flush()
                print(f"task{task_id}/s{state_id}/st{stream} "
                      f"{'grasp_aware' if use_grasp_aware else 'B0'} : "
                      f"succ={int(r['success'])} steps={r['steps']} mug={r['max_mug_disp_mm']}mm "
                      f"forbidden={r['forbidden_steps']}", flush=True)
    policy.close()
    print(f"\n结果: {OUT}")


if __name__ == "__main__":
    main()
