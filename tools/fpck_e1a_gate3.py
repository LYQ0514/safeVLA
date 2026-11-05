"""E1a 门槛 3：独立 goal evaluator 与 native success 的 20 条无干预 baseline 对照。

配置（裁决指定）：
  suite=safelibero_long, level=I, task0, init_state_id=0..19
  frozen π0.5, replan_steps=5, 无干预（直接执行 nominal）

每步记录：
  native_success、每个 goal clause truth、automaton mode/achieved_mask、
  q_acc/q_bad/q_unknown、首次 native/evaluator 分歧步。

输出：data/audit/fpck_e1a_gate3.json（20 条 episode 的逐步 + 汇总）
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
from src.fpck.goal_evaluator import eval_goal_clauses, native_success, goal_state_clauses, canonical_clause  # noqa: E402
from src.fpck.task_automaton import TaskAutomaton, TaskEvent, AutomatonState  # noqa: E402
from src.pnsg.role_manifest import build_role_manifest  # noqa: E402
from tools.g1a_soup_trace import make_env  # noqa: E402

SUITE = "safelibero_long"
LEVEL = "I"
TASK_ID = 0
IMAGE_SIZE = 224
REPLAN = 5
MAX_STEPS = 550
NUM_STEPS_WAIT = 20
N_EPISODES = 20  # init_state_id = 0..19


def make_env_local(ts, task, state_id):
    import pathlib
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256,
                             controller="OSC_POSE")
    env.seed(7)
    env.reset()
    env.set_init_state(ts.get_task_init_states(TASK_ID)[state_id])
    dummy = np.zeros(7, dtype=np.float32); dummy[-1] = -1.0
    for _ in range(NUM_STEPS_WAIT):
        env.step(dummy)
    return env


def main():
    from libero.libero import benchmark
    ts = benchmark.get_benchmark_dict()[SUITE](safety_level=LEVEL)
    task = ts.get_task(TASK_ID)
    prompt = task.language

    # 任务适配记录（manifest）：提取 manipulated_targets + goal clauses
    bddl = task.bddl_file
    import pathlib
    from libero.libero import get_libero_path
    bddl_path = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / bddl
    manifest = build_role_manifest(bddl_path)
    manipulated_targets = [inst for inst, info in manifest["objects"].items()
                           if "MANIPULATED_TARGET" in info["roles"]]
    # goal_keys 用 canonical_clause 统一规范化（与 eval_goal_clauses 的 key 一致）
    goal_keys = [canonical_clause(s) for s in manifest["goal_clauses"]]

    policy = PnsgPolicy()
    results = []

    for state_id in range(N_EPISODES):
        env = make_env_local(ts, task, state_id)
        obs = env.env._get_observations()
        auto = TaskAutomaton(manipulated_targets, goal_keys)

        steps_log = []
        first_divergence_step = None
        success_native_final = False
        chunk = None
        final_step = 0
        for step in range(MAX_STEPS):
            if step % REPLAN == 0:
                q = step // REPLAN
                z0, _ = sample_z(nominal_seed(SUITE, TASK_ID, state_id, 0, q))
                po = R.obs_to_policy_obs(obs, prompt, image_size=IMAGE_SIZE)
                chunk = np.clip(policy.generate(z0, po), -1, 1)[:REPLAN]
            action = chunk[step % REPLAN]
            # 无干预：直接执行 nominal
            obs, _r, _d, _i = env.step(action)

            native = native_success(env)
            clause_truth = eval_goal_clauses(env)
            ev = TaskEvent(
                step=step,
                goal_clause_truth=clause_truth,
                native_success=native,
            )
            auto.process(ev)
            mode = auto.state.mode

            # 首次分歧：native success 与独立 evaluator 全部 clause 为 true 不一致
            evaluator_all_true = all(v is True for v in clause_truth.values()) if clause_truth else False
            if first_divergence_step is None and (native != evaluator_all_true):
                first_divergence_step = step

            steps_log.append({
                "step": step,
                "native_success": native,
                "clause_truth": clause_truth,
                "mode": mode,
                "achieved_mask": dict(auto.state.achieved_mask),
            })
            if native:
                success_native_final = True
                final_step = step
                break
            final_step = step

        results.append({
            "state_id": state_id,
            "native_success": success_native_final,
            "final_step": final_step,
            "first_divergence_step": first_divergence_step,
            "final_mode": auto.state.mode,
            "final_achieved_mask": dict(auto.state.achieved_mask),
            "steps_log": steps_log,
        })
        env.close()
        print(f"state_id={state_id:2d}: native_success={success_native_final} "
              f"final_mode={auto.state.mode} first_divergence={first_divergence_step}",
              flush=True)

    policy.close()

    out = os.path.join(ROOT, "data", "audit", "fpck_e1a_gate3.json")
    with open(out, "w") as f:
        json.dump({"manipulated_targets": manipulated_targets,
                   "goal_clauses": manifest["goal_clauses"],
                   "episodes": results}, f, ensure_ascii=False, indent=2)

    # 汇总
    n_div = sum(1 for r in results if r["first_divergence_step"] is not None)
    print(f"\n=== 门槛 3 汇总 ===")
    print(f"episodes: {len(results)}")
    print(f"native success: {sum(1 for r in results if r['native_success'])}/{len(results)}")
    print(f"有 native/evaluator 分歧的 episode: {n_div}/{len(results)}")
    print(f"结果: {out}")


if __name__ == "__main__":
    main()
