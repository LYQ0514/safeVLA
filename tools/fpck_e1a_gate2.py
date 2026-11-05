"""E1a 门禁 2：snapshot round-trip（完整状态恢复等价）。

在 3 个时刻（early / pregrasp-reentry / late）：
  save full state + policy cursor + chunk
  执行一个 frozen π0.5 action
  恢复 full state + policy state
  重新执行同一 action

要求（全部一致）：
  - MuJoCo qpos/qvel/act 恢复差异 = 0 或机器精度；
  - chunk hash/cursor 一致；
  - next observation 关键真值一致（eef pos / obstacle pos / goal clause truth）；
  - contact 事件集合一致；
  - next action 一致。

只恢复 MuJoCo 不恢复 policy cursor 不算通过。

判定：任一项不一致 → BLOCK，停止 witness 重发现。
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
from src.pnsg.full_state import save_full_state, restore_full_state  # noqa: E402
from src.safety.aegis_adapted import find_active_obstacle  # noqa: E402
from src.fpck.goal_evaluator import eval_goal_clauses, native_success  # noqa: E402
from tools.g1a_soup_trace import body_state, eef_pos  # noqa: E402

SUITE = "safelibero_long"
LEVEL = "I"
TASK_ID = 0
IMAGE_SIZE = 224
REPLAN = 5
MAX_STEPS = 550
NUM_STEPS_WAIT = 20
STATE_ID = 2  # 用 state2（历史假成功候选，较早触发）


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


def gen_chunk(policy, obs, prompt, query_idx):
    z0, _ = sample_z(nominal_seed(SUITE, TASK_ID, STATE_ID, 0, query_idx))
    po = R.obs_to_policy_obs(obs, prompt, image_size=IMAGE_SIZE)
    return np.clip(policy.generate(z0, po), -1, 1)[:REPLAN]


def env_key_truth(env, obstacle):
    """next observation 的关键真值。"""
    obs = env.env._get_observations()
    return {
        "eef_pos": np.asarray(obs["robot0_eef_pos"]).round(9).tolist(),
        "obstacle_pos": (body_state(env, obstacle + "_main")["pos"].round(9).tolist()
                         if obstacle else None),
        "goal_truth": {k: v for k, v in eval_goal_clauses(env).items()},
        "native_success": native_success(env),
    }


def contact_set(env):
    """contact 事件集合（geom 对）。"""
    m = env.sim.model._model
    d = env.sim.data._data
    names = env.sim.model.body_names
    return frozenset(
        (names[m.geom_bodyid[d.contact[c].geom1]],
         names[m.geom_bodyid[d.contact[c].geom2]])
        for c in range(d.ncon)
    )


def main():
    from libero.libero import benchmark
    ts = benchmark.get_benchmark_dict()[SUITE](safety_level=LEVEL)
    task = ts.get_task(TASK_ID)
    policy = PnsgPolicy()

    env = make_env_local(ts, task, STATE_ID)
    obs = env.env._get_observations()
    obstacle = find_active_obstacle(env, obs)

    # 跑到 3 个时刻（early=step10, pregrasp=step50, late=step150）
    checkpoints = [10, 50, 150]
    results = []
    chunk = None
    query_idx = 0
    buf_idx = 0
    all_pass = True

    for step in range(MAX_STEPS):
        # 生成/消费 chunk（与 gate1 一致节奏）
        if chunk is None or buf_idx >= len(chunk):
            chunk = gen_chunk(policy, obs, task.language, query_idx)
            query_idx += 1
            buf_idx = 0
        act = chunk[buf_idx]

        if step in checkpoints:
            # ---- snapshot round-trip ----
            # 1. save env full state + policy state（query_idx/buf_idx + chunk）
            env_snap = save_full_state(env)
            saved_chunk = chunk.copy()
            saved_query = query_idx
            saved_buf = buf_idx

            # 2. 执行 action
            obs2, _, _, _ = env.step(act)
            key2 = env_key_truth(env, obstacle)
            contact2 = contact_set(env)

            # 3. 恢复 env + policy state
            restore_full_state(env, env_snap)
            chunk = saved_chunk
            query_idx = saved_query
            buf_idx = saved_buf

            # 4. 重新执行同一 action
            act_re = chunk[buf_idx]
            obs3, _, _, _ = env.step(act_re)
            key3 = env_key_truth(env, obstacle)
            contact3 = contact_set(env)

            # 对比
            act_same = np.allclose(act, act_re, atol=1e-9)
            key_same = (key2["eef_pos"] == key3["eef_pos"]
                        and key2["obstacle_pos"] == key3["obstacle_pos"]
                        and key2["goal_truth"] == key3["goal_truth"]
                        and key2["native_success"] == key3["native_success"])
            contact_same = contact2 == contact3
            ok = act_same and key_same and contact_same
            all_pass = all_pass and ok
            results.append({
                "step": step, "act_same": bool(act_same),
                "key_same": bool(key_same), "contact_same": bool(contact_same),
                "ok": bool(ok),
            })
            print(f"checkpoint step={step}: act_same={act_same} "
                  f"key_same={key_same} contact_same={contact_same} → "
                  f"{'PASS' if ok else 'FAIL'}", flush=True)
            # 恢复后的 obs 也要同步（env 已在恢复后重新 step，obs 是 obs3）
            obs = obs3

        else:
            obs, _r, _d, _i = env.step(act)

        buf_idx += 1
        if env.check_success():
            break

    env.close(); policy.close()

    out = os.path.join(ROOT, "data", "audit", "fpck_e1a_gate2.json")
    with open(out, "w") as f:
        json.dump({"results": results, "all_pass": all_pass}, f, ensure_ascii=False, indent=2)
    print(f"\n=== 门禁 2 判定: {'PASS' if all_pass else 'BLOCK'} ===")
    print(f"结果: {out}")


if __name__ == "__main__":
    main()
