"""E1a 门禁 1：透明采集等价（collector 开/关动作逐步一致）。

配置：long/I/task0, init 0-4, replan_steps=5, frozen π0.5。
要求：collector 开启与关闭，动作逐步 max-abs ≤ 1e-7，且 native success/步数/
CAR/chunk hash/buffer index 序列一致，不增加推理次数、不改变随机流、不额外 env.step。

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
from src.fpck.collector import Collector, default_anchor  # noqa: E402
from src.safety.aegis_adapted import find_active_obstacle  # noqa: E402
from tools.g1a_soup_trace import body_state  # noqa: E402

SUITE = "safelibero_long"
LEVEL = "I"
TASK_ID = 0
IMAGE_SIZE = 224
REPLAN = 5
MAX_STEPS = 550
NUM_STEPS_WAIT = 20
N_EP = 5  # init 0-4
SPEC_VERSION = "fpck-task-automaton-v1.2"


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


def run_episode(policy, ts, task, state_id, use_collector, log_path):
    """跑一条 episode，返回 (actions_list, chunk_hash_seq, buffer_idx_seq, success, steps, car_disp)"""
    env = make_env_local(ts, task, state_id)
    obs = env.env._get_observations()
    anchor = default_anchor(SUITE, LEVEL, TASK_ID, state_id, 0, task.language,
                            "manifest", "contract", SPEC_VERSION, REPLAN)
    collector = Collector(policy, anchor, SUITE, TASK_ID, state_id, 0,
                          IMAGE_SIZE, REPLAN, log_path) if use_collector else None

    actions = []
    chunk_hashes = []
    buffer_indices = []
    success = False
    steps = 0
    obstacle = find_active_obstacle(env, obs)
    p0 = body_state(env, obstacle + "_main")["pos"].copy() if obstacle else None
    max_disp = 0.0
    n_generate = 0
    chunk = None
    chunk_q = 0
    buf_idx = 0

    for step in range(MAX_STEPS):
        if use_collector:
            if collector.chunk is None or collector.cursor.buffer_index >= len(collector.chunk):
                collector.generate_chunk(obs, task.language)
                n_generate += 1
            act = collector.nominal_action(obs, task.language)
            chunk_hashes.append(collector.chunk.tobytes().hex()[:16])
            buffer_indices.append(collector.cursor.buffer_index - 1)
        else:
            # 与 collector 相同节奏：只在 chunk 耗尽时生成（透明对照）
            if chunk is None or buf_idx >= len(chunk):
                q = chunk_q
                z0, _ = sample_z(nominal_seed(SUITE, TASK_ID, state_id, 0, q))
                po = R.obs_to_policy_obs(obs, task.language, image_size=IMAGE_SIZE)
                chunk = np.clip(policy.generate(z0, po), -1, 1)[:REPLAN]
                chunk_q += 1
                buf_idx = 0
                n_generate += 1
            act = chunk[buf_idx]
            buf_idx += 1
            chunk_hashes.append(chunk.tobytes().hex()[:16])
            buffer_indices.append(buf_idx - 1)

        actions.append(act)
        obs, _r, _d, _i = env.step(act)
        steps += 1
        if p0 is not None:
            max_disp = max(max_disp, float(np.sum(np.abs(body_state(env, obstacle + "_main")["pos"] - p0))))
        if env.check_success():
            success = True
            break

    env.close()
    return actions, chunk_hashes, buffer_indices, success, steps, max_disp, n_generate


def main():
    from libero.libero import benchmark
    ts = benchmark.get_benchmark_dict()[SUITE](safety_level=LEVEL)
    task = ts.get_task(TASK_ID)
    policy = PnsgPolicy()

    results = []
    all_pass = True
    for state_id in range(N_EP):
        # 关 collector
        a_off, h_off, b_off, s_off, st_off, d_off, g_off = run_episode(
            policy, ts, task, state_id, False, None)
        # 开 collector（用临时 log）
        log_path = f"/tmp/fpck_gate1_s{state_id}.jsonl"
        if os.path.exists(log_path):
            os.remove(log_path)
        a_on, h_on, b_on, s_on, st_on, d_on, g_on = run_episode(
            policy, ts, task, state_id, True, log_path)

        # 对比
        max_act_err = max(float(np.max(np.abs(x - y)))
                          for x, y in zip(a_off, a_on)) if a_off else 0.0
        same_len = len(a_off) == len(a_on)
        same_act = max_act_err <= 1e-7
        same_success = s_off == s_on
        same_steps = st_off == st_on
        same_disp = abs(d_off - d_on) <= 1e-7
        same_hash = h_off == h_on
        same_buf = b_off == b_on
        same_gen = g_off == g_on  # 推理次数

        ok = (same_len and same_act and same_success and same_steps and same_disp
              and same_hash and same_buf and same_gen)
        all_pass = all_pass and ok
        results.append({
            "state_id": state_id,
            "max_act_err": max_act_err,
            "same_len": same_len, "same_act": same_act,
            "same_success": same_success, "same_steps": same_steps,
            "same_disp": same_disp, "same_hash": same_hash,
            "same_buf": same_buf, "same_gen": same_gen,
            "ok": ok,
        })
        print(f"state_id={state_id}: act_err={max_act_err:.2e} "
              f"success={same_success} steps={same_steps} hash={same_hash} "
              f"buf={same_buf} gen={same_gen} → {'PASS' if ok else 'FAIL'}",
              flush=True)

    policy.close()

    out = os.path.join(ROOT, "data", "audit", "fpck_e1a_gate1.json")
    with open(out, "w") as f:
        json.dump({"results": results, "all_pass": all_pass}, f, ensure_ascii=False, indent=2)
    print(f"\n=== 门禁 1 判定: {'PASS' if all_pass else 'BLOCK'} ===")
    print(f"结果: {out}")


if __name__ == "__main__":
    main()
