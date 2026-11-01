"""AEGIS long 最小采集器（设计采集，非方法性能证据）。

用途：判断 long 上是否存在值得解决的推进问题（AEGIS「安全但无用」失效模式
是否真实可观测）。成本上限冻结：3 个 episode 起步，最多 6 个，或累计 3 GPU 小时。

采集对象：long task0（子目标可直接观测）。方法：当前原始 AEGIS。

每个重规划决策点保存：
- 完整 snapshot、实际 deployment obs、z0、nominal/AEGIS 动作及哈希
- 两个原生子目标 predicate 的真值（用环境 _eval_predicate，非物体中心近似）
- 目标物体位置、抓持状态、阶段内连续距离
- AEGIS 是否介入、修正幅度、QP 状态
- 成功/停止/失败原因、policy/过滤/记录耗时

结果身份：设计采集，不作为方法性能证据；历史 summary 不用于挑有利 episode。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
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

from src.pnsg.crn import nominal_seed, sample_z, z_sha256, action_sha256  # noqa: E402
from src.pnsg.full_state import save_full_state  # noqa: E402
from src.pnsg.policy_adapter import PnsgPolicy  # noqa: E402
from src.safety.aegis_adapted import AegisSafetyLayer  # noqa: E402

SUITE = "safelibero_long"
LEVEL = "I"
TASK_ID = 0
REPLICATE_ID = 0
NUM_STEPS_WAIT = 10
REPLAN_STEPS = 5          # AEGIS 每 5 步重规划（与历史评测一致）
MAX_STEPS = 550           # long 套件上限（沿用现有评测）
IMAGE_SIZE = 224          # 策略输入分辨率

AUDIT = pathlib.Path(ROOT, "data", "audit")
OUT_ROOT = AUDIT / "aegis_long_design"

# 硬成本上限
MAX_EPISODES = 6
GPU_HOUR_BUDGET = 3.0 * 3600.0


def sha(arr):
    return hashlib.sha256(np.asarray(arr).tobytes()).hexdigest()


def make_env(ts, task, state_id):
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


def subgoal_values(env):
    """用环境原生 _eval_predicate 求每个子目标真值（不预设顺序）。"""
    inner = env.env
    gs = inner.parsed_problem["goal_state"]
    out = []
    for st in gs:
        pred = st[0]
        obj = st[1] if len(st) >= 2 else None
        out.append({
            "predicate": pred,
            "args": st[1:],
            "value": bool(inner._eval_predicate(st)),
        })
    return out


def grasp_state(env, obs):
    """抓持状态：目标物体是否被夹爪抓取（近似：目标物体 z 高于桌面 + 接近 eef）。"""
    # 用 gripper 开合 + 目标物体位置粗判：抓持 = 夹爪闭合 且 物体靠近 eef 且离桌面
    gq = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64)
    gripper_closed = bool(np.mean(gq) < -0.5)  # qpos 负 = 闭合（LIBERO 语义）
    return {"gripper_qpos": gq.tolist(), "gripper_closed": gripper_closed}


def episode_dirs():
    return sorted([p for p in (OUT_ROOT).glob("ep*") if p.is_dir()])


def run_episode(ep, ts, task, policy, out_root, log_cb=None):
    """跑一个 episode，在每个重规划决策点记录完整状态。返回 episode 记录。"""
    env = make_env(ts, task, ep)
    layer = AegisSafetyLayer(env)

    ep_dir = out_root / f"ep{ep:02d}"
    ep_dir.mkdir(parents=True, exist_ok=True)

    obs = env.env._get_observations()
    prompt = task.language

    # 初始障碍物定位
    layer.reset(obs, prompt)

    decisions = []  # 每个重规划决策点的记录
    subgoal_history = []

    t_start = time.time()
    done = False
    success = False
    final_reason = "max_steps"

    for step in range(MAX_STEPS):
        is_replan = (step % REPLAN_STEPS == 0)

        if is_replan:
            # --- 决策点：保存完整状态（不做 filter，避免重复推进 AEGIS 状态）---
            t_pol0 = time.time()
            snapshot = save_full_state(env)
            obs_deploy = dict(obs)  # deployment obs = env.step 返回的缓存
            z0, _ = sample_z(nominal_seed(SUITE, TASK_ID, ep, REPLICATE_ID, step // REPLAN_STEPS))
            po = R.obs_to_policy_obs(obs_deploy, prompt, image_size=IMAGE_SIZE)
            raw = policy.generate(z0, po)
            nominal = np.clip(raw, -1, 1)
            t_pol = time.time() - t_pol0

            sg = subgoal_values(env)
            subgoal_history.append({"step": step, "subgoals": sg})

            objs = {}
            for k in ("alphabet_soup_1", "cream_cheese_1", "basket_1"):
                if k + "_pos" in obs_deploy:
                    objs[k + "_pos"] = np.asarray(obs_deploy[k + "_pos"]).tolist()

            # 决策点保存 AEGIS 状态（filter 尚未在本决策点推进）
            aegis_state = layer.save_state()

            decisions.append({
                "step": step,
                "snapshot": snapshot,
                "obs": obs_deploy,
                "z0": z0,
                "z0_sha256": z_sha256(z0),
                "raw": raw,
                "nominal": nominal,
                "nominal_sha256": action_sha256(nominal),
                "subgoals": sg,
                "objects": objs,
                "grasp": grasp_state(env, obs_deploy),
                "aegis_state": aegis_state,
                "t_policy_s": t_pol,
            })

        # --- 执行当前动作（经 AEGIS 过滤；每个执行步只调用一次 filter）---
        t_f0 = time.time()
        act_exec, info_exec = layer.filter(nominal[step % REPLAN_STEPS], obs)
        t_filter = time.time() - t_f0
        # 记录本次 filter 的 info 到当前决策点（该决策点覆盖 REPLAN_STEPS 个执行步）
        # 首个执行步的 filter info 最有代表性，附到决策点记录
        if step % REPLAN_STEPS == 0:
            decisions[-1]["aegis_first_filter"] = {
                "action": act_exec.tolist(),
                "info": {k: v for k, v in info_exec.items() if k != "h"},
                "h": info_exec.get("h"),
                "t_filter_s": t_filter,
            }

        obs, _r, _d, _i = env.step(act_exec)

        if _d or env.check_success():
            success = True
            final_reason = "success"
            done = True
            break

    t_total = time.time() - t_start

    # 结束时的最终子目标状态
    final_subgoals = subgoal_values(env)

    ep_record = {
        "episode": ep,
        "success": success,
        "final_reason": final_reason,
        "steps": step + 1,
        "t_total_s": t_total,
        "final_subgoals": final_subgoals,
        "n_decisions": len(decisions),
        "aegis_stats": layer.stats(),
        "subgoal_history": subgoal_history,
    }
    env.close()

    # 落盘：decisions 里的 snapshot/obs/z0/raw/nominal 存 npz，其余存 json
    for i, d in enumerate(decisions):
        step = d["step"]
        npz_path = ep_dir / f"decision_{step:03d}.npz"
        obs_arrays = {k: np.asarray(v) for k, v in d["obs"].items() if v is not None}
        # AEGIS 状态里的 ndarray 字段展开成独立键保存
        aegis_state = d.get("aegis_state", {})
        aegis_arrays = {}
        for k, v in aegis_state.items():
            if isinstance(v, np.ndarray):
                aegis_arrays["aegis_" + k] = v
        np.savez(
            npz_path,
            snapshot=d["snapshot"]["flattened"],
            z0=d["z0"],
            raw=d["raw"],
            nominal=d["nominal"],
            **obs_arrays,
            **aegis_arrays,
        )
        # json 只存标量/哈希/轻量字段
        d_json = {k: v for k, v in d.items()
                  if k not in ("snapshot", "obs", "z0", "raw", "nominal", "aegis_state")}
        # aegis_state 里的标量字段进 json，数组字段已在 npz
        aegis_scalar = {k: (v if not isinstance(v, np.ndarray) else None)
                        for k, v in aegis_state.items()}
        d_json["aegis_state_scalar"] = aegis_scalar
        (ep_dir / f"decision_{step:03d}.json").write_text(
            json.dumps(d_json, ensure_ascii=False, indent=2, default=str))

    ep_record["decision_steps"] = [d["step"] for d in decisions]
    (ep_dir / "episode.json").write_text(
        json.dumps(ep_record, ensure_ascii=False, indent=2, default=str))

    if log_cb:
        log_cb(f"ep{ep:02d}: success={success} reason={final_reason} "
               f"steps={step+1} decisions={len(decisions)} t={t_total:.0f}s "
               f"aegis={layer.stats()}")
    return ep_record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-ep", type=int, default=0)
    ap.add_argument("--max-episodes", type=int, default=MAX_EPISODES)
    ap.add_argument("--gpu-hour-budget", type=float, default=3.0)
    args = ap.parse_args()

    from libero.libero import benchmark
    ts = benchmark.get_benchmark_dict()[SUITE](safety_level=LEVEL)
    task = ts.get_task(TASK_ID)
    policy = PnsgPolicy()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    budget_s = args.gpu_hour_budget * 3600.0

    print(f"=== AEGIS long 最小采集器（task0，上限 {args.max_episodes} episode / "
          f"{args.gpu_hour_budget} GPU 小时）===")

    records = []
    elapsed = 0.0
    try:
        for ep in range(args.start_ep, args.start_ep + args.max_episodes):
            if elapsed >= budget_s:
                print(f"\n已达 GPU 小时预算 {args.gpu_hour_budget}h，停止。")
                break
            print(f"\n--- episode {ep} ---")
            rec = run_episode(ep, ts, task, policy, OUT_ROOT,
                              log_cb=lambda m: print("  " + m))
            records.append(rec)
            elapsed += rec["t_total_s"]
    finally:
        policy.close()

    summary = {
        "suite": SUITE, "task_id": TASK_ID, "level": LEVEL,
        "replan_steps": REPLAN_STEPS, "max_steps": MAX_STEPS,
        "identity": "design_collection",  # 设计采集，非方法性能证据
        "gpu_hour_budget": args.gpu_hour_budget,
        "elapsed_s": elapsed,
        "n_episodes": len(records),
        "episodes": records,
    }
    (OUT_ROOT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    print(f"\n=== 采集完成：{len(records)} episode，耗时 {elapsed:.0f}s ===")
    print(f"汇总: {OUT_ROOT / 'summary.json'}")


if __name__ == "__main__":
    main()
