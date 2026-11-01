"""Source steering + AEGIS 批量评测。

在「源空间 steering 提案 → AEGIS CBF-QP 过滤」串接下跑 SafeLIBERO 套件，
输出与 AEGIS 历史评测同口径的四指标（TSR/CAR/CAR>5mm/SSR/ETS）。

口径与 `run_eval_aegis.py` 一致（CollisionTracker 官方 1mm 位移、replan=5、
1024 分辨率对应的策略输入 224、DEFAULT_MAX_STEPS）。区别仅在策略外壳：
π0.5 动作经源空间 steering（危险时）后再过 AEGIS。

用法：
    python tools/run_eval_source_aegis.py --suite safelibero_spatial --level I \
        --tasks 0 1 2 3 --episodes 0..49 --out data/eval/source_aegis_spatial
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
import warnings

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset" / "SafeLIBERO" / "safelibero"))
sys.path.insert(0, "/data/2608.SDL/MojocoProj/src/pi05")

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np  # noqa: E402

DEFAULT_MAX_STEPS = {
    "safelibero_spatial": 300,
    "safelibero_object": 300,
    "safelibero_goal": 300,
    "safelibero_long": 550,
}


def run_one(env, ts, task, task_id, episode, policy, max_steps):
    """跑单个 episode，复用统一口径。"""
    from src.eval_safelibero import CollisionTracker

    prompt = task.language
    env.reset()
    env.set_init_state(ts.get_task_init_states(task_id)[episode])

    dummy = np.zeros(7, dtype=np.float32); dummy[-1] = -1.0
    for _ in range(10):
        env.step(dummy)

    obs = env.env._get_observations()
    policy.reset(obs, prompt, episode_idx=episode)
    tracker = CollisionTracker(env, obs)

    success = False
    steps = 0
    t0 = time.time()
    for step in range(max_steps):
        action = policy.act(obs, prompt)
        obs, _r, done, _i = env.step(action)
        tracker.update(obs, step)
        steps += 1
        if done or env.check_success():
            success = True
            break

    return {
        "task_id": task_id,
        "task_language": prompt,
        "episode_idx": episode,
        "success": success,
        "steps": steps,
        "duration_s": time.time() - t0,
        "collided": tracker.collided,
        "collision_step": tracker.collision_step,
        "obstacle": tracker.name,
        "max_displacement": tracker.max_displacement,
        "source_aegis": policy.stats(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="safelibero_spatial")
    ap.add_argument("--level", default="I")
    ap.add_argument("--tasks", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--episodes", type=int, nargs="+", default=list(range(50)))
    ap.add_argument("--replan-steps", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-episodes", type=int, default=None)
    args = ap.parse_args()

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    from src.eval_safelibero import save_results, summarize
    from src.safety.source_aegis import SourceAegisPolicy

    ts = benchmark.get_benchmark_dict()[args.suite](safety_level=args.level)
    task_suite = ts

    out_dir = pathlib.Path(args.out or (ROOT / "data" / "eval" / f"source_aegis_{args.suite}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    done_keys = set()
    all_results = []
    if args.resume and (out_dir / "summary.json").exists():
        prev = json.loads((out_dir / "summary.json").read_text())
        all_results = prev.get("episodes", [])
        for e in all_results:
            done_keys.add((e["task_id"], e["episode_idx"]))
        print(f"[source_aegis] 续跑：已有 {len(done_keys)} 条")

    # 每个 task 一个 env（不同 bddl）
    envs = {}
    policies = {}

    ran = 0
    try:
        for task_id in args.tasks:
            task = task_suite.get_task(task_id)
            bddl = (pathlib.Path(get_libero_path("bddl_files"))
                    / task.problem_folder / task.bddl_file)
            env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256,
                                     camera_widths=256, controller="OSC_POSE")
            env.seed(7)
            policy = SourceAegisPolicy(env, args.suite, task_id,
                                       replan_steps=args.replan_steps)
            envs[task_id] = env
            policies[task_id] = policy
            max_steps = args.max_steps or DEFAULT_MAX_STEPS[args.suite]

            print(f"\n=== task {task_id}: {task.language} ===", flush=True)
            for ep in args.episodes:
                if (task_id, ep) in done_keys:
                    continue
                if args.max_episodes and ran >= args.max_episodes:
                    break
                r = run_one(env, task_suite, task, task_id, ep, policy, max_steps)
                all_results.append(r)
                ran += 1
                print(f"  ep{ep:2d} {'OK ' if r['success'] else 'FAIL'} "
                      f"{'COLLIDE' if r['collided'] else 'clean  '} "
                      f"steps={r['steps']:3d} disp={r['max_displacement']*1000:.2f}mm "
                      f"steer={r['source_aegis']['steer_count']}",
                      flush=True)
                if len(all_results) % 10 == 0:
                    save_results(all_results, {"overall": summarize(all_results)},
                                 out_dir / "summary.json")
    finally:
        for p in policies.values():
            try:
                p.close()
            except Exception:
                pass
        for e in envs.values():
            try:
                e.close()
            except Exception:
                pass

    overall = summarize(all_results)
    save_results(all_results, {"overall": overall}, out_dir / "summary.json")
    print(f"\n=== Source+AEGIS {args.suite} 总计 ===")
    print(f"  episodes={overall['episodes']} TSR={overall['TSR']:.1%} "
          f"CAR={overall['CAR']:.1%} CAR>5mm={overall['CAR_severe']:.1%} "
          f"SSR={overall['SSR']:.1%} ETS={overall['ETS']:.1f}")
    print(f"  结果: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
