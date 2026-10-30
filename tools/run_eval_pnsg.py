#!/usr/bin/env python
"""PNSG 完整 episode 评测入口。

在真实危险 episode 上跑完整 PNSG 循环，对比 π0.5 基线的 TSR/CAR。

用法：
    cd /data/safeVLA && source tools/activate_env.sh
    python tools/run_eval_pnsg.py --suite safelibero_spatial --level I --tasks 3 --episodes 44
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset" / "SafeLIBERO" / "safelibero"))

from tools.run_eval import DEFAULT_MAX_STEPS, LEVELS, SUITES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="safelibero_spatial", choices=SUITES)
    ap.add_argument("--level", default="I", choices=LEVELS)
    ap.add_argument("--tasks", type=int, nargs="+", default=[3])
    ap.add_argument("--episodes", type=int, nargs="+", default=[44])
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--execute-steps", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not os.environ.get("LIBERO_CONFIG_PATH"):
        sys.exit("先 source tools/activate_env.sh")

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    import numpy as np  # noqa: PLC0415

    from libero.libero import benchmark, get_libero_path  # noqa: PLC0415
    from libero.libero.envs import OffScreenRenderEnv  # noqa: PLC0415

    from src.eval_safelibero import run_episode, summarize  # noqa: PLC0415
    from src.pnsg.pnsg_policy import PnsgSteeringPolicy  # noqa: PLC0415

    out_dir = pathlib.Path(args.out or (ROOT / "data" / "eval" / "pnsg"))
    out_dir.mkdir(parents=True, exist_ok=True)

    benchmark_dict = benchmark.get_benchmark_dict()
    all_results = []

    for suite_name, level in [(args.suite, args.level)]:
        task_suite = benchmark_dict[suite_name](safety_level=level)
        for task_id in args.tasks:
            task = task_suite.get_task(task_id)
            bddl = (pathlib.Path(get_libero_path("bddl_files"))
                    / task.problem_folder / task.bddl_file)
            env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256,
                                     controller="OSC_POSE")
            # libero 初始化采样偶尔 RandomizationError，重试
            for attempt in range(3):
                try:
                    env.seed(7)
                    break
                except Exception as e:
                    if attempt == 2:
                        raise
                    print(f"  [warn] env 初始化重试 {attempt+1}: {type(e).__name__}")
                    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256,
                                             controller="OSC_POSE")
            max_steps = args.max_steps or DEFAULT_MAX_STEPS[suite_name]
            policy = PnsgSteeringPolicy(env, execute_steps=args.execute_steps)
            try:
                for ep in args.episodes:
                    print(f"\n=== {suite_name} L{level} task{task_id} ep{ep} ===")
                    r = run_episode(env, task_suite, task_id, ep, policy,
                                    max_steps=max_steps, num_steps_wait=10,
                                    collect_frames=False)
                    st = policy.stats()
                    r.update({"suite": suite_name, "level": level, "pnsg": st})
                    all_results.append(r)
                    print(f"  结果: {'OK' if r['success'] else 'FAIL'} "
                          f"{'COLLIDE' if r['collided'] else 'clean'} "
                          f"steps={r['steps']} disp={r['max_displacement']:.4f}")
                    print(f"  PNSG 统计: {st}")
            finally:
                policy.close()
                env.close()

    s = summarize(all_results)
    print(f"\n=== PNSG 汇总 ===")
    print(f"  episodes={s['episodes']} TSR={s['TSR']:.1%} CAR={s['CAR']:.1%} "
          f"CAR>5mm={s['CAR_severe']:.1%} SSR={s['SSR']:.1%} ETS={s['ETS']:.1f}")

    out_path = out_dir / "summary.json"
    with open(out_path, "w") as f:
        json.dump({"episodes": all_results}, f, ensure_ascii=False, indent=2)
    print(f"  结果: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
