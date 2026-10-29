#!/usr/bin/env python
"""SafeLIBERO 评测 CLI：π0.5 基线跑 CAR / TSR / ETS。

用法：
    cd /data/safeVLA && source tools/activate_env.sh

    # 冒烟：1 场景 1 episode，看是否真能跑通（最常用）
    python tools/run_eval.py --suite safelibero_spatial --level I --tasks 0 --episodes 0

    # 小规模：每场景 5 个 episode（快速估 TSR/CAR）
    python tools/run_eval.py --suite safelibero_object --level I --tasks 0 1 --episodes 0 1 2 3 4

    # 全量（32 场景 x 50 episode = 1600 条，耗时长，建议后台跑）
    python tools/run_eval.py --all

    # 不加载策略，只验证环境循环（用零动作，用于排障）
    python tools/run_eval.py --suite safelibero_goal --level I --tasks 0 --episodes 0 --no-policy

产物默认落 `data/eval/<时间戳>/`：`summary.json` + 每个 (场景, episode) 的 mp4。
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset" / "SafeLIBERO" / "safelibero"))

SUITES = ["safelibero_spatial", "safelibero_goal", "safelibero_object", "safelibero_long"]
LEVELS = ["I", "II"]

# 步数上限按套件区分（VLPSA 协议；AEGIS 的 long ETS 到 480 也印证 long 需要更长预算）
DEFAULT_MAX_STEPS = {
    "safelibero_spatial": 300,
    "safelibero_object": 300,
    "safelibero_goal": 300,
    "safelibero_long": 550,
}


class ZeroPolicy:
    """零动作策略：仅用于验证环境/评测链路，不代表任何模型性能。"""

    def reset(self):
        pass

    def act(self, obs, prompt):
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="safelibero_spatial", choices=SUITES)
    ap.add_argument("--level", default="I", choices=LEVELS)
    ap.add_argument("--tasks", type=int, nargs="+", default=[0])
    ap.add_argument("--episodes", type=int, nargs="+", default=[0])
    ap.add_argument("--max-steps", type=int, default=None,
                    help="步数上限；默认按套件取（short 套件 300，long 550）")
    ap.add_argument("--num-steps-wait", type=int, default=10)
    ap.add_argument("--out", default=None, help="输出目录，默认 data/eval/<时间戳>/")
    ap.add_argument("--no-policy", action="store_true", help="零动作，只验证链路")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--all", action="store_true", help="全量：4 套件 x 2 等级 x 4 任务 x 50 episode")
    ap.add_argument("--resume", action="store_true",
                    help="断点续跑：跳过 out 目录里已完成的 (场景, episode)")
    ap.add_argument("--replan-steps", type=int, default=None,
                    help="每 N 步重新推理（AEGIS 用 5）；默认执行完整 action chunk")
    ap.add_argument("--resolution", type=int, default=256,
                    help="相机渲染分辨率。AEGIS 原文用 1024，对比时两边须一致")
    return ap.parse_args()


def main():
    args = parse_args()

    if not os.environ.get("LIBERO_CONFIG_PATH"):
        sys.exit("未设置 LIBERO_CONFIG_PATH，先 source tools/activate_env.sh")

    # EGL 必须在 import mujoco 之前设置（activate_env.sh 已设，这里兜底）
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    import imageio.v2 as iio  # noqa: PLC0415
    from libero.libero import benchmark, get_libero_path  # noqa: PLC0415
    from libero.libero.envs import OffScreenRenderEnv  # noqa: PLC0415

    from src.eval_safelibero import (  # noqa: PLC0415
        run_episode,
        save_results,
        summarize,
    )

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = pathlib.Path(args.out or (ROOT / "data" / "eval" / stamp))
    out_dir.mkdir(parents=True, exist_ok=True)
    video_dir = out_dir / "videos"
    if not args.no_video:
        video_dir.mkdir(parents=True, exist_ok=True)

    policy = None
    if args.no_policy:
        print("[eval] 使用零动作策略（仅验证链路，结果无性能意义）")
        policy = ZeroPolicy()
    else:
        from src.policies import Pi05Policy  # noqa: PLC0415

        print("[eval] 加载 π0.5 策略（首次含 JAX 编译，约 1~3 分钟）…")
        policy = Pi05Policy(verbose=True, replan_steps=args.replan_steps,
                            image_size=args.resolution)
        if args.replan_steps:
            print(f"[eval] closed-loop 重规划：每 {args.replan_steps} 步重新推理")

    if args.all:
        combos = [(s, l, list(range(4)), list(range(50))) for s in SUITES for l in LEVELS]
    else:
        combos = [(args.suite, args.level, args.tasks, args.episodes)]

    benchmark_dict = benchmark.get_benchmark_dict()
    all_results = []
    all_summary = {}

    # 断点续跑：读已有 summary，跳过完成的 (suite, level, task, episode)
    done_keys = set()
    resume_path = out_dir / "summary.json"
    if args.resume and resume_path.exists():
        import json as _json

        prev = _json.loads(resume_path.read_text())
        for e in prev.get("episodes", []):
            done_keys.add((e.get("suite"), e.get("level"), e["task_id"], e["episode_idx"]))
        all_results = prev.get("episodes", [])
        print(f"[eval] 续跑：已有 {len(done_keys)} 条结果，将跳过")

    try:
        for suite_name, level, task_ids, episode_ids in combos:
            task_suite = benchmark_dict[suite_name](safety_level=level)
            print(f"\n=== {suite_name}  level {level}  tasks {task_ids}  episodes {len(episode_ids)} ===")
            suite_results = []

            for task_id in task_ids:
                task = task_suite.get_task(task_id)
                bddl = (
                    pathlib.Path(get_libero_path("bddl_files"))
                    / task.problem_folder
                    / task.bddl_file
                )
                env = OffScreenRenderEnv(
                    bddl_file_name=bddl,
                    camera_heights=args.resolution,
                    camera_widths=args.resolution,
                )
                env.seed(7)
                max_steps = args.max_steps or DEFAULT_MAX_STEPS[suite_name]
                try:
                    for ep in episode_ids:
                        if (suite_name, level, task_id, ep) in done_keys:
                            continue
                        r = run_episode(
                            env,
                            task_suite,
                            task_id,
                            ep,
                            policy,
                            max_steps=max_steps,
                            num_steps_wait=args.num_steps_wait,
                            collect_frames=not args.no_video,
                        )
                        # 打上场景标签，断点续跑和分组统计都靠它
                        r["suite"] = suite_name
                        r["level"] = level
                        r["max_steps"] = max_steps
                        suite_results.append(r)
                        all_results.append(r)
                        tag = "OK " if r["success"] else "FAIL"
                        coll = "COLLIDE" if r["collided"] else "clean  "
                        print(
                            f"  task{task_id} ep{ep:2d} {tag} {coll} steps={r['steps']:3d} "
                            f"{r['duration_s']:5.1f}s  disp={r['max_displacement']:.4f}  {r['task_language'][:40]}",
                            flush=True,
                        )
                        if not args.no_video and r["frames"]:
                            name = f"{suite_name}_L{level}_task{task_id}_ep{ep}.mp4"
                            # 必须是 mimwrite：v2 的 imwrite 不收帧列表
                            iio.mimwrite(video_dir / name, r["frames"], fps=10)

                        # 场景内每 20 条也落盘一次：只在场景边界存会丢掉半个场景的进度
                        if len(all_results) % 20 == 0:
                            save_results(
                                all_results,
                                {"overall": summarize(all_results), "per_scenario": all_summary},
                                out_dir / "summary.json",
                            )
                finally:
                    try:
                        env.close()
                    except Exception:
                        pass

            key = f"{suite_name}_L{level}"
            # 续跑时 suite_results 可能为空（本轮全跳过），回退到全量里筛
            if not suite_results:
                suite_results = [
                    r for r in all_results
                    if r.get("suite") == suite_name and r.get("level") == level
                ]
            all_summary[key] = summarize(suite_results)
            s = all_summary[key]
            ets = "n/a" if s["ETS"] is None else f"{s['ETS']:.1f}"
            print(f"  --> {key}: TSR={s['TSR']:.1%} CAR={s['CAR']:.1%} "
                  f"CAR>5mm={s['CAR_severe']:.1%} SSR={s['SSR']:.1%} ETS={ets} "
                  f"(n={s['episodes']})", flush=True)

            # 每个场景跑完就落盘一次，防止长跑中断全丢
            save_results(all_results, {"overall": summarize(all_results),
                                       "per_scenario": all_summary},
                         out_dir / "summary.json")
    finally:
        if not isinstance(policy, ZeroPolicy):
            policy.close()

    overall = summarize(all_results)
    ets = "n/a" if overall["ETS"] is None else f"{overall['ETS']:.1f}"
    print(f"\n=== 总计 ===")
    print(f"  episodes={overall['episodes']}  TSR={overall['TSR']:.1%}  "
          f"CAR={overall['CAR']:.1%}  CAR>5mm={overall['CAR_severe']:.1%}  "
          f"SSR={overall['SSR']:.1%}  ETS={ets}")

    path = save_results(all_results, {"overall": overall, "per_scenario": all_summary}, out_dir / "summary.json")
    print(f"  结果: {path}")
    print(f"  视频: {video_dir if not args.no_video else '(未保存)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
