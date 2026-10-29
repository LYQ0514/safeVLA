"""AEGIS 安全层接入统一评测框架。

与 `tools/run_eval.py` 的区别只在策略外壳：在 π0.5 动作送进 env 之前，
先过一遍 AEGIS 的 CBF-QP 安全滤波。其余（场景、初始状态、碰撞判定、指标）
与基线完全一致，保证可比。

用法：
    cd /data/safeVLA && source tools/activate_env.sh

    # 冒烟：1 场景 1 episode
    python tools/run_eval_aegis.py --suite safelibero_spatial --level I \
        --tasks 0 --episodes 0 --resolution 1024

    # 全量对比（与 pi05_1024_baseline 同条件）
    setsid nohup python tools/run_eval_aegis.py --all --no-video \
        --resolution 1024 --out data/eval/aegis_1024 --resume \
        > .cache-local/eval_aegis.log 2>&1 < /dev/null &
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset" / "SafeLIBERO" / "safelibero"))

from tools.run_eval import DEFAULT_MAX_STEPS, LEVELS, SUITES  # noqa: E402


class AegisPolicy:
    """π0.5 + AEGIS 安全层。

    `act()` 返回 7 维动作：策略原始动作经 CBF-QP 投影后的结果。
    每次 episode 开始要调 `reset(obs, prompt)` 让安全层重新定位障碍物。
    """

    def __init__(self, base_policy, env, verbose=False):
        from src.safety.aegis_adapted import AegisSafetyLayer

        self.base = base_policy
        self.env = env
        self.layer = AegisSafetyLayer(env, verbose=verbose)
        self.verbose = verbose
        self.enabled = True
        self._infos = []

    def reset(self, obs=None, prompt=""):
        self.base.reset()
        self._infos = []
        if obs is None:
            return
        ok = self.layer.reset(obs, prompt)
        self.enabled = ok
        if not ok and self.verbose:
            print("[aegis] 本 episode 未找到障碍物 → 退化为基线策略")

    def act(self, obs, prompt):
        action = self.base.act(obs, prompt)
        if not self.enabled:
            return action
        safe, info = self.layer.filter(action, obs)
        self._infos.append(info)
        return safe

    def stats(self):
        s = self.layer.stats()
        s["enabled_episode"] = self.enabled
        s["steps"] = len(self._infos)
        return s

    def close(self):
        self.base.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="safelibero_spatial", choices=SUITES)
    ap.add_argument("--level", default="I", choices=LEVELS)
    ap.add_argument("--tasks", type=int, nargs="+", default=[0])
    ap.add_argument("--episodes", type=int, nargs="+", default=[0])
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--num-steps-wait", type=int, default=10)
    ap.add_argument("--resolution", type=int, default=1024,
                    help="AEGIS 原文用 1024；与基线对比时须一致")
    ap.add_argument("--replan-steps", type=int, default=5,
                    help="AEGIS 原文默认 5")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-episodes", type=int, default=None,
                    help="调试用：最多跑多少条就停")
    ap.add_argument("--h-gain", type=float, default=None,
                    help="消融：CBF 约束的 h 增益（上游固定 10）")
    ap.add_argument("--small-eef", action="store_true",
                    help="消融：忽略上游大椭球规则，所有任务用小椭球")
    args = ap.parse_args()

    if not os.environ.get("LIBERO_CONFIG_PATH"):
        sys.exit("先 source tools/activate_env.sh")

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    import datetime as dt
    import json as _json

    import imageio.v2 as iio  # noqa: PLC0415
    from libero.libero import benchmark, get_libero_path  # noqa: PLC0415
    from libero.libero.envs import OffScreenRenderEnv  # noqa: PLC0415

    from src.eval_safelibero import run_episode, save_results, summarize  # noqa: PLC0415
    from src.policies import Pi05Policy  # noqa: PLC0415
    import src.safety.aegis_adapted as _aeg  # noqa: PLC0415

    if args.h_gain is not None:
        _aeg.set_h_gain(args.h_gain)
        print(f"[aegis] 消融：h 增益 = {args.h_gain}（上游 10）")
    if args.small_eef:
        _aeg.set_use_small_eef(True)
        print("[aegis] 消融：忽略大椭球规则，统一用小椭球")

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = pathlib.Path(args.out or (ROOT / "data" / "eval" / f"aegis_{stamp}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    video_dir = out_dir / "videos"
    if not args.no_video:
        video_dir.mkdir(parents=True, exist_ok=True)

    print(f"[aegis] 加载 π0.5（分辨率 {args.resolution}, replan={args.replan_steps}）…")
    base = Pi05Policy(verbose=True, replan_steps=args.replan_steps,
                      image_size=args.resolution)

    if args.all:
        combos = [(s, l, list(range(4)), list(range(50))) for s in SUITES for l in LEVELS]
    else:
        combos = [(args.suite, args.level, args.tasks, args.episodes)]

    benchmark_dict = benchmark.get_benchmark_dict()
    all_results, all_summary = [], {}
    done_keys = set()
    if args.resume and (out_dir / "summary.json").exists():
        prev = _json.loads((out_dir / "summary.json").read_text())
        all_results = prev.get("episodes", [])
        for e in all_results:
            done_keys.add((e.get("suite"), e.get("level"), e["task_id"], e["episode_idx"]))
        print(f"[aegis] 续跑：已有 {len(done_keys)} 条")

    ran = 0
    try:
        for suite_name, level, task_ids, episode_ids in combos:
            task_suite = benchmark_dict[suite_name](safety_level=level)
            print(f"\n=== {suite_name} level {level} ===", flush=True)
            suite_results = []

            for task_id in task_ids:
                task = task_suite.get_task(task_id)
                bddl = (pathlib.Path(get_libero_path("bddl_files"))
                        / task.problem_folder / task.bddl_file)
                env = OffScreenRenderEnv(
                    bddl_file_name=bddl,
                    camera_heights=args.resolution,
                    camera_widths=args.resolution,
                )
                env.seed(7)
                max_steps = args.max_steps or DEFAULT_MAX_STEPS[suite_name]
                policy = AegisPolicy(base, env, verbose=False)
                try:
                    for ep in episode_ids:
                        if (suite_name, level, task_id, ep) in done_keys:
                            continue
                        if args.max_episodes and ran >= args.max_episodes:
                            raise SystemExit(0)

                        # run_episode 内部会调用 policy.reset()，但安全层需要 obs
                        r = run_episode(
                            env, task_suite, task_id, ep, policy,
                            max_steps=max_steps,
                            num_steps_wait=args.num_steps_wait,
                            collect_frames=not args.no_video,
                        )
                        st = policy.stats()
                        r.update({"suite": suite_name, "level": level,
                                  "max_steps": max_steps, "aegis": st})
                        suite_results.append(r)
                        all_results.append(r)
                        ran += 1
                        print(f"  task{task_id} ep{ep:2d} "
                              f"{'OK ' if r['success'] else 'FAIL'} "
                              f"{'COLLIDE' if r['collided'] else 'clean  '} "
                              f"steps={r['steps']:3d} disp={r['max_displacement']:.4f} "
                              f"qp={st['qp_calls']} infeas={st['infeasible_rate']:.0%}",
                              flush=True)

                        if not args.no_video and r["frames"]:
                            iio.mimwrite(video_dir / f"{suite_name}_L{level}_t{task_id}_e{ep}.mp4",
                                         r["frames"], fps=10)
                        if len(all_results) % 20 == 0:
                            save_results(all_results, {
                                "overall": summarize(all_results),
                                "per_scenario": all_summary,
                            }, out_dir / "summary.json")
                finally:
                    try:
                        env.close()
                    except Exception:
                        pass

            key = f"{suite_name}_L{level}"
            if not suite_results:
                suite_results = [r for r in all_results
                                 if r.get("suite") == suite_name and r.get("level") == level]
            all_summary[key] = summarize(suite_results)
            s = all_summary[key]
            print(f"  --> {key}: TSR={s['TSR']:.1%} CAR={s['CAR']:.1%} "
                  f"CAR>5mm={s['CAR_severe']:.1%} SSR={s['SSR']:.1%} "
                  f"ETS={s['ETS']:.1f} (n={s['episodes']})", flush=True)
            save_results(all_results, {"overall": summarize(all_results),
                                       "per_scenario": all_summary},
                         out_dir / "summary.json")
    finally:
        base.close()

    overall = summarize(all_results)
    print(f"\n=== AEGIS 总计 ===")
    print(f"  episodes={overall['episodes']} TSR={overall['TSR']:.1%} "
          f"CAR={overall['CAR']:.1%} CAR>5mm={overall['CAR_severe']:.1%} "
          f"SSR={overall['SSR']:.1%} ETS={overall['ETS']:.1f}")
    save_results(all_results, {"overall": overall, "per_scenario": all_summary},
                 out_dir / "summary.json")
    print(f"  结果: {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
