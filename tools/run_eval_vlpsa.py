#!/usr/bin/env python
"""VLPSA 安全层评测入口（方案 B：核心思想复现）。

复用 AEGIS 评测框架的结构，把安全层换成 VLPSA（PSF + 全身采样）。
与 π0.5 基线、AEGIS 共用同一套场景初始化、碰撞判定、指标。

用法：
    cd /data/safeVLA && source tools/activate_env.sh

    # 冒烟：1 场景 1 episode
    python tools/run_eval_vlpsa.py --suite safelibero_spatial --level I \
        --tasks 0 --episodes 0 --resolution 1024

    # 全量
    setsid nohup python tools/run_eval_vlpsa.py --all --no-video \
        --resolution 1024 --out data/eval/vlpsa_1024 --resume \
        > .cache-local/eval_vlpsa.log 2>&1 < /dev/null &
"""

from __future__ import annotations

import argparse
import datetime as dt
import json as _json
import os
import pathlib
import sys
import warnings

import numpy as np  # noqa: F401  (VlpsaPolicy 用到)

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset" / "SafeLIBERO" / "safelibero"))

from tools.run_eval import DEFAULT_MAX_STEPS, LEVELS, SUITES  # noqa: E402


class VlpsaPolicy:
    """π0.5 + VLPSA 安全层（PSF + 全身采样 + CLF 轨迹追踪）。

    论文把 VLA 视为高层规划器：action chunk 插值成期望轨迹 x_d(t)，
    CLF 追踪这条轨迹。因此本类维护 chunk 状态，把累积的期望位置喂给安全层。
    """

    # action[:3] 归一化速度 → 末端速度（m/s）。实测 OSC_POSE 下 action=1 → 0.2078 m/s
    EEF_VEL_SCALE = 0.2
    DT = 0.05  # control_freq=20Hz 的控制周期

    def __init__(self, base_policy, env, spacing=0.03, cbf_gain=1.0, verbose=False,
                 clf_gain=1.0, slack_penalty=1.0, buffer=0.0, grasp_protection=True):
        from src.safety.vlpsa_layer import VlpsaLayer

        self.base = base_policy
        self.env = env
        self.layer = VlpsaLayer(env, spacing=spacing, cbf_gain=cbf_gain,
                                verbose=verbose, clf_gain=clf_gain,
                                slack_penalty=slack_penalty, buffer=buffer,
                                grasp_protection=grasp_protection)
        # 控制器类型决定 QP 在关节空间还是末端空间
        cname = type(env.robots[0].controller).__name__
        self.joint_space = (cname == "JointVelocityController")
        self.verbose = verbose
        self._last_infer = -1
        self._x_start = None
        self._cum = None

    def reset(self, obs=None, prompt=""):
        self.base.reset()
        self._last_infer = -1
        self._x_start = None
        self._cum = None
        if obs is None:
            return
        self.layer.reset(obs, prompt)

    def act(self, obs, prompt):
        x_ee = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
        # 检测重新推理：chunk 边界重置轨迹起点
        if self.base.infer_count != self._last_infer:
            self._last_infer = self.base.infer_count
            self._x_start = x_ee.copy()
            self._cum = np.zeros(3)

        action = self.base.act(obs, prompt)
        # 期望位置累积：x_desired += 末端速度 × dt
        self._cum = self._cum + np.asarray(action[:3]) * self.EEF_VEL_SCALE * self.DT
        x_desired = self._x_start + self._cum

        if self.layer.clf_gain <= 0:
            x_desired = None  # CLF 关闭，纯 CBF 投影
        safe, info = self.layer.filter(action, obs, x_desired=x_desired,
                                       action_scale=self.EEF_VEL_SCALE,
                                       joint_space=self.joint_space)
        return safe

    def stats(self):
        return self.layer.stats()

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
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--replan-steps", type=int, default=5)
    ap.add_argument("--spacing", type=float, default=0.03, help="occupancy 体素边长(m)")
    ap.add_argument("--cbf-gain", type=float, default=1.0, help="CBF α(h)=γh 的 γ")
    ap.add_argument("--clf-gain", type=float, default=1.0, help="CLF 追踪增益 γ_c")
    ap.add_argument("--no-clf", action="store_true", help="关闭 CLF，退化为纯 CBF 投影")
    ap.add_argument("--buffer", type=float, default=0.0, help="occupancy ε-膨胀（米）")
    ap.add_argument("--no-grasp", action="store_true", help="关闭持物保护")
    ap.add_argument("--controller", default="OSC_POSE", choices=["OSC_POSE", "JOINT_VELOCITY"],
                    help="底层控制器。VLPSA 论文用 JOINT_VELOCITY（关节速度空间 QP）")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-episodes", type=int, default=None)
    args = ap.parse_args()

    if not os.environ.get("LIBERO_CONFIG_PATH"):
        sys.exit("先 source tools/activate_env.sh")

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    import imageio.v2 as iio  # noqa: PLC0415
    from libero.libero import benchmark, get_libero_path  # noqa: PLC0415
    from libero.libero.envs import OffScreenRenderEnv  # noqa: PLC0415

    from src.eval_safelibero import run_episode, save_results, summarize  # noqa: PLC0415
    from src.policies import Pi05Policy  # noqa: PLC0415

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = pathlib.Path(args.out or (ROOT / "data" / "eval" / f"vlpsa_{stamp}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    video_dir = out_dir / "videos"
    if not args.no_video:
        video_dir.mkdir(parents=True, exist_ok=True)

    print(f"[vlpsa] 加载 π0.5（分辨率 {args.resolution}, replan={args.replan_steps}, "
          f"spacing={args.spacing}, cbf_gain={args.cbf_gain}）…")
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
        print(f"[vlpsa] 续跑：已有 {len(done_keys)} 条")

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
                    controller=args.controller,
                )
                env.seed(7)
                max_steps = args.max_steps or DEFAULT_MAX_STEPS[suite_name]
                policy = VlpsaPolicy(base, env, spacing=args.spacing,
                                     cbf_gain=args.cbf_gain,
                                     clf_gain=0.0 if args.no_clf else args.clf_gain,
                                     buffer=args.buffer,
                                     grasp_protection=not args.no_grasp)
                try:
                    for ep in episode_ids:
                        if (suite_name, level, task_id, ep) in done_keys:
                            continue
                        if args.max_episodes and ran >= args.max_episodes:
                            raise SystemExit(0)

                        r = run_episode(
                            env, task_suite, task_id, ep, policy,
                            max_steps=max_steps,
                            num_steps_wait=args.num_steps_wait,
                            collect_frames=not args.no_video,
                        )
                        st = policy.stats()
                        r.update({"suite": suite_name, "level": level,
                                  "max_steps": max_steps, "vlpsa": st})
                        suite_results.append(r)
                        all_results.append(r)
                        ran += 1
                        print(f"  task{task_id} ep{ep:2d} "
                              f"{'OK ' if r['success'] else 'FAIL'} "
                              f"{'COLLIDE' if r['collided'] else 'clean  '} "
                              f"steps={r['steps']:3d} disp={r['max_displacement']:.4f} "
                              f"intv={st['intervened']}/{st['qp_calls']}",
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
    print(f"\n=== VLPSA 总计 ===")
    print(f"  episodes={overall['episodes']} TSR={overall['TSR']:.1%} "
          f"CAR={overall['CAR']:.1%} CAR>5mm={overall['CAR_severe']:.1%} "
          f"SSR={overall['SSR']:.1%} ETS={overall['ETS']:.1f}")
    save_results(all_results, {"overall": overall, "per_scenario": all_summary},
                 out_dir / "summary.json")
    print(f"  结果: {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
