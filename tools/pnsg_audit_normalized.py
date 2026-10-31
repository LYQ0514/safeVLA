"""control-step 级归一化对照：Y_sub vs Y_end 四格表 + pair 分布。

修正此前「547 vs 24 = 漏检 23 倍」的统计单位错误。
按想法v4.md：真正证明漏检的是 Y_sub=1 且 Y_end=0 的 control-step 数。

输出：
- 四格表（TN/FP/FN/TP，单位 = control-step）
- pair 分布（unique forbidden geom pairs）
- 连续接触事件数（同一 pair 连续 substep 合并）
"""

from __future__ import annotations

import os
import sys
import warnings
from collections import Counter

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "dataset", "SafeLIBERO", "safelibero"))
sys.path.insert(0, "/data/2608.SDL/MojocoProj/src/pi05")

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
import pathlib  # noqa: E402

import pi05_libero_runner as R  # noqa: E402


def main():
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    from src.pnsg.contact_audit import ContactEventRecorder
    from src.pnsg.policy_adapter import PnsgPolicy

    suite, level, task_id, ep = "safelibero_spatial", "I", 0, 0
    ts = benchmark.get_benchmark_dict()[suite](safety_level=level)
    task = ts.get_task(task_id)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256,
                             controller="OSC_POSE")
    env.seed(7)
    obs = env.reset()
    obs = env.set_init_state(ts.get_task_init_states(task_id)[ep])
    dummy = np.array([0.0] * 6 + [-1.0])
    for _ in range(10):
        obs, _r, _d, _i = env.step(dummy)

    prompt = task.language
    policy = PnsgPolicy()
    rec = ContactEventRecorder(env)

    # 四格表计数（单位 = control-step）
    tn = fp = fn = tp = 0
    pair_counter = Counter()
    contiguous_events = 0
    prev_pairs = set()  # 上一个 control step 的 pair，用于连续事件合并

    n_steps = 100  # 200 步
    print(f"场景 task{task_id} ep{ep}，跑 {n_steps} 个 control-step\n")
    for step in range(n_steps):
        po = R.obs_to_policy_obs(obs, prompt, image_size=224)
        z0 = policy.sample_noise(seed=step // 2)
        ext = policy.generate(z0, po)
        action = np.clip(ext[step % 2], -1, 1)  # 复用 chunk，简化

        obs, summary = rec.substep_step(action)
        y_sub = summary["any_substep_contact"]
        y_end = summary["endpoint_contact"]

        # 四格表
        if y_sub and y_end:
            tp += 1
        elif y_sub and not y_end:
            fn += 1  # 瞬时接触被 step-end 漏掉
        elif not y_sub and y_end:
            fp += 1
        else:
            tn += 1

        # pair 分布
        for p in summary["unique_forbidden_pairs"]:
            pair_counter[p] += 1

        # 连续事件：本步有新 pair 且上一步没有
        if summary["unique_forbidden_pairs"] and not (summary["unique_forbidden_pairs"] & prev_pairs):
            contiguous_events += 1
        prev_pairs = summary["unique_forbidden_pairs"]

    print("=== control-step 级四格表 ===")
    print(f"{'':>12} {'Y_end=0':>12} {'Y_end=1':>12}")
    print(f"{'Y_sub=0':>12} {tn:>12} {fp:>12}")
    print(f"{'Y_sub=1':>12} {fn:>12} {tp:>12}")
    print(f"\n  TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(f"  step-end 漏检率（FN / (FN+TP)）= {fn/(fn+tp)*100:.1f}% (n={fn+tp})")
    print(f"  总 contact steps（Y_sub=1）= {fn+tp} / {n_steps}")

    print("\n=== unique forbidden geom pair 分布（top 15）===")
    for (g1, g2), cnt in pair_counter.most_common(15):
        b1 = env.sim.model.body_names[rec._m.geom_bodyid[g1]]
        b2 = env.sim.model.body_names[rec._m.geom_bodyid[g2]]
        print(f"  {b1} <-> {b2}: {cnt} steps")

    print(f"\n=== 连续接触事件数（同一 pair 连续 substep 合并）===")
    print(f"  {contiguous_events} 个")

    policy.close()
    env.close()


if __name__ == "__main__":
    main()
