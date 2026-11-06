"""sticky 映射精确表征（裁决 §一/§二）。

补齐全部 9 个 0.25 量化级别，区分 settling transient 与 hold drift，
从 open/closed 两侧接近检测迟滞。

输出：
- 每 level 的 steady aperture median/min/max/drift；
- steps_to_settle（aperture 进入目标带且速度足够小后的步数）；
- 双方向迟滞。
"""

from __future__ import annotations

import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = "/data/safeVLA"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "dataset", "SafeLIBERO", "safelibero"))
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

CA_LEVELS = [-1.00, -0.75, -0.50, -0.25, 0.00, 0.25, 0.50, 0.75, 1.00]
SETTLE_VEL_TOL = 0.0005   # aperture 速度（每 control step）低于此视为 settle
HOLD_STEPS = 30
N_REPEATS = 3


def make_env():
    from libero.libero import benchmark
    from tools.g1a_soup_trace import make_env as _make
    ts = benchmark.get_benchmark_dict()["safelibero_long"](safety_level="I")
    task = ts.get_task(0)
    return _make(ts, task, 0)


def aperture(env):
    obs = env.env._get_observations()
    gq = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64)
    return float(gq[0] - gq[1])


def current_action(env):
    return np.asarray(env.env.robots[0].gripper.current_action).copy()


def servo_to_ca(env, target_ca, max_steps=40):
    """控制 current_action 到 target_ca。"""
    for i in range(max_steps):
        c = current_action(env)[0]
        if abs(c - target_ca) < 0.13:
            return True, i
        act = np.zeros(7)
        act[-1] = 1.0 if c > target_ca else -1.0
        env.step(act)
    return False, max_steps


def measure_steady(env, target_ca, from_side):
    """从 from_side 接近 target_ca，分离 settling 与 hold drift。"""
    env.env.reset()
    # 从方向接近：closed 先闭合到 -1，open 先张开到 +1
    if from_side == "closed":
        for _ in range(15):
            env.step(np.array([0, 0, 0, 0, 0, 0, 1.0]))
    else:
        for _ in range(15):
            env.step(np.array([0, 0, 0, 0, 0, 0, -1.0]))
    ok, steps = servo_to_ca(env, target_ca)
    if not ok:
        return None

    # settling：等 aperture 速度 < SETTLE_VEL_TOL
    prev_a = aperture(env)
    settle_steps = 0
    for i in range(60):
        env.step(np.zeros(7))
        a = aperture(env)
        if abs(a - prev_a) < SETTLE_VEL_TOL:
            settle_steps = i
            break
        prev_a = a

    # hold：固定窗口记录 drift
    aps = []
    for _ in range(HOLD_STEPS):
        env.step(np.zeros(7))
        aps.append(aperture(env))
    aps = np.array(aps)
    return {
        "target_ca": target_ca, "from_side": from_side,
        "servo_steps": steps, "settle_steps": settle_steps,
        "steady_median": float(np.median(aps)),
        "steady_min": float(aps.min()),
        "steady_max": float(aps.max()),
        "steady_drift": float(aps.max() - aps.min()),
    }


def main():
    env = make_env()
    results = []
    for ca in CA_LEVELS:
        for side in ["open", "closed"]:
            for rep in range(N_REPEATS):
                r = measure_steady(env, ca, side)
                if r is not None:
                    r["rep"] = rep
                    results.append(r)
    env.close()

    # 汇总（按 ca + side 聚合）
    from collections import defaultdict
    agg = defaultdict(list)
    for r in results:
        agg[(r["target_ca"], r["from_side"])].append(r)

    print("=== sticky 映射精确表征（settle 分离）===")
    print(f"{'ca':>6} {'side':>6} {'median':>8} {'drift':>8} {'settle':>7} {'servo':>6}")
    summary = {}
    for (ca, side), recs in sorted(agg.items()):
        med = np.median([r["steady_median"] for r in recs])
        drift = np.max([r["steady_drift"] for r in recs])
        settle = np.max([r["settle_steps"] for r in recs])
        servo = np.max([r["servo_steps"] for r in recs])
        summary[f"{ca}_{side}"] = {
            "median_aperture_mm": round(float(med)*1000, 1),
            "max_drift_mm": round(float(drift)*1000, 2),
            "max_settle_steps": int(settle),
            "max_servo_steps": int(servo),
        }
        print(f"{ca:>6} {side:>6} {med*1000:>7.1f} {drift*1000:>7.2f} {settle:>7} {servo:>6}")

    # 迟滞检测（同 ca 两侧 median 差）
    print("\n=== 迟滞（open vs closed 接近的 median 差）===")
    for ca in CA_LEVELS:
        open_meds = [r["steady_median"] for r in agg.get((ca, "open"), [])]
        closed_meds = [r["steady_median"] for r in agg.get((ca, "closed"), [])]
        if open_meds and closed_meds:
            diff = abs(np.median(open_meds) - np.median(closed_meds)) * 1000
            print(f"  ca={ca}: |open-closed| = {diff:.2f}mm")

    out_dir = os.path.join(ROOT, ".cache-local", "fpck_characterization", "mvp")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sticky_mapping_artifact_v1.json")
    with open(out_path, "w") as f:
        json.dump({"development_only": True,
                   "results": results, "summary": summary}, f,
                  ensure_ascii=False, indent=2)
    print(f"\n→ {out_path}")


if __name__ == "__main__":
    main()
