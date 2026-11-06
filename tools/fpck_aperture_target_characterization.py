"""空载 aperture target 表征 + 目标感知 servo 验证（裁决 §三/§四）。

验证：sticky current_action 能否停在中间值，并让实际 aperture 稳定保持。
关键：控制 current_action（sticky target），而非 actual aperture（有滞后）。

流程：
1. 记录 current_action 量化步长（实测 0.25）；
2. 对目标 aperture {20, 43, 60}mm，从 open/closed 两侧接近；
3. current_action 停在对应目标值后，action=0 保持 20 步；
4. 记录 drift/oscillation。
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
    g = env.env.robots[0].gripper
    return np.asarray(g.current_action).copy()


def grip_action(grip):
    act = np.zeros(7, dtype=np.float64)
    act[-1] = grip
    return act


def servo_to_target(env, target_aperture, from_side, max_steps=40):
    """目标感知 servo：控制 current_action 使 actual aperture 接近 target。

    from_side: "open"（从 open 接近）或 "closed"（从 closed 接近）。
    返回 (converged, steps, final_aperture, trace)。
    """
    # 关键：用 current_action 的单调性（close 使 aperture 减小，open 使增大）
    # 根据 actual aperture 与 target 的符号差决定 pulse 方向，
    # 但步长要小到不越过 target（sticky 步长 0.25）。
    trace = []
    for i in range(max_steps):
        a = aperture(env)
        if abs(a - target_aperture) <= 0.003:  # 3mm 容差（先粗测）
            return True, i, a, trace
        # 决定方向：aperture 太大→close；太小→open
        direction = 1.0 if a > target_aperture else -1.0
        act = grip_action(direction)
        env.step(act)
        trace.append({"step": i, "aperture": aperture(env),
                      "current_action": current_action(env).tolist()})
    return False, max_steps, aperture(env), trace


def hold_test(env, n_steps=20):
    """action=0 保持，记录 aperture drift。"""
    apertures = []
    for _ in range(n_steps):
        act = grip_action(0.0)
        env.step(act)
        apertures.append(aperture(env))
    return apertures


def main():
    env = make_env()
    targets = [0.020, 0.043, 0.060]
    results = {}
    for target in targets:
        for side in ["open", "closed"]:
            env.env.reset()
            # 从 side 接近：closed 先闭合，open 先张开
            if side == "closed":
                for _ in range(15):
                    env.step(grip_action(1.0))
            else:
                for _ in range(15):
                    env.step(grip_action(-1.0))
            conv, steps, a_final, trace = servo_to_target(env, target, side)
            # hold
            hold_aps = hold_test(env)
            drift = max(hold_aps) - min(hold_aps)
            results[f"{target*1000:.0f}mm_{side}"] = {
                "converged": conv, "steps": steps,
                "aperture_at_converge": a_final,
                "hold_aperture_range": [min(hold_aps), max(hold_aps)],
                "hold_drift": drift,
            }
            print(f"target={target*1000:.0f}mm from {side}: conv={conv} "
                  f"steps={steps} a={a_final*1000:.1f}mm "
                  f"hold_drift={drift*1000:.2f}mm")
    env.close()

    out_dir = os.path.join(ROOT, ".cache-local", "fpck_characterization", "mvp")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "aperture_target_characterization.json"), "w") as f:
        json.dump({"development_only": True, "results": results}, f,
                  ensure_ascii=False, indent=2)
    print("\n→ 完成")


if __name__ == "__main__":
    main()
