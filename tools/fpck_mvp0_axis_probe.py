"""MVP-0：单轴位置探测（验证 OSC_POSE 坐标系与比例）。

裁决 §三：先确认 +x/+y/+z 命令 → world x/y/z 增加，不凭文档假设。
用 env.step(action) 发送小幅单轴动作，记录 EEF 实际移动。

DEVELOPMENT_ONLY：cream_cheese 开发对象，不进入表征选择/统计。
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
    ts = benchmark.get_benchmark_dict()["safelibero_spatial"](safety_level="I")
    task = ts.get_task(0)
    return _make(ts, task, 0)


def eef_pos(env):
    return np.asarray(env.sim.data._data.site_xpos[
        env.env.robots[0].eef_site_id]).copy()


def single_axis_probe(env, axis_idx, sign, n_steps=10, action_mag=0.2):
    """沿 axis_idx 方向发送 sign*action_mag 的动作，记录 EEF 位移。"""
    env.env.reset()
    for _ in range(15):
        env.step(np.zeros(7))  # 先稳定（grip=0 保持）
    p0 = eef_pos(env)
    for i in range(n_steps):
        act = np.zeros(7)
        act[axis_idx] = sign * action_mag
        env.step(act)
    p1 = eef_pos(env)
    return p0, p1, (p1 - p0)


def main():
    env = make_env()
    axes = ["x", "y", "z"]
    results = []
    for ai, name in enumerate(axes):
        for sign in [+1.0, -1.0]:
            p0, p1, delta = single_axis_probe(env, ai, sign)
            results.append({
                "axis": name, "sign": sign,
                "eef_start": p0.tolist(), "eef_end": p1.tolist(),
                "delta_world": delta.tolist(),
            })
            print(f"{name}{'+' if sign>0 else '-'} 动作: "
                  f"EEF {p0.round(4)} → {p1.round(4)}  Δworld={delta.round(4)}")
    env.close()

    out_dir = os.path.join(ROOT, ".cache-local", "fpck_characterization", "mvp")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "mvp0_axis_probe.json"), "w") as f:
        json.dump({"development_only": True,
                   "not_included_in_characterization": True,
                   "results": results}, f, ensure_ascii=False, indent=2)
    print("\n=== 结论 ===")
    for r in results:
        d = r["delta_world"]
        print(f"  {r['axis']}{r['sign']:+}: Δworld=({d[0]:+.4f},{d[1]:+.4f},{d[2]:+.4f})")


if __name__ == "__main__":
    main()
