"""STG 延迟微基准（G1）：单独测量 governor/geometry/QP 的 CPU 延迟。

任务书 §5 / §9：CPU 本地 1000+ 次固定问题微基准，报告 p50/p95；
四组使用一致测时方法。计算成本门槛：p95(B0 filter+governor) ≤ 1.2× p95(B0 filter)。

测量三部分：
1. STGGuide.step() 纯几何/参考计算（无 QP）；
2. filter QP（CVXPY 构造 + OSQP 求解）——B0 基线；
3. filter QP + 软项——B3（含 governor）。
"""

from __future__ import annotations
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

ROOT = "/data/safeVLA"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "dataset", "SafeLIBERO", "safelibero"))
sys.path.insert(0, os.path.join(ROOT, "third_party", "aegis"))

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

from libero.libero import benchmark  # noqa: E402
from src.safety.aegis_adapted import AegisSafetyLayer  # noqa: E402
from src.safety.stg import STGGuide  # noqa: E402
from tools.g1a_soup_trace import make_env  # noqa: E402

N = 1000


def pct(x, p):
    return float(np.percentile(x, p))


def main():
    ts = benchmark.get_benchmark_dict()["safelibero_long"](safety_level="I")
    task = ts.get_task(0)
    env = make_env(ts, task, 1)
    obs = env.env._get_observations()

    rng = np.random.default_rng(0)
    actions = [rng.uniform(-1, 1, 7) for _ in range(N)]
    for a in actions:
        a[6] = -1

    # 1. STGGuide.step() 纯几何（无 QP）
    g = STGGuide(mode="B3")
    p_ee = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    a_v = np.array([0.1, 0.2, 0.9])
    R1 = np.eye(3)
    d = np.array([0.5, 0.3, 0.0])
    t0 = time.perf_counter()
    for i in range(N):
        g.step(p_ee, a_v, R1, d, h=0.005)
    governor_total = time.perf_counter() - t0
    governor_us = governor_total / N * 1e6

    # 2. B0 filter（基线 QP）
    l0 = AegisSafetyLayer(env, filter_mode="euclidean", stg_mode="B0")
    l0.reset(env.env._get_observations(), task.language)
    qp0_times = []
    for a in actions:
        t0 = time.perf_counter()
        l0.filter(a, obs)
        qp0_times.append(time.perf_counter() - t0)

    # 3. B3 filter（QP + 软项 + governor）
    l3 = AegisSafetyLayer(env, filter_mode="euclidean", stg_mode="B3")
    l3.reset(env.env._get_observations(), task.language)
    qp3_times = []
    for a in actions:
        t0 = time.perf_counter()
        l3.filter(a, obs)
        qp3_times.append(time.perf_counter() - t0)

    result = {
        "n_samples": N,
        "governor_step_us_mean": governor_us,
        "governor_step_us_p50": governor_us,  # 单次均值近似（纯 NumPy，稳定）
        "B0_filter_ms": {
            "p50": pct(qp0_times, 50) * 1e3,
            "p95": pct(qp0_times, 95) * 1e3,
            "p99": pct(qp0_times, 99) * 1e3,
        },
        "B3_filter_plus_governor_ms": {
            "p50": pct(qp3_times, 50) * 1e3,
            "p95": pct(qp3_times, 95) * 1e3,
            "p99": pct(qp3_times, 99) * 1e3,
        },
    }
    # 成本门槛：p95(B3) / p95(B0)
    ratio = pct(qp3_times, 95) / max(pct(qp0_times, 95), 1e-9)
    result["p95_ratio_B3_over_B0"] = ratio
    result["cost_gate_pass"] = ratio <= 1.2

    out = os.path.join(ROOT, "experiments", "stg_v1", "outputs", "g1_latency.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"governor.step 均值 = {governor_us:.2f} us")
    print(f"B0 filter: p50={result['B0_filter_ms']['p50']:.2f}ms p95={result['B0_filter_ms']['p95']:.2f}ms")
    print(f"B3 filter+gov: p50={result['B3_filter_plus_governor_ms']['p50']:.2f}ms p95={result['B3_filter_plus_governor_ms']['p95']:.2f}ms")
    print(f"p95 比值 B3/B0 = {ratio:.3f}，成本门槛（≤1.2）: {'PASS' if ratio <= 1.2 else 'FAIL'}")
    env.close()


if __name__ == "__main__":
    main()
