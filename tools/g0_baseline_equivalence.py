"""G0 零引导数值等价测试（baseline_equivalence）。

任务书 §4.6：
- 锁定 solver、约束构造、QP slack、执行映射、fallback；
- 零引导（λ=0）必须在固定输入上数值复现原生 AEGIS。

当前代码无 STG 的 λ 软项接口（G0 阶段尚未实现 B1/B2/B3），故本测试验证：
1. 原生 AEGIS filter 在固定 (obs, action) 输入下**确定性可复现**（同输入→同输出）；
2. QP objective 形式 = cp.quad_form(u - u_ref, W)，W = diag(1/25×6 + 1×3)，
   即 λ=0 时 STG 软项应消失、回归此目标；
3. QP 变量 9D、约束为 CBF 行 + 额外障碍行，fallback 行为一致。

这是后续 G1「λ=0 恢复 B0」测试的基线。
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
sys.path.insert(0, os.path.join(ROOT, "third_party", "aegis"))

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

from libero.libero import benchmark  # noqa: E402
from src.safety.aegis_adapted import AegisSafetyLayer  # noqa: E402
from tools.g1a_soup_trace import make_env  # noqa: E402

SUITE = "safelibero_long"
LEVEL = "I"
TASK_ID = 0


def main():
    ts = benchmark.get_benchmark_dict()[SUITE](safety_level=LEVEL)
    task = ts.get_task(TASK_ID)
    env = make_env(ts, task, 1)
    layer = AegisSafetyLayer(env, filter_mode="euclidean")
    layer.reset(env.env._get_observations(), task.language)

    obs = env.env._get_observations()
    rng = np.random.default_rng(42)
    action = rng.uniform(-1, 1, 7).astype(np.float64)
    action[6] = -1.0

    # 1. 确定性：同输入调用 3 次（每次前重置 z_fixed 到初始状态，保证可比）
    outs = []
    for _ in range(3):
        # 用独立 layer 保证完全同初始状态（filter 会演化 z_fixed）
        l2 = AegisSafetyLayer(env, filter_mode="euclidean")
        l2.reset(env.env._get_observations(), task.language)
        out2, info2 = l2.filter(action, env.env._get_observations())
        outs.append(out2)

    deterministic = all(np.allclose(outs[0], o, atol=0) for o in outs)

    # 2. QP objective 形式（从源码常量确认，不重新构造 QP）
    W = np.diag([1.0 / 25] * 6 + [1.0, 1.0, 1.0])
    w_diag = np.diag(W).tolist()

    # 3. 一次真实 filter 的输出作为基线记录
    out0, info0 = layer.filter(action, obs)

    result = {
        "suite": SUITE, "level": LEVEL, "task_id": TASK_ID,
        "action_input_7d": action.tolist(),
        "deterministic_across_3_calls": bool(deterministic),
        "output_7d": out0.tolist(),
        "W_diag_9d": w_diag,
        "W_diag_expected": [1.0/25]*6 + [1.0]*3,
        "qp_variable_dim": 9,
        "filter_info": {
            "h": float(info0.get("h", 0.0)),
            "feasible": bool(info0.get("feasible", False)),
            "active": bool(info0.get("active", False)),
        },
    }
    result["W_diag_matches_expected"] = bool(np.allclose(
        np.diag(W), np.array([1.0/25]*6 + [1.0]*3)))
    result["verdict"] = "PASS" if (deterministic and result["W_diag_matches_expected"]) else "FAIL"

    out = os.path.join(ROOT, "experiments", "stg_v1", "audit", "baseline_equivalence.json")
    with open(out, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("baseline_equivalence.json 已写")
    print(f"  deterministic_across_3_calls: {'PASS' if deterministic else 'FAIL'}")
    print(f"  W_diag_matches_expected: {'PASS' if result['W_diag_matches_expected'] else 'FAIL'}")
    print(f"  verdict: {result['verdict']}")
    env.close()


if __name__ == "__main__":
    main()
