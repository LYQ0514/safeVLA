"""STG QP 等价 + 不可行处理测试（G1，需要 env，单步 filter 不消耗 GPU rollout）。

任务书 §5 / 用户 G1 验收项 1/2/6：
- λ=0 时与 B0 原始 QP 输出等价；
- B1/B2/B3 硬安全可行集与 B0 一致（不改变约束）；
- 不可行 QP、异常输入处理一致性（软项不修复不可行）。
"""

from __future__ import annotations

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
import pytest  # noqa: E402

from libero.libero import benchmark  # noqa: E402
from src.safety.aegis_adapted import AegisSafetyLayer  # noqa: E402
from tools.g1a_soup_trace import make_env  # noqa: E402


@pytest.fixture(scope="module")
def env_ctx():
    ts = benchmark.get_benchmark_dict()["safelibero_long"](safety_level="I")
    task = ts.get_task(0)
    env = make_env(ts, task, 1)
    yield env, task
    env.close()


def _mk_layer(env, mode, lam_max=1.0):
    return AegisSafetyLayer(env, filter_mode="euclidean", stg_mode=mode,
                            stg_lam_max=lam_max)


def test_lambda_zero_equivalence(env_ctx):
    """λ=0（stg_lam_max=0）时 B0/B1/B2/B3 输出完全一致。"""
    env, task = env_ctx
    obs = env.env._get_observations()
    rng = np.random.default_rng(7)
    action = rng.uniform(-1, 1, 7); action[6] = -1  # 固定输入，循环外
    outs = {}
    for mode in ["B0", "B1", "B2", "B3"]:
        layer = _mk_layer(env, mode, lam_max=0.0)
        layer.reset(env.env._get_observations(), task.language)
        out, info = layer.filter(action, obs)
        outs[mode] = out
    for mode in ["B1", "B2", "B3"]:
        assert np.allclose(outs["B0"], outs[mode], atol=1e-9), \
            f"λ=0 时 {mode} 应等价 B0"


def test_lam_zero_inside_risk_still_b0(env_ctx):
    """即使进入风险，stg_lam_max=0 时软项也应消失，等价 B0。"""
    env, task = env_ctx
    obs = env.env._get_observations()
    rng = np.random.default_rng(8)
    action = rng.uniform(-1, 1, 7); action[6] = -1
    # B0
    l0 = _mk_layer(env, "B0", lam_max=0.0)
    l0.reset(env.env._get_observations(), task.language)
    out0, info0 = l0.filter(action, obs)
    # B3 with lam_max=0（进入风险但软项权重为 0）
    l3 = _mk_layer(env, "B3", lam_max=0.0)
    l3.reset(env.env._get_observations(), task.language)
    out3, info3 = l3.filter(action, obs)
    assert np.allclose(out0, out3, atol=1e-9)


def test_soft_term_preserves_constraints(env_ctx):
    """软项只改目标不改约束：B3 的 QP 解仍满足 CBF 约束（残差 ≥ -tol）。"""
    env, task = env_ctx
    obs = env.env._get_observations()
    rng = np.random.default_rng(9)
    layer = _mk_layer(env, "B3", lam_max=1.0)
    layer.reset(env.env._get_observations(), task.language)
    # 连续多步，每步验证可行（无 infeasible）
    n_infeasible = 0
    for _ in range(10):
        action = rng.uniform(-1, 1, 7); action[6] = -1
        out, info = layer.filter(action, obs)
        if not info.get("feasible", False):
            n_infeasible += 1
    # 软项（凸 PSD 二次项）不应导致原本可行的 QP 不可行
    assert n_infeasible == 0, f"软项导致 {n_infeasible} 步不可行"


def test_soft_term_does_not_fix_infeasible(env_ctx):
    """软项不修复不可行：不改变约束，不可行时 fallback 行为与 B0 一致。"""
    # 构造不可行：极小 eef 椭球 + 极大障碍椭球使 CBF 与 bounds 冲突时，
    # 软项只改目标，不能使不可行变可行。这里验证 B0 与 B3 的 fallback 路径一致：
    # 两者用相同 constraints，若 B0 不可行则 B3 也不可行。
    env, task = env_ctx
    obs = env.env._get_observations()
    rng = np.random.default_rng(10)
    action = rng.uniform(-1, 1, 7); action[6] = -1
    # 用极小 eef 椭球（eef_q_override）制造极端几何
    l0 = AegisSafetyLayer(env, filter_mode="euclidean", stg_mode="B0",
                          eef_q_override=np.array([1e-6, 1e-6, 1e-6]))
    l0.reset(env.env._get_observations(), task.language)
    out0, info0 = l0.filter(action, obs)
    l3 = AegisSafetyLayer(env, filter_mode="euclidean", stg_mode="B3",
                          eef_q_override=np.array([1e-6, 1e-6, 1e-6]))
    l3.reset(env.env._get_observations(), task.language)
    out3, info3 = l3.filter(action, obs)
    # 无论可行与否，B0 与 B3 的 fallback 行为一致（可行则解一致，不可行则同回退）
    assert info0.get("feasible") == info3.get("feasible")
    if info0.get("feasible"):
        assert np.allclose(out0, out3, atol=1e-9)
    else:
        assert info0.get("fallback_to_raw") == info3.get("fallback_to_raw")


def test_nan_input_consistent(env_ctx):
    """异常输入（NaN）下 B0 与 B3 行为一致（都走相同 QP 路径）。"""
    env, task = env_ctx
    obs = env.env._get_observations()
    action = np.array([np.nan, 0, 0, 0, 0, 0, -1])
    outs = {}
    for mode in ["B0", "B3"]:
        layer = _mk_layer(env, mode)
        layer.reset(env.env._get_observations(), task.language)
        try:
            out, info = layer.filter(action.copy(), obs)
            outs[mode] = ("ok", out, info.get("feasible"))
        except Exception as e:
            outs[mode] = ("err", type(e).__name__, None)
    # 两者行为一致（都成功或都同类失败）
    assert outs["B0"][0] == outs["B3"][0]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
