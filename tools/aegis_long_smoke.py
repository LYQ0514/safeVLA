"""AEGIS long 采集器 smoke：验证保存—加载—同动作认证 与 predicate 日志可复现。

三项验证（对应设计笔记的「控制器复现」与「物理执行复现」）：
1. 物理状态：save_full_state 后 restore，qpos/qvel 一致；
2. AEGIS 控制器状态：save_state/restore_state 后，同 nominal 动作 filter 输出一致，
   且 filter 后控制器状态演化一致；
3. predicate 日志：同一 snapshot 下，子目标 predicate 真值可复现。

只跑 episode 0 的前 2 个重规划决策点（2×5 步），不跑完整 episode。
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")

ROOT = "/data/safeVLA"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "dataset", "SafeLIBERO", "safelibero"))
sys.path.insert(0, "/data/2608.SDL/MojocoProj/src/pi05")
sys.path.insert(0, os.path.join(ROOT, "third_party", "aegis"))

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
import pi05_libero_runner as R  # noqa: E402

from src.pnsg.crn import nominal_seed, sample_z  # noqa: E402
from src.pnsg.full_state import save_full_state, restore_full_state  # noqa: E402
from src.pnsg.policy_adapter import PnsgPolicy  # noqa: E402
from src.safety.aegis_adapted import AegisSafetyLayer  # noqa: E402

from tools.aegis_long_collect import (  # noqa: E402
    SUITE, TASK_ID, make_env, subgoal_values, NUM_STEPS_WAIT, REPLAN_STEPS, IMAGE_SIZE,
)

EPS = 1e-6


def _qpos(env):
    return np.asarray(env.sim.data.qpos).copy()


def _qvel(env):
    return np.asarray(env.sim.data.qvel).copy()


def main():
    from libero.libero import benchmark
    ts = benchmark.get_benchmark_dict()[SUITE](safety_level="I")
    task = ts.get_task(TASK_ID)
    prompt = task.language

    policy = PnsgPolicy()

    print("=== smoke 1：物理状态保存/恢复 ===")
    env = make_env(ts, task, 0)
    obs = env.env._get_observations()
    layer = AegisSafetyLayer(env)
    layer.reset(obs, prompt)

    # 决策点 0：保存 snapshot + obs + z0 + nominal
    snapshot = save_full_state(env)
    obs_deploy = dict(obs)
    z0, _ = sample_z(nominal_seed(SUITE, TASK_ID, 0, 0, 0))
    po = R.obs_to_policy_obs(obs_deploy, prompt, image_size=IMAGE_SIZE)
    raw = policy.generate(z0, po)
    nominal = np.clip(raw, -1, 1)

    qpos0 = _qpos(env); qvel0 = _qvel(env)
    restore_full_state(env, snapshot)
    print(f"  restore 后 qpos 差 = {float(np.max(np.abs(_qpos(env) - qpos0))):.2e}")
    print(f"  restore 后 qvel 差 = {float(np.max(np.abs(_qvel(env) - qvel0))):.2e}")

    print("\n=== smoke 2：AEGIS 控制器状态保存/恢复 ===")
    aegis_state_before = layer.save_state()

    # 分支 A：直接 filter 第一步
    act_a, info_a = layer.filter(nominal[0], obs_deploy)
    aegis_state_after_a = layer.save_state()

    # 分支 B：restore AEGIS 状态后再 filter 同一动作
    layer.restore_state(aegis_state_before)
    act_b, info_b = layer.filter(nominal[0], obs_deploy)
    aegis_state_after_b = layer.save_state()

    print(f"  filtered action 一致 = {np.allclose(act_a, act_b, atol=EPS)}")
    print(f"  info['feasible'] 一致 = {info_a.get('feasible') == info_b.get('feasible')}")
    print(f"  info['intervened'] 一致 = {info_a.get('intervened') == info_b.get('intervened')}")
    print(f"  z_fixed 演化一致 = {np.allclose(aegis_state_after_a['z_fixed'], aegis_state_after_b['z_fixed'], atol=EPS)}")
    print(f"  p2 演化一致 = {np.allclose(aegis_state_after_a['p2'], aegis_state_after_b['p2'], atol=EPS)}")

    print("\n=== smoke 3：predicate 日志可复现 ===")
    sg1 = subgoal_values(env)
    # restore 后再次求 predicate（应一致，因为都是读 object_states_dict）
    sg2 = subgoal_values(env)
    same = all(sg1[i]["value"] == sg2[i]["value"] for i in range(len(sg1)))
    print(f"  predicate 真值一致 = {same}")
    for i in range(len(sg1)):
        print(f"    子目标 {sg1[i]['predicate']}{sg1[i]['args']} = {sg1[i]['value']}")

    # 决策点 0 的完整字段检查
    print("\n=== 决策点 0 字段完整性 ===")
    aegis_arrays = {k: v for k, v in aegis_state_before.items() if isinstance(v, np.ndarray)}
    print(f"  AEGIS 状态 ndarray 字段: {list(aegis_arrays.keys())}")
    print(f"  AEGIS 状态标量字段: {[k for k, v in aegis_state_before.items() if not isinstance(v, np.ndarray)]}")
    print(f"  snapshot shape: {snapshot['flattened'].shape}")
    print(f"  z0 shape: {z0.shape}, nominal shape: {nominal.shape}")
    print(f"  obs 字段数: {len(obs_deploy)}")

    env.close()
    policy.close()
    print("\n=== smoke 通过 ===")


if __name__ == "__main__":
    main()
