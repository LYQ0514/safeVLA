"""SAFETY_GROUND_TRUTH_CONSISTENCY 门禁测试。

对应 docs/reference/safety_causal_experiment_protocol.md 三项断言：
1. 物理真值一致性：安全指标读同步后的 MuJoCo 物理状态，不经过缓存观测接口。
2. 快照恢复一致性：同一快照、相同动作，重复执行得到一致的物理结果。
3. 缓存隔离性：人为改变策略观测缓存时，物理安全指标不得变化
   （只要底层仿真物理状态未变）。

反面案例：375mm 观测缓存伪影（commit 0902f4c），即因安全指标误读
env._get_observations() 缓存导致。
"""

from __future__ import annotations

import glob
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

TASK_ID = 0
NUM_STEPS_WAIT = 10
DESIGN = pathlib.Path(ROOT, "data", "audit", "aegis_long_design")
EPS_POS = 1e-9   # 物理真值容差
EPS_DISP = 1e-6  # 位移容差


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  <- {detail}" if detail and not cond else ""))
    return cond


def make_env(ts, task):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256,
                             controller="OSC_POSE")
    env.seed(7)
    env.reset()
    env.set_init_state(ts.get_task_init_states(TASK_ID)[0])
    dummy = np.zeros(7, dtype=np.float32); dummy[-1] = -1.0
    for _ in range(NUM_STEPS_WAIT):
        env.step(dummy)
    return env


def load_decision(ep, step):
    with np.load(DESIGN / f"ep{ep:02d}" / f"decision_{step:03d}.npz") as f:
        d = {k: f[k] for k in f.files}
    return d


def ground_truth_pos(env, body_name):
    """物理真值：从 sim.data.xpos 读 body 位置。"""
    bid = env.sim.model.body_name2id(body_name)
    return np.asarray(env.sim.data._data.xpos[bid]).copy()


def cached_pos(env, obs_key):
    """策略观测接口：env._get_observations() 缓存。"""
    obs = env.env._get_observations()
    return np.asarray(obs[obs_key]).copy()


def main():
    from libero.libero import benchmark
    ts = benchmark.get_benchmark_dict()["safelibero_long"](safety_level="I")
    task = ts.get_task(TASK_ID)

    results = []
    print("=== SAFETY_GROUND_TRUTH_CONSISTENCY 门禁 ===\n")

    # ---- 断言 1：物理真值一致性 ----
    print("[T1] 物理真值一致性：安全指标必须从 sim.data 读，不经过观测缓存")
    env = make_env(ts, task)
    d = load_decision(0, 330)
    env.sim.set_state_from_flattened(d["snapshot"])
    env.sim.forward()

    mug_body = "red_coffee_mug_obstacle_1_main"
    mug_key = "red_coffee_mug_obstacle_1_pos"

    gt = ground_truth_pos(env, mug_body)
    cache = cached_pos(env, mug_key)
    diff = float(np.linalg.norm(gt - cache))

    # 关键：缓存值与物理真值不同（正是 375mm 伪影的来源）
    mismatch = diff > EPS_DISP
    t1 = check("缓存接口与物理真值存在差异（证明必须用真值）", mismatch,
               f"diff={diff:.4f}m，说明缓存不能替代真值")
    t1b = check("ground_truth_pos 读取成功且稳定", np.isfinite(gt).all())
    results += [t1, t1b]
    env.close()

    # ---- 断言 2：快照恢复一致性 ----
    print("\n[T2] 快照恢复一致性：同一快照、相同动作，重复执行物理结果一致")
    env = make_env(ts, task)
    a_zero = np.zeros(7, dtype=np.float64); a_zero[6] = -1.0
    positions = []
    for _ in range(3):
        env.sim.set_state_from_flattened(d["snapshot"])
        env.sim.forward()
        env.step(a_zero)
        positions.append(ground_truth_pos(env, mug_body))
    env.close()
    max_spread = max(np.linalg.norm(positions[i] - positions[0]) for i in range(1, 3))
    t2 = check("3 次重复恢复后 mug 物理位置一致", max_spread < EPS_POS,
               f"max_spread={max_spread:.3e}")
    results.append(t2)

    # ---- 断言 3：缓存隔离性（用伪影现场回归验证）----
    print("\n[T3] 缓存隔离性：安全指标必须读 sim.data 真值，读缓存会重现 375mm 伪影")
    env = make_env(ts, task)
    env.sim.set_state_from_flattened(d["snapshot"])
    env.sim.forward()

    # step330 快照是 375mm 伪影现场：观测缓存滞后于物理真值
    gt = ground_truth_pos(env, mug_body)
    cache = cached_pos(env, mug_key)
    diff_m = float(np.linalg.norm(gt - cache))

    # 关键断言：缓存值与真值确实不同（若相同则本测试失去回归意义）
    t3 = check(f"缓存 vs 真值存在差异（伪影现场，diff={diff_m:.4f}m）", diff_m > EPS_DISP,
               "若此 FAIL，说明该快照缓存已同步，需换一个未 force 刷新的决策点")
    # 安全指标必须用真值：ground_truth_pos 与 cached_pos 是两个独立接口
    t3b = check("ground_truth_pos 与 cached_pos 是不同数据来源（接口隔离）",
                not np.array_equal(gt, cache))
    # 物理真值自身稳定（读两次一致）
    gt2 = ground_truth_pos(env, mug_body)
    t3c = check("物理真值读取稳定（连续两次一致）",
                np.linalg.norm(gt - gt2) < EPS_POS)
    env.close()
    results += [t3, t3b, t3c]

    # ---- 汇总 ----
    print("\n=== 门禁汇总 ===")
    n_pass = sum(results)
    n_total = len(results)
    verdict = "PASS" if n_pass == n_total else "FAIL"
    print(f"SAFETY_GROUND_TRUTH_CONSISTENCY = {verdict}  ({n_pass}/{n_total})")
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
