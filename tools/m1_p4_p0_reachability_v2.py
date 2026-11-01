"""M1-P4-P0 v2：安全预抓取可达性（多起点 IK + RRT-Connect）。

修正 v1 的单起点 IK 缺陷：Panda 7-DOF 冗余臂的单起点 DLS IK 会收敛到特定肘部
构型，可能「肘部穿过 mug」。v2 用多起点 IK 探索肘部零空间，只有找到**无碰撞**的
IK 解才作为 RRT 目标。

对每组（S01/st3、S04/st3、S09/st3）：
- 重跑 HVNP 到首次 NVS，捕获安全起始构型 q_start；
- 多起点 IK：对每个预抓取位姿做 N 次随机起点 IK，统计无碰撞解；
- 若有无碰撞 IK 解 → RRT-Connect 从 q_start 到该解（5000 迭代）；
- 报告是否找到连续无碰撞关节空间路径。
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time
import warnings

warnings.filterwarnings("ignore")

ROOT = "/data/safeVLA"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "dataset", "SafeLIBERO", "safelibero"))
sys.path.insert(0, "/data/2608.SDL/MojocoProj/src/pi05")
sys.path.insert(0, os.path.join(ROOT, "third_party", "aegis"))

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
import mujoco  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

from src.pnsg.policy_adapter import PnsgPolicy  # noqa: E402
from src.safety.aegis_adapted import AegisSafetyLayer  # noqa: E402
from tools.g1a_soup_trace import make_env, body_state  # noqa: E402
from tools.m1_p1_design_gate_v2 import step_tracked, build_body_geoms, generate_block  # noqa: E402
from tools.m1_p1_design_gate_v3 import disp_from_p0  # noqa: E402
from tools.m1_p2_p1a_pilot import (  # noqa: E402
    select_action, Z_DIR, L_DIR, AVOID_MAG, AVOID_STEPS,
    SUITE, LEVEL, TASK_ID, REPLAN_STEPS, SWITCH, PROTECTED, DELTA_CAR, EPS,
)
from tools.m1_p4_p0_reachability import (  # noqa: E402
    ik, joint_limits, set_arm_q, get_arm_q, collision_free, reconstruct_nvs_state,
)

SOUP_BODY = "alphabet_soup_1_main"
MUG_BODY = "red_coffee_mug_obstacle_1_main"
CASES = [(1, 3), (4, 3), (9, 3)]

# 预抓取位姿（位置偏移；+y 远离 mug，mug 在 -y 侧）
PREGRASP_OFFSETS = [
    np.array([0.0, 0.0, 0.14]),    # 正上方
    np.array([0.0, 0.0, 0.09]),    # 正上方低
    np.array([0.0, 0.06, 0.12]),   # +y 侧
    np.array([0.0, 0.10, 0.14]),   # +y 更远更高
    np.array([0.04, 0.08, 0.13]),  # +x 稍 +y
    np.array([0.06, 0.0, 0.12]),   # +x 侧
]

IK_RESTARTS = 30
RRT_ITER = 5000


def ik_multistart(env, tgt_pos, tgt_ori, rg, og, n=IK_RESTARTS, seed=0):
    """多起点 IK，返回 (free_solutions, best_pos_err, any_ik_ok)。"""
    lo, hi = joint_limits(env)
    rng = np.random.default_rng(seed)
    free_sols = []
    any_ok = False
    best_err = np.inf
    for _ in range(n):
        qinit = rng.uniform(lo, hi)
        set_arm_q(env, qinit)
        mujoco.mj_forward(env.sim.model._model, env.sim.data._data)
        q, ok, perr = ik(env, tgt_pos, tgt_ori, rg, og)
        if ok and perr < 0.004:
            any_ok = True
            best_err = min(best_err, perr)
            if collision_free(env, q, rg, og):
                free_sols.append(q.copy())
    return free_sols, best_err, any_ok


def rrt_connect(env, rg, og, q_start, q_goal, max_iter=RRT_ITER, step=0.15,
                goal_bias=0.15, seed=0):
    """双向 RRT-Connect（含边稠密碰撞检查）。"""
    lo, hi = joint_limits(env)
    rng = np.random.default_rng(seed)

    class Tree:
        def __init__(self, root):
            self.nodes = [np.asarray(root, float).copy()]
            self.parent = {0: -1}

        def add(self, q, p):
            i = len(self.nodes)
            self.nodes.append(np.asarray(q, float).copy())
            self.parent[i] = p
            return i

    def sample():
        return q_goal.copy() if rng.random() < goal_bias else rng.uniform(lo, hi)

    def nearest(tree, q):
        return int(np.argmin([np.linalg.norm(n - q) for n in tree.nodes]))

    def steer(a, b):
        d = b - a
        nn = np.linalg.norm(d)
        if nn < 1e-9:
            return a.copy(), False
        return a + d / nn * min(nn, step), nn <= step

    def extend(tree, qt):
        ni = nearest(tree, qt)
        qn, reached = steer(tree.nodes[ni], qt)
        if not collision_free(env, qn, rg, og):
            return None, False
        return tree.add(qn, ni), reached

    def connect(tree, q):
        while True:
            i, reached = extend(tree, q)
            if i is None:
                return None
            if reached:
                return i

    def path_of(tree, i):
        p = [tree.nodes[i]]
        while tree.parent[i] != -1:
            i = tree.parent[i]
            p.append(tree.nodes[i])
        return p[::-1]

    Ta, Tb = Tree(q_start), Tree(q_goal)
    for _ in range(max_iter):
        ia, _ = extend(Ta, sample())
        if ia is not None:
            ib = connect(Tb, Ta.nodes[ia])
            if ib is not None:
                pa = path_of(Ta, ia)
                pb = path_of(Tb, ib)
                full = pa + pb[1:]
                # 边稠密碰撞检查
                dense = [full[0]]
                for a, b in zip(full[:-1], full[1:]):
                    k = int(np.linalg.norm(b - a) / 0.02) + 1
                    for j in range(1, k + 1):
                        dense.append(a + (b - a) * (j / k))
                for q in dense:
                    if not collision_free(env, q, rg, og):
                        return None, False
                return full, True
        Ta, Tb = Tb, Ta
    return None, False


def main():
    from libero.libero import benchmark
    ts = benchmark.get_benchmark_dict()[SUITE](safety_level=LEVEL)
    task = ts.get_task(TASK_ID)
    prompt = task.language
    pol = PnsgPolicy()

    print("=== M1-P4-P0 v2：多起点 IK + RRT-Connect 可达性 ===", flush=True)
    out_all = {}

    for state_id, stream in CASES:
        print(f"\n--- S{state_id:02d}/st{stream} ---", flush=True)
        t0 = time.time()
        env, layer, scratch, layer_s, rg, og, p0, soup_pos, nvs_step, Dmax = \
            reconstruct_nvs_state(ts, task, prompt, pol, state_id, stream)
        print(f"  NVS @ step {nvs_step}，Dmax={Dmax*1000:.3f}mm", flush=True)
        print(f"  soup={np.round(soup_pos,4)}  mug={np.round(p0,4)}", flush=True)

        obs = env.env._get_observations()
        init_quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float64)
        tgt_ori = R.from_quat([init_quat[1], init_quat[2], init_quat[3], init_quat[0]]).as_matrix()

        q_start = get_arm_q(env)

        # Phase B：多起点 IK
        print(f"  [Phase B] 多起点 IK（{len(PREGRASP_OFFSETS)} 位姿 × {IK_RESTARTS} 起点）:", flush=True)
        free_by_offset = {}
        for off in PREGRASP_OFFSETS:
            tgt = soup_pos + off
            free_sols, best_err, any_ok = ik_multistart(env, tgt, tgt_ori, rg, og)
            free_by_offset[tuple(off.tolist())] = len(free_sols)
            tag = f"{len(free_sols)}/{IK_RESTARTS} 无碰撞解" if any_ok else "IK 全失败"
            print(f"    off={np.round(off,3)} → {tag} (best_err={best_err*1000:.1f}mm)", flush=True)

        # 选无碰撞解最多的位姿，取一个解做 RRT 目标
        best_off = max(PREGRASP_OFFSETS, key=lambda o: free_by_offset[tuple(o.tolist())])
        n_free_best = free_by_offset[tuple(best_off.tolist())]
        rrt_found = False
        rrt_path_len = 0
        if n_free_best == 0:
            print(f"  → 所有位姿均无无碰撞 IK 解", flush=True)
        else:
            # 重新取 best_off 的一个无碰撞解
            free_sols, _, _ = ik_multistart(env, soup_pos + best_off, tgt_ori, rg, og)
            q_goal = free_sols[0]
            print(f"  → 选 {np.round(best_off,3)}（{n_free_best} 无碰撞解）作 RRT 目标", flush=True)
            print(f"  [Phase C] RRT-Connect（{RRT_ITER} 迭代）...", flush=True)
            t_rrt = time.time()
            path, rrt_found = rrt_connect(env, rg, og, q_start, q_goal)
            rrt_path_len = len(path) if path else 0
            print(f"  RRT {'找到' if rrt_found else '未找到'} 路径（{rrt_path_len} 路点，{time.time()-t_rrt:.1f}s）", flush=True)

        out_all[f"{state_id}/{stream}"] = {
            "nvs_step": nvs_step, "Dmax": Dmax,
            "soup_pos": soup_pos.tolist(), "mug_pos": p0.tolist(),
            "q_start": q_start.tolist(),
            "free_by_offset": {str(k): v for k, v in free_by_offset.items()},
            "best_offset": best_off.tolist(), "n_free_best": n_free_best,
            "rrt_found": rrt_found, "rrt_path_len": rrt_path_len,
        }
        env.close(); scratch.close()
        print(f"  [耗时 {time.time()-t0:.1f}s]", flush=True)

    out = pathlib.Path(ROOT, "data", "audit", "m1_p4_p0_reachability_v2.json")
    out.write_text(json.dumps(out_all, ensure_ascii=False, indent=2, default=str))
    print(f"\n结果: {out}")
    pol.close()


if __name__ == "__main__":
    main()
