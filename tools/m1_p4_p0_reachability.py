"""M1-P4-P0：安全预抓取可达性验证（特权 Oracle）。

从 E1 预注册 3 个失败组合（S01/st3 早NVS、S04/st3 晚NVS、S09/st3 早NVS+vanilla失败），
对每组：
- Phase A：重跑 HVNP 到首次 NVS，捕获安全起始关节构型 + soup/mug 真值位姿；
- Phase B：由 soup 真值构造有限预抓取位姿 → IK（DLS）→ 全身碰撞检查；
- Phase C：RRT-Connect 关节空间规划（避开 mug）+ 边稠密碰撞检查；
- （若找到路径）Phase D：真实控制器执行，检查 OSC 动力学/禁止接触/mug 位移。

安全契约：robot↔obstacle 禁止接触 + mug 历史位移 < 1mm - eps。

特权信息（MuJoCo 真值位姿/几何/关节状态）仅作离线可行性判定，非在线算法。
"""

from __future__ import annotations

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
    SUITE, LEVEL, TASK_ID, REPLAN_STEPS, IMAGE_SIZE, SWITCH, H, PROTECTED, DELTA_CAR, EPS,
)

SOUP_BODY = "alphabet_soup_1_main"
MUG_BODY = "red_coffee_mug_obstacle_1_main"

# 预注册组合（E1 失败案例）
CASES = [(1, 3), (4, 3), (9, 3)]

# 预抓取位姿：相对 soup 的偏移（位置），方向固定为初始 eef 方向
PREGRASP_OFFSETS = [
    np.array([0.0, 0.0, 0.14]),   # 正上方 14cm
    np.array([0.0, 0.0, 0.09]),   # 正上方 9cm
    np.array([0.06, 0.0, 0.12]),  # +x 侧
    np.array([0.0, 0.06, 0.12]),  # +y 侧（远离 mug，mug 在 -y）
    np.array([-0.06, 0.0, 0.12]), # -x 侧
    np.array([0.0, -0.06, 0.12]), # -y 侧（朝 mug）
]


def set_arm_q(env, q):
    """设置 7 个臂关节 qpos，保留其余（夹爪/物体）。"""
    d = env.sim.data._data
    d.qpos[:7] = q


def get_arm_q(env):
    return np.asarray(env.sim.data._data.qpos[:7]).copy()


def robot_obstacle_contact(env, rg, og):
    """当前状态 robot↔obstacle 接触（经 mj_collision）。"""
    m = env.sim.model._model
    d = env.sim.data._data
    mujoco.mj_collision(m, d)
    for c in range(d.ncon):
        g1, g2 = d.contact[c].geom1, d.contact[c].geom2
        if (g1 in rg and g2 in og) or (g2 in rg and g1 in og):
            return True
    return False


def eef_pose(env):
    c = env.robots[0].controller
    return c.ee_pos.copy(), c.ee_ori_mat.copy()


def ik(env, target_pos, target_ori, rg, og, max_iter=800, tol=1e-3):
    """DLS IK：7 关节到目标 6-DOF 位姿，返回 (q, success, pos_err)。"""
    c = env.robots[0].controller
    q = get_arm_q(env)
    lam = 0.1
    for _ in range(max_iter):
        set_arm_q(env, q)
        mujoco.mj_forward(env.sim.model._model, env.sim.data._data)
        c.update(force=True)
        J = c.J_full  # 6x7
        p_err = target_pos - c.ee_pos
        # 朝向误差（轴角）
        R_err = target_ori @ c.ee_ori_mat.T
        ori_err = R.from_matrix(R_err).as_rotvec()
        err = np.concatenate([p_err, ori_err])
        if np.linalg.norm(p_err) < tol and np.linalg.norm(ori_err) < 0.05:
            set_arm_q(env, q)
            mujoco.mj_forward(env.sim.model._model, env.sim.data._data)
            return q, True, float(np.linalg.norm(p_err))
        JtJ = J @ J.T  # 6x6
        dq = J.T @ np.linalg.solve(JtJ + lam * lam * np.eye(6), err)
        dq = np.clip(dq, -0.15, 0.15)
        q = q + dq
        q = np.clip(q, *joint_limits(env))
    set_arm_q(env, q)
    mujoco.mj_forward(env.sim.model._model, env.sim.data._data)
    return q, False, float(np.linalg.norm(target_pos - c.ee_pos))


def joint_limits(env):
    m = env.sim.model._model
    lo = np.array([m.jnt_range[i][0] for i in range(7)])
    hi = np.array([m.jnt_range[i][1] for i in range(7)])
    return lo, hi


def collision_free(env, q, rg, og):
    """关节构型 q 是否无 robot↔obstacle 接触（含关节限位检查）。"""
    lo, hi = joint_limits(env)
    if np.any(q < lo - 1e-6) or np.any(q > hi + 1e-6):
        return False
    set_arm_q(env, q)
    mujoco.mj_forward(env.sim.model._model, env.sim.data._data)
    return not robot_obstacle_contact(env, rg, og)


def rrt_connect(env, rg, og, q_start, q_goal, max_iter=3000, step=0.15,
                goal_bias=0.1, rng=np.random.default_rng(0)):
    """双向 RRT-Connect，返回 (path, found)。path 为关节构型列表。"""
    lo, hi = joint_limits(env)

    class Tree:
        def __init__(self, root):
            self.nodes = [root]
            self.parent = {0: -1}

        def add(self, q, parent_idx):
            idx = len(self.nodes)
            self.nodes.append(q.copy())
            self.parent[idx] = parent_idx
            return idx

    def sample():
        if rng.random() < goal_bias:
            return q_goal.copy()
        return rng.uniform(lo, hi)

    def nearest(tree, q):
        dists = [np.linalg.norm(n - q) for n in tree.nodes]
        return int(np.argmin(dists))

    def steer(q_from, q_to):
        d = q_to - q_from
        n = np.linalg.norm(d)
        if n < 1e-9:
            return q_from.copy(), False
        if n <= step:
            return q_to.copy(), True
        return q_from + (d / n) * step, False

    def extend(tree, q_target):
        """extend toward q_target; returns (idx, reached)."""
        near = nearest(tree, q_target)
        q_new, reached = steer(tree.nodes[near], q_target)
        if not collision_free(env, q_new, rg, og):
            return None, False
        idx = tree.add(q_new, near)
        return idx, reached

    def connect(tree, q):
        while True:
            idx, reached = extend(tree, q)
            if idx is None:
                return None
            if reached:
                return idx

    def path_from(tree, idx):
        path = [tree.nodes[idx]]
        while tree.parent[idx] != -1:
            idx = tree.parent[idx]
            path.append(tree.nodes[idx])
        return list(reversed(path))

    Ta = Tree(q_start)
    Tb = Tree(q_goal)
    for _ in range(max_iter):
        q_rand = sample()
        ia, _ = extend(Ta, q_rand)
        if ia is not None:
            ib = connect(Tb, Ta.nodes[ia])
            if ib is not None:
                pa = path_from(Ta, ia)
                pb = path_from(Tb, ib)
                # pb 已含 ib 起点（=Ta.nodes[ia] 的镜像），拼接时去重
                full = pa[:-1] + pb
                # 路径稠密碰撞检查（对连续运动）
                dense = []
                for i in range(len(full) - 1):
                    seg = [full[i]]
                    n = int(np.linalg.norm(full[i+1] - full[i]) / 0.02) + 1
                    for k in range(1, n + 1):
                        seg.append(full[i] + (full[i+1] - full[i]) * (k / n))
                    dense.extend(seg)
                dense.append(full[-1])
                for q in dense:
                    if not collision_free(env, q, rg, og):
                        return None, False
                return full, True
        # swap
        Ta, Tb = Tb, Ta
    return None, False


def reconstruct_nvs_state(ts, task, prompt, pol, state_id, stream):
    """重跑 HVNP 到首次 NVS，返回 (env, layer, scratch, layer_s, rg, og, p0, soup_pos, nvs_step, Dmax)。"""
    env = make_env(ts, task, state_id)
    layer = AegisSafetyLayer(env)
    scratch = make_env(ts, task, state_id)
    layer_s = AegisSafetyLayer(scratch)
    obs = env.env._get_observations()
    layer.reset(obs, prompt)
    rg, og = build_body_geoms(env)
    p0 = body_state(env, MUG_BODY)["pos"].copy()
    soup_pos = body_state(env, SOUP_BODY)["pos"].copy()
    Dmax = 0.0
    nominal = None
    avoid_remaining = 0
    avoid_dir = None
    nvs_step = None
    for step in range(550):
        if step == SWITCH:
            layer.reset(obs, prompt)
        if step % REPLAN_STEPS == 0:
            nominal = generate_block(pol, obs, prompt, step // REPLAN_STEPS)
        nom_a = nominal[step % REPLAN_STEPS]
        if avoid_remaining > 0:
            act = np.clip(nom_a + AVOID_MAG * avoid_dir, -1, 1)
            avoid_remaining -= 1
        else:
            chosen, verdicts = select_action(scratch, layer_s, pol, prompt, env, layer,
                                             nominal, step, Dmax, rg, og, p0)
            if chosen == "NO_VERIFIED_SEQUENCE":
                nvs_step = step
                break
            if chosen == "N":
                act = nom_a
            elif chosen == "Hold+N":
                act = np.zeros(7)
            elif chosen == "A":
                act = layer.filter(nom_a, obs)[0]
            else:
                act = np.clip(nom_a + AVOID_MAG * (Z_DIR if chosen == "Z+A" else L_DIR), -1, 1)
                avoid_remaining = AVOID_STEPS - 1
                avoid_dir = Z_DIR if chosen == "Z+A" else L_DIR
        obs, _r, _d, _i, ca, cl, _ = step_tracked(env, act, rg, og)
        Dmax = max(Dmax, disp_from_p0(env, p0))
    return env, layer, scratch, layer_s, rg, og, p0, soup_pos, nvs_step, Dmax


def main():
    from libero.libero import benchmark
    ts = benchmark.get_benchmark_dict()[SUITE](safety_level=LEVEL)
    task = ts.get_task(TASK_ID)
    prompt = task.language
    pol = PnsgPolicy()

    print("=== M1-P4-P0：安全预抓取可达性验证 ===", flush=True)
    out_all = {}

    for state_id, stream in CASES:
        print(f"\n--- S{state_id:02d}/st{stream} ---", flush=True)
        t0 = time.time()
        env, layer, scratch, layer_s, rg, og, p0, soup_pos, nvs_step, Dmax = \
            reconstruct_nvs_state(ts, task, prompt, pol, state_id, stream)
        print(f"  NVS @ step {nvs_step}，Dmax={Dmax*1000:.3f}mm", flush=True)
        print(f"  起始关节构型 q_start={np.round(get_arm_q(env),3)}", flush=True)
        print(f"  soup={np.round(soup_pos,4)}  mug={np.round(p0,4)}", flush=True)

        # 起始 eef 方向（作为预抓取方向）
        obs = env.env._get_observations()
        init_quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float64)  # wxyz
        target_ori = R.from_quat([init_quat[1], init_quat[2], init_quat[3], init_quat[0]]).as_matrix()

        q_start = get_arm_q(env)
        lo, hi = joint_limits(env)

        # Phase B：IK + 碰撞检查
        print(f"  [Phase B] IK + 碰撞检查（{len(PREGRASP_OFFSETS)} 个预抓取位姿）:", flush=True)
        ik_results = []
        for off in PREGRASP_OFFSETS:
            tgt_pos = soup_pos + off
            q, ok, perr = ik(env, tgt_pos, target_ori, rg, og)
            free = collision_free(env, q, rg, og)
            ik_results.append({"offset": off.tolist(), "ok": ok, "pos_err": perr,
                               "collision_free": free})
            tag = "✓可达+无碰撞" if (ok and free) else ("✗IK失败" if not ok else "✗碰撞")
            print(f"    off={np.round(off,3)} → {tag} (pos_err={perr*1000:.1f}mm)", flush=True)

        reachable_free = [r for r in ik_results if r["ok"] and r["collision_free"]]
        if not reachable_free:
            print(f"  → 无可达且无碰撞的预抓取位姿", flush=True)
            out_all[f"{state_id}/{stream}"] = {"nvs_step": nvs_step, "ik": ik_results,
                                               "rrt_found": False, "note": "无可达预抓取位姿"}
            env.close(); scratch.close()
            continue

        # 选 pos_err 最小的可达无碰撞位姿做 RRT 目标
        best = min(reachable_free, key=lambda r: r["pos_err"])
        tgt_pos = soup_pos + np.array(best["offset"])
        q_goal, _, _ = ik(env, tgt_pos, target_ori, rg, og)
        print(f"  → 选 {np.round(best['offset'],3)} 作 RRT 目标（pos_err={best['pos_err']*1000:.1f}mm）", flush=True)

        # Phase C：RRT-Connect
        print(f"  [Phase C] RRT-Connect 关节空间规划...", flush=True)
        t_rrt = time.time()
        path, found = rrt_connect(env, rg, og, q_start, q_goal)
        print(f"  RRT {'找到' if found else '未找到'} 路径（{len(path) if path else 0} 个路点，{time.time()-t_rrt:.1f}s）", flush=True)

        out_all[f"{state_id}/{stream}"] = {
            "nvs_step": nvs_step, "ik": ik_results,
            "best_offset": best["offset"], "rrt_found": found,
            "rrt_path_len": len(path) if path else 0,
        }
        env.close(); scratch.close()
        print(f"  [耗时 {time.time()-t0:.1f}s]", flush=True)

    out = pathlib.Path(ROOT, "data", "audit", "m1_p4_p0_reachability.json")
    import json
    out.write_text(json.dumps(out_all, ensure_ascii=False, indent=2, default=str))
    print(f"\n结果: {out}")
    pol.close()


if __name__ == "__main__":
    main()
