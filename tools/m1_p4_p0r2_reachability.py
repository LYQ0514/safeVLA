"""M1-P4-P0r2：Grasp-Orientation-Aware Safe Reachability。

Gate A 已确认：+X 位置可达（位置-only IK 50/60），6D 失败源于朝向约束 + DLS 局部极小。

本脚本 Gate B/C/D：
- Gate B：13 个预注册抓取朝向（绕 x/y 轴倾斜 ±30/±60/±90° + 基准朝下）；
- Gate C：每个朝向 × 每个位置 × 多起点 IK，完整契约碰撞检查，分类 SAFE_PREGRASP /
  SUPPORT_COLLISION / PROTECTED_OBJECT_COLLISION / SELF_COLLISION / IK_NONCONVERGED；
- Gate D：State02 正对照 + 安全目标 RRT-Connect。

朝向任务有效性：绕 x 轴倾斜 ≤60° 视为「可能侧抓」；绕 y 轴倾斜 ≤60° 视为「可能
俯仰抓」；90° 视为「侧向/横向抓」（可能无法包围 soup，标记 task-questionable）。
"""

from __future__ import annotations
import json, os, sys, time, warnings
warnings.filterwarnings("ignore")
ROOT = "/data/safeVLA"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "dataset", "SafeLIBERO", "safelibero"))
sys.path.insert(0, "/data/2608.SDL/MojocoProj/src/pi05")
sys.path.insert(0, os.path.join(ROOT, "third_party", "aegis"))
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as Rot
from libero.libero import benchmark
from src.pnsg.policy_adapter import PnsgPolicy
from src.safety.contract_objects import (robot_geoms, forbidden_contact_geoms,
                                         classify_body, monitor_contacts)
from tools.g1a_soup_trace import make_env, body_state
from tools.m1_p4_p0e_joint_execution import (reconstruct_nvs, SOUP_BODY, MUG_BODY,
                                            IK_RESTARTS, RRT_ITER)
from tools.m1_p4_p0_reachability import joint_limits, set_arm_q, get_arm_q, ik

CASES = [(1, 3), (4, 3), (9, 3)]
POSITIVE_CONTROL = (2, 0)  # State02（P1b 成功）

# 位置候选（沿用 P0r）
APPROACH_OFFSETS = [
    (np.array([0.0, 0.0, 0.14]), "正上方"),
    (np.array([0.0, 0.06, 0.12]), "+Y侧向"),
    (np.array([0.06, 0.0, 0.12]), "+X侧向"),
    (np.array([0.0, 0.10, 0.14]), "+Y远高"),
    (np.array([0.04, 0.08, 0.16]), "斜向抬升"),
    (np.array([0.0, 0.10, 0.18]), "高抬升"),
]

# 13 个预注册朝向：基准朝下 + 绕 x/y 轴 ±30/±60/±90
def build_orientations(base_ori):
    tilt_specs = [(None, 0.0)]  # 基准朝下
    for axis in ("x", "y"):
        for ang in (30, 60, 90):
            for sign in (+1, -1):
                tilt_specs.append((axis, sign * ang))
    oris = []
    for axis, ang in tilt_specs:
        if axis is None:
            R_tilt = np.eye(3)
        elif axis == "x":
            R_tilt = Rot.from_euler("x", np.radians(ang)).as_matrix()
        elif axis == "y":
            R_tilt = Rot.from_euler("y", np.radians(ang)).as_matrix()
        else:
            R_tilt = np.eye(3)
        oris.append((axis, ang, base_ori @ R_tilt))
    return oris


def classify_ik(env, q, ok, perr, rg, fg):
    lo, hi = joint_limits(env)
    if not ok or perr >= 0.004:
        return "IK_NONCONVERGED"
    if np.any(q < lo - 1e-6) or np.any(q > hi + 1e-6):
        return "JOINT_LIMIT"
    set_arm_q(env, q)
    mujoco.mj_forward(env.sim.model._model, env.sim.data._data)
    m = env.sim.model._model
    d = env.sim.data._data
    # 用正 margin 查询（与 min_gap_full 一致），避免 margin=0 的宽相位漏检
    saved = {g: m.geom_margin[g] for g in list(rg) + list(fg)}
    for g in list(rg) + list(fg):
        m.geom_margin[g] = 0.05
    try:
        mujoco.mj_collision(m, d)
        names = env.sim.model.body_names
        out = "SAFE_PREGRASP"
        for c in range(d.ncon):
            g1, g2 = d.contact[c].geom1, d.contact[c].geom2
            b1 = names[m.geom_bodyid[g1]]
            b2 = names[m.geom_bodyid[g2]]
            is_r1 = g1 in rg
            is_r2 = g2 in rg
            if is_r1 and is_r2:
                return "SELF_COLLISION"
            elif is_r1 or is_r2:
                env_body = b1 if is_r2 else b2
                role, _ = classify_body(env_body)
                if role == "PROTECTED_SUPPORT":
                    return "SUPPORT_COLLISION"
                elif role == "PROTECTED":
                    return "PROTECTED_OBJECT_COLLISION"
        return out
    finally:
        for g, v in saved.items():
            m.geom_margin[g] = v


def min_gap_full(env, q, rg, fg):
    set_arm_q(env, q)
    m = env.sim.model._model
    d = env.sim.data._data
    mujoco.mj_forward(m, d)
    saved = {g: m.geom_margin[g] for g in list(rg) + list(fg)}
    for g in list(rg) + list(fg):
        m.geom_margin[g] = 0.05
    try:
        mujoco.mj_collision(m, d)
        best = None
        for c in range(d.ncon):
            g1, g2 = d.contact[c].geom1, d.contact[c].geom2
            if (g1 in rg and g2 in fg) or (g2 in rg and g1 in fg):
                dd = float(d.contact[c].dist)
                best = dd if best is None else min(best, dd)
        return best
    finally:
        for g, v in saved.items():
            m.geom_margin[g] = v


def rrt_full(env, rg, fg, q_start, q_goal, max_iter=RRT_ITER, seed=0):
    lo, hi = joint_limits(env)
    rng = np.random.default_rng(seed)

    class Tree:
        def __init__(self, root):
            self.nodes = [np.asarray(root, float).copy()]
            self.parent = {0: -1}
        def add(self, q, p):
            i = len(self.nodes); self.nodes.append(np.asarray(q, float).copy()); self.parent[i] = p; return i

    def free(q):
        if np.any(q < lo - 1e-6) or np.any(q > hi + 1e-6):
            return False
        set_arm_q(env, q); mujoco.mj_forward(env.sim.model._model, env.sim.data._data)
        m = env.sim.model._model; d = env.sim.data._data
        mujoco.mj_collision(m, d)
        for c in range(d.ncon):
            g1, g2 = d.contact[c].geom1, d.contact[c].geom2
            if (g1 in rg and g2 in fg) or (g2 in rg and g1 in fg):
                return False
        return True

    def sample():
        return q_goal.copy() if rng.random() < 0.15 else rng.uniform(lo, hi)

    def nearest(tree, q):
        return int(np.argmin([np.linalg.norm(n - q) for n in tree.nodes]))

    def steer(a, b):
        d = b - a; nn = np.linalg.norm(d)
        if nn < 1e-9: return a.copy(), False
        return a + d / nn * min(nn, 0.15), nn <= 0.15

    def extend(tree, qt):
        ni = nearest(tree, qt)
        qn, reached = steer(tree.nodes[ni], qt)
        if not free(qn): return None, False
        return tree.add(qn, ni), reached

    def connect(tree, q):
        while True:
            i, reached = extend(tree, q)
            if i is None: return None
            if reached: return i

    def path_of(tree, i):
        p = [tree.nodes[i]]
        while tree.parent[i] != -1:
            i = tree.parent[i]; p.append(tree.nodes[i])
        return p[::-1]

    Ta, Tb = Tree(q_start), Tree(q_goal)
    for _ in range(max_iter):
        ia, _ = extend(Ta, sample())
        if ia is not None:
            ib = connect(Tb, Ta.nodes[ia])
            if ib is not None:
                full = path_of(Ta, ia) + path_of(Tb, ib)[1:]
                dense = [full[0]]
                for a, b in zip(full[:-1], full[1:]):
                    k = int(np.linalg.norm(b - a) / 0.02) + 1
                    for j in range(1, k + 1):
                        dense.append(a + (b - a) * (j / k))
                for q in dense:
                    if not free(q):
                        return None, False
                return full, True
        Ta, Tb = Tb, Ta
    return None, False


def run_case(ts, task, prompt, pol, state_id, stream):
    env, layer, scratch, layer_s, rg_old, og, p0, soup_pos, nvs_step, Dmax = \
        reconstruct_nvs(ts, task, prompt, pol, state_id, stream)
    rg = robot_geoms(env)
    fg = forbidden_contact_geoms(env)
    q_start = get_arm_q(env)
    obs = env.env._get_observations()
    iq = np.asarray(obs["robot0_eef_quat"])
    base_ori = Rot.from_quat([iq[1], iq[2], iq[3], iq[0]]).as_matrix()
    oris = build_orientations(base_ori)
    lo, hi = joint_limits(env)

    results = {"nvs_step": nvs_step, "safe_pregrasps": [], "ori_stats": []}

    for oi, (axis, ang, R_tilt_abs) in enumerate(oris):
        tilt_label = "基准朝下" if oi == 0 else f"{axis}{ang}°"
        for off, off_desc in APPROACH_OFFSETS:
            tgt = soup_pos + off
            tgt_ori = R_tilt_abs
            rng = np.random.default_rng(oi * 100 + len(APPROACH_OFFSETS))
            fail_types = {}
            best_safe = None
            for _ in range(30):  # 30 起点/组合（控制总预算）
                qinit = rng.uniform(lo, hi)
                set_arm_q(env, qinit)
                mujoco.mj_forward(env.sim.model._model, env.sim.data._data)
                q, ok, perr = ik(env, tgt, tgt_ori, rg, fg)
                r = classify_ik(env, q, ok, perr, rg, fg)
                if r == "SAFE_PREGRASP":
                    gap = min_gap_full(env, q, rg, fg)
                    gap_v = 0.05 if gap is None else gap  # None = >50mm，视为最大安全
                    if best_safe is None or gap_v > best_safe[1]:
                        best_safe = (q.copy(), gap_v)
                else:
                    fail_types[r] = fail_types.get(r, 0) + 1
            if best_safe is not None:
                results["safe_pregrasps"].append({
                    "ori_idx": oi, "tilt": tilt_label, "offset": off.tolist(), "offset_desc": off_desc,
                    "q": best_safe[0].tolist(), "gap_mm": best_safe[1] * 1000,
                })
            results["ori_stats"].append({"ori_idx": oi, "tilt": tilt_label,
                                         "offset": off.tolist(), "offset_desc": off_desc,
                                         "fail_types": fail_types,
                                         "n_safe": 1 if best_safe else 0})

    # 汇总
    n_safe = len(results["safe_pregrasps"])
    env.close(); scratch.close()
    return results, n_safe, q_start, rg, fg, soup_pos, base_ori


def main():
    from libero.libero import benchmark
    ts = benchmark.get_benchmark_dict()['safelibero_long'](safety_level='I')
    task = ts.get_task(0)
    prompt = task.language
    pol = PnsgPolicy()

    print("=== M1-P4-P0r2：抓取朝向感知的安全可达性 ===", flush=True)
    out_all = {}

    for state_id, stream in CASES:
        print(f"\n--- S{state_id:02d}/st{stream} ---", flush=True)
        t0 = time.time()
        results, n_safe, q_start, rg, fg, soup_pos, base_ori = \
            run_case(ts, task, prompt, pol, state_id, stream)
        # 失败类型统计
        fail_total = {}
        for s in results["ori_stats"]:
            for k, v in s["fail_types"].items():
                fail_total[k] = fail_total.get(k, 0) + v
        n_safe_total = sum(s["n_safe"] for s in results["ori_stats"])
        print(f"  安全预抓取配置: {n_safe_total} 个（分布见 JSON）", flush=True)
        print(f"  失败类型: {fail_total}", flush=True)
        out_all[f"{state_id}/{stream}"] = {
            "nvs_step": results["nvs_step"], "n_safe_total": n_safe_total,
            "fail_types": fail_total, "safe_pregrasps": results["safe_pregrasps"],
        }
        print(f"  [耗时 {time.time()-t0:.1f}s]", flush=True)

    # State02 正对照
    print(f"\n--- State02/stream0（正对照）---", flush=True)
    t0 = time.time()
    results, n_safe, q_start, rg, fg, soup_pos, base_ori = \
        run_case(ts, task, prompt, pol, POSITIVE_CONTROL[0], POSITIVE_CONTROL[1])
    fail_total = {}
    for s in results["ori_stats"]:
        for k, v in s["fail_types"].items():
            fail_total[k] = fail_total.get(k, 0) + v
    n_safe_total = sum(s["n_safe"] for s in results["ori_stats"])
    print(f"  安全预抓取配置: {n_safe_total} 个", flush=True)
    print(f"  失败类型: {fail_total}", flush=True)
    out_all["2/0_positive_control"] = {
        "nvs_step": results["nvs_step"], "n_safe_total": n_safe_total,
        "fail_types": fail_total, "safe_pregrasps": results["safe_pregrasps"],
    }
    print(f"  [耗时 {time.time()-t0:.1f}s]", flush=True)

    out = json.dumps(out_all, ensure_ascii=False, indent=2, default=str)
    with open(os.path.join(ROOT, "data", "audit", "m1_p4_p0r2_reachability.json"), "w") as f:
        f.write(out)
    print(f"\n结果: data/audit/m1_p4_p0r2_reachability.json", flush=True)
    pol.close()

if __name__ == "__main__":
    main()
