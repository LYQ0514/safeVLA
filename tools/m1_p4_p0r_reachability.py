"""M1-P4-P0r：Complete-Contract Safe Pregrasp Reachability。

在完整安全契约（含 mug 底座 box_small_base_1_main）下，对 S01/S04/S09 重新搜索
无碰撞预抓取配置 + RRT 可达路径。

关键改动（相对 P4-P0）：
1. 用 src/safety/contract_objects.py 的角色清单（含底座）作为唯一安全对象来源；
2. 扩展接近姿态集合：上方 / +Y / -Y / +X / -X / 斜向抬升；
3. 区分 IK 失败类型：未收敛 / 关节越界 / robot 自碰撞 / robot-底座接触 / robot-mug接触；
4. RRT 用完整契约的稠密边检查，报告最小几何间隙。

验收：找到「无碰撞 IK + RRT 路径」→ 恢复 P0e；只有 IK 无碰撞但 RRT 失败→ 报告搜索不足；
全部 IK 碰撞 → 记录搜索空间边界，不宣称物理不可达。
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
from src.pnsg.policy_adapter import PnsgPolicy
from src.safety.contract_objects import (robot_geoms, forbidden_contact_geoms,
                                         protected_geoms, classify_body, monitor_contacts)
from tools.g1a_soup_trace import make_env, body_state
from tools.m1_p4_p0e_joint_execution import (reconstruct_nvs, MUG_BODY, SOUP_BODY,
                                            IK_RESTARTS, RRT_ITER)
from tools.m1_p4_p0_reachability import joint_limits, set_arm_q, get_arm_q, ik

CASES = [(1, 3), (4, 3), (9, 3)]

# 预注册接近姿态集合（位置偏移相对 soup，+ 姿态变化）
# (offset, 说明)
APPROACHES = [
    (np.array([0.0, 0.0, 0.14]), "正上方 z+0.14"),
    (np.array([0.0, 0.0, 0.09]), "正上方 z+0.09"),
    (np.array([0.0, 0.06, 0.12]), "+Y 侧向"),
    (np.array([0.0, -0.06, 0.12]), "-Y 侧向(朝底座)"),
    (np.array([0.06, 0.0, 0.12]), "+X 侧向"),
    (np.array([-0.06, 0.0, 0.12]), "-X 侧向"),
    (np.array([0.0, 0.10, 0.14]), "+Y 远更高"),
    (np.array([0.04, 0.08, 0.16]), "斜向抬升(+x+y+z)"),
    (np.array([-0.04, 0.08, 0.16]), "斜向抬升(-x+y+z)"),
    (np.array([0.0, 0.10, 0.18]), "高抬升 z+0.18"),
]


def classify_ik_failure(env, q, ok, perr, rg, fg, self_geoms):
    """区分 IK/碰撞失败类型。"""
    lo, hi = joint_limits(env)
    if not ok:
        return "IK未收敛"
    if perr >= 0.004:
        return "IK未收敛"
    if np.any(q < lo - 1e-6) or np.any(q > hi + 1e-6):
        return "关节越界"
    set_arm_q(env, q)
    mujoco.mj_forward(env.sim.model._model, env.sim.data._data)
    m = env.sim.model._model
    d = env.sim.data._data
    mujoco.mj_collision(m, d)
    names = env.sim.model.body_names
    robot_base = False
    robot_protected = False
    robot_self = False
    for c in range(d.ncon):
        g1, g2 = d.contact[c].geom1, d.contact[c].geom2
        b1 = names[m.geom_bodyid[g1]]
        b2 = names[m.geom_bodyid[g2]]
        is_r1 = g1 in rg
        is_r2 = g2 in rg
        if is_r1 and is_r2:
            robot_self = True
        elif is_r1 or is_r2:
            env_body = b1 if is_r2 else b2
            role, _ = classify_body(env_body)
            if role == "PROTECTED_SUPPORT":
                robot_base = True
            elif role == "PROTECTED":
                robot_protected = True
    if robot_base:
        return "robot-底座接触"
    if robot_protected:
        return "robot-mug/受保护接触"
    if robot_self:
        return "robot自碰撞"
    return "无碰撞"


def collision_free_full(env, q, rg, fg):
    lo, hi = joint_limits(env)
    if np.any(q < lo - 1e-6) or np.any(q > hi + 1e-6):
        return False
    set_arm_q(env, q)
    mujoco.mj_forward(env.sim.model._model, env.sim.data._data)
    m = env.sim.model._model
    d = env.sim.data._data
    mujoco.mj_collision(m, d)
    for c in range(d.ncon):
        g1, g2 = d.contact[c].geom1, d.contact[c].geom2
        if (g1 in rg and g2 in fg) or (g2 in rg and g1 in fg):
            return False
    return True


def min_gap(env, q, rg, fg):
    """计算 robot 与 forbidden 几何的最小表面间隙（margin 查询）。"""
    set_arm_q(env, q)
    m = env.sim.model._model
    d = env.sim.data._data
    mujoco.mj_forward(m, d)
    # 用 margin 扩大查询
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
        return best  # None = gap > 0.05
    finally:
        for g, v in saved.items():
            m.geom_margin[g] = v


def rrt_full(env, rg, fg, q_start, q_goal, max_iter=RRT_ITER, seed=0):
    """完整契约双向 RRT（含边稠密检查）。"""
    lo, hi = joint_limits(env)
    rng = np.random.default_rng(seed)

    class Tree:
        def __init__(self, root):
            self.nodes = [np.asarray(root, float).copy()]
            self.parent = {0: -1}
        def add(self, q, p):
            i = len(self.nodes); self.nodes.append(np.asarray(q, float).copy()); self.parent[i] = p; return i

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
        if not collision_free_full(env, qn, rg, fg): return None, False
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
                pa = path_of(Ta, ia); pb = path_of(Tb, ib)
                full = pa + pb[1:]
                dense = [full[0]]
                for a, b in zip(full[:-1], full[1:]):
                    k = int(np.linalg.norm(b - a) / 0.02) + 1
                    for j in range(1, k + 1):
                        dense.append(a + (b - a) * (j / k))
                for q in dense:
                    if not collision_free_full(env, q, rg, fg):
                        return None, False
                return full, True
        Ta, Tb = Tb, Ta
    return None, False


def main():
    from libero.libero import benchmark
    ts = benchmark.get_benchmark_dict()['safelibero_long'](safety_level='I')
    task = ts.get_task(0)
    prompt = task.language
    pol = PnsgPolicy()

    print("=== M1-P4-P0r：完整契约安全预抓取可达性 ===", flush=True)
    out_all = {}

    for state_id, stream in CASES:
        print(f"\n--- S{state_id:02d}/st{stream} ---", flush=True)
        t0 = time.time()
        env, layer, scratch, layer_s, rg_old, og, p0, soup_pos, nvs_step, Dmax = \
            reconstruct_nvs(ts, task, prompt, pol, state_id, stream)
        rg = robot_geoms(env)
        fg = forbidden_contact_geoms(env)
        q_start = get_arm_q(env)
        obs = env.env._get_observations()
        iq = np.asarray(obs["robot0_eef_quat"])
        tgt_ori = Rot.from_quat([iq[1], iq[2], iq[3], iq[0]]).as_matrix()

        print(f"  q_start 无碰撞(完整契约)={collision_free_full(env, q_start, rg, fg)}", flush=True)
        print(f"  [IK 搜索] {len(APPROACHES)} 个接近姿态 × {IK_RESTARTS} 起点:", flush=True)

        free_sols_all = []  # (approach_idx, q, min_gap)
        approach_summary = []
        for ai, (off, desc) in enumerate(APPROACHES):
            tgt = soup_pos + off
            lo, hi = joint_limits(env)
            rng = np.random.default_rng(ai)  # 固定种子
            free_cnt = 0
            fail_types = {}
            best_free = None
            for _ in range(IK_RESTARTS):
                qinit = rng.uniform(lo, hi)
                set_arm_q(env, qinit)
                mujoco.mj_forward(env.sim.model._model, env.sim.data._data)
                q, ok, perr = ik(env, tgt, tgt_ori, rg, fg)
                fail = classify_ik_failure(env, q, ok, perr, rg, fg, rg)
                if fail == "无碰撞":
                    free_cnt += 1
                    gap = min_gap(env, q, rg, fg)
                    free_sols_all.append((ai, q.copy(), gap))
                    if best_free is None or gap > best_free[1]:
                        best_free = (q.copy(), gap)
                else:
                    fail_types[fail] = fail_types.get(fail, 0) + 1
            approach_summary.append({
                "desc": desc, "offset": off.tolist(), "free_cnt": free_cnt,
                "fail_types": fail_types,
                "best_gap_mm": None if best_free is None else best_free[1] * 1000,
            })
            tag = f"{free_cnt}/{IK_RESTARTS}" if free_cnt > 0 else "0"
            print(f"    {desc:18s} off={np.round(off,3)}: 无碰撞 {tag}  失败类型={fail_types}", flush=True)

        out_all[f"{state_id}/{stream}"] = {
            "nvs_step": nvs_step, "approaches": approach_summary,
            "n_free_ik_total": len(free_sols_all),
        }

        if free_sols_all:
            # 选最大 gap 的 free 解做 RRT 目标
            ai_best, q_goal, gap_best = max(free_sols_all, key=lambda x: x[2])
            print(f"  → 选 {APPROACHES[ai_best][1]} 作 RRT 目标（gap={gap_best*1000:.1f}mm）", flush=True)
            print(f"  [RRT] 完整契约 {RRT_ITER} 迭代...", flush=True)
            found = False
            path = None
            used_seed = None
            for seed in range(5):
                p, f = rrt_full(env, rg, fg, q_start, q_goal, max_iter=RRT_ITER, seed=seed)
                if f:
                    found = True; path = p; used_seed = seed; break
            out_all[f"{state_id}/{stream}"]["rrt_found"] = found
            out_all[f"{state_id}/{stream}"]["rrt_seed"] = used_seed
            out_all[f"{state_id}/{stream}"]["rrt_path_len"] = len(path) if path else 0
            if found:
                # 报告路径最小间隙
                gaps = []
                for q in path[::max(1, len(path)//50)]:
                    g = min_gap(env, q, rg, fg)
                    if g is not None:
                        gaps.append(g)
                min_gap_mm = (min(gaps) * 1000) if gaps else None
                out_all[f"{state_id}/{stream}"]["path_min_gap_mm"] = min_gap_mm
                print(f"  RRT 找到路径（{len(path)} 路点，seed={used_seed}，路径最小间隙={min_gap_mm if min_gap_mm is not None else '>50'}mm）", flush=True)
            else:
                print(f"  RRT 未找到路径（5 seeds 均失败）", flush=True)
        else:
            print(f"  → 全部接近姿态均无无碰撞 IK 解", flush=True)

        env.close(); scratch.close()
        print(f"  [耗时 {time.time()-t0:.1f}s]", flush=True)

    out = json.dumps(out_all, ensure_ascii=False, indent=2, default=str)
    with open(os.path.join(ROOT, "data", "audit", "m1_p4_p0r_reachability.json"), "w") as f:
        f.write(out)
    print(f"\n结果: data/audit/m1_p4_p0r_reachability.json", flush=True)
    pol.close()

if __name__ == "__main__":
    main()
