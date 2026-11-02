"""M1-P4-P0r2b：朝向分布 + 抓取入口验证 + RRT 连通性（逐状态原子落盘）。

在 P0r2 确认「安全预抓取配置存在」的基础上，补齐三个层次：
1. 朝向分布：20/19/18 个安全解集中在哪些倾斜角度；
2. 抓取入口验证：从预抓取位姿沿接近方向短距离进入，检查是否新产生 link5/link6 扫过底座；
3. RRT 连通性：对任务有效安全解，完整契约 RRT 从 q_start 连接。

工程改进：逐状态跑完立即原子写 JSON（临时文件 + os.replace），单个状态崩溃不丢已完成的。

抓取入口验证定义（保守、可复现）：
- 预抓取位姿 q_pre，接近方向 = 夹爪局部 -z（朝下），沿 -z 进入 Δ=3cm 到 6 个中间位姿；
- 每个中间位姿重新 IK（保持朝向），检查完整契约碰撞；
- 若全部中间位姿无 SUPPORT_COLLISION / PROTECTED_OBJECT_COLLISION，则入口有效。
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
from src.safety.contract_objects import robot_geoms, forbidden_contact_geoms, classify_body
from tools.g1a_soup_trace import make_env, body_state
from tools.m1_p4_p0e_joint_execution import (reconstruct_nvs, SOUP_BODY, MUG_BODY,
                                            IK_RESTARTS, RRT_ITER)
from tools.m1_p4_p0_reachability import joint_limits, set_arm_q, get_arm_q, ik
from tools.m1_p4_p0r2_reachability import (build_orientations, APPROACH_OFFSETS,
                                           classify_ik, min_gap_full, rrt_full)

CASES = [(1, 3), (4, 3), (9, 3)]
POSITIVE_CONTROL = (2, 0)
OUT_PATH = os.path.join(ROOT, "data", "audit", "m1_p4_p0r2b_reachability.json")


def atomic_write(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, path)


def grasp_entry_check(env, q_pre, tgt_ori, rg, fg, soup_pos):
    """从 q_pre 沿夹爪 -z 方向进入 soup 的短距离入口验证。

    返回 (entry_ok, entry_collisions)：entry_collisions 为各中间位姿的失败类型。
    """
    c = env.robots[0].controller
    set_arm_q(env, q_pre)
    mujoco.mj_forward(env.sim.model._model, env.sim.data._data)
    c.update(force=True)
    # 接近方向 = 夹爪局部 -z（世界坐标）
    Rz = c.ee_ori_mat[:, 2]  # 世界 z 轴 = 局部 z 的第3列
    approach_dir = -Rz  # 朝下
    entry_collisions = []
    entry_ok = True
    # 沿接近方向进入 6 个中间位姿（Δ=0.5cm 每步，共 3cm）
    for step in range(1, 7):
        delta = 0.005 * step  # 0.5cm * step
        tgt_pos = c.ee_pos + approach_dir * delta
        q, ok, perr = ik(env, tgt_pos, tgt_ori, rg, fg)
        r = classify_ik(env, q, ok, perr, rg, fg)
        if r in ("SUPPORT_COLLISION", "PROTECTED_OBJECT_COLLISION"):
            entry_ok = False
            entry_collisions.append({"step": step, "delta_mm": delta * 1000, "type": r})
        elif r != "SAFE_PREGRASP":
            entry_collisions.append({"step": step, "delta_mm": delta * 1000, "type": r})
    return entry_ok, entry_collisions


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

    safe_pregrasps = []
    for oi, (axis, ang, R_tilt_abs) in enumerate(oris):
        tilt_label = "基准朝下" if oi == 0 else f"{axis}{ang}°"
        for oi_off, (off, off_desc) in enumerate(APPROACH_OFFSETS):
            tgt = soup_pos + off
            tgt_ori = R_tilt_abs
            rng = np.random.default_rng(oi * 100 + oi_off)
            best_safe = None
            for _ in range(30):
                qinit = rng.uniform(lo, hi)
                set_arm_q(env, qinit)
                mujoco.mj_forward(env.sim.model._model, env.sim.data._data)
                q, ok, perr = ik(env, tgt, tgt_ori, rg, fg)
                r = classify_ik(env, q, ok, perr, rg, fg)
                if r == "SAFE_PREGRASP":
                    gap = min_gap_full(env, q, rg, fg)
                    gap_v = 0.05 if gap is None else gap
                    if best_safe is None or gap_v > best_safe[1]:
                        best_safe = (q.copy(), gap_v)
            if best_safe is not None:
                safe_pregrasps.append({
                    "tilt": tilt_label, "offset": off.tolist(), "offset_desc": off_desc,
                    "q": best_safe[0].tolist(), "gap_mm": best_safe[1] * 1000,
                    "tgt_ori": tgt_ori.tolist(),
                })

    # 朝向分布
    tilt_dist = {}
    for sp in safe_pregrasps:
        tilt_dist[sp["tilt"]] = tilt_dist.get(sp["tilt"], 0) + 1

    # 入口验证（对每个安全解）
    entry_results = []
    for sp in safe_pregrasps:
        q_pre = np.array(sp["q"])
        tgt_ori = np.array(sp["tgt_ori"])
        entry_ok, entry_col = grasp_entry_check(env, q_pre, tgt_ori, rg, fg, soup_pos)
        entry_results.append({"tilt": sp["tilt"], "offset_desc": sp["offset_desc"],
                              "entry_ok": entry_ok, "entry_collisions": entry_col})

    n_entry_ok = sum(1 for e in entry_results if e["entry_ok"])

    # RRT 连通性：对 entry_ok 的安全解，按 gap 降序选前 3 个跑 RRT
    rrt_results = []
    entry_ok_sp = [sp for sp, e in zip(safe_pregrasps, entry_results) if e["entry_ok"]]
    entry_ok_sp.sort(key=lambda sp: -sp["gap_mm"])
    for sp in entry_ok_sp[:3]:
        q_goal = np.array(sp["q"])
        found = False
        used_seed = None
        path_len = 0
        for seed in range(5):
            p, f = rrt_full(env, rg, fg, q_start, q_goal, max_iter=RRT_ITER, seed=seed)
            if f:
                found = True; used_seed = seed; path_len = len(p); break
        rrt_results.append({"tilt": sp["tilt"], "offset_desc": sp["offset_desc"],
                            "rrt_found": found, "seed": used_seed, "path_len": path_len})

    env.close(); scratch.close()
    return {
        "nvs_step": nvs_step, "n_safe_total": len(safe_pregrasps),
        "tilt_distribution": tilt_dist,
        "n_entry_ok": n_entry_ok, "entry_results": entry_results,
        "rrt_results": rrt_results,
        "safe_pregrasps": safe_pregrasps,
    }


def main():
    from libero.libero import benchmark
    ts = benchmark.get_benchmark_dict()['safelibero_long'](safety_level='I')
    task = ts.get_task(0)
    prompt = task.language
    pol = PnsgPolicy()

    print("=== M1-P4-P0r2b：朝向分布 + 入口 + RRT ===", flush=True)
    out_all = {}
    if os.path.exists(OUT_PATH):
        out_all = json.load(open(OUT_PATH))

    for state_id, stream in CASES + [POSITIVE_CONTROL]:
        key = f"{state_id}/{stream}"
        if key in out_all and out_all[key].get("done"):
            print(f"\n--- {key} 已完成，跳过 ---", flush=True)
            continue
        label = "正对照" if (state_id, stream) == POSITIVE_CONTROL else ""
        print(f"\n--- S{state_id:02d}/st{stream} {label} ---", flush=True)
        t0 = time.time()
        r = run_case(ts, task, prompt, pol, state_id, stream)
        r["done"] = True
        r["wall_s"] = time.time() - t0
        out_all[key] = r
        atomic_write(out_all, OUT_PATH)  # 逐状态原子落盘
        print(f"  n_safe={r['n_safe_total']} 朝向分布={r['tilt_distribution']}", flush=True)
        print(f"  入口有效={r['n_entry_ok']}/{r['n_safe_total']}", flush=True)
        print(f"  RRT: {r['rrt_results']}", flush=True)
        print(f"  [耗时 {r['wall_s']:.1f}s，已落盘]", flush=True)

    print(f"\n结果: {OUT_PATH}", flush=True)
    pol.close()

if __name__ == "__main__":
    main()
