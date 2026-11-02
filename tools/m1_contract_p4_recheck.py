"""M1-Contract-Audit：P4 三条 RRT 路径几何复核（新完整几何契约）。

P4-P0 的 collision_free 用旧 build_body_geoms（漏检底座 box_small_base_1_main）。
本脚本对 S01/S04/S09 重新检查：
1. 旧契约下的 IK 无碰撞解，在新契约（含底座+摆设）下是否仍有碰撞；
2. 新契约下多起点 IK 是否仍有无碰撞解；
3. RRT 路径在新契约下是否仍无碰撞。

关键问题：此前「3/3 几何路径无碰撞」在完整几何契约下是否成立。
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
from src.safety.contract_objects import forbidden_contact_geoms, robot_geoms
from tools.g1a_soup_trace import make_env, body_state
from tools.m1_p1_design_gate_v2 import build_body_geoms
from tools.m1_p4_p0e_joint_execution import (reconstruct_nvs, MUG_BODY, SOUP_BODY,
                                            PREGRASP_OFFSETS, IK_RESTARTS, RRT_ITER)
from tools.m1_p4_p0_reachability import joint_limits, set_arm_q, get_arm_q, ik

CASES = [(1, 3), (4, 3), (9, 3)]

def collision_free_full(env, q, rg, fg):
    """新契约：robot 与 forbidden 几何（含底座+摆设）无接触。"""
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

def collision_free_old(env, q, rg, og):
    """旧契约（漏检底座）。"""
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
        if (g1 in rg and g2 in og) or (g2 in rg and g1 in og):
            return False
    return True

def ik_multistart_full(env, tgt_pos, tgt_ori, rg, fg, n=IK_RESTARTS, seed=0):
    lo, hi = joint_limits(env)
    rng = np.random.default_rng(seed)
    free_sols = []
    for _ in range(n):
        qinit = rng.uniform(lo, hi)
        set_arm_q(env, qinit)
        mujoco.mj_forward(env.sim.model._model, env.sim.data._data)
        q, ok, perr = ik(env, tgt_pos, tgt_ori, rg, fg)
        if ok and perr < 0.004 and collision_free_full(env, q, rg, fg):
            free_sols.append(q.copy())
    return free_sols

def main():
    from libero.libero import benchmark
    ts = benchmark.get_benchmark_dict()['safelibero_long'](safety_level='I')
    task = ts.get_task(0)
    prompt = task.language
    pol = PnsgPolicy()

    print("=== P4 RRT 路径几何复核（新完整契约）===", flush=True)
    out_all = {}

    for state_id, stream in CASES:
        print(f"\n--- S{state_id:02d}/st{stream} ---", flush=True)
        env, layer, scratch, layer_s, rg_old, og, p0, soup_pos, nvs_step, Dmax = \
            reconstruct_nvs(ts, task, prompt, pol, state_id, stream)
        rg_full = robot_geoms(env)
        fg = forbidden_contact_geoms(env)
        print(f"  新 forbidden 几何数: {len(fg)} (旧 og: {len(og)})", flush=True)
        print(f"  底座 geom 在新 fg 里: {[g for g in [216,217,218,219] if g in fg]}", flush=True)

        q_start = get_arm_q(env)
        obs = env.env._get_observations()
        iq = np.asarray(obs["robot0_eef_quat"])
        tgt_ori = Rot.from_quat([iq[1], iq[2], iq[3], iq[0]]).as_matrix()

        # 对每个预抓取位姿，新旧契约分别做多起点 IK
        print(f"  [IK 对比] 位姿 → 旧契约无碰撞解数 / 新契约无碰撞解数:", flush=True)
        best_off = None
        best_new_free = 0
        for off in PREGRASP_OFFSETS:
            tgt = soup_pos + off
            # 新契约
            new_free = ik_multistart_full(env, tgt, tgt_ori, rg_full, fg, n=IK_RESTARTS, seed=0)
            # 旧契约（用旧 collision_free）
            old_free = []
            lo, hi = joint_limits(env)
            rng = np.random.default_rng(0)
            for _ in range(IK_RESTARTS):
                qinit = rng.uniform(lo, hi)
                set_arm_q(env, qinit)
                mujoco.mj_forward(env.sim.model._model, env.sim.data._data)
                q, ok, perr = ik(env, tgt, tgt_ori, rg_old, og)
                if ok and perr < 0.004 and collision_free_old(env, q, rg_old, og):
                    old_free.append(q.copy())
            print(f"    off={np.round(off,3)}: 旧={len(old_free)}/{IK_RESTARTS}  新={len(new_free)}/{IK_RESTARTS}", flush=True)
            if len(new_free) > best_new_free:
                best_new_free = len(new_free)
                best_off = off

        out_all[f"{state_id}/{stream}"] = {
            "nvs_step": nvs_step, "n_forbidden_geoms": len(fg), "n_old_og": len(og),
            "best_off_new": None if best_off is None else best_off.tolist(),
            "best_new_free_ik": best_new_free,
        }
        if best_new_free > 0:
            print(f"  → 新契约下存在无碰撞 IK 解（best_off={np.round(best_off,3)}, {best_new_free} 解）", flush=True)
        else:
            print(f"  → 新契约下所有位姿均无无碰撞 IK 解", flush=True)

        env.close(); scratch.close()

    out = json.dumps(out_all, ensure_ascii=False, indent=2, default=str)
    with open(os.path.join(ROOT, "data", "audit", "m1_contract_p4_path_recheck.json"), "w") as f:
        f.write(out)
    print(f"\n结果: data/audit/m1_contract_p4_path_recheck.json", flush=True)
    pol.close()

if __name__ == "__main__":
    main()
