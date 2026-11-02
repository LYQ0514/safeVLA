"""M1-P4-P0r2 Gate A：+X 方向的 IK 失败分离诊断。

对 P0r 中 +X 侧向（off=[0.06,0,0.12]）连续 IK 未收敛的目标位置，做三个固定检查：
- A1 位置-only IK：只约束 TCP 位置，放开朝向，判断位置是否可达；
- A2 原始 6D IK：重现失败；
- A3 数值诊断：记录位置/朝向残差、关节限位命中、Jacobian 奇异值、终止原因。

注意 robosuite 的 eef 是 gripper0_grip_site（非 eef_site），P0r 的 ik 用 c.ee_pos。
位置-only IK 用 J_pos（3x7）做 DLS。
"""

from __future__ import annotations
import json, os, sys, warnings
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
from src.safety.contract_objects import robot_geoms, forbidden_contact_geoms
from tools.g1a_soup_trace import make_env, body_state
from tools.m1_p4_p0e_joint_execution import reconstruct_nvs, SOUP_BODY, MUG_BODY
from tools.m1_p4_p0_reachability import joint_limits, set_arm_q, get_arm_q, ik

def ik_pos_only(env, target_pos, max_iter=1000, tol=1e-3):
    """位置-only DLS IK（3 维目标，放开朝向）。返回 (q, success, pos_err, ori_free_ok)。"""
    c = env.robots[0].controller
    q = get_arm_q(env)
    lam = 0.1
    for it in range(max_iter):
        set_arm_q(env, q)
        mujoco.mj_forward(env.sim.model._model, env.sim.data._data)
        c.update(force=True)
        J = c.J_pos  # 3x7
        p_err = target_pos - c.ee_pos
        if np.linalg.norm(p_err) < tol:
            return q, True, float(np.linalg.norm(p_err))
        JJt = J @ J.T  # 3x3
        dq = J.T @ np.linalg.solve(JJt + lam * lam * np.eye(3), p_err)
        dq = np.clip(dq, -0.15, 0.15)
        q = q + dq
        q = np.clip(q, *joint_limits(env))
    return q, False, float(np.linalg.norm(target_pos - c.ee_pos))

def ik_6d_diag(env, target_pos, target_ori, max_iter=800, tol=1e-3):
    """原始 6D IK + 数值诊断。返回 dict。"""
    c = env.robots[0].controller
    q = get_arm_q(env)
    lam = 0.1
    diag = {"iters": 0, "final_pos_err": None, "final_ori_err": None,
            "joint_limit_hits": 0, "min_singular": None, "terminate_reason": "max_iter"}
    for it in range(max_iter):
        set_arm_q(env, q)
        mujoco.mj_forward(env.sim.model._model, env.sim.data._data)
        c.update(force=True)
        J = c.J_full  # 6x7
        p_err = target_pos - c.ee_pos
        R_err = target_ori @ c.ee_ori_mat.T
        ori_err = Rot.from_matrix(R_err).as_rotvec()
        # 奇异值
        sv = np.linalg.svd(J, compute_uv=False)
        diag["min_singular"] = float(sv.min()) if diag["min_singular"] is None else min(diag["min_singular"], float(sv.min()))
        if np.linalg.norm(p_err) < tol and np.linalg.norm(ori_err) < 0.05:
            diag["iters"] = it
            diag["terminate_reason"] = "converged"
            break
        err = np.concatenate([p_err, ori_err])
        JJt = J @ J.T
        dq = J.T @ np.linalg.solve(JJt + lam * lam * np.eye(6), err)
        # 检查关节限位命中
        q_before = q.copy()
        q = np.clip(q + np.clip(dq, -0.15, 0.15), *joint_limits(env))
        if np.any(q != np.clip(q_before + np.clip(dq, -0.15, 0.15), *joint_limits(env))):
            diag["joint_limit_hits"] += 1
        diag["iters"] = it
    set_arm_q(env, q)
    mujoco.mj_forward(env.sim.model._model, env.sim.data._data)
    c.update(force=True)
    diag["final_pos_err"] = float(np.linalg.norm(target_pos - c.ee_pos))
    diag["final_ori_err"] = float(np.linalg.norm(Rot.from_matrix(target_ori @ c.ee_ori_mat.T).as_rotvec()))
    return diag

def main():
    from libero.libero import benchmark
    ts = benchmark.get_benchmark_dict()['safelibero_long'](safety_level='I')
    task = ts.get_task(0)
    prompt = task.language
    pol = PnsgPolicy()

    print("=== P0r2 Gate A：+X IK 失败分离诊断 ===", flush=True)
    state_id, stream = 1, 3
    env, layer, scratch, layer_s, rg_old, og, p0, soup_pos, nvs_step, Dmax = \
        reconstruct_nvs(ts, task, prompt, pol, state_id, stream)
    rg = robot_geoms(env)
    fg = forbidden_contact_geoms(env)
    obs = env.env._get_observations()
    iq = np.asarray(obs["robot0_eef_quat"])
    tgt_ori = Rot.from_quat([iq[1], iq[2], iq[3], iq[0]]).as_matrix()

    # +X 侧向目标位置（P0r 中 60/60 未收敛）
    target_pos = soup_pos + np.array([0.06, 0.0, 0.12])
    print(f"目标位置 target_pos = {np.round(target_pos,4)}", flush=True)
    print(f"soup_pos = {np.round(soup_pos,4)}", flush=True)

    # A1：位置-only IK（多起点）
    lo, hi = joint_limits(env)
    print(f"\n[A1] 位置-only IK（60 起点）:", flush=True)
    pos_ok_cnt = 0
    pos_errs = []
    for seed in range(60):
        rng = np.random.default_rng(seed)
        qinit = rng.uniform(lo, hi)
        set_arm_q(env, qinit)
        mujoco.mj_forward(env.sim.model._model, env.sim.data._data)
        q, ok, perr = ik_pos_only(env, target_pos)
        if ok:
            pos_ok_cnt += 1
        pos_errs.append(perr)
    print(f"  位置-only IK 成功: {pos_ok_cnt}/60", flush=True)
    print(f"  位置残差分布: min={np.min(pos_errs)*1000:.1f}mm max={np.max(pos_errs)*1000:.1f}mm "
          f"中位={np.median(pos_errs)*1000:.1f}mm", flush=True)

    # A2：原始 6D IK（重现失败）
    print(f"\n[A2] 原始 6D IK（60 起点）:", flush=True)
    ok6d = 0
    for seed in range(60):
        rng = np.random.default_rng(seed)
        qinit = rng.uniform(lo, hi)
        set_arm_q(env, qinit)
        mujoco.mj_forward(env.sim.model._model, env.sim.data._data)
        q, ok, perr = ik(env, target_pos, tgt_ori, rg, fg)
        if ok:
            ok6d += 1
    print(f"  6D IK 成功: {ok6d}/60", flush=True)

    # A3：数值诊断（单个代表起点）
    print(f"\n[A3] 数值诊断（单起点）:", flush=True)
    set_arm_q(env, np.random.default_rng(0).uniform(lo, hi))
    mujoco.mj_forward(env.sim.model._model, env.sim.data._data)
    diag = ik_6d_diag(env, target_pos, tgt_ori)
    print(f"  终止原因: {diag['terminate_reason']}", flush=True)
    print(f"  最终位置残差: {diag['final_pos_err']*1000:.1f}mm", flush=True)
    print(f"  最终朝向残差: {diag['final_ori_err']*1000:.1f}rad ({diag['final_ori_err']*57.3:.1f}deg)", flush=True)
    print(f"  关节限位命中次数: {diag['joint_limit_hits']}", flush=True)
    print(f"  最小 Jacobian 奇异值: {diag['min_singular']:.4f}", flush=True)

    out = json.dumps({
        "target_pos": target_pos.tolist(), "soup_pos": soup_pos.tolist(),
        "A1_pos_ik_ok": pos_ok_cnt, "A1_pos_errs_mm": [e*1000 for e in pos_errs],
        "A2_6d_ik_ok": ok6d, "A3_diag": diag,
    }, ensure_ascii=False, indent=2)
    with open(os.path.join(ROOT, "data", "audit", "m1_p4_p0r2_gateA.json"), "w") as f:
        f.write(out)
    print(f"\n结果: data/audit/m1_p4_p0r2_gateA.json", flush=True)
    pol.close(); env.close(); scratch.close()

if __name__ == "__main__":
    main()
