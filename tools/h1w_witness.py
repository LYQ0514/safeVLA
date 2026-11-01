"""H1-W：同状态三动作非凸性见证（Unsafe-Midpoint Witness）。

物理见证空间：固定夹爪的 6 维外部可执行动作。
代理约束空间：AEGIS 9 维 QP 可行域投影到 6 维。

流程：
1. 执行一致性检查：同一 snapshot 重复执行同一动作，验证物理结果稳定；
   验证外部动作 ↔ u_v/u_omega 双向映射、夹爪一致、物体位移基准不随分支重置。
2. Unsafe-Midpoint Witness 搜索：以不安全动作 a_M 为中心，a_L=a_M+δ, a_R=a_M-δ，
   找 Φ(a_L)=1, Φ(a_R)=1, Φ(a_M)=0 的见证。
3. 对见证做 QP 投影成员判定（a_L/a_R/a_M 是否属于 AEGIS 投影可行域）。

物理分支直接执行候选动作，不经过 AEGIS filter。

数据：仅 task0 快照（aegis_long_design/ep00-05/decision_*.npz）。
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

from src.safety.aegis_adapted import (  # noqa: E402
    AegisSafetyLayer, _quat_wxyz_to_rot,
    QP_ACTION_SCALE, V_REF_GAIN, OMEGA_REF_GAIN,
)

DESIGN = pathlib.Path(ROOT, "data", "audit", "aegis_long_design")
TASK_ID = 0
NUM_STEPS_WAIT = 10
DISPLACEMENT_EPS = 0.001  # 官方 CAR：L1 位移 > 1mm 碰撞

# 工作空间内障碍物（long task0 只有这两个在 ws 内）
WS_OBSTACLE_KEYS = ["red_coffee_mug_obstacle_1_pos", "box_small_base_1_pos"]
ALL_OBSTACLE_POS_KEYS = [
    "moka_pot_small_obstacle_1_pos", "white_storage_box_obstacle_1_pos",
    "milk_small_obstacle_1_pos", "wine_bottle_small_obstacle_1_pos",
    "red_coffee_mug_obstacle_1_pos", "yellow_book_obstacle_1_pos",
    "box_small_base_1_pos",
]


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
    """加载决策点的 snapshot + obs + aegis 状态。"""
    with np.load(DESIGN / f"ep{ep:02d}" / f"decision_{step:03d}.npz") as f:
        d = {k: f[k] for k in f.files}
    j = json.loads((DESIGN / f"ep{ep:02d}" / f"decision_{step:03d}.json").read_text())
    return d, j


def restore_env(env, d):
    """从快照恢复完整状态 + obs。"""
    env.sim.set_state_from_flattened(d["snapshot"])
    env.sim.forward()
    obs = {k: d[k] for k in d if k in ("robot0_eef_pos", "robot0_eef_quat",
                                        "robot0_gripper_qpos", "red_coffee_mug_obstacle_1_pos",
                                        "box_small_base_1_pos")
           or k in ALL_OBSTACLE_POS_KEYS}
    # 完整 obs 键（用于 AEGIS layer 需要的 _pos / _quat）
    return obs


def restore_aegis(layer, d, j):
    """恢复 AEGIS 控制器状态。"""
    st = {}
    sc = j["aegis_state_scalar"]
    st["obstacle_name"] = sc["obstacle_name"]
    st["feasible_count"] = sc["feasible_count"]
    st["infeasible_count"] = sc["infeasible_count"]
    st["intervene_count"] = sc["intervene_count"]
    for k in ["p2", "R2", "Q2_diag", "z_fixed", "Q1_diag", "_last_p1"]:
        key = "aegis_" + k
        st[k] = d[key] if key in d else None
    layer.restore_state(st)


def physical_safety(env, d, action_6, H, gripper_cmd, contract):
    """从 snapshot 恢复，直接执行 action_6（H 步），判定位移/接触契约。

    action_6: (6,) 连续动作。gripper_cmd: 夹爪（固定）。
    直接 env.step，不经过 AEGIS filter。
    返回 dict: {collided, max_disp, contact, ...}
    """
    env.sim.set_state_from_flattened(d["snapshot"])
    env.sim.forward()

    # 记录初始障碍物位置（位移基准，不随分支重置）
    init_pos = {}
    for k in WS_OBSTACLE_KEYS:
        if k in d:
            init_pos[k] = np.asarray(d[k], dtype=np.float64).copy()

    # 机器人/障碍物碰撞几何（接触契约）
    m = env.sim.model._model
    robot_geoms = {g for g in range(m.ngeom) if m.geom_contype[g] > 0 and
                   (env.sim.model.body_names[m.geom_bodyid[g]].startswith("robot0") or
                    env.sim.model.body_names[m.geom_bodyid[g]].startswith("gripper0"))}
    obstacle_names = set(n.replace("_joint0", "") for n in env.sim.model.joint_names if "obstacle" in n)
    obstacle_geoms = {g for g in range(m.ngeom) if m.geom_contype[g] > 0 and
                      any(env.sim.model.body_names[m.geom_bodyid[g]].startswith(o) for o in obstacle_names)}

    contact_collided = False
    max_disp = 0.0
    for step in range(H):
        act = np.zeros(7, dtype=np.float64)
        act[:6] = action_6
        act[6] = gripper_cmd
        env.step(act)
        # 接触判定
        dd = env.sim.data._data
        for c in range(dd.ncon):
            g1, g2 = dd.contact[c].geom1, dd.contact[c].geom2
            if (g1 in robot_geoms and g2 in obstacle_geoms) or (g2 in robot_geoms and g1 in obstacle_geoms):
                contact_collided = True
        # 位移判定（读工作空间内障碍物位置）
        for k in WS_OBSTACLE_KEYS:
            if k in init_pos:
                pass  # 位移在循环结束后统一从 obs_after 读

    # 位移：执行后读障碍物最新位置
    obs_after = env.env._get_observations()
    for k in WS_OBSTACLE_KEYS:
        if k in init_pos and k in obs_after:
            disp = float(np.sum(np.abs(np.asarray(obs_after[k]) - init_pos[k])))
            max_disp = max(max_disp, disp)

    disp_collided = max_disp > DISPLACEMENT_EPS

    if contract == "disp":
        collided = disp_collided
    elif contract == "contact":
        collided = contact_collided
    else:
        raise ValueError(contract)

    return {"collided": collided, "max_disp": max_disp,
            "contact_collided": contact_collided, "disp_collided": disp_collided}


def consistency_check(ts, task, ep, step):
    """执行一致性：同一动作重复执行，物理结果是否稳定。

    判定：collided 布尔一致 + max_disp 在容差内（浮点末位差不算不一致）。
    """
    d, j = load_decision(ep, step)
    env = make_env(ts, task)
    action_6 = np.zeros(6)
    gripper_cmd = -1.0
    results = []
    for _ in range(3):
        r = physical_safety(env, d, action_6, H=1, gripper_cmd=gripper_cmd, contract="disp")
        results.append(r)
    env.close()
    collided = [r["collided"] for r in results]
    disp = [r["max_disp"] for r in results]
    collided_same = all(c == collided[0] for c in collided)
    disp_same = all(abs(d - disp[0]) < 1e-9 for d in disp)
    same = collided_same and disp_same
    return same, [(r["collided"], r["max_disp"]) for r in results]


def qp_membership(d, j, action_6):
    """判定 action_6 是否属于 AEGIS 投影 QP 可行域（u_z 存在性）。

    外部动作 → u_v = 5 R1^T a_v, u_omega = 5 a_omega。
    QP 可行域约束：a_u_v @ u[:3] + a_u_omega @ u[3:6] + a_uz @ u[6:] + 10h >= 0。
    检查是否存在 u_z 使约束满足（这里简化：约束对 u_z 是线性的 a_uz @ u_z，
    若 a_uz 非零则总能选 u_z 使约束满足；若 a_uz 全零则约束退化）。
    """
    from utils import compute_h_coeffs_3d
    env = None  # 不需要 env，用 obs 里的量重建
    # 需要 R1（eef 旋转）、p1（eef 中心）、p2、Q1、Q2、z_fixed
    # 从 d（obs）和 aegis 状态取
    obs_eef_quat = d["robot0_eef_quat"]
    R1 = _quat_wxyz_to_rot(np.asarray(obs_eef_quat, dtype=np.float64))
    a_v_ext = action_6[:3]
    a_omega_ext = action_6[3:6]
    u_v = V_REF_GAIN * (R1.T @ a_v_ext)
    u_omega = OMEGA_REF_GAIN * a_omega_ext

    sc = j["aegis_state_scalar"]
    obstacle_name = sc["obstacle_name"]
    # p2, Q2, R2 从 aegis 状态
    p2 = d["aegis_p2"] if "aegis_p2" in d else None
    Q2 = d["aegis_Q2_diag"] if "aegis_Q2_diag" in d else None
    R2 = d["aegis_R2"] if "aegis_R2" in d else None
    z_fixed = d["aegis_z_fixed"] if "aegis_z_fixed" in d else None
    Q1 = d["aegis_Q1_diag"] if "aegis_Q1_diag" in d else None
    if p2 is None or Q2 is None or R2 is None or z_fixed is None or Q1 is None:
        return {"status": "missing_aegis_state"}

    # p1 = eef 中心（eef pos 沿局部 z 偏移 -0.08）
    eef_pos = np.asarray(d["robot0_eef_pos"], dtype=np.float64)
    p1 = eef_pos + R1 @ np.array([0.0, 0.0, -0.08])

    a_v, a_omega, a_uz, h, mu_row = compute_h_coeffs_3d(
        p1, Q1, R1, p2, Q2, R2, z_fixed)
    a_u_v = QP_ACTION_SCALE * a_v
    a_u_omega = QP_ACTION_SCALE * a_omega

    m_physical = float(a_u_v @ u_v + a_u_omega @ u_omega + 10.0 * h)
    # a_uz 对 u_z 的线性项
    a_uz_vec = a_uz  # (3,)
    # 约束：a_uz @ u_z + m_physical >= 0
    # u_z 存在性：若 a_uz 非零，总能选 u_z 使约束满足（u_z 无界）
    # 因此 membership 取决于是否存在 u_z；若 a_uz 非零则恒可行
    a_uz_norm = float(np.linalg.norm(a_uz_vec))
    if a_uz_norm > 1e-12:
        feasible = True  # 总能选 u_z
    else:
        feasible = (m_physical >= 0)

    return {
        "status": "ok",
        "m_physical": m_physical,
        "m_full_feasible": feasible,
        "a_uz_norm": a_uz_norm,
        "h": float(h),
    }


def search_witness(ts, task, ep, step, contract, H, delta_list, budget=10):
    """Unsafe-Midpoint Witness 搜索。

    以 nominal 动作为中心 a_M，尝试 δ 方向找 a_L=a_M+δ, a_R=a_M-δ 满足
    Φ(a_L)=1, Φ(a_R)=1, Φ(a_M)=0。
    """
    d, j = load_decision(ep, step)
    env = make_env(ts, task)
    # a_M = nominal 第 1 步的 6 维（但需要外部动作空间；用零动作或 nominal？）
    # 这里 a_M 用 nominal 的 pose 部分
    nominal = d["nominal"]  # (10,7)
    a_M = nominal[0][:6].astype(np.float64)
    gripper_cmd = nominal[0][6]

    # 先判 a_M 安全性
    r_M = physical_safety(env, d, a_M, H, gripper_cmd, contract)

    witnesses = []
    for delta in delta_list:
        # delta 是 6 维扰动向量
        a_L = a_M + delta
        a_R = a_M - delta
        # clip 到 [-1,1]
        a_L = np.clip(a_L, -1, 1)
        a_R = np.clip(a_R, -1, 1)
        r_L = physical_safety(env, d, a_L, H, gripper_cmd, contract)
        r_R = physical_safety(env, d, a_R, H, gripper_cmd, contract)
        # 见证：a_M 不安全，两侧安全
        if (not r_M["collided"]) and (not r_L["collided"]) and r_R["collided"]:
            # 注意：这里要找的是 a_M 不安全、两侧安全，即 Φ(a_M)=0, Φ(a_L)=1, Φ(a_R)=1
            pass
        # 正确判定：Φ(a_M)=0 且 Φ(a_L)=1 且 Φ(a_R)=1
        if r_M["collided"] and (not r_L["collided"]) and (not r_R["collided"]):
            # 这个 delta 找到见证
            witnesses.append({
                "a_M": a_M.tolist(), "a_L": a_L.tolist(), "a_R": a_R.tolist(),
                "delta": delta.tolist(),
                "r_M": r_M, "r_L": r_L, "r_R": r_R,
            })

    env.close()
    return witnesses, {"a_M_safe": not r_M["collided"], "r_M": r_M}


def main():
    import argparse
    from libero.libero import benchmark

    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", type=int, default=0)
    ap.add_argument("--step", type=int, default=0)
    ap.add_argument("--H", type=int, default=1)
    args = ap.parse_args()

    ts = benchmark.get_benchmark_dict()["safelibero_long"](safety_level="I")
    task = ts.get_task(TASK_ID)

    print(f"=== H1-W 执行一致性检查 (ep{args.ep} step{args.step}) ===")
    same, results = consistency_check(ts, task, args.ep, args.step)
    print(f"  同一动作 3 次重复: 一致={same}  结果={results}")
    if not same:
        print("  ⚠️ 执行不一致，先定位基础设施问题，不允许用不一致结果判非凸。")
        return

    print(f"\n=== Unsafe-Midpoint Witness 搜索 (H={args.H}) ===")
    # δ 方向列表：先低维切片（xyz 平移、xyz 旋转各方向）
    deltas = []
    for i in range(3):
        for mag in [0.05, 0.1, 0.2, 0.3, 0.5]:
            delta = np.zeros(6)
            delta[i] = mag
            deltas.append(delta)
    for i in range(3, 6):
        for mag in [0.1, 0.2, 0.4, 0.6]:
            delta = np.zeros(6)
            delta[i] = mag
            deltas.append(delta)

    for contract in ["disp", "contact"]:
        witnesses, info = search_witness(ts, task, args.ep, args.step, contract, args.H, deltas)
        print(f"\n  [{contract} 契约] a_M 安全={info['a_M_safe']} 见证数={len(witnesses)}")
        for w in witnesses[:3]:
            print(f"    δ={np.round(w['delta'],3)} | Φ(a_L)={not w['r_L']['collided']} "
                  f"Φ(a_M)={not w['r_M']['collided']} Φ(a_R)={not w['r_R']['collided']}")
            # QP 成员判定
            for name, act in [("a_L", w["a_L"]), ("a_M", w["a_M"]), ("a_R", w["a_R"])]:
                qp = qp_membership(load_decision(args.ep, args.step)[0],
                                   load_decision(args.ep, args.step)[1],
                                   np.array(act))
                print(f"      {name} QP: {qp['status']} m_physical={qp.get('m_physical'):.3f} "
                      f"feasible={qp.get('m_full_feasible')}")


if __name__ == "__main__":
    main()
