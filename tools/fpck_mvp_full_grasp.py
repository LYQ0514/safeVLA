"""MVP-3~6：完整脚本化抓取门禁（cream_cheese 开发对象）。

整合 v2 协议全流程：
  MOVE_TO_TRANSIT_HEIGHT → ALIGN_ORIENTATION → HORIZONTAL_TRANSIT
  → DESCEND_TO_PREGRASP → DESCEND_TO_GRASP → CLOSE → VERIFY_BILATERAL → LIFT → HOLD

DEVELOPMENT_ONLY。验证工程骨架，不参与对象选择。
"""

from __future__ import annotations

import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = "/data/safeVLA"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "dataset", "SafeLIBERO", "safelibero"))
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

# v2 协议参数
TRANSIT_CLEARANCE = 0.10
POS_TOL = 0.002
ORI_TOL = 0.035
MAX_TRANS = 0.01
MAX_ROT = 0.05
ACTION_CLIP = 0.2
ACTION_ORI_CLIP = 0.1
DESCENT_MAX_TRANS = 0.005
HARD_CAP = 150
STALL_WINDOW = 10
STALL_MIN_REDUCTION = 0.001
LIFT_HEIGHT = 0.05
H_LIFT_MIN = 0.04
HOLD_STEPS = 10
STABLE_WINDOW = 5


def make_env():
    from libero.libero import benchmark
    from tools.g1a_soup_trace import make_env as _make
    ts = benchmark.get_benchmark_dict()["safelibero_long"](safety_level="I")
    task = ts.get_task(0)
    return _make(ts, task, 0)


def eef_pos(env):
    return np.asarray(env.sim.data._data.site_xpos[
        env.env.robots[0].eef_site_id]).copy()


def body_pos(env, body_name):
    from tools.g1a_soup_trace import body_state
    return body_state(env, body_name)["pos"].copy()


def aperture(env):
    obs = env.env._get_observations()
    gq = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64)
    return float(gq[0] - gq[1])


def finger_pos(env, side):
    d = env.sim.data._data
    m = env.sim.model
    # 用 finger tip（真正接触对象的部分），非 finger body
    # tip: gripper0_finger_joint1_tip (左), gripper0_finger_joint2_tip (右)
    bn = f"gripper0_finger_joint{1 if side == 'left' else 2}_tip"
    bid = m.body_name2id(bn)
    return np.asarray(d.xpos[bid]).copy()


def tip_midpoint(env):
    """左右 finger tip 的中点（抓取参考点）。"""
    return (finger_pos(env, "left") + finger_pos(env, "right")) / 2.0


def eef_to_tip_offset(env):
    """EEF site 到 finger tip 中点的世界系偏移（用于 grasp 高度修正）。"""
    return tip_midpoint(env) - eef_pos(env)


def closing_axis(env):
    v = finger_pos(env, "right") - finger_pos(env, "left")
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.zeros(3)


def eef_rot(env):
    d = env.sim.data._data
    m = env.sim.model
    eef_body = env.env.robots[0].robot_model.eef_name
    bid = m.body_name2id(eef_body)
    return np.asarray(d.xmat[bid]).reshape(3, 3)


def mat2axisangle(R):
    R = np.clip(R, -1.0, 1.0)
    cos = (np.trace(R) - 1.0) / 2.0
    cos = np.clip(cos, -1.0, 1.0)
    angle = np.arccos(cos)
    if angle < 1e-8:
        return np.zeros(3), 0.0
    if angle > np.pi - 1e-6:
        axis = np.sqrt(np.clip((np.diag(R) + 1) / 2, 0, None))
        return axis / (np.linalg.norm(axis) + 1e-9), angle
    so3 = (R - R.T) / (2 * np.sin(angle))
    axis = np.array([so3[2, 1], so3[0, 2], so3[1, 0]])
    n = np.linalg.norm(axis)
    return axis / (n + 1e-9), angle


def build_target_rot(closing_world, approach_world=np.array([0.0, 0.0, -1.0])):
    c = closing_world / np.linalg.norm(closing_world)
    a = approach_world / np.linalg.norm(approach_world)
    body_y = -c  # closing axis = body 局部 -y
    body_z = a
    body_x = np.cross(body_y, body_z)
    body_x = body_x / np.linalg.norm(body_x)
    body_y = np.cross(body_z, body_x)
    body_y = body_y / np.linalg.norm(body_y)
    return np.column_stack([body_x, body_y, body_z])


def nav_phase(env, target, freeze, max_trans=MAX_TRANS, ori_hold=True,
              gripper_mode="open"):
    """位置导航（freeze 轴不动），返回 (label, steps)。

    gripper_mode: "open"(act[-1]=-1), "hold"(act[-1]=0 保持当前), "close"(act[-1]=+1)。
    """
    axis_idx = {"x": 0, "y": 1, "z": 2}
    frozen = {axis_idx[a] for a in freeze}
    grip = {"open": -1.0, "hold": 0.0, "close": 1.0}[gripper_mode]
    err_hist = []
    for i in range(HARD_CAP):
        err = target - eef_pos(env)
        for ai in frozen:
            err[ai] = 0.0
        if np.linalg.norm(err) < POS_TOL:
            return "CONVERGED", i
        act = np.zeros(7)
        act[:3] = np.clip(np.clip(err, -max_trans, max_trans) / 0.05,
                          -ACTION_CLIP, ACTION_CLIP)
        act[-1] = grip
        env.step(act)
        err_hist.append(float(np.linalg.norm(target - eef_pos(env))))
        # stall detector：仅误差 > 2*POS_TOL 时判停滞（协议 v2 §8 容差区关闭）
        if (len(err_hist) >= STALL_WINDOW
                and np.linalg.norm(err) > 2 * POS_TOL):
            if err_hist[-STALL_WINDOW] - err_hist[-1] < STALL_MIN_REDUCTION:
                return "STALLED", i
    return "TIMEOUT", HARD_CAP


def align_phase(env, closing_target):
    """姿态对齐，返回 (label, steps)。"""
    for i in range(HARD_CAP):
        c_cur = closing_axis(env)
        dot = np.clip(np.dot(c_cur, closing_target), -1.0, 1.0)
        ang_err = np.arccos(dot)
        if ang_err < ORI_TOL:
            return "CONVERGED", i
        R_target = build_target_rot(closing_target)
        R_err = R_target @ eef_rot(env).T
        axis, angle = mat2axisangle(R_err)
        act = np.zeros(7)
        act[3:6] = np.clip(axis * min(angle, MAX_ROT) / 0.5,
                           -ACTION_ORI_CLIP, ACTION_ORI_CLIP)
        act[-1] = -1.0
        env.step(act)
    return "TIMEOUT", HARD_CAP


def check_finger_contacts(env, obj_body):
    """返回 (left, right, non_finger) 接触状态。只处理 robot↔object 接触。"""
    d = env.sim.data._data
    m = env.sim.model
    names = env.sim.model.body_names
    left = right = non_finger = False
    for c in range(d.ncon):
        g1, g2 = d.contact[c].geom1, d.contact[c].geom2
        b1, b2 = names[m.geom_bodyid[g1]], names[m.geom_bodyid[g2]]
        if obj_body not in (b1, b2):
            continue
        # 确定 robot side 和 object side（跳过对象自接触 b1==b2）
        if b1 == b2:
            continue  # 对象自接触，非 robot 接触
        robot = b1 if b1.startswith(("robot0", "gripper0")) else b2
        if not robot.startswith(("robot0", "gripper0")):
            continue  # 两个都不是 robot（如对象↔对象），跳过
        if "leftfinger" in robot or "finger_joint1" in robot:
            left = True
        elif "rightfinger" in robot or "finger_joint2" in robot:
            right = True
        else:
            non_finger = True
    return left, right, non_finger


def main():
    env = make_env()
    obj_body = "cream_cheese_1_main"
    obj_p0 = body_pos(env, obj_body)

    # 几何：cream_cheese 最小水平轴 = world x（body-local size x=0.0089 半轴）
    closing_target = np.array([1.0, 0.0, 0.0])

    eef0 = eef_pos(env)
    z_transit = max(eef0[2], obj_p0[2] + 0.04 + TRANSIT_CLEARANCE)
    z_pregrasp = obj_p0[2] + TRANSIT_CLEARANCE

    log = {}

    # 1. transit height
    l, s = nav_phase(env, np.array([eef0[0], eef0[1], z_transit]), freeze=("x", "y"))
    log["transit_height"] = {"label": l, "steps": s}

    # 2. align orientation
    l, s = align_phase(env, closing_target)
    log["align_orientation"] = {"label": l, "steps": s}

    # 3. horizontal transit
    l, s = nav_phase(env, np.array([obj_p0[0], obj_p0[1], z_transit]), freeze=("z",))
    log["horizontal_transit"] = {"label": l, "steps": s}

    # 4. descend to pregrasp
    l, s = nav_phase(env, np.array([obj_p0[0], obj_p0[1], z_pregrasp]), freeze=("x", "y"))
    log["descend_pregrasp"] = {"label": l, "steps": s}

    # 5. descend to grasp（慢速，到手指 tip 对齐对象中心高度）
    # finger tip 在 EEF 下方约 0.011m，故 EEF 目标 z = 对象中心 z + tip_offset
    tip_off = eef_to_tip_offset(env)
    z_grasp = obj_p0[2] - tip_off[2]  # 让 tip 中点 z ≈ 对象中心 z
    l, s = nav_phase(env, np.array([obj_p0[0], obj_p0[1], z_grasp]),
                     freeze=("x", "y"), max_trans=DESCENT_MAX_TRANS)
    log["descend_grasp"] = {"label": l, "steps": s, "tip_offset_z": float(tip_off[2])}

    # 6. close
    close_steps = 0
    close_ok = False
    for i in range(22):  # H_close_empty + ΔH = 17 + 5
        act = np.zeros(7)
        act[-1] = 1.0  # close
        env.step(act)
        close_steps += 1
        lf, rf, nf = check_finger_contacts(env, obj_body)
        if lf and rf and not nf:
            a = aperture(env)
            close_ok = True
            break
    log["close"] = {"label": "OK" if close_ok else "FAIL", "steps": close_steps,
                    "bilateral": close_ok}

    # 7. lift
    obj_before_lift = body_pos(env, obj_body)
    eef_before_lift = eef_pos(env)
    lift_target = eef_before_lift + np.array([0, 0, LIFT_HEIGHT])
    l, s = nav_phase(env, lift_target, freeze=("x", "y"),
                     max_trans=DESCENT_MAX_TRANS)
    obj_after_lift = body_pos(env, obj_body)
    actual_lift = obj_after_lift[2] - obj_before_lift[2]
    log["lift"] = {"label": l, "steps": s, "object_lift_m": float(actual_lift)}

    # 8. hold + 稳定判据
    rel_pose_errs = []
    bilateral_held = True
    for i in range(HOLD_STEPS):
        act = np.zeros(7)
        act[-1] = 1.0  # 保持 close
        env.step(act)
        lf, rf, nf = check_finger_contacts(env, obj_body)
        if not (lf and rf):
            bilateral_held = False
        # 目标-EEF 相对位置（用 object pos - eef pos）
        rel = body_pos(env, obj_body) - eef_pos(env)
        rel_pose_errs.append(float(np.linalg.norm(rel)))
    # 最后 stable_window 步相对位姿稳定
    last_rel = rel_pose_errs[-STABLE_WINDOW:]
    rel_stable = (max(last_rel) - min(last_rel)) < 0.002
    eff_ap = aperture(env)
    log["hold"] = {"bilateral_held": bilateral_held,
                   "relative_pose_stable": rel_stable,
                   "rel_err_range_m": float(max(last_rel) - min(last_rel)),
                   "effective_aperture": float(eff_ap)}

    # 汇总
    print("=== cream_cheese 完整抓取门禁 ===")
    for phase, r in log.items():
        print(f"  {phase}: {r}")
    obj_final = body_pos(env, obj_body)
    print(f"对象初始 z={obj_p0[2]:.4f} 最终 z={obj_final[2]:.4f} "
          f"实际抬升={(obj_final[2]-obj_p0[2])*1000:.1f}mm")

    env.close()
    out_dir = os.path.join(ROOT, ".cache-local", "fpck_characterization", "mvp")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "mvp_full_grasp.json"), "w") as f:
        json.dump({"development_only": True, "log": log}, f,
                  ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
