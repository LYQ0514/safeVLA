"""脚本化抓取可行性门禁 v3 正式实现。

协议：docs/method/scripted_grasp_feasibility_protocol_v3.md
protocol_v3_sha256=e042e0ef7df3a47c8fc38ec7848ee5ffdf9443cae394faf548f06fab41186ff3

实现 v3 状态机：
  INIT → MOVE_TO_TRANSIT_HEIGHT → ALIGN_ORIENTATION → HORIZONTAL_TRANSIT
  → DESCEND_TO_PREGRASP → PRESHAPE_AT_SAFE_HEIGHT → DESCEND_WITH_PRESHAPE
  → FINAL_CLOSE → VERIFY_GRASP → LIFT → HOLD → PASS/FAIL

纯脚本化 env.step 控制，不读 VLA，不特调对象。DEVELOPMENT_ONLY 对象不进入选择。
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from typing import Optional

warnings.filterwarnings("ignore")
ROOT = "/data/safeVLA"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "dataset", "SafeLIBERO", "safelibero"))
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

# v3 协议常量
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
EFFECTIVE_MIN_PROGRESS = 0.002
SETTLE_MARGIN = 20
DELTA_CLEAR = 0.005       # 预闭合每侧 clearance
EPS_A = 0.002             # aperture 容差
LIFT_HEIGHT = 0.05
H_LIFT_MIN = 0.04
HOLD_STEPS = 10
STABLE_WINDOW = 5
CLOSE_TIMEOUT = 22        # H_close_empty(17) + ΔH(5)

# 失败原因
FAILURES = [
    "PREGRASP_UNREACHABLE", "ORIENTATION_UNREACHABLE", "PREGRASP_TIMEOUT",
    "DESCENT_TIMEOUT", "PREMATURE_FINGER_CONTACT", "PREGRASP_NON_FINGER_CONTACT",
    "CLOSE_COMMAND_NOT_APPLIED", "NO_APERTURE_RESPONSE", "NO_LEFT_FINGER_CONTACT",
    "NO_RIGHT_FINGER_CONTACT", "NO_BILATERAL_CONTACT",
    "NON_FINGER_ROBOT_TARGET_CONTACT", "CLOSE_TIMEOUT", "OBJECT_NOT_LIFTED",
    "SUPPORT_CONTACT_PERSISTED", "OBJECT_DROPPED", "RELATIVE_POSE_UNSTABLE",
    "BILATERAL_CONTACT_LOST", "FORBIDDEN_ENVIRONMENT_CONTACT",
    "EFFECTIVE_APERTURE_UNAVAILABLE", "IDENTITY_FAILURE", "RUNTIME_FAILURE",
    "OBJECT_EJECTED_DURING_FINAL_CLOSE", "PRESHAPE_APERTURE_UNREACHABLE",
    "NAVIGATION_STALLED", "TRANSIT_HEIGHT_UNREACHABLE", "TRANSIT_PATH_BLOCKED",
    "SCRIPTED_GRASP_NONDETERMINISTIC",
]


def make_env(suite, task_ref):
    """task_ref 可以是 task_id(int) 或 task_name(str)。独立实例化（不 set_init_state）。

    不复用 g1a_soup_trace.make_env（那个硬编码 TASK_ID=0，multi-object 任务会
    broadcast error）。这里只要 reset 后的 env，对象位置由 BDDL 初始场景决定。
    """
    from libero.libero import benchmark
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    import pathlib
    ts = benchmark.get_benchmark_dict()[suite](safety_level="I")
    if isinstance(task_ref, str):
        task_id = None
        for i in range(ts.n_tasks):
            if ts.get_task(i).name == task_ref:
                task_id = i
                break
        if task_id is None:
            raise ValueError(f"task {task_ref} 不在 suite {suite}")
        task = ts.get_task(task_id)
    else:
        task_id = task_ref
        task = ts.get_task(task_id)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256,
                             camera_widths=256, controller="OSC_POSE")
    env.seed(7)
    env.reset()
    return env


def eef_pos(env):
    return np.asarray(env.sim.data._data.site_xpos[
        env.env.robots[0].eef_site_id]).copy()


def body_pos(env, body_name):
    from tools.g1a_soup_trace import body_state
    return body_state(env, body_name)["pos"].copy()


def current_action(env):
    """sticky gripper 的 current_action（sticky target），量化步长 0.25。"""
    return np.asarray(env.env.robots[0].gripper.current_action).copy()


def aperture(env):
    obs = env.env._get_observations()
    gq = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64)
    return float(gq[0] - gq[1])


# current_action → 稳定 aperture 映射（空载表征，见 fpck_aperture_target_characterization）
# ca: 1.00→79.5mm, 0.75→69.5, 0.50→59.5, 0.25→49.5, 0.00→40.3,
#     -0.25→30.5, -0.50→20.5, -0.75→10.5, -1.00→1.0
CA_TO_APERTURE = {
    1.00: 0.0795, 0.75: 0.0695, 0.50: 0.0595, 0.25: 0.0495, 0.00: 0.0403,
    -0.25: 0.0305, -0.50: 0.0205, -0.75: 0.0105, -1.00: 0.0010,
}

# 预闭合：手指 tip 半宽（tip 内侧间距 = aperture - 2*tip_half_width）
TIP_HALF_WIDTH = 0.004


def preshape_target_ca(obj_width):
    """由对象宽度反求 current_action 目标。

    约束：手指 tip 内侧间距 > 对象宽度，即 aperture > obj_width + 2*tip_half_width。
    选满足此约束的最小 current_action 档位（裁决 §三 argmin 语义）。
    """
    needed_aperture = obj_width + 2 * TIP_HALF_WIDTH
    # 从最小 aperture（ca=-1.0）往上找第一个 > needed 的（最小满足）
    for ca in [-1.00, -0.75, -0.50, -0.25, 0.00, 0.25, 0.50, 0.75, 1.00]:
        if CA_TO_APERTURE[ca] > needed_aperture:
            return ca
    return None  # 无足够大档位（对象超过 full open 可夹）


def servo_current_action(env, target_ca, max_steps=30):
    """控制 current_action 到 target_ca（±0.13），返回 (converged, steps)。"""
    for i in range(max_steps):
        c = current_action(env)[0]
        if abs(c - target_ca) < 0.13:
            return True, i
        act = np.zeros(7)
        act[-1] = 1.0 if c > target_ca else -1.0
        env.step(act)
    return False, max_steps


def finger_pos(env, side):
    d = env.sim.data._data
    m = env.sim.model
    bn = f"gripper0_finger_joint{1 if side == 'left' else 2}_tip"
    bid = m.body_name2id(bn)
    return np.asarray(d.xpos[bid]).copy()


def tip_midpoint(env):
    return (finger_pos(env, "left") + finger_pos(env, "right")) / 2.0


def closing_axis(env):
    v = finger_pos(env, "right") - finger_pos(env, "left")
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.zeros(3)


def eef_rot(env):
    d = env.sim.data._data
    m = env.sim.model
    bid = m.body_name2id(env.env.robots[0].robot_model.eef_name)
    return np.asarray(d.xmat[bid]).reshape(3, 3)


def mat2axisangle(R):
    R = np.clip(R, -1.0, 1.0)
    cos = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
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


def check_finger_contacts(env, obj_body):
    d = env.sim.data._data
    m = env.sim.model
    names = env.sim.model.body_names
    left = right = non_finger = False
    for c in range(d.ncon):
        g1, g2 = d.contact[c].geom1, d.contact[c].geom2
        b1, b2 = names[m.geom_bodyid[g1]], names[m.geom_bodyid[g2]]
        if obj_body not in (b1, b2):
            continue
        if b1 == b2:
            continue  # 对象自接触
        robot = b1 if b1.startswith(("robot0", "gripper0")) else b2
        if not robot.startswith(("robot0", "gripper0")):
            continue
        if "leftfinger" in robot or "finger_joint1" in robot:
            left = True
        elif "rightfinger" in robot or "finger_joint2" in robot:
            right = True
        else:
            non_finger = True
    return left, right, non_finger


def nav_phase(env, target, freeze, max_trans=MAX_TRANS, gripper_mode="open"):
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
        if (len(err_hist) >= STALL_WINDOW
                and np.linalg.norm(err) > 2 * POS_TOL):
            if err_hist[-STALL_WINDOW] - err_hist[-1] < STALL_MIN_REDUCTION:
                return "STALLED", i
    return "TIMEOUT", HARD_CAP


def align_phase(env, closing_target):
    for i in range(HARD_CAP):
        c_cur = closing_axis(env)
        ang_err = np.arccos(np.clip(np.dot(c_cur, closing_target), -1.0, 1.0))
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


class GraspGateResult:
    def __init__(self):
        self.template = None
        self.passed = False
        self._failure_reason = None
        self.outcome = None  # PASS / PHYSICAL_INELIGIBLE / TEMPLATE_FAIL / CONTROLLER_INCONCLUSIVE
        self.phase_log = {}
        self.effective_aperture = None
        self.object_lift_m = 0.0
        self.bilateral_held = False
        self.relative_pose_stable = False

    @property
    def failure_reason(self):
        return self._failure_reason

    @failure_reason.setter
    def failure_reason(self, value):
        self._failure_reason = value
        self.outcome = classify_outcome(value)


# 结果分类（裁决 §四）：不能把 controller 失败解释成物理不可抓
_PHYSICAL_INELIGIBLE_REASONS = {
    "PRESHAPE_APERTURE_UNREACHABLE",
    "TRANSIT_HEIGHT_UNREACHABLE",
}
_CONTROLLER_INCONCLUSIVE_REASONS = {
    "DESCENT_TIMEOUT", "PREGRASP_TIMEOUT", "NAVIGATION_STALLED",
    "ORIENTATION_UNREACHABLE", "PREGRASP_UNREACHABLE",
    "TRANSIT_PATH_BLOCKED",
}
# 其余 → TEMPLATE_FAIL（控制器完成动作但未建立稳定抓取）


def classify_outcome(failure_reason):
    """由 failure_reason 分类结果（裁决 §四四类）。"""
    if failure_reason is None:
        return "PASS"
    if failure_reason in _PHYSICAL_INELIGIBLE_REASONS:
        return "PHYSICAL_INELIGIBLE"
    if failure_reason in _CONTROLLER_INCONCLUSIVE_REASONS:
        return "CONTROLLER_INCONCLUSIVE"
    return "TEMPLATE_FAIL"


def run_grasp_gate(env, obj_body, closing_target, obj_width, grasp_center_xy=None):
    """对单个模板运行完整门禁。返回 GraspGateResult。

    closing_target: 世界系 closing axis 单位向量。
    obj_width: 沿 closing axis 的几何宽度（m）。
    grasp_center_xy: 对象抓取中心 xy（None 用 body xpos xy）。
    """
    r = GraspGateResult()
    obj_p0 = body_pos(env, obj_body)
    cx, cy = (grasp_center_xy if grasp_center_xy is not None
              else (obj_p0[0], obj_p0[1]))

    # 预闭合：由对象宽度反求 current_action 目标（裁决 §四 目标感知 servo）
    target_ca = preshape_target_ca(obj_width)
    if target_ca is None:
        r.failure_reason = "PRESHAPE_APERTURE_UNREACHABLE"
        return r

    eef0 = eef_pos(env)
    z_transit = max(eef0[2], obj_p0[2] + 0.04 + TRANSIT_CLEARANCE)
    if z_transit > 1.3:  # 粗略工作空间上限（后续正式运行时用模型导出）
        r.failure_reason = "TRANSIT_HEIGHT_UNREACHABLE"
        return r

    # 1. transit height
    l, s = nav_phase(env, np.array([eef0[0], eef0[1], z_transit]), freeze=("x", "y"))
    r.phase_log["transit_height"] = (l, s)

    # 2. align
    l, s = align_phase(env, closing_target)
    r.phase_log["align"] = (l, s)

    # 3. horizontal transit
    l, s = nav_phase(env, np.array([cx, cy, z_transit]), freeze=("z",))
    r.phase_log["horizontal"] = (l, s)

    # 4. descend to pregrasp
    l, s = nav_phase(env, np.array([cx, cy, obj_p0[2] + TRANSIT_CLEARANCE]),
                     freeze=("x", "y"))
    r.phase_log["descend_pregrasp"] = (l, s)

    # 5. preshape：servo current_action 到目标（裁决 §四，控制 sticky target 而非 actual）
    servo_ok, servo_steps = servo_current_action(env, target_ca)
    if not servo_ok:
        r.failure_reason = "PRESHAPE_APERTURE_UNREACHABLE"
        return r
    # 检查 preshape 后无接触（手指在对象上方）
    lf, rf, nf = check_finger_contacts(env, obj_body)
    if lf or rf or nf:
        r.failure_reason = "PREMATURE_FINGER_CONTACT" if (lf or rf) else "PREGRASP_NON_FINGER_CONTACT"
        r.phase_log["preshape"] = ("CONTACT", servo_steps)
        return r
    r.phase_log["preshape"] = ("OK", servo_steps, current_action(env)[0], aperture(env))

    # 6. descend：pose 收敛到 grasp pose（裁决 §七：不等待 contact，pose 到即进入 final close）
    tip_off = eef_to_tip_offset(env)
    z_grasp = obj_p0[2] - tip_off[2]
    l, s = nav_phase(env, np.array([cx, cy, z_grasp]), freeze=("x", "y"),
                     max_trans=DESCENT_MAX_TRANS, gripper_mode="hold")
    r.phase_log["descend"] = (l, s, aperture(env))
    if l != "CONVERGED":
        r.failure_reason = "DESCENT_TIMEOUT"
        return r

    # 7. final close（夹紧：双指 + aperture 稳定）
    close_ok = False
    prev_a = []
    for i in range(CLOSE_TIMEOUT):
        act = np.zeros(7)
        act[-1] = 1.0
        env.step(act)
        lf, rf, nf = check_finger_contacts(env, obj_body)
        a = aperture(env)
        prev_a.append(a)
        if nf:
            r.failure_reason = "NON_FINGER_ROBOT_TARGET_CONTACT"
            return r
        if lf and rf and len(prev_a) >= 3 and max(prev_a[-3:]) - min(prev_a[-3:]) < 0.001:
            close_ok = True
            r.phase_log["final_close"] = ("OK", i, a)
            break
    if not close_ok:
        r.failure_reason = "CLOSE_TIMEOUT" if (lf and rf) else "NO_BILATERAL_CONTACT"
        return r

    # 8. verify + lift
    obj_before = body_pos(env, obj_body)[2]
    l, s = nav_phase(env, eef_pos(env) + np.array([0, 0, LIFT_HEIGHT]),
                     freeze=("x", "y"), max_trans=DESCENT_MAX_TRANS, gripper_mode="close")
    obj_after = body_pos(env, obj_body)[2]
    r.object_lift_m = obj_after - obj_before
    r.phase_log["lift"] = (l, s, r.object_lift_m)
    if r.object_lift_m < H_LIFT_MIN:
        r.failure_reason = "OBJECT_NOT_LIFTED"
        return r

    # 9. hold
    bilateral_held = True
    rel_errs = []
    for i in range(HOLD_STEPS):
        act = np.zeros(7)
        act[-1] = 1.0
        env.step(act)
        lf, rf, nf = check_finger_contacts(env, obj_body)
        if not (lf and rf):
            bilateral_held = False
        rel_errs.append(float(np.linalg.norm(
            body_pos(env, obj_body) - eef_pos(env))))
    last = rel_errs[-STABLE_WINDOW:]
    r.bilateral_held = bilateral_held
    r.relative_pose_stable = (max(last) - min(last)) < 0.002
    r.effective_aperture = aperture(env)
    r.phase_log["hold"] = (bilateral_held, r.relative_pose_stable, r.effective_aperture)

    if bilateral_held and r.relative_pose_stable:
        r.passed = True
        r.outcome = "PASS"
    else:
        r.failure_reason = "OBJECT_DROPPED" if not bilateral_held else "RELATIVE_POSE_UNSTABLE"
    return r


def eef_to_tip_offset(env):
    return tip_midpoint(env) - eef_pos(env)


if __name__ == "__main__":
    # 开发自测：cream_cheese 沿 y 轴
    env = make_env("safelibero_long", 0)
    obj_body = "cream_cheese_1_main"
    r = run_grasp_gate(env, obj_body, np.array([0.0, 1.0, 0.0]), 0.0426)
    print("=== cream_cheese 沿 y 门禁 ===")
    print(f"passed={r.passed} failure={r.failure_reason}")
    for k, v in r.phase_log.items():
        print(f"  {k}: {v}")
    print(f"lift={r.object_lift_m*1000:.1f}mm aperture={r.effective_aperture}")
    env.close()
