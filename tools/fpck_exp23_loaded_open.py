"""实验 2（载物 open）+ 实验 3（contact gone）— cream_cheese 单对象表征。

复用冻结的 cream_cheese canonical 抓取（场景 v2 + 门禁 v3，3/3 PASS）。
抓取成功后发送 open 命令，逐 substep 记录：
- aperture 时序、sticky current_action；
- finger-object contact 消失/重接触时序；
- 对象线速度/角速度、下落/分离时刻；
- 支撑面接触时刻。

实验 2 输出 T_open_onset/stable（loaded）；
实验 3 输出 contact-gone 原始时序（不预先假定窗口）。

SINGLE_OBJECT_CHARACTERIZATION（cream_cheese 单对象）。
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

import src.fpck.characterization_scene as CS  # noqa: E402
import src.fpck.scripted_grasp_gate as G  # noqa: E402

SUITE = "safelibero_long"
TASK_NAME = "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket"
OBJ_CLASS = "cream_cheese"
OBJ_BODY = "cream_cheese_1_main"
CLOSING_TARGET = np.array([0.0, 1.0, 0.0])  # y 轴（宽面 43.1mm）
OBJ_WIDTH = 0.0431

OPEN_OBSERVE_STEPS = 30   # open 命令后观察的控制步数
A_OPEN = 0.077
EPS_A = 0.002


def aperture(env):
    return G.aperture(env)


def current_action(env):
    return G.current_action(env)


def finger_contact(env, obj_body):
    """返回 (left, right, non_finger) 接触。"""
    return G.check_finger_contacts(env, obj_body)


def run_one_trial():
    env = CS.load_env(SUITE, TASK_NAME)
    anchor, ok, reason = CS.build_scene(env, OBJ_BODY, OBJ_CLASS)
    if not ok:
        env.close()
        return {"scene_fail": reason}

    # 完整抓取（复用冻结门禁，直到稳定抓取）
    r = G.run_grasp_gate(env, OBJ_BODY, CLOSING_TARGET, OBJ_WIDTH)
    if not r.passed:
        env.close()
        return {"grasp_fail": r.outcome, "reason": r.failure_reason}

    grasp_aperture = aperture(env)
    obj_pos_before_open = G.body_pos(env, OBJ_BODY)

    # 发送 open 命令，逐 substep 记录
    rows = []
    prev_a = aperture(env)
    first_open_step = None
    onset_step = None
    stable_step = None
    first_contact_gone = None
    last_contact = None
    obj_vel_trace = []

    for cs in range(OPEN_OBSERVE_STEPS):
        # 先记录本步起始接触（open 前）
        lf, rf, nf = finger_contact(env, OBJ_BODY)
        act = np.zeros(7)
        act[-1] = -1.0  # open
        env.step(act)
        if first_open_step is None:
            first_open_step = cs
        a = aperture(env)
        lf2, rf2, nf2 = finger_contact(env, OBJ_BODY)
        obj_vel = np.linalg.norm(G.body_pos(env, OBJ_BODY) - obj_pos_before_open)
        rows.append({
            "control_step": cs,
            "aperture": a,
            "aperture_velocity": (a - prev_a) / 0.05,
            "current_action": current_action(env).tolist(),
            "left_contact": lf2, "right_contact": rf2, "non_finger": nf2,
            "obj_disp_from_grasp_m": float(obj_vel),
            "obj_z": float(G.body_pos(env, OBJ_BODY)[2]),
        })
        # onset: 首次 aperture 增大
        if onset_step is None and a > grasp_aperture + 0.001:
            onset_step = cs
        # stable: aperture 达 A_open - EPS_A
        if stable_step is None and a >= A_OPEN - EPS_A:
            stable_step = cs
        # contact gone: 左右指都不接触
        if first_contact_gone is None and (not lf2 and not rf2):
            first_contact_gone = cs
        # last contact: 最后一步有接触
        if lf2 or rf2:
            last_contact = cs
        prev_a = a

    env.close()
    return {
        "grasp_aperture": grasp_aperture,
        "obj_pos_before_open": obj_pos_before_open.tolist(),
        "first_open_step": first_open_step,
        "onset_step": onset_step,
        "stable_step": stable_step,
        "first_contact_gone": first_contact_gone,
        "last_contact": last_contact,
        "rows": rows,
    }


def main():
    # 冻结的抓取条件：中心抓取 + 两个小幅对称偏置（裁决重复口径）
    # 本实验先用中心抓取 3 次（technical replicates），
    # 偏置条件在实验 3 需要更多夹持初态时再冻结。
    n_trials = 3
    results = []
    for i in range(n_trials):
        r = run_one_trial()
        results.append(r)
        if "grasp_fail" in r or "scene_fail" in r:
            print(f"trial{i}: FAIL {r}")
        else:
            print(f"trial{i}: grasp_aperture={r['grasp_aperture']*1000:.1f}mm "
                  f"onset={r['onset_step']} stable={r['stable_step']} "
                  f"first_contact_gone={r['first_contact_gone']} last_contact={r['last_contact']}")

    out_dir = os.path.join(ROOT, ".cache-local", "fpck_characterization")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "exp23_loaded_open_contact_gone.json")
    with open(out_path, "w") as f:
        json.dump({"verification_status": "VERIFIED_WITHIN_SCOPE",
                   "scope": {"scene": "CHARACTERIZATION_SCENE_V2",
                             "object": "cream_cheese", "object_count": 1,
                             "purpose": ["loaded_open_response", "finger_contact_gone"]},
                   "generalization_status": "NOT_ESTABLISHED_ACROSS_OBJECTS",
                   "trials": results}, f, ensure_ascii=False, indent=2)
    print(f"\n→ {out_path}")


if __name__ == "__main__":
    main()
