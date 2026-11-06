"""MVP 方案 B（预闭合）：cream_cheese 开发对象抓取验证。

标准平行夹爪流程：
  导航到 pregrasp（手指 open，tip 在对象上方）→ 姿态对齐
  → 预闭合到 对象宽+余量 → 下降到对象中心 → 完全闭合 → 抬升 → 保持

修正 v1/v2 的「降到对象中心再闭合」顺序错误。
DEVELOPMENT_ONLY。
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

# 复用导航/几何函数
import tools.fpck_mvp_full_grasp as G  # noqa: E402

PRE_CLOSE_MARGIN = 0.03   # 预闭合余量（> tip 半宽 0.008，保证下降不接触）
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


def close_to(env, target_aperture, n_steps=30):
    """闭合夹爪直到 aperture <= target（或 timeout）。"""
    for i in range(n_steps):
        act = np.zeros(7)
        act[-1] = 1.0  # close
        env.step(act)
        if G.aperture(env) <= target_aperture:
            return True, i + 1
    return False, n_steps


def main():
    env = make_env()
    obj_body = "cream_cheese_1_main"
    obj_p0 = G.body_pos(env, obj_body)

    # cream_cheese 几何：沿 x 最小轴宽 = 2*0.0089 = 0.0178m
    obj_width = 0.0178
    closing_target = np.array([1.0, 0.0, 0.0])

    eef0 = G.eef_pos(env)
    z_transit = max(eef0[2], obj_p0[2] + 0.04 + 0.10)
    z_pregrasp = obj_p0[2] + 0.10

    log = {}

    # 1-4. 导航到 pregrasp（手指 open，tip 在对象上方 0.10m）
    G.nav_phase(env, np.array([eef0[0], eef0[1], z_transit]), freeze=("x", "y"))
    G.align_phase(env, closing_target)
    G.nav_phase(env, np.array([obj_p0[0], obj_p0[1], z_transit]), freeze=("z",))
    l, s = G.nav_phase(env, np.array([obj_p0[0], obj_p0[1], z_pregrasp]),
                       freeze=("x", "y"))
    log["descend_pregrasp"] = {"label": l, "steps": s}

    # 5. 预闭合到 对象宽 + 余量（手指在对象上方，不接触）
    pre_aperture = obj_width + PRE_CLOSE_MARGIN
    ok, steps = close_to(env, pre_aperture)
    log["pre_close"] = {"ok": ok, "steps": steps,
                        "target_aperture": pre_aperture,
                        "actual_aperture": G.aperture(env)}
    # 预闭合后检查是否接触（不应接触）
    lf, rf, nf = G.check_finger_contacts(env, obj_body)
    log["pre_close_contact"] = {"L": lf, "R": rf, "non_finger": nf}

    # 6. 下降到 grasp 高度（手指接触对象即停，不追求 tip 精确对齐对象中心）
    tip_off = G.eef_to_tip_offset(env)
    z_grasp = obj_p0[2] - tip_off[2]  # tip 目标到对象中心
    descend_ok = False
    descend_steps = 0
    for i in range(150):
        err = np.array([obj_p0[0], obj_p0[1], z_grasp]) - G.eef_pos(env)
        lf, rf, nf = G.check_finger_contacts(env, obj_body)
        # grasp 高度到达：手指接触对象（接触即停，避免 tip 精确对齐被对象挡住）
        if lf or rf:
            descend_ok = True
            descend_steps = i
            break
        act = np.zeros(7)
        act[:3] = np.clip(np.clip(err, -0.005, 0.005) / 0.05, -0.2, 0.2)
        act[-1] = 0.0  # 保持预闭合
        env.step(act)
        descend_steps = i
    log["descend_grasp"] = {"ok": descend_ok, "steps": descend_steps,
                            "tip_z": float(G.tip_midpoint(env)[2]),
                            "object_z": float(obj_p0[2])}

    # 7. 完全闭合（形成夹持：双指接触 + aperture 连续稳定不再减小）
    close_ok = False
    close_steps = 0
    prev_apertures = []
    for i in range(22):
        act = np.zeros(7)
        act[-1] = 1.0
        env.step(act)
        close_steps += 1
        lf, rf, nf = G.check_finger_contacts(env, obj_body)
        a = G.aperture(env)
        prev_apertures.append(a)
        # 夹持形成：双指接触 + 最近 3 步 aperture 变化 < 0.001（真正夹紧，非仅碰到）
        if lf and rf and not nf and len(prev_apertures) >= 3:
            delta = max(prev_apertures[-3:]) - min(prev_apertures[-3:])
            if delta < 0.001:
                close_ok = True
                break
    log["close"] = {"ok": close_ok, "steps": close_steps,
                    "aperture": G.aperture(env),
                    "object_width": obj_width}

    # 8. 抬升（保持 close 夹持）
    obj_before_lift = G.body_pos(env, obj_body)
    eef_lift_target = G.eef_pos(env) + np.array([0, 0, LIFT_HEIGHT])
    l, s = G.nav_phase(env, eef_lift_target, freeze=("x", "y"),
                       max_trans=0.005, gripper_mode="close")
    obj_after_lift = G.body_pos(env, obj_body)
    log["lift"] = {"label": l, "steps": s,
                   "object_lift_m": float(obj_after_lift[2] - obj_before_lift[2])}

    # 9. 保持
    bilateral_held = True
    rel_pose_errs = []
    for i in range(HOLD_STEPS):
        act = np.zeros(7)
        act[-1] = 1.0
        env.step(act)
        lf, rf, nf = G.check_finger_contacts(env, obj_body)
        if not (lf and rf):
            bilateral_held = False
        rel = G.body_pos(env, obj_body) - G.eef_pos(env)
        rel_pose_errs.append(float(np.linalg.norm(rel)))
    last_rel = rel_pose_errs[-STABLE_WINDOW:]
    rel_stable = (max(last_rel) - min(last_rel)) < 0.002
    log["hold"] = {"bilateral_held": bilateral_held,
                   "relative_pose_stable": rel_stable,
                   "rel_err_range_m": float(max(last_rel) - min(last_rel)),
                   "effective_aperture": G.aperture(env)}

    obj_final = G.body_pos(env, obj_body)
    print("=== cream_cheese 方案 B 抓取 ===")
    for phase, r in log.items():
        print(f"  {phase}: {r}")
    print(f"对象初始 z={obj_p0[2]:.4f} 最终 z={obj_final[2]:.4f} "
          f"实际抬升={(obj_final[2]-obj_p0[2])*1000:.1f}mm")

    env.close()
    out_dir = os.path.join(ROOT, ".cache-local", "fpck_characterization", "mvp")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "mvp_planB_grasp.json"), "w") as f:
        json.dump({"development_only": True, "plan": "B_pre_close",
                   "log": log}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
