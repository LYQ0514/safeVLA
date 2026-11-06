"""MVP v2 分阶段导航验证（cream_cheese 开发对象）。

按 scripted_grasp_feasibility_protocol_v2 的导航状态机：
  MOVE_TO_TRANSIT_HEIGHT → ALIGN_ORIENTATION → HORIZONTAL_TRANSIT → DESCEND_TO_PREGRASP

每阶段只控必要自由度，距离相关 deadline + stall detector。
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

# v2 协议参数
TRANSIT_CLEARANCE = 0.10
POS_TOL = 0.002
MAX_TRANS = 0.01
ACTION_CLIP = 0.2
EFFECTIVE_MIN_PROGRESS = 0.002
SETTLE_MARGIN = 20
HARD_CAP = 150
STALL_WINDOW = 10
STALL_MIN_REDUCTION = 0.001


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


def navigate_phase(env, target_pos, freeze_axes=None, max_steps=None):
    """导航到 target_pos，freeze_axes 指定保持不变的轴（如 ('x','y') 只控 z）。

    freeze_axes: 集合，这些轴的动作固定为 0（保持当前）。
    返回 (label, converged, steps, err_trace)。
    """
    if max_steps is None:
        max_steps = HARD_CAP
    axis_idx = {"x": 0, "y": 1, "z": 2}
    frozen = {axis_idx[a] for a in (freeze_axes or [])}
    err_trace = []
    for i in range(max_steps):
        err = target_pos - eef_pos(env)
        # freeze 轴误差清零（不控制）
        for ai in frozen:
            err[ai] = 0.0
        if np.linalg.norm(err) < POS_TOL:
            return "CONVERGED", i, err_trace
        # stall detector（容差区外）
        act = np.zeros(7)
        act[:3] = np.clip(np.clip(err, -MAX_TRANS, MAX_TRANS) / 0.05,
                          -ACTION_CLIP, ACTION_CLIP)
        act[-1] = -1.0  # open
        env.step(act)
        err_trace.append(float(np.linalg.norm(target_pos - eef_pos(env))))
        # stall: 最近 W 步误差下降 < 阈值
        if len(err_trace) >= STALL_WINDOW:
            delta = err_trace[-STALL_WINDOW] - err_trace[-1]
            if delta < STALL_MIN_REDUCTION:
                return "STALLED", i, err_trace
    return "TIMEOUT", max_steps, err_trace


def main():
    env = make_env()
    obj_body = "cream_cheese_1_main"
    obj_p = body_pos(env, obj_body)
    obj_p0 = obj_p.copy()
    eef0 = eef_pos(env)

    # transit height（协议 §6）
    z_obj_top = obj_p[2] + 0.04  # cream_cheese 半高约 0.04（几何近似，此处用 body 高度）
    z_transit = max(eef0[2], z_obj_top + TRANSIT_CLEARANCE)

    results = {}

    # MVP-1a: MOVE_TO_TRANSIT_HEIGHT（只控 z）
    t1 = np.array([eef0[0], eef0[1], z_transit])
    label, steps, tr = navigate_phase(env, t1, freeze_axes=("x", "y"))
    results["transit_height"] = {"target_z": z_transit, "label": label,
                                  "steps": steps, "final_eef": eef_pos(env).tolist()}
    print(f"MOVE_TO_TRANSIT_HEIGHT: {label} steps={steps} (目标 z={z_transit:.3f})")

    # MVP-1c: HORIZONTAL_TRANSIT（只控 x,y，z 保持 transit）
    # grasp center = 对象 xy
    t2 = np.array([obj_p[0], obj_p[1], z_transit])
    label, steps, tr = navigate_phase(env, t2, freeze_axes=("z",))
    results["horizontal_transit"] = {"target": t2.tolist(), "label": label,
                                      "steps": steps, "final_eef": eef_pos(env).tolist()}
    print(f"HORIZONTAL_TRANSIT: {label} steps={steps} (目标 xy=({obj_p[0]:.3f},{obj_p[1]:.3f}))")

    # MVP-1d: DESCEND_TO_PREGRASP（只控 z，到 grasp+0.10）
    z_pregrasp = obj_p[2] + TRANSIT_CLEARANCE
    t3 = np.array([obj_p[0], obj_p[1], z_pregrasp])
    label, steps, tr = navigate_phase(env, t3, freeze_axes=("x", "y"))
    results["descend_to_pregrasp"] = {"target": t3.tolist(), "label": label,
                                       "steps": steps, "final_eef": eef_pos(env).tolist()}
    print(f"DESCEND_TO_PREGRASP: {label} steps={steps} (目标 z={z_pregrasp:.3f})")

    # 对象是否移动 + 接触检查
    obj_moved = float(np.linalg.norm(body_pos(env, obj_body) - obj_p0))
    d = env.sim.data._data
    m = env.sim.model
    names = env.sim.model.body_names
    obj_contacts = []
    for c in range(d.ncon):
        g1, g2 = d.contact[c].geom1, d.contact[c].geom2
        b1, b2 = names[m.geom_bodyid[g1]], names[m.geom_bodyid[g2]]
        if obj_body in (b1, b2):
            other = b2 if b1 == obj_body else b1
            if other.startswith(("robot0", "gripper0")):
                obj_contacts.append((other, obj_body))
    print(f"对象移动: {obj_moved*1000:.3f}mm  对象↔robot 接触: {obj_contacts if obj_contacts else '无'}")
    results["object_displacement_mm"] = obj_moved * 1000
    results["obj_contacts"] = obj_contacts

    env.close()

    out_dir = os.path.join(ROOT, ".cache-local", "fpck_characterization", "mvp")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "mvp2_phased_navigation.json"), "w") as f:
        json.dump({"development_only": True, "protocol_v2_sha256":
                   "9fa4d0cac3d24c34bbd48f76ad69cbef445bf53d60ba7a75e87dafa5156583f9",
                   "results": results}, f, ensure_ascii=False, indent=2)
    print("\n→ 完成")


if __name__ == "__main__":
    main()
