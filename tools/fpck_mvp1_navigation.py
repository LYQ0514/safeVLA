"""MVP-1：安全 waypoint 导航（导航到 cream_cheese 上方 pregrasp，不下降不抓取）。

裁决 §二/§四：分两段 waypoint，验证闭环收敛 + 无接触 + 对象不动。
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

PREGRASP_CLEARANCE = 0.10
POS_TOL = 0.002
MAX_TRANS = 0.01  # m/step（脚本导航上限）
ACTION_CLIP = 0.2  # 归一化动作上限（0.01/0.05）
MAX_STEPS = 40


def make_env():
    from libero.libero import benchmark
    from tools.g1a_soup_trace import make_env as _make
    ts = benchmark.get_benchmark_dict()["safelibero_long"](safety_level="I")
    task = ts.get_task(0)  # SCENE1 cream_cheese
    return _make(ts, task, 0)


def eef_pos(env):
    return np.asarray(env.sim.data._data.site_xpos[
        env.env.robots[0].eef_site_id]).copy()


def body_pos(env, body_name):
    from tools.g1a_soup_trace import body_state
    return body_state(env, body_name)["pos"].copy()


def navigate_to(env, target_pos, max_steps=MAX_STEPS):
    """闭环导航到 target_pos，返回 (converged, steps, trace)。"""
    trace = []
    for i in range(max_steps):
        err = target_pos - eef_pos(env)
        if np.linalg.norm(err) < POS_TOL:
            return True, i, trace
        act = np.zeros(7)
        act[:3] = np.clip(np.clip(err, -MAX_TRANS, MAX_TRANS) / 0.05,
                          -ACTION_CLIP, ACTION_CLIP)
        act[-1] = -1.0  # 保持 open
        env.step(act)
        trace.append({"step": i, "eef": eef_pos(env).tolist(),
                      "err_norm": float(np.linalg.norm(target_pos - eef_pos(env)))})
    return False, max_steps, trace


def main():
    env = make_env()
    obj_body = "cream_cheese_1_main"
    obj_p0 = body_pos(env, obj_body)

    # waypoint 1: 当前 EEF 上方 +0.03m（小单轴测试）
    p_start = eef_pos(env)
    wp1 = p_start + np.array([0, 0, 0.03])
    # waypoint 2: 对象上方 pregrasp
    wp2 = obj_p0 + np.array([0, 0, PREGRASP_CLEARANCE])

    results = {}
    # waypoint 1
    conv1, steps1, tr1 = navigate_to(env, wp1)
    obj_after1 = body_pos(env, obj_body)
    obj_moved1 = float(np.linalg.norm(obj_after1 - obj_p0))

    # waypoint 2
    conv2, steps2, tr2 = navigate_to(env, wp2)
    obj_after2 = body_pos(env, obj_body)
    obj_moved2 = float(np.linalg.norm(obj_after2 - obj_p0))

    print("=== MVP-1 安全 waypoint 导航 ===")
    print(f"对象初始 pos: {obj_p0.round(4)}")
    print(f"waypoint1 (+0.03m): 收敛={conv1} 步数={steps1} 对象移动={obj_moved1*1000:.2f}mm")
    print(f"waypoint2 (pregrasp): 收敛={conv2} 步数={steps2} 对象移动={obj_moved2*1000:.2f}mm")
    print(f"最终 EEF: {eef_pos(env).round(4)} 目标: {wp2.round(4)}")

    # 接触检查
    d = env.sim.data._data
    m = env.sim.model
    names = env.sim.model.body_names
    robot_prefix = ("robot0", "gripper0")
    obj_contacts = []
    for c in range(d.ncon):
        g1, g2 = d.contact[c].geom1, d.contact[c].geom2
        b1, b2 = names[m.geom_bodyid[g1]], names[m.geom_bodyid[g2]]
        if obj_body in (b1, b2):
            other = b2 if b1 == obj_body else b1
            if other.startswith(robot_prefix):
                obj_contacts.append((other, obj_body))
    print(f"对象↔robot 接触: {obj_contacts if obj_contacts else '无'}")

    env.close()

    out_dir = os.path.join(ROOT, ".cache-local", "fpck_characterization", "mvp")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "mvp1_navigation.json"), "w") as f:
        json.dump({"development_only": True,
                   "wp1": {"converged": conv1, "steps": steps1, "obj_moved_mm": obj_moved1*1000},
                   "wp2": {"converged": conv2, "steps": steps2, "obj_moved_mm": obj_moved2*1000},
                   "obj_contacts": obj_contacts},
                  f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
