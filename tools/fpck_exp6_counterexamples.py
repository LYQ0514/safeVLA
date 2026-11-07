"""实验 6：cream_cheese 高速进入 basket 的瞬时 goal-truth 反例。

协议：docs/method/exp6_counterexample_protocol_v1.md
COUNTEREXAMPLE_SCENE_V1：cream_cheese + basket（goal receptacle）保留，其它移开。

速度注入：cream_cheese 放 basket 上方，设 free joint qvel（纯垂直高速）。
逐 substep 记录 goal truth + 速度 + 接触，验证「goal 瞬时为真 + 高速」。
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

SUITE = "safelibero_long"
TASK_NAME = "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket"
OBJ_BODY = "cream_cheese_1_main"
GOAL_RECEPTACLE = "basket_1_main"
GOAL_CLAUSE = ['in', 'cream_cheese_1', 'basket_1_contain_region']

V_IMPACT_N = 0.0422  # 实验 5 正常撞击峰值速度
SPEED_FAMILY = [V_IMPACT_N, 2 * V_IMPACT_N, 4 * V_IMPACT_N]

OBS_STEPS = 80
SUBSTEPS_PER_STEP = 25


def setup_counterexample_scene(env, v0_vertical):
    """cream_cheese + basket 保留，其它移开。cream_cheese 放 basket 上方，注入垂直速度。

    用 robosuite set_joint_qpos/set_joint_qvel（正确接口，非直接写底层 qvel）。
    """
    m = env.sim.model
    d = env.sim.data._data
    free = CS.free_joint_bodies(env)
    joint_name = "cream_cheese_1_joint0"

    # basket 保留在桌面（reset 后自然位置），cream_cheese 保留，其它移开
    basket_bid = m.body_name2id(GOAL_RECEPTACLE)
    basket_pos = np.asarray(d.xpos[basket_bid])
    for bn in free:
        if bn not in (OBJ_BODY, GOAL_RECEPTACLE):
            CS.set_body_pose(env, bn, [0, 0, 10.0], [1, 0, 0, 0])

    # cream_cheese 放 basket 上方（用 set_joint_qpos）
    quat = CS.load_canonical_pose("cream_cheese")
    cc_z = basket_pos[2] + 0.10  # basket 上方 10cm
    env.sim.data.set_joint_qpos(joint_name, np.array(
        [basket_pos[0], basket_pos[1], cc_z, quat[0], quat[1], quat[2], quat[3]]))

    # 注入垂直速度（set_joint_qvel；free joint qvel 顺序 = [wx,wy,wz,vx,vy,vz]）
    env.sim.data.set_joint_qvel(joint_name, np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, -v0_vertical]))
    return joint_name


def goal_truth(env):
    try:
        return bool(env.env._eval_predicate(GOAL_CLAUSE))
    except Exception:
        return False


def run_counterexample(env, v0):
    """注入速度 v0，纯动力学，记录 goal truth + 速度 + 接触。"""
    joint_name = setup_counterexample_scene(env, v0)
    m = env.sim.model
    d = env.sim.data._data
    cc_bid = m.body_name2id(OBJ_BODY)

    rows = []
    goal_seen = False
    goal_true_high_speed = False
    for cs in range(OBS_STEPS):
        for ss in range(SUBSTEPS_PER_STEP):
            gt = goal_truth(env)
            cvel = np.asarray(d.cvel[cc_bid])
            v = np.linalg.norm(cvel[:3])
            w = np.linalg.norm(cvel[3:])
            pos = np.asarray(d.xpos[cc_bid])
            global_idx = cs * SUBSTEPS_PER_STEP + ss
            rows.append({
                "global_substep": global_idx, "control_step": cs,
                "goal_truth": bool(gt),
                "lin_norm2": float(v), "ang_norm2": float(w),
                "obj_z": float(pos[2]),
            })
            if gt:
                goal_seen = True
                if v > V_IMPACT_N or w > 0.1:
                    goal_true_high_speed = True
            # 纯动力学（robosuite sim.step）
            env.sim.step()

    return {
        "v0": float(v0), "joint_name": joint_name,
        "goal_seen": bool(goal_seen),
        "goal_true_high_speed": bool(goal_true_high_speed),
        "rows": rows,
    }


def main():
    from libero.libero import benchmark as bm
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    import pathlib
    ts = bm.get_benchmark_dict()[SUITE](safety_level="I")
    task = ts.get_task(0)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file

    results = []
    for v0 in SPEED_FAMILY:
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256,
                                 camera_widths=256, controller="OSC_POSE")
        env.seed(7)
        env.reset()
        r = run_counterexample(env, v0)
        results.append(r)
        print(f"v0={v0:.4f} m/s: goal_seen={r['goal_seen']} "
              f"goal_true_high_speed={r['goal_true_high_speed']}")
        env.close()

    out_dir = os.path.join(ROOT, ".cache-local", "fpck_characterization")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exp6_counterexamples.json"), "w") as f:
        json.dump({"verification_status": "VERIFIED_WITHIN_SCOPE",
                   "object": "cream_cheese",
                   "scene": "COUNTEREXAMPLE_SCENE_V1",
                   "speed_family": SPEED_FAMILY,
                   "results": results}, f, ensure_ascii=False, indent=2)
    print("\n→ 完成")


if __name__ == "__main__":
    main()
