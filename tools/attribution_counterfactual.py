"""step330 同状态动力学归因实验。

反事实：同一快照，分别执行 a_M 与 a_0（零外部动作，夹爪一致），
记录 mug 的终点位置、速度、逐 substep 接触与冲量。

回答：mug 的 4.58mm 运动是「当前动作造成」还是「历史残余惯性」。

物理真值只读同步后的 MuJoCo 状态（sim.data.xpos/cvel），
绝不用 env._get_observations 缓存替代安全真值。
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

from tools.h1w_witness import load_decision, make_env  # noqa: E402

TASK_ID = 0
NUM_STEPS_WAIT = 10


def run_branch(ts, task, ep, step, action_6, gripper_cmd):
    """从快照执行 action_6（经 controller），逐 substep 记录 mug 真值运动。"""
    d, j = load_decision(ep, step)
    env = make_env(ts, task)
    env.sim.set_state_from_flattened(d["snapshot"])
    env.sim.forward()

    inner = env.env
    m = env.sim.model._model
    mug_bid = env.sim.model.body_name2id("red_coffee_mug_obstacle_1_main")
    box_bid = env.sim.model.body_name2id("box_small_base_1_main")
    cream_bid = env.sim.model.body_name2id("cream_cheese_1_main")

    act = np.zeros(7, dtype=np.float64)
    act[:6] = action_6
    act[6] = gripper_cmd

    p0_mug = np.asarray(env.sim.data._data.xpos[mug_bid]).copy()
    v0_mug = np.asarray(env.sim.data._data.cvel[mug_bid]).copy()

    def short_body(g):
        bn = env.sim.model.body_names[m.geom_bodyid[g]]
        if bn.startswith(("robot0", "gripper0")):
            return "ROBOT"
        for k in ("cream_cheese", "coffee_mug", "box_small_base", "soup", "tomato", "ketchup", "basket"):
            if k in bn:
                return k
        if "table" in bn or "living_room" in bn:
            return "table"
        return bn

    records = []
    contact_seen = set()
    for i in range(int(inner.control_timestep / inner.model_timestep)):
        policy_step = (i == 0)
        env.sim.forward()
        inner._pre_action(act, policy_step)
        env.sim.step()
        dd = env.sim.data._data
        p_mug = np.asarray(dd.xpos[mug_bid]).copy()
        v_mug = np.asarray(dd.cvel[mug_bid]).copy()
        # 接触对（涉及 mug 或 box 或 cream）
        for c in range(dd.ncon):
            g1, g2 = dd.contact[c].geom1, dd.contact[c].geom2
            b1, b2 = short_body(g1), short_body(g2)
            pair = tuple(sorted([b1, b2]))
            contact_seen.add(pair)
        records.append({
            "sub": i,
            "mug_disp_mm": float(np.linalg.norm(p_mug - p0_mug) * 1000),
            "mug_vel": float(np.linalg.norm(v_mug[:3])),
            "mug_pos": p_mug.tolist(),
        })

    p_end_mug = np.asarray(env.sim.data._data.xpos[mug_bid]).copy()
    v_end_mug = np.asarray(env.sim.data._data.cvel[mug_bid]).copy()

    # cream 是否被带动
    p_end_cream = np.asarray(env.sim.data._data.xpos[cream_bid]).copy()
    p0_cream = None  # 后面补

    env.close()
    return {
        "records": records,
        "p0_mug": p0_mug, "p_end_mug": p_end_mug,
        "v0_mug": v0_mug[:3], "v_end_mug": v_end_mug[:3],
        "mug_disp_mm": float(np.linalg.norm(p_end_mug - p0_mug) * 1000),
        "contacts_seen": sorted(contact_seen),
    }


def main():
    import argparse
    from libero.libero import benchmark

    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", type=int, default=0)
    ap.add_argument("--step", type=int, default=330)
    args = ap.parse_args()

    ts = benchmark.get_benchmark_dict()["safelibero_long"](safety_level="I")
    task = ts.get_task(TASK_ID)

    d, j = load_decision(args.ep, args.step)
    a_M = d["nominal"][0][:6].astype(np.float64)
    gripper = d["nominal"][0][6]
    a_0 = np.zeros(6, dtype=np.float64)

    print(f"=== step{args.step} 同状态动力学归因实验 ===")
    print(f"a_M = {np.round(a_M, 4)}")
    print(f"a_0 = {a_0}")
    print()

    res_M = run_branch(ts, task, args.ep, args.step, a_M, gripper)
    res_0 = run_branch(ts, task, args.ep, args.step, a_0, gripper)

    print("=== 结果对比 ===")
    print(f"{'量':>28} {'a_M':>12} {'a_0':>12}")
    print(f"{'mug 初速度 |v0| (m/s)':>28} {np.linalg.norm(res_M['v0_mug']):>12.4f} {np.linalg.norm(res_0['v0_mug']):>12.4f}")
    print(f"{'mug 终点位移 (mm)':>28} {res_M['mug_disp_mm']:>12.3f} {res_0['mug_disp_mm']:>12.3f}")
    print(f"{'mug 末速度 |v| (m/s)':>28} {np.linalg.norm(res_M['v_end_mug']):>12.4f} {np.linalg.norm(res_0['v_end_mug']):>12.4f}")
    print()

    # 因果差异
    delta_causal = abs(res_M['mug_disp_mm'] - res_0['mug_disp_mm'])
    print(f"Δ_causal = |disp(a_M) - disp(a_0)| = {delta_causal:.3f} mm")
    print()
    print("=== 接触对（a_M 分支）===")
    for c in res_M['contacts_seen']:
        print("  ", c)
    print("=== 接触对（a_0 分支）===")
    for c in res_0['contacts_seen']:
        print("  ", c)

    # 判决
    print()
    if delta_causal < 0.5:
        print("判决：两种动作下 mug 运动近乎一致 → 历史动力学/共同环境作用，不归因当前 a_M")
    else:
        print("判决：a_M 明显改变 mug 运动 → 需查间接接触链")


if __name__ == "__main__":
    main()
