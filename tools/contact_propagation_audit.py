"""ep00 step330 单案例接触传播审计。

重建执行 a_M 时 cream_cheese 的 375mm 运动，逐 substep 记录：
- cream_cheese 位置/线速度/角速度
- 所有涉及 cream_cheese 的接触对（不只 FORBIDDEN）
- 接触顺序（robot / table / mug / 其他物体）

只使用现有快照，不新增整条 rollout。
判定 375mm 是「合法接触带动」还是「失控加速」。
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


def substep_rebuild(ts, task, ep, step, action_6, gripper_cmd, record_all=True):
    """逐 substep 执行 action，记录 cream_cheese 运动 + 接触链。"""
    d, j = load_decision(ep, step)
    env = make_env(ts, task)
    env.sim.set_state_from_flattened(d["snapshot"])
    env.sim.forward()

    m = env.sim.model._model
    dd = env.sim.data._data
    cream_bid = env.sim.model.body_name2id("cream_cheese_1_main")

    # 物体 body 名 → 简称（用于接触链记录）
    def short_body(g):
        bn = env.sim.model.body_names[m.geom_bodyid[g]]
        for prefix in ("robot0", "gripper0"):
            if bn.startswith(prefix):
                return "ROBOT:" + bn
        if "cream_cheese" in bn:
            return "cream_cheese"
        if "mug" in bn or "coffee" in bn:
            return "mug"
        if "box" in bn:
            return "box_base"
        if "table" in bn or "living_room" in bn:
            return "table"
        if "soup" in bn:
            return "soup"
        if "tomato" in bn:
            return "tomato"
        if "ketchup" in bn:
            return "ketchup"
        if "basket" in bn:
            return "basket"
        return bn

    # 初始 cream 状态
    p0 = np.asarray(dd.xpos[cream_bid]).copy()
    v0 = np.asarray(dd.cvel[cream_bid]).copy()

    # 逐个 substep
    records = []
    contact_pairs_seen = []
    dt = m.opt.timestep

    for sub in range(25):
        # 逐 substep 推进（不通过 env.step，直接 sim 层面）
        # robosuite 的 step 是 controller 输出 torque + sim.step，这里我们只关心物理
        # 但直接 sim.step 需要先 forward + 施加控制
        # 简化：用 env.step 会一次性走 25 substep，看不到中间过程
        # 所以这里用底层：先记录当前状态，再 sim.forward + sim.step
        dd = env.sim.data._data
        p = np.asarray(dd.xpos[cream_bid]).copy()
        v = np.asarray(dd.cvel[cream_bid]).copy()
        # 记录接触
        contacts_this_sub = []
        for c in range(dd.ncon):
            g1, g2 = dd.contact[c].geom1, dd.contact[c].geom2
            b1, b2 = short_body(g1), short_body(g2)
            if "cream_cheese" in (b1, b2):
                contacts_this_sub.append((b1, b2))
        records.append({
            "sub": sub,
            "pos": p.tolist(),
            "lin_vel": v[:3].tolist(),
            "ang_vel": v[3:].tolist(),
            "disp_from_start_mm": float(np.linalg.norm(p - p0) * 1000),
            "cream_contacts": contacts_this_sub,
        })
        # 推进一个 substep（先 forward 让 contact 更新，再施加控制/步进）
        # 这里需要正确推进：robosuite 的 _pre_action 施加 controller torque
        # 我们直接 sim.forward + sim.step 跳过 controller，物理上等价于零扭矩推进
        # 这不是 a_M 的真实执行！
        # 要真实执行 a_M，必须走 controller
        break  # 占位，实际执行在下方

    env.close()
    return records, {"p0": p0, "v0": v0}


def real_execute_with_controller(ts, task, ep, step, action_6, gripper_cmd):
    """通过 robosuite controller 真实执行，但逐 substep 观测。

    robosuite 的 env.step 内部循环 25 substep，只返回最终 obs。
    要观测中间过程，需要手动复刻 step 的内部循环。
    """
    d, j = load_decision(ep, step)
    env = make_env(ts, task)
    env.sim.set_state_from_flattened(d["snapshot"])
    env.sim.forward()

    inner = env.env
    m = env.sim.model._model
    cream_bid = env.sim.model.body_name2id("cream_cheese_1_main")

    act = np.zeros(7, dtype=np.float64)
    act[:6] = action_6
    act[6] = gripper_cmd

    # 初始 cream 状态
    p0 = np.asarray(env.sim.data._data.xpos[cream_bid]).copy()

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

    # 复刻 robosuite step 内部循环：25 substep
    records = []
    # 第一步 policy_step=True 会施加 controller
    for i in range(int(inner.control_timestep / inner.model_timestep)):
        policy_step = (i == 0)
        env.sim.forward()
        inner._pre_action(act, policy_step)
        env.sim.step()
        # 观测 cream 状态
        dd = env.sim.data._data
        p = np.asarray(dd.xpos[cream_bid]).copy()
        v = np.asarray(dd.cvel[cream_bid]).copy()
        contacts = []
        for c in range(dd.ncon):
            g1, g2 = dd.contact[c].geom1, dd.contact[c].geom2
            b1, b2 = short_body(g1), short_body(g2)
            if "cream" in (b1, b2):
                contacts.append((b1, b2))
        records.append({
            "sub": i,
            "pos": p.tolist(),
            "lin_vel": [float(x) for x in v[:3]],
            "disp_from_start_mm": float(np.linalg.norm(p - p0) * 1000),
            "cream_contacts": contacts,
        })

    env.close()
    return records, p0


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

    print(f"=== ep{args.ep} step{args.step} 单案例接触传播审计 ===")
    print(f"a_M = {np.round(a_M, 4)}, gripper = {gripper}")
    print()

    records, p0 = real_execute_with_controller(ts, task, args.ep, args.step, a_M, gripper)

    # 汇总
    disp = [r["disp_from_start_mm"] for r in records]
    print(f"cream_cheese 起始位置: {np.round(p0, 4)}")
    print(f"25 substep 内位移: {disp[0]:.3f} -> {disp[-1]:.3f} mm")
    print(f"最大单 substep 位移: {max(np.diff(disp)):.3f} mm")
    print()
    print("=== 逐 substep 位移 + 线速度 + 接触（每 5 substep 抽样）===")
    print(f"{'sub':>4} {'disp_mm':>9} {'lin_vel':>20} {'cream_contacts'}")
    for r in records[::5]:
        contacts = str(r["cream_contacts"]) if r["cream_contacts"] else "无"
        print(f"{r['sub']:>4} {r['disp_from_start_mm']:>9.2f} {str(np.round(r['lin_vel'],2)):>20} {contacts}")

    # 找第一个出现 cream 接触的 substep
    first_contact = next((r for r in records if r["cream_contacts"]), None)
    print()
    if first_contact:
        print(f"首次出现 cream_cheese 接触 @ substep {first_contact['sub']}, "
              f"接触对 {first_contact['cream_contacts']}, 此时位移 {first_contact['disp_from_start_mm']:.2f}mm")
    else:
        print("整个控制步内 cream_cheese 无任何接触对（可能是 solver 数值异常）")

    # 峰值速度
    max_v = max(np.linalg.norm(r["lin_vel"]) for r in records)
    print(f"cream_cheese 峰值线速度: {max_v:.3f} m/s")


if __name__ == "__main__":
    main()
