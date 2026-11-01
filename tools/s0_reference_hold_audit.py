"""S0：Reference-Hold Execution Safety Audit（修复版）。

修复 S0-F1/F2/F3：
- F1/F2：不重新调 filter。直接用历史保存的 aegis_first_filter.action
  （采集时 filter 出的 safe action）执行 25 substep。避免 filter 状态被
  重复调用推进，保证安全过滤器身份与历史一致。
- F3：物理表面距离用 native contact dist（接触时），非接触时显式标记
  「分离但未知正距离」，不用中心距冒充 clearance。

输出：25 substep 的接触判定 + 力矩饱和 + 起点/终点/最小接触 dist。
"""

from __future__ import annotations

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


def build_geom_sets(env):
    """robot / obstacle 的碰撞 geom 集合（contype>0）。"""
    m = env.sim.model._model
    robot_geoms = {g for g in range(m.ngeom) if m.geom_contype[g] > 0 and
                   (env.sim.model.body_names[m.geom_bodyid[g]].startswith("robot0") or
                    env.sim.model.body_names[m.geom_bodyid[g]].startswith("gripper0"))}
    obstacle_names = set(n.replace("_joint0", "") for n in env.sim.model.joint_names if "obstacle" in n)
    obstacle_geoms = {g for g in range(m.ngeom) if m.geom_contype[g] > 0 and
                      any(env.sim.model.body_names[m.geom_bodyid[g]].startswith(o) for o in obstacle_names)}
    return robot_geoms, obstacle_geoms


def native_contact_state(env, robot_geoms, obstacle_geoms):
    """读当前状态 robot-obstacle 接触：返回 (bool, min_dist)。

    min_dist = 接触对的最小 dist（≤0 穿透），无接触时 None。
    """
    dd = env.sim.data._data
    dists = []
    for c in range(dd.ncon):
        g1, g2 = dd.contact[c].geom1, dd.contact[c].geom2
        if (g1 in robot_geoms and g2 in obstacle_geoms) or (g2 in robot_geoms and g1 in obstacle_geoms):
            dists.append(float(dd.contact[c].dist))
    if not dists:
        return False, None  # 无接触，分离但未知正距离
    return True, min(dists)


def run_s0(ts, task, ep, step):
    """用历史保存的 safe action 执行 25 substep，记录接触 + 力矩。"""
    d, j = load_decision(ep, step)
    safe_action = np.asarray(j["aegis_first_filter"]["action"], dtype=np.float64)
    hist_h = j["aegis_first_filter"]["h"]

    env = make_env(ts, task)
    env.sim.set_state_from_flattened(d["snapshot"])
    env.sim.forward()

    robot_geoms, obstacle_geoms = build_geom_sets(env)
    inner = env.env

    # 起点接触
    start_contact, start_dist = native_contact_state(env, robot_geoms, obstacle_geoms)

    records = []
    for i in range(int(inner.control_timestep / inner.model_timestep)):
        policy_step = (i == 0)
        env.sim.forward()
        inner._pre_action(safe_action, policy_step)
        env.sim.step()
        contact, dist = native_contact_state(env, robot_geoms, obstacle_geoms)
        ctrl = np.asarray(env.sim.data._data.ctrl).copy()
        records.append({
            "sub": i,
            "contact": contact,
            "dist": dist,  # None = 分离未知正距离
            "ctrl_absmax": float(np.max(np.abs(ctrl))) if ctrl.size else 0.0,
        })

    env.close()
    return {
        "ep": ep, "step": step,
        "hist_h": hist_h,
        "safe_action": safe_action.tolist(),
        "start_contact": start_contact, "start_dist": start_dist,
        "records": records,
    }


def main():
    import argparse
    from libero.libero import benchmark

    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", type=int, default=4)
    ap.add_argument("--step", type=int, default=40)
    args = ap.parse_args()

    ts = benchmark.get_benchmark_dict()["safelibero_long"](safety_level="I")
    task = ts.get_task(TASK_ID)

    print(f"=== S0 修复版：ep{args.ep} step{args.step} ===")
    # 3 次重复（同 safe action 重放）
    results = [run_s0(ts, task, args.ep, args.step) for _ in range(3)]

    r0 = results[0]
    print(f"历史 h = {r0['hist_h']:.4f}")
    print(f"起点接触 = {r0['start_contact']} (dist={r0['start_dist']})")
    print()

    # 接触判定轨迹（run0）
    contact_subs = [r["sub"] for r in r0["records"] if r["contact"]]
    print(f"执行区间内出现接触的 substep: {contact_subs if contact_subs else '无'}")
    print(f"力矩 absmax 轨迹（每5步）: {[round(r['ctrl_absmax'],3) for r in r0['records'][::5]]}")

    # 3 次重复一致性
    all_contact_patterns = [
        tuple(r["contact"] for r in res["records"]) for res in results
    ]
    consistent = all(p == all_contact_patterns[0] for p in all_contact_patterns)
    print(f"\n3 次重复接触模式一致 = {consistent}")

    # 判定
    if not contact_subs:
        print("判定：执行区间全程无 robot-obstacle 接触（分离，正距离未知）")
    elif r0["start_contact"]:
        print("判定：起点已有接触，不适用 S0（起点不干净）")
    else:
        print(f"判定：起点干净、执行区间 {contact_subs} 出现接触 → 采样间违规见证")

    # 力矩饱和诊断：请求 vs applied 需在 _pre_action 内读 controller.torques（下一层）
    print("\n注：力矩饱和(requested vs applied)需读 controller.torques，本版记录 ctrl 下发值。")


if __name__ == "__main__":
    main()
