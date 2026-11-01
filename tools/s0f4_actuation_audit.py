"""S0-F4：Actuation Realization Audit（执行器实现一致性审计）。

逐 substep 记录力矩执行链：
  r_k → g_k → tau_requested(OSC 求解, clip 前) → tau_clipped → ctrl → qfrc_actuator

以及 q/qvel/goal_pos/ee_pos，计算：
  Δτ_j = tau_requested - tau_clipped（限幅偏差）
  ρ_sat = 25 substep 中发生限幅的比例

回答：我们一直讨论的力矩饱和，在实际执行中究竟有没有发生？

不新增状态搜索；用历史保存的 safe action 重放，不推进 AEGIS filter 状态。
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


def run_actuation_audit(ts, task, ep, step, eps_tau=1e-6):
    d, j = load_decision(ep, step)
    safe_action = np.asarray(j["aegis_first_filter"]["action"], dtype=np.float64)
    hist_h = j["aegis_first_filter"]["h"]

    env = make_env(ts, task)
    env.sim.set_state_from_flattened(d["snapshot"])
    env.sim.forward()

    robot = env.robots[0]
    controller = robot.controller
    inner = env.env
    joint_idx = robot.controller.qvel_index  # 关节速度索引

    records = []
    for i in range(int(inner.control_timestep / inner.model_timestep)):
        policy_step = (i == 0)
        env.sim.forward()

        # 复刻 SingleArm.control 的力矩链，但在 clip 前读取 requested
        if policy_step:
            controller.set_goal(safe_action[: controller.control_dim])
        tau_requested = np.asarray(controller.run_controller()).copy()  # clip 前

        low, high = robot.torque_limits
        tau_clipped = np.clip(tau_requested, low, high)

        # 下发（模拟 control 的下发路径）
        robot.sim.data.ctrl[robot._ref_joint_actuator_indexes] = tau_clipped

        # 记录 q/qvel/goal/ee（sim.step 前）
        q = np.asarray(env.sim.data.qpos[robot.controller.qpos_index]).copy()
        qvel = np.asarray(env.sim.data.qvel[joint_idx]).copy()
        goal_pos = np.asarray(controller.goal_pos).copy()
        ee_pos = np.asarray(controller.ee_pos).copy()

        # sim.step
        env.sim.step()

        # qfrc_actuator（sim.step 后）
        qfrc_actuator = np.asarray(env.sim.data._data.qfrc_actuator[joint_idx]).copy()

        delta_tau = tau_requested - tau_clipped
        saturated = np.max(np.abs(delta_tau)) > eps_tau

        records.append({
            "sub": i,
            "tau_requested_absmax": float(np.max(np.abs(tau_requested))),
            "tau_clipped_absmax": float(np.max(np.abs(tau_clipped))),
            "delta_tau_absmax": float(np.max(np.abs(delta_tau))),
            "saturated": bool(saturated),
            "n_saturated_joints": int(np.sum(np.abs(delta_tau) > eps_tau)),
            "qfrc_actuator_absmax": float(np.max(np.abs(qfrc_actuator))),
            "tracking_err_m": float(np.linalg.norm(goal_pos - ee_pos)),
        })

    env.close()
    return {
        "ep": ep, "step": step, "hist_h": hist_h,
        "records": records,
        "n_dim": len(tau_requested),
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

    print(f"=== S0-F4：Actuation Realization Audit ep{args.ep} step{args.step} ===")
    res = run_actuation_audit(ts, task, args.ep, args.step)
    print(f"历史 h = {res['hist_h']:.4f}，力矩维度 = {res['n_dim']}")
    print()

    recs = res["records"]
    n_sat = sum(1 for r in recs if r["saturated"])
    rho_sat = n_sat / len(recs)
    print(f"限幅发生 substep 数 = {n_sat}/25，饱和比例 ρ_sat = {rho_sat:.2f}")
    print()
    print(f"{'sub':>4} {'req_absmax':>11} {'clip_absmax':>11} {'Δτ_absmax':>11} {'饱和关节':>8} {'tracking_err':>12} {'qfrc_absmax':>12}")
    for r in recs[::5]:
        print(f"{r['sub']:>4} {r['tau_requested_absmax']:>11.3f} {r['tau_clipped_absmax']:>11.3f} "
              f"{r['delta_tau_absmax']:>11.4f} {r['n_saturated_joints']:>8} {r['tracking_err_m']:>12.4f} "
              f"{r['qfrc_actuator_absmax']:>12.3f}")

    # 核心诊断
    print()
    if rho_sat == 0:
        print("判决：25 substep 全程无力矩限幅 → 当前案例不支持「力矩饱和导致执行偏差」")
    else:
        print(f"判决：存在限幅（ρ_sat={rho_sat:.2f}），但限幅≠安全失效，需进一步分析对轨迹影响")

    # requested vs qfrc_actuator 的差异（gear/映射）
    print()
    print("注：tau_clipped 是下发 ctrl 值；qfrc_actuator 是广义驱动力，二者可能因 actuator gear 映射不同。")


if __name__ == "__main__":
    main()
