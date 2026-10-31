"""PNSG 正对照：真实危险状态下的源空间 vs 动作空间修正（完整版）。

单一脚本完成：找危险状态 → 三候选对照 → 输出对照表。

三候选（首轮 direct proposal，α=1，不搜索）：
- Nominal：保存的名义动作
- Source repair：同一 z0 求 Δz，重新前向冻结策略（pose-only steering）
- Action repair：直接修正名义动作的 pose 部分

协议（想法v12）：
- 危险状态 = 真实 π0.5 rollout 中，碰撞前最近一次 query snapshot
- 入选：初始无 robot-obstacle + nominal 前 2 步 C_2 FAIL(robot-obstacle)
- 不把"代理能检测"作为入选条件
- 共用同一线性问题 C,b；源空间 A=CJ，动作空间 A=C
- gripper 序列保持不变（pose-only steering）
- 真实安全用完整 C_2，接触作为解释
- 进度用执行后测量，不冒充 P_H
- 源修正超 r_budget=12 记 OVER_BUDGET
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time
import warnings

warnings.filterwarnings("ignore")

ROOT = "/data/safeVLA"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "dataset", "SafeLIBERO", "safelibero"))
sys.path.insert(0, "/data/2608.SDL/MojocoProj/src/pi05")

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
import pi05_libero_runner as R  # noqa: E402

from src.pnsg.crn import nominal_seed, sample_z  # noqa: E402
from src.pnsg.source_solver import minimum_norm_halfspaces  # noqa: E402
from src.pnsg.full_state import save_full_state, restore_full_state  # noqa: E402
from src.pnsg.certifier import Certifier  # noqa: E402
from src.pnsg.geometry_proxy import (  # noqa: E402
    build_control_points, build_obstacles, selected_constraints, compute_clearance,
)

SUITE = "safelibero_spatial"
LEVEL = "I"
TASK_ID = 0
REPLICATE_ID = 0
EXECUTE_STEPS = 2
NUM_STEPS_WAIT = 10
R_BUDGET = 12.0
H_FLOOR = 0.005
MAX_CONSTRAINTS = 3
MARGIN = 0.01


def make_env(ts, task, state_id):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256,
                             controller="OSC_POSE")
    env.seed(7)
    obs = env.reset()
    obs = env.set_init_state(ts.get_task_init_states(TASK_ID)[state_id])
    dummy = np.zeros(7, dtype=np.float32); dummy[-1] = -1.0
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = env.step(dummy)
    return env, obs


def find_dangerous_state(ts, task, policy, state_ids, max_scan=60):
    """按固定顺序找第一个危险状态。"""
    from src.eval_safelibero import active_obstacle

    for state_id in state_ids:
        env, obs = make_env(ts, task, state_id)
        obst = active_obstacle(env, obs)
        certifier = Certifier(env)
        prompt = task.language

        for step in range(max_scan):
            query = step // EXECUTE_STEPS
            if step % EXECUTE_STEPS == 0:
                snapshot = save_full_state(env)
                z0, _ = sample_z(nominal_seed(SUITE, TASK_ID, state_id, REPLICATE_ID, query))
                po = R.obs_to_policy_obs(obs, prompt, image_size=224)
                raw = policy.generate(z0, po)
                nominal = np.clip(raw, -1, 1)
                obs_at_query = dict(obs)

                # shadow 认证 nominal 前 2 步
                restore_full_state(env, snapshot)
                c2 = certifier.certify_prefix(nominal[:EXECUTE_STEPS])
                if not c2.pass_ and "robot-obstacle" in c2.reason:
                    # 满足入选条件（初始无接触 + nominal 前2步产生 robot-obstacle）
                    restore_full_state(env, snapshot)
                    env.close()
                    return {
                        "state_id": state_id, "query": query, "snapshot": snapshot,
                        "obs": obs_at_query, "z0": z0, "raw": raw, "nominal": nominal,
                        "prompt": prompt, "obstacle": obst,
                        "c2_nominal_reason": c2.reason,
                    }
                restore_full_state(env, snapshot)

            action = nominal[step % EXECUTE_STEPS]
            obs, _, _, _ = env.step(action)

        env.close()
    return None


def evaluate_candidate(env_factory, snapshot, actions, cps, obstacles, prompt, z0=None, nominal=None):
    """从 snapshot 恢复，执行候选前 2 步，返回安全 + 净空结果。

    真实净空 = MuJoCo native 几何（CertifyResult.min_robot_obstacle_dist），
    不是球体代理。代理净空单独报告（compute_clearance）。
    """
    env, _ = env_factory()
    restore_full_state(env, snapshot)
    certifier = Certifier(env)

    # C_2 认证（内部执行 2 步，返回 native 最小净空）
    c2 = certifier.certify_prefix(actions[:EXECUTE_STEPS])

    # 代理净空（球体代理，用于对比 proxy vs native）
    clear = compute_clearance(cps, obstacles, actions[:, :6])
    min_proxy_clearance = float(clear.min())

    env.close()
    return {
        "c2_pass": c2.pass_,
        "c2_reason": c2.reason,
        "min_proxy_clearance": min_proxy_clearance,
        "min_native_clearance": c2.min_robot_obstacle_dist,  # native 几何（或 None）
    }


def main():
    from libero.libero import benchmark
    from src.pnsg.policy_adapter import PnsgPolicy
    from src.pnsg.contact_audit import load_role_manifest

    manifest_all = load_role_manifest(os.path.join(ROOT, "data", "task_object_role_manifest.json"))
    ts = benchmark.get_benchmark_dict()[SUITE](safety_level=LEVEL)
    task = ts.get_task(TASK_ID)
    manifest = manifest_all[SUITE][task.name]

    policy = PnsgPolicy()

    print("=== 找危险状态 ===")
    ds = find_dangerous_state(ts, task, policy, list(range(50)))
    if ds is None:
        print("未找到危险状态")
        policy.close()
        return

    print(f"危险状态: state_id={ds['state_id']}, query={ds['query']}, "
          f"obstacle={ds['obstacle']}")
    print(f"  nominal C_2 原因: {ds['c2_nominal_reason']}")

    # 重建 env + 控制点/障碍物（用于净空）
    def env_factory():
        return make_env(ts, task, ds["state_id"])

    env, obs = env_factory()
    restore_full_state(env, ds["snapshot"])
    cps = build_control_points(env, ds["obs"])
    obstacles = build_obstacles(env, ds["obs"])
    env.close()

    # 构造线性问题（在 nominal 上，一次）
    nominal_pose = ds["nominal"][:, :6]
    rows = selected_constraints(cps, obstacles, nominal_pose,
                                h_floor=H_FLOOR, max_constraints=MAX_CONSTRAINTS, margin=MARGIN)
    print(f"\n线性问题：选中 {len(rows)} 条约束")

    if not rows:
        print("⚠️ 代理没有检测到约束（代理漏报）——如实报告，不换状态")
        # 仍继续记录 nominal 结果
        C = np.zeros((0, 60)); b = np.zeros(0)
    else:
        C = np.vstack([c for c, _ in rows])
        b = np.array([bb for _, bb in rows])

    # 求 J（源空间需要）
    _, _, _, ext_J = policy.generate_with_jacobian(ds["z0"],
                                                   R.obs_to_policy_obs(ds["obs"], ds["prompt"], image_size=224))

    # ---- 三候选 ----
    candidates = {"nominal": {"actions": ds["nominal"], "label": "Nominal",
                              "c_src": 0.0, "delta_pose_norm": 0.0, "status": "ok"}}

    # Source repair
    if len(rows) > 0:
        A_src = C @ ext_J  # (m, 320)
        dz, rcrit = minimum_norm_halfspaces(A_src, b)
        if dz is None:
            candidates["source"] = {"actions": None, "label": "Source repair",
                                    "c_src": None, "delta_pose_norm": None,
                                    "status": "INFEASIBLE"}
        else:
            over_budget = float(np.linalg.norm(dz)) > R_BUDGET
            # 重新前向 z0+Δz
            z_cand = ds["z0"] + dz.reshape(ds["z0"].shape)
            src_raw = policy.generate(z_cand, R.obs_to_policy_obs(ds["obs"], ds["prompt"], image_size=224))
            src_clipped = np.clip(src_raw, -1, 1)
            # pose-only：恢复 nominal gripper
            src_clipped[:, 6] = ds["nominal"][:, 6]
            candidates["source"] = {"actions": src_clipped, "label": "Source repair",
                                    "c_src": 0.5 * float(np.linalg.norm(dz)**2),
                                    "delta_pose_norm": float(np.linalg.norm(dz)),
                                    "status": "OVER_BUDGET" if over_budget else "ok"}

    # Action repair
    if len(rows) > 0:
        da, rcrit_a = minimum_norm_halfspaces(C, b)  # A=C，动作空间
        if da is None:
            candidates["action"] = {"actions": None, "label": "Action repair",
                                    "c_src": None, "delta_pose_norm": None,
                                    "status": "INFEASIBLE"}
        else:
            act_pose = nominal_pose + da.reshape(nominal_pose.shape)
            act_actions = ds["nominal"].copy()
            act_actions[:, :6] = act_pose
            act_actions = np.clip(act_actions, -1, 1)
            # pose-only：恢复 nominal gripper
            act_actions[:, 6] = ds["nominal"][:, 6]
            candidates["action"] = {"actions": act_actions, "label": "Action repair",
                                    "c_src": float("nan"),
                                    "delta_pose_norm": float(np.linalg.norm(da)),
                                    "status": "ok"}

    # ---- 评估三候选 ----
    print(f"\n{'候选':<16} {'状态':>12} {'代理净空mm':>10} {'native净空mm':>12} {'C_2':>8} {'原因':>30}")
    results = {}
    for key, cand in candidates.items():
        if cand["actions"] is None:
            print(f"{cand['label']:<16} {cand['status']:>12}")
            results[key] = {"status": cand["status"]}
            continue
        ev = evaluate_candidate(env_factory, ds["snapshot"], cand["actions"],
                                cps, obstacles, ds["prompt"])
        mc = ev["min_proxy_clearance"] * 1000 if ev["min_proxy_clearance"] is not None else float('nan')
        nc = ev["min_native_clearance"] * 1000 if ev["min_native_clearance"] is not None else float('nan')
        nc_str = f"{nc:>12.1f}" if ev["min_native_clearance"] is not None else f"{'无接触':>12}"
        print(f"{cand['label']:<16} {cand['status']:>12} {mc:>10.1f} {nc_str} "
              f"{str(ev['c2_pass']):>8} {ev['c2_reason']:>30}")
        results[key] = {
            "status": cand["status"],
            "c2_pass": ev["c2_pass"], "c2_reason": ev["c2_reason"],
            "min_proxy_clearance": ev["min_proxy_clearance"],
            "min_native_clearance": ev["min_native_clearance"],
            "delta_pose_norm": cand["delta_pose_norm"],
            "c_src": cand["c_src"],
        }

    # 保存
    out = pathlib.Path(ROOT, "data", "audit", "positive_control_state0.json")
    out.write_text(json.dumps({
        "state_id": ds["state_id"], "query": ds["query"], "obstacle": ds["obstacle"],
        "n_constraints": len(rows),
        "nominal_c2_reason": ds["c2_nominal_reason"],
        "results": results,
    }, ensure_ascii=False, indent=2, default=str))
    print(f"\n结果: {out}")

    policy.close()


if __name__ == "__main__":
    main()
