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

    # ---- 求原始解 Δz* 和 Δa* ----
    BETA = [0.25, 0.5, 0.75, 1.0]
    policy_obs_fn = lambda: R.obs_to_policy_obs(ds["obs"], ds["prompt"], image_size=224)

    source_info = None
    action_info = None
    if len(rows) > 0:
        # 源空间
        A_src = C @ ext_J  # (m, 320)
        dz, _ = minimum_norm_halfspaces(A_src, b)
        if dz is not None:
            r_star = float(np.linalg.norm(dz))
            source_info = {"dz": dz, "r_star": r_star}
        # 动作空间
        da, _ = minimum_norm_halfspaces(C, b)  # A=C
        if da is not None:
            action_info = {"da": da}

    # ---- 生成预算内候选（α_j = β_j · min(1, 12/r*)）----
    # Nominal（β=0 基准）
    all_candidates = [
        {"group": "nominal", "beta": 0.0, "actions": ds["nominal"],
         "label": "Nominal", "delta_norm": 0.0, "c_src": 0.0, "c_act": 0.0},
    ]

    # Source 候选（预算内）
    source_candidates = []
    if source_info is not None:
        dz = source_info["dz"]
        r_star = source_info["r_star"]
        alpha_scale = min(1.0, R_BUDGET / r_star)
        for beta in BETA:
            alpha = beta * alpha_scale
            z_cand = ds["z0"] + (alpha * dz).reshape(ds["z0"].shape)
            src_raw = policy.generate(z_cand, policy_obs_fn())
            src_clipped = np.clip(src_raw, -1, 1)
            src_clipped[:, 6] = ds["nominal"][:, 6]  # pose-only，恢复 gripper
            delta_norm = float(np.linalg.norm(alpha * dz))
            c_src = 0.5 * delta_norm**2
            # 动作修正代价：执行动作 vs nominal 的加权修正
            c_act = float(np.linalg.norm(src_clipped[:, :6] - ds["nominal"][:, :6]))
            source_candidates.append({
                "group": "source", "beta": beta, "alpha": alpha,
                "actions": src_clipped, "label": f"Source β={beta}",
                "delta_norm": delta_norm, "c_src": c_src, "c_act": c_act,
            })
        all_candidates.extend(source_candidates)

    # Action 候选
    action_candidates = []
    if action_info is not None:
        da = action_info["da"]
        for beta in BETA:
            act_pose = nominal_pose + beta * da.reshape(nominal_pose.shape)
            act_actions = ds["nominal"].copy()
            act_actions[:, :6] = act_pose
            act_actions = np.clip(act_actions, -1, 1)
            act_actions[:, 6] = ds["nominal"][:, 6]  # pose-only
            delta_norm = float(np.linalg.norm(beta * da))
            c_act = float(np.linalg.norm(act_actions[:, :6] - ds["nominal"][:, :6]))
            action_candidates.append({
                "group": "action", "beta": beta, "actions": act_actions,
                "label": f"Action β={beta}", "delta_norm": delta_norm,
                "c_src": float("nan"), "c_act": c_act,
            })
        all_candidates.extend(action_candidates)

    # ---- 评估所有候选 ----
    print(f"\n{'候选':<14} {'β':>4} {'α':>6} {'代理净空mm':>10} {'native净空mm':>12} {'C_2':>6} {'原因':>28}")
    eval_results = {}
    for cand in all_candidates:
        t0 = time.time()
        ev = evaluate_candidate(env_factory, ds["snapshot"], cand["actions"],
                                cps, obstacles, ds["prompt"])
        ev["eval_time_s"] = time.time() - t0
        cand["eval"] = ev
        mc = ev["min_proxy_clearance"] * 1000 if ev["min_proxy_clearance"] is not None else float('nan')
        nc = ev["min_native_clearance"] * 1000 if ev["min_native_clearance"] is not None else None
        nc_str = f"{nc:.1f}" if nc is not None else "无接触"
        alpha_str = f"{cand.get('alpha', 0):.3f}" if cand["group"] == "source" else "-"
        print(f"{cand['label']:<14} {cand['beta']:>4} {alpha_str:>6} {mc:>10.1f} "
              f"{nc_str:>12} {str(ev['c2_pass']):>6} {ev['c2_reason']:>28}")

    # ---- 各组选「修正幅度最小」的 C_2 通过候选 ----
    def pick_best(cands):
        passing = [c for c in cands if c["eval"]["c2_pass"]]
        if not passing:
            return None
        return min(passing, key=lambda c: c["delta_norm"])

    best_source = pick_best(source_candidates)
    best_action = pick_best(action_candidates)
    nominal_ev = all_candidates[0]["eval"]

    print(f"\n=== 各组最优 C_2 通过候选 ===")
    print(f"  Nominal: C_2={nominal_ev['c2_pass']} ({nominal_ev['c2_reason']})")
    src_best_label = "无通过" if best_source is None else (
        best_source['label'] + " (Δz=%.1f, C_src=%.1f)" % (best_source['delta_norm'], best_source['c_src']))
    act_best_label = "无通过" if best_action is None else (
        best_action['label'] + " (Δa=%.2f)" % best_action['delta_norm'])
    print(f"  Source:  {src_best_label}")
    print(f"  Action:  {act_best_label}")

    # 保存
    out = pathlib.Path(ROOT, "data", "audit", "positive_control_state0_alpha.json")
    cand_records = []
    for cand in all_candidates:
        rec = {k: cand[k] for k in ("group", "beta", "label", "delta_norm", "c_src", "c_act")
               if k in cand}
        rec["c2_pass"] = cand["eval"]["c2_pass"]
        rec["c2_reason"] = cand["eval"]["c2_reason"]
        rec["min_proxy_clearance"] = cand["eval"]["min_proxy_clearance"]
        rec["min_native_clearance"] = cand["eval"]["min_native_clearance"]
        rec["eval_time_s"] = cand["eval"]["eval_time_s"]
        cand_records.append(rec)
    payload = {
        "state_id": ds["state_id"], "query": ds["query"], "obstacle": ds["obstacle"],
        "n_constraints": len(rows),
        "nominal_c2_reason": ds["c2_nominal_reason"],
        "r_budget": R_BUDGET,
        "r_star": source_info["r_star"] if source_info else None,
        "source_budget_scale": (min(1.0, R_BUDGET / source_info["r_star"])
                                if source_info else None),
        "candidates": cand_records,
        "best_source": (best_source["label"] if best_source else None),
        "best_action": (best_action["label"] if best_action else None),
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    print(f"\n结果: {out}")

    policy.close()


if __name__ == "__main__":
    main()
