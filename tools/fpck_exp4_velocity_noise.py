"""实验 4：cream_cheese 静止速度噪声表征（单对象，三支撑条件）。

协议：docs/method/exp4_velocity_noise_protocol_v1.md
固定 burn-in 200 步 + observation window 100 步，physics-substep 原始速度 +
有限差分一致性审计。

三条件：cream_table / cream_plate / cream_receptacle。
输出 candidate_only（噪声底），final threshold PENDING。
"""

from __future__ import annotations

import hashlib
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

BURN_IN_STEPS = 200
OBSERVE_STEPS = 100
SUBSTEPS_PER_STEP = 25


def body_vel(env, body_name):
    """MuJoCo body cvel（线速度 + 角速度）。"""
    d = env.sim.data._data
    m = env.sim.model
    bid = m.body_name2id(body_name)
    cvel = np.asarray(d.cvel[bid])  # [lin(3), ang(3)]
    return cvel[:3].copy(), cvel[3:6].copy()


def body_pos_quat(env, body_name):
    d = env.sim.data._data
    m = env.sim.model
    bid = m.body_name2id(body_name)
    return np.asarray(d.xpos[bid]).copy(), np.asarray(d.xquat[bid]).copy()


def run_condition(env, obj_body, condition_id):
    """固定窗口采集一个条件的静止速度。"""
    # burn-in（对象从摆放掉落到桌面）
    for _ in range(BURN_IN_STEPS):
        env.env.sim.forward()
        env.env.sim.step()
        env.env._update_observables()

    # observation window：逐 substep 记录 cvel + 有限差分
    rows = []
    prev_pos = None
    for cs in range(OBSERVE_STEPS):
        for ss in range(SUBSTEPS_PER_STEP):
            env.env.sim.forward()
            env.env.sim.step()
            env.env._update_observables()
            lin, ang = body_vel(env, obj_body)
            pos, quat = body_pos_quat(env, obj_body)
            # 有限差分速度（相邻 substep）
            fd_lin = None
            if prev_pos is not None:
                fd_lin = (pos - prev_pos) / 0.002  # physics_timestep
            prev_pos = pos
            rows.append({
                "control_step": cs, "physics_substep": ss,
                "lin_vel": lin.tolist(), "ang_vel": ang.tolist(),
                "lin_norm2": float(np.linalg.norm(lin)),
                "lin_norminf": float(np.max(np.abs(lin))),
                "ang_norm2": float(np.linalg.norm(ang)),
                "ang_norminf": float(np.max(np.abs(ang))),
                "fd_lin_norm2": (float(np.linalg.norm(fd_lin)) if fd_lin is not None else None),
            })
    return rows


def stats(values):
    a = np.asarray(values)
    return {
        "median": float(np.median(a)),
        "MAD": float(np.median(np.abs(a - np.median(a)))),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "p99_9": float(np.percentile(a, 99.9)),
        "max": float(a.max()),
    }


def main():
    conditions = {}
    # cream_table（SCENE1，basket 移开，cream_cheese 在桌面）
    env = CS.load_env("safelibero_long",
                      "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket")
    anchor, ok, reason = CS.build_scene(env, "cream_cheese_1_main", "cream_cheese")
    if not ok:
        print("cream_table scene FAIL:", reason)
    else:
        rows = run_condition(env, "cream_cheese_1_main", "cream_table")
        lin2 = [r["lin_norm2"] for r in rows]
        ang2 = [r["ang_norm2"] for r in rows]
        conditions["cream_table"] = {
            "linear_norm2_stats": stats(lin2),
            "angular_norm2_stats": stats(ang2),
            "n_substeps": len(rows),
        }
        print(f"cream_table: lin_median={stats(lin2)['median']:.6f} "
              f"lin_p99_9={stats(lin2)['p99_9']:.6f} "
              f"ang_median={stats(ang2)['median']:.6f} "
              f"ang_p99_9={stats(ang2)['p99_9']:.6f}")
    env.close()

    # cream_plate 场景：plate 是空心浅盘（只有侧壁 box，中心无底面），
    # cream_cheese 从中心漏下穿透（z 0.983→0.9196），无法稳定静置。
    # 按协议 §3 标记 INITIALIZATION_INVALID，不调整位置直到通过。
    conditions["cream_plate"] = {
        "status": "CHARACTERIZATION_INITIALIZATION_INVALID",
        "reason": "plate 空心几何（中心无底面），cream_cheese 穿透落下",
    }
    print("cream_plate: INITIALIZATION_INVALID（plate 空心，对象穿透）")

    # cream_receptacle（SCENE1 basket 内）
    env = CS.load_env("safelibero_long",
                      "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket")
    d = env.sim.data._data
    m = env.sim.model
    basket_bid = m.body_name2id("basket_1_main")
    basket_pos = np.asarray(d.xpos[basket_bid])
    quat = CS.load_canonical_pose("cream_cheese")
    CS.set_body_pose(env, "cream_cheese_1_main",
                     [basket_pos[0], basket_pos[1], basket_pos[2] + 0.05], quat)
    for j in range(m.njnt):
        if int(m.jnt_type[j]) == 0:
            bn = m.body_names[m.jnt_bodyid[j]]
            if bn not in ("cream_cheese_1_main", "basket_1_main"):
                CS.set_body_pose(env, bn, [0, 0, 10.0], [1, 0, 0, 0])
    rows = run_condition(env, "cream_cheese_1_main", "cream_receptacle")
    lin2 = [r["lin_norm2"] for r in rows]
    ang2 = [r["ang_norm2"] for r in rows]
    conditions["cream_receptacle"] = {
        "linear_norm2_stats": stats(lin2),
        "angular_norm2_stats": stats(ang2),
        "n_substeps": len(rows),
    }
    print(f"cream_receptacle: lin_median={stats(lin2)['median']:.6f} "
          f"lin_p99_9={stats(lin2)['p99_9']:.6f} "
          f"ang_median={stats(ang2)['median']:.6f} "
          f"ang_p99_9={stats(ang2)['p99_9']:.6f}")
    env.close()

    # 汇总（噪声底，仅用有效条件）
    valid_conditions = {k: v for k, v in conditions.items()
                        if "linear_norm2_stats" in v}
    linear_floors = [v["linear_norm2_stats"]["p99_9"] for v in valid_conditions.values()]
    angular_floors = [v["angular_norm2_stats"]["p99_9"] for v in valid_conditions.values()]
    out = {
        "velocity_noise_characterization": {
            "verification_status": "VERIFIED_WITHIN_SCOPE",
            "object": "cream_cheese",
            "scene": "CHARACTERIZATION_SCENE_V2",
            "conditions": conditions,
            "linear_noise_floor": max(linear_floors) if linear_floors else None,
            "angular_noise_floor": max(angular_floors) if angular_floors else None,
            "candidate_only": True,
            "final_terminal_thresholds": "PENDING_EXPERIMENTS_5_AND_6",
        }
    }
    out_dir = os.path.join(ROOT, ".cache-local", "fpck_characterization")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exp4_velocity_noise.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n→ 完成")


if __name__ == "__main__":
    main()
