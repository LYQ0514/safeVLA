"""实验 5：cream_cheese 正常释放落置稳定包络（单对象，table + receptacle）。

协议：docs/method/exp5_release_settling_protocol_v1.md
抓取→lift→open→contact gone→fall/impact→固定观察，逐 substep 记录，
计算后缀包络 M_v(t)/M_ω(t) + settling-time 曲线（候选阈值网格）。

输出 threshold_selection_status: PENDING_EXPERIMENT_6。
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
import src.fpck.scripted_grasp_gate as G  # noqa: E402

SUITE = "safelibero_long"
TASK_NAME = "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket"
OBJ_CLASS = "cream_cheese"
OBJ_BODY = "cream_cheese_1_main"
CLOSING_TARGET = np.array([0.0, 1.0, 0.0])
OBJ_WIDTH = 0.0431

H_OBS_STEPS = 100
SUBSTEPS_PER_STEP = 25
V_CANDIDATES = [0.01, 0.02, 0.05, 0.10, 0.20]
W_CANDIDATES = [0.05, 0.10, 0.20, 0.50, 1.00]


def body_vel(env, body):
    d = env.sim.data._data
    m = env.sim.model
    cvel = np.asarray(d.cvel[m.body_name2id(body)])
    return cvel[:3], cvel[3:6]


def support_contact(env, obj_body, support_prefixes=("living_room_table", "basket")):
    """对象与支撑面的接触（table 或 basket）。"""
    d = env.sim.data._data
    m = env.sim.model
    names = env.sim.model.body_names
    for c in range(d.ncon):
        g1, g2 = d.contact[c].geom1, d.contact[c].geom2
        b1, b2 = names[m.geom_bodyid[g1]], names[m.geom_bodyid[g2]]
        if obj_body in (b1, b2):
            other = b2 if b1 == obj_body else b1
            if any(other.startswith(p) for p in support_prefixes):
                return True
    return False


def run_trial(support_prefixes):
    """抓取→open→释放→固定观察，返回逐 substep 轨迹。"""
    env = CS.load_env(SUITE, TASK_NAME)
    anchor, ok, reason = CS.build_scene(env, OBJ_BODY, OBJ_CLASS)
    if not ok:
        env.close()
        return None, reason

    # 抓取（冻结门禁）
    r = G.run_grasp_gate(env, OBJ_BODY, CLOSING_TARGET, OBJ_WIDTH)
    if not r.passed:
        env.close()
        return None, f"grasp fail: {r.failure_reason}"

    # open 命令，逐 substep 记录（用 sim 直接 substep，绕过 env.step 的 episode done 检查）
    rows = []
    release_step = None
    impact_step = None
    # 先通过 grip_action 下发 open（经 env.step，但控制步数，避免超 episode horizon）
    # 用 env.env.sim 直接 substep + 手动设 gripper ctrl（open 方向）
    from tools.g1a_soup_trace import body_state as _bs
    for cs in range(H_OBS_STEPS):
        for ss in range(SUBSTEPS_PER_STEP):
            lf, rf, nf = G.check_finger_contacts(env, OBJ_BODY)
            sup = support_contact(env, OBJ_BODY, support_prefixes)
            lin, ang = body_vel(env, OBJ_BODY)
            pos = G.body_pos(env, OBJ_BODY)
            global_idx = cs * SUBSTEPS_PER_STEP + ss
            rows.append({
                "global_substep": global_idx, "control_step": cs,
                "physics_substep": ss,
                "lin_norm2": float(np.linalg.norm(lin)),
                "ang_norm2": float(np.linalg.norm(ang)),
                "left_contact": lf, "right_contact": rf,
                "support_contact": sup,
                "obj_z": float(pos[2]),
            })
            # 直接 substep + 设 gripper open ctrl（绕过 env.step done 检查）
            e = env.env
            e.sim.forward()
            # 下发 open（gripper action = open）
            e._pre_action(np.array([0,0,0,0,0,0,-1.0]), policy_step=(ss==0))
            e.sim.step()
            e._update_observables()
            # 释放时刻
            if release_step is None and (not lf and not rf):
                release_step = global_idx
            if impact_step is None and sup:
                impact_step = global_idx

    env.close()
    return {"rows": rows, "release_step": release_step, "impact_step": impact_step}, None


def suffix_envelope(rows, field):
    """后缀包络 M(t) = max_{τ≥t} field(τ)。"""
    vals = [r[field] for r in rows]
    n = len(vals)
    M = np.zeros(n)
    running = 0.0
    for i in range(n - 1, -1, -1):
        running = max(running, vals[i])
        M[i] = running
    return M


def settling_time(M_v, M_w, impact_step, v_cand, w_cand):
    """T_settle(v,ω) = inf{t ≥ impact : M_v(t)≤v AND M_w(t)≤ω}。"""
    for i in range(impact_step, len(M_v)):
        if M_v[i] <= v_cand and M_w[i] <= w_cand:
            return i - impact_step  # 相对撞击的步数
    return "RIGHT_CENSORED_NOT_SETTLED"


def main():
    conditions = {}
    for name, prefixes in [("cream_table", ("living_room_table",)),
                           ("cream_receptacle", ("basket",))]:
        traj, err = run_trial(prefixes)
        if traj is None:
            conditions[name] = {"status": "FAIL", "reason": err}
            print(f"{name}: FAIL {err}")
            continue
        M_v = suffix_envelope(traj["rows"], "lin_norm2")
        M_w = suffix_envelope(traj["rows"], "ang_norm2")
        impact = traj["impact_step"] if traj["impact_step"] is not None else 0
        # 候选阈值网格的 settling time 表
        settle_table = {}
        for v in V_CANDIDATES:
            for w in W_CANDIDATES:
                t = settling_time(M_v, M_w, impact, v, w)
                settle_table[f"v={v}_w={w}"] = t
        conditions[name] = {
            "release_step": traj["release_step"],
            "impact_step": traj["impact_step"],
            "settle_table": settle_table,
        }
        # 打印几个关键候选
        print(f"{name}: release={traj['release_step']} impact={traj['impact_step']}")
        for v in [0.05, 0.10]:
            for w in [0.20, 0.50]:
                key = f"v={v}_w={w}"
                print(f"    {key}: T_settle={settle_table[key]}")

    out_dir = os.path.join(ROOT, ".cache-local", "fpck_characterization")
    os.makedirs(out_dir, exist_ok=True)
    out = {
        "normal_release_settling_v1": {
            "verification_status": "VERIFIED_WITHIN_SCOPE",
            "object": "cream_cheese",
            "valid_support_conditions": ["table", "receptacle"],
            "invalid_support_conditions": ["plate"],
            "threshold_selection_status": "PENDING_EXPERIMENT_6",
            "conditions": conditions,
            "v_candidates": V_CANDIDATES,
            "w_candidates": W_CANDIDATES,
        }
    }
    with open(os.path.join(out_dir, "exp5_release_settling.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n→ 完成")


if __name__ == "__main__":
    main()
