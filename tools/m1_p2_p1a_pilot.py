"""M1-P2-P1a：HVNP 自主闭环 Pilot（step0–99）。

从 episode 初始状态开始，HVNP 每个控制步自主选择动作：
- Nominal-first Lazy Verification：先预测 N 的 H=10 安全；若安全，直接执行 N。
- 否则验证替代候选 {A, Z+A, L+A, Hold+N}，选安全且干预最小（J_dev）者。
- 两步避让（Z+A/L+A）剩余阶段纳入控制器状态，不静默取消。
- 无安全候选 → NO_VERIFIED_SEQUENCE + 安全终止。

架构：live env（只执行）+ scratch env（只预测，全状态 fork）。
安全契约：全身禁止接触（sim.data 真值）+ 历史 CAR（Dmax < 1mm - eps）。
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
sys.path.insert(0, os.path.join(ROOT, "third_party", "aegis"))

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
import mujoco  # noqa: E402

from src.pnsg.policy_adapter import PnsgPolicy  # noqa: E402
from src.safety.aegis_adapted import AegisSafetyLayer  # noqa: E402
from tools.g1a_soup_trace import make_env, body_state, eef_pos, subgoal_values  # noqa: E402
from tools.m1_p1_design_gate_v2 import step_tracked, build_body_geoms, generate_block  # noqa: E402
from tools.m1_p1_design_gate_v3 import disp_from_p0, d_soup  # noqa: E402

SUITE = "safelibero_long"
LEVEL = "I"
TASK_ID = 0
STATE_ID = 2
STREAM = 0
REPLAN_STEPS = 5
IMAGE_SIZE = 224
SWITCH = 40
H = 10
PROTECTED = "red_coffee_mug_obstacle_1_main"
DELTA_CAR = 0.001
EPS = 0.0001
PILOT_STEPS = 100

Z_DIR = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
L_DIR = np.array([-0.024, -0.992, 0.099, 0.0, 0.0, 0.0, 0.0])
AVOID_MAG = 0.4
AVOID_STEPS = 2  # 两步避让

SPEC = mujoco.mjtState.mjSTATE_INTEGRATION

# 候选顺序（用于 J_dev 平序，实际按最小干预排序）
ALT_CANDIDATES = ["A", "Z+A", "L+A", "Hold+N"]


def save_full(env, layer):
    m = env.sim.model._model
    d = env.sim.data._data
    size = mujoco.mj_stateSize(m, SPEC)
    buf = np.zeros(size, dtype=np.float64)
    mujoco.mj_getState(m, d, buf, SPEC)
    c = env.robots[0].controller
    return {"mj": buf,
            "goal_pos": c.goal_pos.copy(), "goal_ori": c.goal_ori.copy(),
            "relative_ori": c.relative_ori.copy() if c.relative_ori is not None else None,
            "new_update": c.new_update,
            "aegis": layer.save_state()}


def restore_full(env, layer, st):
    m = env.sim.model._model
    d = env.sim.data._data
    mujoco.mj_setState(m, d, st["mj"], SPEC)
    mujoco.mj_forward(m, d)
    c = env.robots[0].controller
    c.goal_pos[:] = st["goal_pos"]
    c.goal_ori[:] = st["goal_ori"]
    if st["relative_ori"] is not None:
        c.relative_ori[:] = st["relative_ori"]
    c.new_update = st["new_update"]
    layer.restore_state(st["aegis"])


def verify_candidate(scratch, layer_s, pol, prompt, snapshot, cand, nom_block,
                     start_step, Dmax0, rg, og, p0):
    """在 scratch env 从 snapshot 恢复，模拟候选 H=10 步，返回判决。"""
    restore_full(scratch, layer_s, snapshot)
    obs = scratch.env._get_observations(force_update=True)
    nominal = nom_block.copy()

    first_contact = None
    first_step_contact = False
    contact_steps = 0
    Dmax = Dmax0
    J_dev = 0.0

    for t in range(H):
        step = start_step + t
        if step % REPLAN_STEPS == 0 and t > 0:
            nominal = generate_block(pol, obs, prompt, step // REPLAN_STEPS)
        nom_a = nominal[step % REPLAN_STEPS]

        if cand == "N":
            act = nom_a
        elif cand == "A":
            act = layer_s.filter(nom_a, obs)[0]
        elif cand == "Z+A":
            act = np.clip(nom_a + AVOID_MAG * Z_DIR, -1, 1) if t < AVOID_STEPS else layer_s.filter(nom_a, obs)[0]
        elif cand == "L+A":
            act = np.clip(nom_a + AVOID_MAG * L_DIR, -1, 1) if t < AVOID_STEPS else layer_s.filter(nom_a, obs)[0]
        elif cand == "Hold+N":
            act = np.zeros(7) if t < AVOID_STEPS else nom_a
        else:
            raise ValueError(cand)

        J_dev += float(np.linalg.norm(np.asarray(act) - np.asarray(nom_a)))
        obs, _r, _d, _i, c_any, c_l6, _ = step_tracked(scratch, act, rg, og)
        if c_any:
            contact_steps += 1
            if first_contact is None:
                first_contact = step
            if t == 0:
                first_step_contact = True
        Dmax = max(Dmax, disp_from_p0(scratch, p0))

    safe = (contact_steps == 0) and (Dmax < DELTA_CAR - EPS)
    return {
        "cand": cand, "safe": safe, "first_contact": first_contact,
        "first_step_contact": first_step_contact, "contact_steps": contact_steps,
        "Dmax": Dmax, "J_dev": J_dev,
    }


def select_action(scratch, layer_s, pol, prompt, live, layer_live, nom_block,
                  step, Dmax, rg, og, p0):
    """Nominal-first lazy verification + 最小干预选择。返回 (chosen, verdicts)。

    只判定候选名，不在此处调用 layer_live.filter（避免污染正式执行）。
    """
    snapshot = save_full(live, layer_live)

    # 1. 名义优先
    vN = verify_candidate(scratch, layer_s, pol, prompt, snapshot, "N", nom_block,
                          step, Dmax, rg, og, p0)
    if vN["safe"]:
        return "N", {"N": vN}

    # 2. 替代候选（AEGIS 从 SWITCH 起可用）
    alts = [c for c in ALT_CANDIDATES if c != "Hold+N"] if step >= SWITCH else ["Hold+N"]
    verdicts = {"N": vN}
    for cand in alts:
        verdicts[cand] = verify_candidate(scratch, layer_s, pol, prompt, snapshot, cand,
                                          nom_block, step, Dmax, rg, og, p0)

    safe_alts = {k: v for k, v in verdicts.items() if k != "N" and v["safe"]}
    if safe_alts:
        best = min(safe_alts, key=lambda k: safe_alts[k]["J_dev"])
        return best, verdicts

    return "NO_VERIFIED_SEQUENCE", verdicts


def main():
    from libero.libero import benchmark
    ts = benchmark.get_benchmark_dict()[SUITE](safety_level=LEVEL)
    task = ts.get_task(TASK_ID)
    prompt = task.language
    pol = PnsgPolicy()

    live = make_env(ts, task, STATE_ID)
    layer_live = AegisSafetyLayer(live)
    scratch = make_env(ts, task, STATE_ID)
    layer_s = AegisSafetyLayer(scratch)

    obs = live.env._get_observations()
    layer_live.reset(obs, prompt)
    rg, og = build_body_geoms(live)
    p0 = body_state(live, PROTECTED)["pos"].copy()

    print("=== M1-P2-P1a：HVNP 自主闭环 Pilot（step0-99）===", flush=True)

    # 控制器状态
    avoid_remaining = 0
    avoid_dir = None
    avoid_name = None

    # 记录
    step_records = []
    Dmax = 0.0
    first_actual_contact = None
    n_nominal_kept = 0
    cand_use = {"N": 0, "A": 0, "Z+A": 0, "L+A": 0, "Hold+N": 0, "cont": 0}
    n_verify_total = 0
    n_nvs = 0
    nvs_steps = []
    n_pred_miss = 0
    nominal = None
    d_soup_init = d_soup(live)
    n_generate_live = 0

    t_start = time.time()
    terminated_reason = "达到 step99"

    for step in range(PILOT_STEPS):
        if step == SWITCH:
            layer_live.reset(obs, prompt)  # 与 m1 B 分支一致：切 AEGIS 时重定位障碍物
        if step % REPLAN_STEPS == 0:
            nominal = generate_block(pol, obs, prompt, step // REPLAN_STEPS)
            n_generate_live += 1
        nom_a = nominal[step % REPLAN_STEPS]

        t0 = time.time()

        if avoid_remaining > 0:
            # 继续两步避让的剩余阶段
            act = np.clip(nom_a + AVOID_MAG * avoid_dir, -1, 1)
            chosen = f"cont:{avoid_name}"
            avoid_remaining -= 1
            verdicts = {}
            n_verified = 0
            pred_first_contact = None
        else:
            chosen, verdicts = select_action(scratch, layer_s, pol, prompt,
                                             live, layer_live, nominal, step,
                                             Dmax, rg, og, p0)
            n_verified = len(verdicts)
            n_verify_total += n_verified
            if chosen == "NO_VERIFIED_SEQUENCE":
                n_nvs += 1
                nvs_steps.append(step)
                terminated_reason = f"NO_VERIFIED_SEQUENCE @ step{step}"
                step_records.append({
                    "step": step, "chosen": chosen, "n_verified": n_verified,
                    "decision_s": time.time() - t0, "terminated": True,
                })
                break
            # 由 chosen 计算实际第一步动作（执行阶段才调用 layer_live.filter）
            if chosen == "N":
                act = nom_a
                pred_first_contact = verdicts["N"]["first_step_contact"]
            elif chosen == "A":
                act = layer_live.filter(nom_a, obs)[0]
                pred_first_contact = verdicts["A"]["first_step_contact"]
            elif chosen == "Z+A":
                act = np.clip(nom_a + AVOID_MAG * Z_DIR, -1, 1)
                avoid_remaining = AVOID_STEPS - 1
                avoid_dir = Z_DIR
                avoid_name = chosen
                pred_first_contact = verdicts["Z+A"]["first_step_contact"]
            elif chosen == "L+A":
                act = np.clip(nom_a + AVOID_MAG * L_DIR, -1, 1)
                avoid_remaining = AVOID_STEPS - 1
                avoid_dir = L_DIR
                avoid_name = chosen
                pred_first_contact = verdicts["L+A"]["first_step_contact"]
            elif chosen == "Hold+N":
                act = np.zeros(7)
                pred_first_contact = verdicts["Hold+N"]["first_step_contact"]
            cand_use[chosen] = cand_use.get(chosen, 0) + 1
            if chosen == "N":
                n_nominal_kept += 1

        # 执行（live 只执行，不 fork）
        obs, _r, _d, _i, c_any, c_l6, _ = step_tracked(live, act, rg, og)
        Dmax = max(Dmax, disp_from_p0(live, p0))
        if c_any and first_actual_contact is None:
            first_actual_contact = step

        # 预测一致性：若预测首步无接触但实际有接触 → 预测 miss
        if pred_first_contact is False and c_any:
            n_pred_miss += 1

        step_records.append({
            "step": step, "chosen": chosen, "n_verified": n_verified,
            "contact_any": c_any, "contact_l6mug": c_l6,
            "Dmax": float(Dmax), "d_soup": d_soup(live),
            "decision_s": time.time() - t0, "terminated": False,
        })

    total_wall = time.time() - t_start
    d_soup_final = d_soup(live)
    sg_final = subgoal_values(live)

    # ---- 汇总 ----
    print("\n=== P1a 结果 ===")
    print(f"执行步数: {len([r for r in step_records if not r.get('terminated')])} / {PILOT_STEPS}")
    print(f"终止原因: {terminated_reason}")
    print(f"真实安全: 首次禁止接触={first_actual_contact}  Dmax={Dmax*1000:.3f}mm  "
          f"CAR={'合格' if Dmax < DELTA_CAR - EPS else '越界'}")
    print(f"Nominal 保留: {n_nominal_kept} 次")
    print(f"候选使用: {cand_use}")
    print(f"预测一致性: 预测miss={n_pred_miss}")
    print(f"候选集失效: NO_VERIFIED_SEQUENCE={n_nvs} 次 @ {nvs_steps if nvs_steps else '-'}")
    print(f"任务进度: d_soup {d_soup_init*1000:.1f}mm → {d_soup_final*1000:.1f}mm")
    print(f"子目标(终): {sg_final}")
    print(f"计算: 总验证={n_verify_total}  π0.5推理(执行)={n_generate_live}  "
          f"总耗时={total_wall:.1f}s")

    # 决策耗时分布
    dts = sorted(r["decision_s"] for r in step_records if not r.get("terminated"))
    if dts:
        import statistics
        print(f"决策耗时: 均值={statistics.mean(dts):.2f}s  "
              f"P95={np.percentile(dts, 95):.2f}s  最大={max(dts):.2f}s")

    out = pathlib.Path(ROOT, "data", "audit", "m1_p2_p1a_pilot.json")
    out.write_text(json.dumps({
        "config": {"H": H, "PILOT_STEPS": PILOT_STEPS, "SWITCH": SWITCH,
                   "DELTA_CAR": DELTA_CAR, "EPS": EPS},
        "summary": {
            "steps_executed": len([r for r in step_records if not r.get("terminated")]),
            "terminated_reason": terminated_reason,
            "first_actual_contact": first_actual_contact, "Dmax": Dmax,
            "n_nominal_kept": n_nominal_kept, "cand_use": cand_use,
            "n_pred_miss": n_pred_miss, "n_nvs": n_nvs, "nvs_steps": nvs_steps,
            "d_soup_init": d_soup_init, "d_soup_final": d_soup_final,
            "subgoal_final": sg_final, "n_verify_total": n_verify_total,
            "n_generate_live": n_generate_live, "total_wall_s": total_wall,
        },
        "records": step_records,
    }, ensure_ascii=False, indent=2, default=str))
    print(f"\n结果: {out}")

    pol.close()
    live.close()
    scratch.close()


if __name__ == "__main__":
    main()
