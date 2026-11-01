"""M1-P2-P1b：HVNP 完整 Episode（Long-I/Task0/State02/Stream0，最多 550 步）。

复用 P1a 的已验证机制（select_action/verify_candidate/save_full/restore_full），
扩展到完整任务：
- 最长 550 步（SafeLIBERO long max_steps），check_success 提前终止。
- 记录 soup/cream 子目标首次满足步、任务成功、CAR、SSR、严格禁止接触。
- 区分「干预事件数」与「非名义实际动作数」。
- 完整逐步决策日志 + 决策耗时分布。

算法设置与 P1a 完全一致，不调任何超参。
"""

from __future__ import annotations

import json
import os
import pathlib
import statistics
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

from src.pnsg.policy_adapter import PnsgPolicy  # noqa: E402
from src.safety.aegis_adapted import AegisSafetyLayer  # noqa: E402
from tools.g1a_soup_trace import make_env, body_state, eef_pos, subgoal_values  # noqa: E402
from tools.m1_p1_design_gate_v2 import step_tracked, build_body_geoms, generate_block  # noqa: E402
from tools.m1_p1_design_gate_v3 import disp_from_p0, d_soup  # noqa: E402
from tools.m1_p2_p1a_pilot import (  # noqa: E402
    select_action,
    Z_DIR, L_DIR, AVOID_MAG, AVOID_STEPS,
    SUITE, LEVEL, TASK_ID, STATE_ID, STREAM, REPLAN_STEPS, IMAGE_SIZE, SWITCH,
    H, PROTECTED, DELTA_CAR, EPS,
)

MAX_STEPS = 550


def contact_details(env, rg, og):
    """当前控制步边界的 robot↔obstacle 接触 body 对（substep 内接触可能已消失）。"""
    m = env.sim.model._model
    dd = env.sim.data._data
    pairs = []
    for c in range(dd.ncon):
        g1, g2 = dd.contact[c].geom1, dd.contact[c].geom2
        if (g1 in rg and g2 in og) or (g2 in rg and g1 in og):
            b1 = env.sim.model.body_names[m.geom_bodyid[g1]]
            b2 = env.sim.model.body_names[m.geom_bodyid[g2]]
            pairs.append(f"{b1}↔{b2}")
    return pairs


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

    print("=== M1-P2-P1b：HVNP 完整 Episode（550 步）===", flush=True)

    # 控制器状态
    avoid_remaining = 0
    avoid_dir = None
    avoid_name = None

    # 记录
    step_records = []
    Dmax = 0.0
    first_actual_contact = None
    first_l6mug = None
    first_contact_details = []
    n_nominal_kept = 0
    n_intervention_events = 0
    n_non_nominal_acts = 0
    cand_use = {"N": 0, "A": 0, "Z+A": 0, "L+A": 0, "Hold+N": 0, "cont": 0}
    n_verify_total = 0
    n_nvs = 0
    nvs_steps = []
    n_pred_miss = 0
    nominal = None
    d_soup_init = d_soup(live)
    n_generate_live = 0
    soup_first_done = None
    cream_first_done = None
    success_step = None

    t_start = time.time()
    terminated_reason = f"达到 {MAX_STEPS} 步上限"
    task_success = False

    for step in range(MAX_STEPS):
        if step == SWITCH:
            layer_live.reset(obs, prompt)
        if step % REPLAN_STEPS == 0:
            nominal = generate_block(pol, obs, prompt, step // REPLAN_STEPS)
            n_generate_live += 1
        nom_a = nominal[step % REPLAN_STEPS]

        t0 = time.time()

        if avoid_remaining > 0:
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
            else:
                n_intervention_events += 1

        # 非名义实际动作计数（含 cont 续步）
        if not np.allclose(np.asarray(act), np.asarray(nom_a)):
            n_non_nominal_acts += 1

        # 执行
        obs, _r, _d, _i, c_any, c_l6, _ = step_tracked(live, act, rg, og)
        Dmax = max(Dmax, disp_from_p0(live, p0))
        if c_any and first_actual_contact is None:
            first_actual_contact = step
            first_contact_details = contact_details(live, rg, og)
        if c_l6 and first_l6mug is None:
            first_l6mug = step

        if pred_first_contact is False and c_any:
            n_pred_miss += 1

        # 子目标 + 成功判断
        sg = subgoal_values(live)
        if sg.get("alphabet_soup_1") and soup_first_done is None:
            soup_first_done = step
        if sg.get("cream_cheese_1") and cream_first_done is None:
            cream_first_done = step
        if live.check_success():
            task_success = True
            success_step = step
            terminated_reason = f"任务成功 @ step{step}"
            step_records.append({
                "step": step, "chosen": chosen, "n_verified": n_verified,
                "contact_any": c_any, "contact_l6mug": c_l6,
                "Dmax": float(Dmax), "d_soup": d_soup(live),
                "decision_s": time.time() - t0, "terminated": False, "success": True,
            })
            break

        step_records.append({
            "step": step, "chosen": chosen, "n_verified": n_verified,
            "contact_any": c_any, "contact_l6mug": c_l6,
            "Dmax": float(Dmax), "d_soup": d_soup(live),
            "decision_s": time.time() - t0, "terminated": False, "success": False,
        })

    total_wall = time.time() - t_start
    d_soup_final = d_soup(live)
    sg_final = subgoal_values(live)
    car_ok = Dmax < DELTA_CAR - EPS
    strict_safe = (first_actual_contact is None)

    # ---- 汇总 ----
    print("\n=== P1b 结果 ===")
    print(f"任务成功: {task_success}  @ step{success_step}")
    print(f"终止原因: {terminated_reason}")
    print(f"执行步数: {len([r for r in step_records if not r.get('terminated')])} / {MAX_STEPS}")
    print(f"真实安全: 首次禁止接触={first_actual_contact} ({first_contact_details if first_contact_details else '无'})  "
          f"首次l6mug={first_l6mug}")
    print(f"Dmax={Dmax*1000:.3f}mm  CAR={'合格' if car_ok else '越界'}  SSR={'成立' if task_success and car_ok else '不成立'}")
    print(f"子目标: soup={'✓' if sg_final.get('alphabet_soup_1') else '✗'}(首@{soup_first_done})  "
          f"cream={'✓' if sg_final.get('cream_cheese_1') else '✗'}(首@{cream_first_done})")
    print(f"Nominal 保留: {n_nominal_kept} 步 / 干预事件 {n_intervention_events} 次 / 非名义动作 {n_non_nominal_acts} 步")
    print(f"候选使用: {cand_use}")
    print(f"预测一致性(首动作): miss={n_pred_miss}")
    print(f"候选集失效: NO_VERIFIED_SEQUENCE={n_nvs} @ {nvs_steps if nvs_steps else '-'}")
    print(f"任务进度: d_soup {d_soup_init*1000:.1f} → {d_soup_final*1000:.1f}mm")
    print(f"计算: 总验证={n_verify_total}  π0.5推理(执行)={n_generate_live}  总耗时={total_wall:.1f}s")

    dts = sorted(r["decision_s"] for r in step_records if not r.get("terminated"))
    if dts:
        print(f"决策耗时: 均值={statistics.mean(dts):.2f}s  P95={np.percentile(dts,95):.2f}s  最大={max(dts):.2f}s")

    out = pathlib.Path(ROOT, "data", "audit", "m1_p2_p1b_full_episode.json")
    out.write_text(json.dumps({
        "config": {"MAX_STEPS": MAX_STEPS, "H": H, "SWITCH": SWITCH,
                   "DELTA_CAR": DELTA_CAR, "EPS": EPS},
        "summary": {
            "task_success": task_success, "success_step": success_step,
            "terminated_reason": terminated_reason,
            "first_actual_contact": first_actual_contact,
            "first_l6mug": first_l6mug, "first_contact_details": first_contact_details,
            "Dmax": Dmax, "car_ok": car_ok,
            "ssr": task_success and car_ok, "strict_safe": strict_safe,
            "soup_first_done": soup_first_done, "cream_first_done": cream_first_done,
            "subgoal_final": sg_final,
            "n_nominal_kept": n_nominal_kept, "n_intervention_events": n_intervention_events,
            "n_non_nominal_acts": n_non_nominal_acts, "cand_use": cand_use,
            "n_pred_miss": n_pred_miss, "n_nvs": n_nvs, "nvs_steps": nvs_steps,
            "d_soup_init": d_soup_init, "d_soup_final": d_soup_final,
            "n_verify_total": n_verify_total, "n_generate_live": n_generate_live,
            "total_wall_s": total_wall,
        },
        "records": step_records,
    }, ensure_ascii=False, indent=2, default=str))
    print(f"\n结果: {out}")

    pol.close()
    live.close()
    scratch.close()


if __name__ == "__main__":
    main()
