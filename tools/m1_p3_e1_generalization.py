"""M1-P3-E1：Same-task Generalization（Long-I/Task0，12 组配对）。

4 state × 3 stream，每组 3 方法配对（Vanilla π0.5 / AEGIS / HVNP），共 36 episodes。
- 相同初始状态 + 匹配 CRN（nominal_seed 共享，stream = replicate_id）。
- 同 worker 内配对，减少环境/模型执行差异。
- 指标：TSR / CAR(1mm) / SSR / StrictSSR / 干预统计 / 子目标 / 决策耗时。
- 保留每组合的转移关系，不只看平均值。

算法设置与 P1b 完全一致，冻结不调参。
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
from tools.g1a_soup_trace import make_env, body_state, subgoal_values  # noqa: E402
from tools.m1_p1_design_gate_v2 import step_tracked, build_body_geoms, generate_block  # noqa: E402
from tools.m1_p1_design_gate_v3 import disp_from_p0, d_soup  # noqa: E402
from tools.m1_p2_p1a_pilot import (  # noqa: E402
    select_action,
    Z_DIR, L_DIR, AVOID_MAG, AVOID_STEPS,
    SUITE, LEVEL, TASK_ID, REPLAN_STEPS, IMAGE_SIZE, SWITCH,
    H, PROTECTED, DELTA_CAR, EPS,
)

MAX_STEPS = 550

# 预注册测试组合（锁定，不按 HVNP 表现选样）
STATE_IDS = [1, 4, 9, 12]
STREAMS = [3, 4, 5]
METHODS = ["vanilla", "aegis", "hvnp"]


def run_episode(ts, task, prompt, pol, state_id, stream, method):
    """运行一条 episode，返回完整指标。"""
    t_start = time.time()
    env = make_env(ts, task, state_id)
    layer = AegisSafetyLayer(env)
    obs = env.env._get_observations()
    layer.reset(obs, prompt)
    rg, og = build_body_geoms(env)
    p0 = body_state(env, PROTECTED)["pos"].copy()

    if method == "hvnp":
        scratch = make_env(ts, task, state_id)
        layer_s = AegisSafetyLayer(scratch)

    Dmax = 0.0
    first_contact = None
    first_contact_details = []
    n_contact_steps = 0
    n_nominal_kept = 0
    n_intervention_events = 0
    n_non_nominal_acts = 0
    n_nominal_unsafe = 0  # N 预测不安全的次数（= 干预事件 + NVS）
    cand_use = {"N": 0, "A": 0, "Z+A": 0, "L+A": 0, "Hold+N": 0, "cont": 0}
    n_nvs = 0
    nvs_step = None
    n_pred_miss = 0
    n_verify_total = 0
    decision_times = []
    nominal = None
    soup_first_done = None
    cream_first_done = None
    success = False
    success_step = None
    terminated_reason = f"达到 {MAX_STEPS} 步"
    n_generate = 0

    # HVNP 控制器状态
    avoid_remaining = 0
    avoid_dir = None
    avoid_name = None

    for step in range(MAX_STEPS):
        if step == SWITCH:
            layer.reset(obs, prompt)
        if step % REPLAN_STEPS == 0:
            nominal = generate_block(pol, obs, prompt, step // REPLAN_STEPS)
            n_generate += 1
        nom_a = nominal[step % REPLAN_STEPS]

        t0 = time.time()

        if method == "vanilla":
            act = nom_a
            chosen = "N"
            n_nominal_kept += 1
            pred_first_contact = None
        elif method == "aegis":
            act = layer.filter(nom_a, obs)[0]
            chosen = "A"
            pred_first_contact = None
        elif method == "hvnp":
            if avoid_remaining > 0:
                act = np.clip(nom_a + AVOID_MAG * avoid_dir, -1, 1)
                chosen = f"cont:{avoid_name}"
                avoid_remaining -= 1
                n_non_nominal_acts += 1
                pred_first_contact = None
            else:
                chosen, verdicts = select_action(scratch, layer_s, pol, prompt,
                                                 env, layer, nominal, step,
                                                 Dmax, rg, og, p0)
                n_verify_total += len(verdicts)
                decision_times.append(time.time() - t0)
                if not verdicts["N"]["safe"]:
                    n_nominal_unsafe += 1
                if chosen == "NO_VERIFIED_SEQUENCE":
                    n_nvs += 1
                    nvs_step = step
                    terminated_reason = f"NO_VERIFIED_SEQUENCE @ step{step}"
                    break
                pred_first_contact = verdicts[chosen]["first_step_contact"]
                if chosen == "N":
                    act = nom_a
                    n_nominal_kept += 1
                elif chosen == "A":
                    act = layer.filter(nom_a, obs)[0]
                elif chosen == "Z+A":
                    act = np.clip(nom_a + AVOID_MAG * Z_DIR, -1, 1)
                    avoid_remaining = AVOID_STEPS - 1
                    avoid_dir = Z_DIR; avoid_name = chosen
                elif chosen == "L+A":
                    act = np.clip(nom_a + AVOID_MAG * L_DIR, -1, 1)
                    avoid_remaining = AVOID_STEPS - 1
                    avoid_dir = L_DIR; avoid_name = chosen
                elif chosen == "Hold+N":
                    act = np.zeros(7)
                cand_use[chosen] = cand_use.get(chosen, 0) + 1
                if chosen != "N":
                    n_intervention_events += 1
                if not np.allclose(np.asarray(act), np.asarray(nom_a)):
                    n_non_nominal_acts += 1
        else:
            raise ValueError(method)

        # 执行
        obs, _r, _d, _i, c_any, c_l6, _ = step_tracked(env, act, rg, og)
        Dmax = max(Dmax, disp_from_p0(env, p0))
        if c_any:
            n_contact_steps += 1
            if first_contact is None:
                first_contact = step
                first_contact_details = _contact_details(env, rg, og)

        # HVNP 预测一致性（仅执行首动作）
        if method == "hvnp" and pred_first_contact is False and c_any:
            n_pred_miss += 1

        sg = subgoal_values(env)
        if sg.get("alphabet_soup_1") and soup_first_done is None:
            soup_first_done = step
        if sg.get("cream_cheese_1") and cream_first_done is None:
            cream_first_done = step
        if env.check_success():
            success = True
            success_step = step
            terminated_reason = f"成功 @ step{step}"
            break

    car_ok = Dmax < DELTA_CAR - EPS
    strict_safe = first_contact is None
    total_wall = time.time() - t_start

    result = {
        "state_id": state_id, "stream": stream, "method": method,
        "success": success, "success_step": success_step,
        "terminated_reason": terminated_reason,
        "steps": step + 1 if not (method == "hvnp" and n_nvs > 0) else step,
        "Dmax": Dmax, "car_ok": car_ok, "ssr": success and car_ok,
        "strict_safe": strict_safe, "strict_ssr": success and car_ok and strict_safe,
        "first_contact": first_contact, "first_contact_details": first_contact_details,
        "n_contact_steps": n_contact_steps,
        "soup_first_done": soup_first_done, "cream_first_done": cream_first_done,
        "n_nominal_kept": n_nominal_kept,
        "n_intervention_events": n_intervention_events,
        "n_non_nominal_acts": n_non_nominal_acts,
        "n_nominal_unsafe": n_nominal_unsafe,
        "cand_use": cand_use, "n_nvs": n_nvs, "nvs_step": nvs_step,
        "n_pred_miss": n_pred_miss, "n_verify_total": n_verify_total,
        "decision_mean_s": statistics.mean(decision_times) if decision_times else None,
        "decision_p95_s": float(np.percentile(decision_times, 95)) if decision_times else None,
        "decision_max_s": max(decision_times) if decision_times else None,
        "wall_s": total_wall, "n_generate": n_generate,
    }
    env.close()
    if method == "hvnp":
        scratch.close()
    return result


def _contact_details(env, rg, og):
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

    print("=== M1-P3-E1：Same-task Generalization（12 组 × 3 方法）===", flush=True)
    print(f"测试组合: state {STATE_IDS} × stream {STREAMS}", flush=True)

    all_results = []
    for state_id in STATE_IDS:
        for stream in STREAMS:
            group = []
            for method in METHODS:
                r = run_episode(ts, task, prompt, pol, state_id, stream, method)
                group.append(r)
                tag = "✓成功" if r["success"] else "✗失败"
                car = "CAR✓" if r["car_ok"] else "CAR✗"
                sc = "零接触" if r["strict_safe"] else f"接触@{r['first_contact']}"
                print(f"  S{state_id:02d}/st{stream}: {method:7s} {tag} {car} {sc} "
                      f"Dmax={r['Dmax']*1000:6.3f}mm 步数={r['steps']}", flush=True)
            all_results.append({"state_id": state_id, "stream": stream, "group": group})

    # ---- 汇总 ----
    def agg(method):
        rs = [g["group"][METHODS.index(method)] for g in all_results]
        n = len(rs)
        tsr = sum(1 for r in rs if r["success"])
        car = sum(1 for r in rs if r["car_ok"])
        ssr = sum(1 for r in rs if r["ssr"])
        strict = sum(1 for r in rs if r["strict_ssr"])
        return {"tsr": tsr, "car": car, "ssr": ssr, "strict": strict, "n": n}

    print("\n=== E1 汇总（原始计数，不宣称显著性）===")
    print(f"{'方法':10s} {'TSR':>8s} {'CAR':>8s} {'SSR':>8s} {'StrictSSR':>10s}")
    for m in METHODS:
        a = agg(m)
        print(f"{m:10s} {a['tsr']:>4d}/{a['n']} {a['car']:>4d}/{a['n']} "
              f"{a['ssr']:>4d}/{a['n']} {a['strict']:>4d}/{a['n']}")

    # HVNP 额外统计
    hvnp_rs = [g["group"][METHODS.index("hvnp")] for g in all_results]
    n_interv = sum(r["n_intervention_events"] for r in hvnp_rs)
    n_nvs = sum(r["n_nvs"] for r in hvnp_rs)
    n_nom_unsafe = sum(r["n_nominal_unsafe"] for r in hvnp_rs)
    n_danger_avoided = n_interv  # 成功避险 = 干预事件（N 不安全 → 选了安全替代）
    print(f"\nHVNP 额外: 总干预事件={n_interv}  总危险名义(N不安全)={n_nom_unsafe}  "
          f"成功避险={n_danger_avoided}  总NO_VERIFIED_SEQUENCE={n_nvs}")
    for r in hvnp_rs:
        if r["n_intervention_events"] > 0 or r["n_nvs"] > 0:
            print(f"  S{r['state_id']:02d}/st{r['stream']}: 危险名义{r['n_nominal_unsafe']}次 "
                  f"干预{r['n_intervention_events']}次 {r['cand_use']} NVS={r['n_nvs']}@step{r['nvs_step']}")

    # 转移关系
    print("\n=== 转移关系（Vanilla → AEGIS → HVNP 成功/安全）===")
    for g in all_results:
        v, a, h = g["group"][0], g["group"][1], g["group"][2]
        print(f"  S{g['state_id']:02d}/st{g['stream']}: "
              f"V[{'✓' if v['success'] else '✗'} CAR{'✓' if v['car_ok'] else '✗'}] "
              f"A[{'✓' if a['success'] else '✗'} CAR{'✓' if a['car_ok'] else '✗'}] "
              f"H[{'✓' if h['success'] else '✗'} CAR{'✓' if h['car_ok'] else '✗'} "
              f"{'零接触' if h['strict_safe'] else '接触'}]")

    out = pathlib.Path(ROOT, "data", "audit", "m1_p3_e1_generalization.json")
    out.write_text(json.dumps({
        "config": {"STATE_IDS": STATE_IDS, "STREAMS": STREAMS, "METHODS": METHODS,
                   "MAX_STEPS": MAX_STEPS, "H": H, "DELTA_CAR": DELTA_CAR, "EPS": EPS},
        "summary": {m: agg(m) for m in METHODS},
        "groups": all_results,
    }, ensure_ascii=False, indent=2, default=str))
    print(f"\n结果: {out}")
    pol.close()


if __name__ == "__main__":
    main()
