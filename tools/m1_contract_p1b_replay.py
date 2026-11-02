"""M1-Contract-Audit：P1b 成功轨迹重放审计（新接触监视器）。

保持 HVNP 决策逻辑与 P1b 完全一致（决策用旧 build_body_geoms），但执行时用
monitor_contacts 逐 substep 审计 robot↔受保护物体/支撑物/摆设 的接触。

目标：验证 P1b 声称的「全程零禁止接触」在完整接触清单下是否仍成立。
"""

from __future__ import annotations
import json, os, sys, time, warnings
warnings.filterwarnings("ignore")
ROOT = "/data/safeVLA"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "dataset", "SafeLIBERO", "safelibero"))
sys.path.insert(0, "/data/2608.SDL/MojocoProj/src/pi05")
sys.path.insert(0, os.path.join(ROOT, "third_party", "aegis"))
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
from src.pnsg.policy_adapter import PnsgPolicy
from src.safety.aegis_adapted import AegisSafetyLayer
from src.safety.contract_objects import monitor_contacts, forbidden_contact_geoms, robot_geoms
from tools.g1a_soup_trace import make_env, body_state
from tools.m1_p1_design_gate_v2 import build_body_geoms, generate_block
from tools.m1_p1_design_gate_v3 import disp_from_p0, d_soup
from tools.m1_p2_p1a_pilot import (select_action, Z_DIR, L_DIR, AVOID_MAG, AVOID_STEPS,
                                   SUITE, LEVEL, TASK_ID, REPLAN_STEPS, IMAGE_SIZE, SWITCH,
                                   H, PROTECTED, DELTA_CAR, EPS)

MAX_STEPS = 550

def exec_audited(env, act, rg_old, og_old):
    """执行 1 个控制步（25 substep），每 substep 用 monitor_contacts 审计 forbidden 接触。

    返回 (obs, forbidden_events)。forbidden_events = [(substep, body1, body2, dist)]。
    """
    e = env.env
    e.timestep += 1
    policy_step = True
    events = []
    m = env.sim.model._model
    d = env.sim.data._data
    for i in range(int(e.control_timestep / e.model_timestep)):
        e.sim.forward()
        e._pre_action(act, policy_step)
        e.sim.step()
        # 新监视器审计（逐 substep）
        mon = monitor_contacts(env)
        for ev in mon["forbidden"] + mon["unclassified"]:
            events.append((i, ev["body1"], ev["body2"], ev["dist"]))
        e._update_observables()
        policy_step = False
    e.cur_time += e.control_timestep
    obs = e._get_observations()
    return obs, events


def main():
    from libero.libero import benchmark
    ts = benchmark.get_benchmark_dict()[SUITE](safety_level=LEVEL)
    task = ts.get_task(TASK_ID)
    prompt = task.language
    pol = PnsgPolicy()

    live = make_env(ts, task, 2)  # P1b 是 State02
    layer_live = AegisSafetyLayer(live)
    scratch = make_env(ts, task, 2)
    layer_s = AegisSafetyLayer(scratch)

    obs = live.env._get_observations()
    layer_live.reset(obs, prompt)
    rg, og = build_body_geoms(live)  # 旧集合（决策用，保持 P1b 轨迹）
    p0 = body_state(live, PROTECTED)["pos"].copy()

    print("=== P1b 轨迹重放审计（新接触监视器）===", flush=True)

    avoid_remaining = 0
    avoid_dir = None
    avoid_name = None
    Dmax = 0.0
    nominal = None
    all_forbidden = []  # (step, substep, body1, body2, dist)
    first_contact_step = None

    t0 = time.time()
    for step in range(MAX_STEPS):
        if step == SWITCH:
            layer_live.reset(obs, prompt)
        if step % REPLAN_STEPS == 0:
            nominal = generate_block(pol, obs, prompt, step // REPLAN_STEPS)
        nom_a = nominal[step % REPLAN_STEPS]

        if avoid_remaining > 0:
            act = np.clip(nom_a + AVOID_MAG * avoid_dir, -1, 1)
            avoid_remaining -= 1
        else:
            chosen, verdicts = select_action(scratch, layer_s, pol, prompt,
                                             live, layer_live, nominal, step,
                                             Dmax, rg, og, p0)
            if chosen == "NO_VERIFIED_SEQUENCE":
                print(f"  NVS @ step{step}", flush=True)
                break
            if chosen == "N":
                act = nom_a
            elif chosen == "A":
                act = layer_live.filter(nom_a, obs)[0]
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

        obs, events = exec_audited(live, act, rg, og)
        for (si, b1, b2, dist) in events:
            all_forbidden.append((step, si, b1, b2, dist))
            if first_contact_step is None:
                first_contact_step = step
        Dmax = max(Dmax, disp_from_p0(live, p0))

        if live.check_success():
            print(f"  任务成功 @ step{step}", flush=True)
            break

    total_wall = time.time() - t0
    print(f"\n=== 审计结果 ===", flush=True)
    print(f"执行步数: {step+1}", flush=True)
    print(f"Dmax = {Dmax*1000:.6f}mm", flush=True)
    print(f"首次 forbidden 接触 @ step {first_contact_step}", flush=True)
    print(f"forbidden 事件总数: {len(all_forbidden)}", flush=True)

    # 去重后的接触对类型
    pair_types = {}
    for (st, si, b1, b2, dist) in all_forbidden:
        key = f"{b1}↔{b2}"
        pair_types[key] = pair_types.get(key, 0) + 1
    print(f"\nforbidden 接触对类型（去重计数）:")
    for k, v in sorted(pair_types.items()):
        print(f"  {k}: {v} 次", flush=True)

    # 前 10 个 forbidden 事件详情
    print(f"\n前 10 个 forbidden 事件:")
    for (st, si, b1, b2, dist) in all_forbidden[:10]:
        print(f"  step{st}/substep{si}: {b1}↔{b2} dist={dist*1000:.3f}mm", flush=True)

    out = json.dumps({
        "steps": step + 1, "Dmax": Dmax, "first_contact_step": first_contact_step,
        "n_forbidden_events": len(all_forbidden), "pair_types": pair_types,
        "first_events": [(st, si, b1, b2, dist) for (st, si, b1, b2, dist) in all_forbidden[:20]],
        "total_wall_s": total_wall,
    }, ensure_ascii=False, indent=2)
    with open(os.path.join(ROOT, "data", "audit", "m1_contract_p1b_replay.json"), "w") as f:
        f.write(out)
    print(f"\n结果: data/audit/m1_contract_p1b_replay.json", flush=True)

    pol.close(); live.close(); scratch.close()

if __name__ == "__main__":
    main()
