"""M1-Contract-Audit：E1 代表案例（S01/st3）HVNP 重放审计。

E1 的 HVNP 决策用旧 build_body_geoms（漏检底座）。重放 S01/st3 的 HVNP，
每步执行用新 monitor_contacts 审计：HVNP 在 NVS 前是否碰了底座/受保护物体。
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
from src.safety.contract_objects import monitor_contacts
from tools.g1a_soup_trace import make_env, body_state
from tools.m1_p1_design_gate_v2 import build_body_geoms
from tools.m1_p1_design_gate_v3 import disp_from_p0
from tools.m1_p2_p1a_pilot import (Z_DIR, L_DIR, AVOID_MAG, AVOID_STEPS,
                                   SUITE, LEVEL, TASK_ID, REPLAN_STEPS, IMAGE_SIZE, SWITCH,
                                   H, PROTECTED, DELTA_CAR, EPS)
from tools.m1_p4_p0e_joint_execution import generate_block, select_action  # 6/14参数修正版

def main():
    from libero.libero import benchmark
    ts = benchmark.get_benchmark_dict()[SUITE](safety_level=LEVEL)
    task = ts.get_task(TASK_ID)
    prompt = task.language
    pol = PnsgPolicy()

    for state_id, stream in [(1, 3), (4, 3), (9, 3)]:
        live = make_env(ts, task, state_id)
        layer_live = AegisSafetyLayer(live)
        scratch = make_env(ts, task, state_id)
        layer_s = AegisSafetyLayer(scratch)
        obs = live.env._get_observations()
        layer_live.reset(obs, prompt)
        rg, og = build_body_geoms(live)
        p0 = body_state(live, PROTECTED)["pos"].copy()

        avoid_remaining = 0; avoid_dir = None; avoid_name = None
        Dmax = 0.0; nominal = None
        all_forbidden = []; first_forbidden_step = None
        nvs_step = None

        for step in range(550):
            if step == SWITCH:
                layer_live.reset(obs, prompt)
            if step % REPLAN_STEPS == 0:
                nominal = generate_block(pol, obs, prompt, step // REPLAN_STEPS, state_id, stream)
            nom_a = nominal[step % REPLAN_STEPS]

            if avoid_remaining > 0:
                act = np.clip(nom_a + AVOID_MAG * avoid_dir, -1, 1)
                avoid_remaining -= 1
            else:
                chosen, verdicts = select_action(scratch, layer_s, pol, prompt,
                                                 live, layer_live, nominal, step,
                                                 Dmax, rg, og, p0, state_id, stream)
                if chosen == "NO_VERIFIED_SEQUENCE":
                    nvs_step = step
                    break
                if chosen == "N":
                    act = nom_a
                elif chosen == "Hold+N":
                    act = np.zeros(7)
                elif chosen == "A":
                    act = layer_live.filter(nom_a, obs)[0]
                else:
                    act = np.clip(nom_a + AVOID_MAG * (Z_DIR if chosen == "Z+A" else L_DIR), -1, 1)
                    avoid_remaining = AVOID_STEPS - 1
                    avoid_dir = Z_DIR if chosen == "Z+A" else L_DIR

            # 执行 + 新 monitor 审计
            e = live.env
            e.timestep += 1
            policy_step = True
            for i in range(int(e.control_timestep / e.model_timestep)):
                e.sim.forward()
                e._pre_action(act, policy_step)
                e.sim.step()
                mon = monitor_contacts(live)
                for ev in mon["forbidden"] + mon["unclassified"]:
                    all_forbidden.append((step, i, ev["body1"], ev["body2"], ev["dist"]))
                    if first_forbidden_step is None:
                        first_forbidden_step = step
                e._update_observables()
                policy_step = False
            e.cur_time += e.control_timestep
            obs = e._get_observations()
            Dmax = max(Dmax, disp_from_p0(live, p0))

        pair_types = {}
        for (st, si, b1, b2, dist) in all_forbidden:
            key = f"{b1}↔{b2}"
            pair_types[key] = pair_types.get(key, 0) + 1

        print(f"S{state_id:02d}/st{stream}: NVS@step{nvs_step} Dmax={Dmax*1000:.4f}mm "
              f"forbidden事件={len(all_forbidden)} 首次@{first_forbidden_step}", flush=True)
        for k, v in sorted(pair_types.items()):
            print(f"    {k}: {v} 次", flush=True)

        live.close(); scratch.close()

    pol.close()

if __name__ == "__main__":
    main()
