"""CCA-ASF E1-B0：Catastrophic CAR Violation Audit（290mm mug 位移定向归因）。

目标：定位首次 CAR 越界（Dmax ≥ 1mm）时刻，逐 substep 捕获：
- 接触链（robot-mug / robot-base / 其他，含 efc_address）
- 名义动作 / 执行动作 / β / 任务方向 d_y
- z_fixed / u 解 / CBF 残差
- mug 位置/速度/Dmax

重复运行 N 次直到复现 mug 大幅位移，每次逐 substep 记录。
不做因果断言，只收集完整证据。
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
sys.path.insert(0, "/data/2608.SDL/MojocoProj/src/pi05")
sys.path.insert(0, os.path.join(ROOT, "third_party", "aegis"))

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
import mujoco  # noqa: E402

from src.policies import Pi05Policy  # noqa: E402
from src.safety.aegis_adapted import AegisSafetyLayer  # noqa: E402
from src.safety.cca_asf.intention import chunk_direction  # noqa: E402
from src.safety.cca_asf.metric import B_DIAG, compute_conflict_beta, map_dz_to_dy  # noqa: E402
from tools.g1a_soup_trace import make_env, body_state  # noqa: E402

SUITE = "safelibero_long"
LEVEL = "I"
TASK_ID = 0
STATE_ID = 1
STREAM = 3
REPLAN = 5
IMAGE_SIZE = 256
BETA_MAX = 4.0
N_REPEATS = 6
MAX_STEPS = 300
CAR_EPS = 0.001  # 1mm

PROTECTED = "red_coffee_mug_obstacle_1_main"


def contact_chain(env):
    """当前 substep 的所有 contact，含 body 对 + dist + efc_address。"""
    m = env.sim.model._model
    d = env.sim.data._data
    out = []
    for c in range(d.ncon):
        g1, g2 = d.contact[c].geom1, d.contact[c].geom2
        b1 = env.sim.model.body_names[m.geom_bodyid[g1]]
        b2 = env.sim.model.body_names[m.geom_bodyid[g2]]
        out.append({
            "pair": f"{b1}↔{b2}",
            "dist": float(d.contact[c].dist),
            "efc": int(d.contact[c].efc_address),
        })
    return out


def run_once(ts, task, prompt, chunk0, meta0):
    """跑一次，逐 substep 检测首次 mug 位移 > 1mm，返回证据或 None。"""
    pol = Pi05Policy(verbose=False, replan_steps=REPLAN, image_size=IMAGE_SIZE)
    env = make_env(ts, task, STATE_ID)
    pol.reset()
    pol.inject_initial_chunk(chunk0, metadata=meta0)
    layer = AegisSafetyLayer(env, filter_mode="cca_asf", beta_max=BETA_MAX)
    layer.record_qp_snapshot = True
    layer.reset(env.env._get_observations(), prompt)

    p0 = body_state(env, PROTECTED)["pos"].copy()
    m = env.sim.model._model
    d = env.sim.data._data

    for step in range(MAX_STEPS):
        obs = env.env._get_observations()
        action = pol.act(obs, prompt)
        chunk = pol.get_current_chunk()
        idx = pol.buffer_index - 1
        d_z = chunk_direction(chunk[max(0, idx):]) if chunk is not None else None
        if d_z is not None:
            layer.set_task_direction(d_z)
        exec_a, info = layer.filter(action, obs)

        # 逐 substep 执行 + 检测
        e = env.env
        e.timestep += 1
        policy_step = True
        for i in range(int(e.control_timestep / e.model_timestep)):
            e.sim.forward()
            e._pre_action(exec_a, policy_step)
            e.sim.step()
            disp = float(np.sum(np.abs(body_state(env, PROTECTED)["pos"] - p0)))
            if disp >= CAR_EPS:
                # 捕获证据
                mug_pos = body_state(env, PROTECTED)["pos"].copy()
                mug_vel = body_state(env, PROTECTED)["vel"].copy()
                ev = {
                    "step": step, "substep": i,
                    "mug_disp_mm": disp * 1000,
                    "mug_pos": mug_pos.tolist(),
                    "mug_vel": mug_vel.tolist(),
                    "nom_action": np.asarray(action).tolist(),
                    "exec_action": np.asarray(exec_a).tolist(),
                    "beta": float(info.get("beta", 0.0)),
                    "task_direction": (None if info.get("task_direction") is None
                                       else info["task_direction"].tolist()),
                    "z_fixed": layer.z_fixed.tolist(),
                    "h": info.get("h"),
                    "feasible": info.get("feasible"),
                    "contacts": contact_chain(env),
                    "qp_snapshot": info.get("qp_snapshot"),
                }
                env.close()
                pol.close()
                return ev
            e._update_observables()
            policy_step = False
        e.cur_time += e.control_timestep
        obs = e._get_observations()

    env.close()
    pol.close()
    return None


def main():
    from libero.libero import benchmark
    ts = benchmark.get_benchmark_dict()[SUITE](safety_level=LEVEL)
    task = ts.get_task(TASK_ID)
    prompt = task.language

    # 生成首 chunk（一次）
    pol_gen = Pi05Policy(verbose=False, replan_steps=REPLAN, image_size=IMAGE_SIZE)
    env0 = make_env(ts, task, STATE_ID)
    pol_gen.reset()
    obs0 = env0.env._get_observations()
    _ = pol_gen.act(obs0, prompt)
    chunk0 = pol_gen.get_current_chunk()
    meta0 = pol_gen.get_chunk_metadata()
    env0.close()

    events = []
    for rep in range(N_REPEATS):
        ev = run_once(ts, task, prompt, chunk0, meta0)
        if ev is not None:
            events.append(ev)
            print(f"rep{rep}: 复现 mug 位移 {ev['mug_disp_mm']:.1f}mm @ step{ev['step']}/substep{ev['substep']} "
                  f"beta={ev['beta']:.2f}", flush=True)
        else:
            print(f"rep{rep}: 未复现（{MAX_STEPS}步内 mug 未超 1mm）", flush=True)

    pol_gen.close()

    # 汇总
    out = {
        "n_repeats": N_REPEATS,
        "n_reproduced": len(events),
        "events": events,
    }
    out_path = os.path.join(ROOT, "data", "audit", "cca_asf_e1_catastrophic_audit.json")
    with open(out_path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n=== B 归因结果 ===")
    print(f"复现 {len(events)}/{N_REPEATS} 次")
    if events:
        # 分析首次事件的接触链
        ev0 = events[0]
        print(f"首次事件: mug 位移 {ev0['mug_disp_mm']:.1f}mm")
        print(f"  beta={ev0['beta']:.2f}, h={ev0['h']}, feasible={ev0['feasible']}")
        print(f"  接触链（前 8）:")
        for c in ev0["contacts"][:8]:
            print(f"    {c['pair']} dist={c['dist']*1000:.3f}mm efc={c['efc']}")
    print(f"\n结果: {out_path}")


if __name__ == "__main__":
    main()
