"""M1-P2-P1a de-risk：scratch env 全状态 fork 的预测判决保真度。

自主闭环需要「从 live env 当前状态 s_t 复制到独立 scratch env，做 H=10 预测」。
此前 fork 的 bit-exact 问题（qfrc_constraint）必须确认：即使 qpos 非 bit-exact，
接触判决（首次 link6-mug 接触 step、Dmax）是否仍与确定性重跑一致。

测试：在 step54 全状态 fork 到 scratch env，跑候选 B（AEGIS 接续）H=10，
对比已知确定性重跑结果（首次 l6mug @ 59，Dmax ~1.0-1.16mm）。
"""

from __future__ import annotations

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

from src.pnsg.policy_adapter import PnsgPolicy  # noqa: E402
from src.safety.aegis_adapted import AegisSafetyLayer  # noqa: E402
from tools.g1a_soup_trace import make_env  # noqa: E402
from tools.m1_p1_design_gate_v2 import (  # noqa: E402
    step_tracked, build_body_geoms, generate_block,
)
from tools.m1_p1_design_gate_v3 import disp_from_p0  # noqa: E402

SUITE = "safelibero_long"
LEVEL = "I"
TASK_ID = 0
STATE_ID = 2
STREAM = 0
REPLAN_STEPS = 5
IMAGE_SIZE = 224
SWITCH = 40
PROTECTED = "red_coffee_mug_obstacle_1_main"
SPEC = mujoco.mjtState.mjSTATE_INTEGRATION


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


def main():
    from libero.libero import benchmark
    ts = benchmark.get_benchmark_dict()[SUITE](safety_level=LEVEL)
    task = ts.get_task(TASK_ID)
    prompt = task.language

    # live env + scratch env（同 make_env）
    live = make_env(ts, task, STATE_ID)
    layer_live = AegisSafetyLayer(live)
    scratch = make_env(ts, task, STATE_ID)
    layer_scratch = AegisSafetyLayer(scratch)

    obs = live.env._get_observations()
    layer_live.reset(obs, prompt)
    rg, og = build_body_geoms(live)
    p0 = None
    pol = PnsgPolicy()

    # 参考：live 确定性跑 B 到 step64，记录首次 l6mug + Dmax
    ref_first_l6 = None
    ref_Dmax = 0.0
    snapshot = None
    nominal = None
    for step in range(64):
        if step == SWITCH:
            layer_live.reset(obs, prompt)
        if step % REPLAN_STEPS == 0:
            nominal = generate_block(pol, obs, prompt, step // REPLAN_STEPS)
        nom_a = nominal[step % REPLAN_STEPS]
        if p0 is None:
            p0 = np.asarray(live.env._get_observations()["red_coffee_mug_obstacle_1_pos"]).copy() \
                if "red_coffee_mug_obstacle_1_pos" in obs else None
        if step == 54:
            # 用 body_state 取 p0
            from tools.g1a_soup_trace import body_state
            p0 = body_state(live, PROTECTED)["pos"].copy()
            snapshot = save_full(live, layer_live)
            snap_nom = nominal.copy()
        act = nom_a if step < SWITCH else layer_live.filter(nom_a, obs)[0]
        obs, _r, _d, _i, ca, cl, _ = step_tracked(live, act, rg, og)
        ref_Dmax = max(ref_Dmax, disp_from_p0(live, p0))
        if cl and ref_first_l6 is None:
            ref_first_l6 = step

    print("=== P1a de-risk：scratch fork 预测保真度 ===")
    print(f"[参考 live 重跑] 首次 l6mug @ {ref_first_l6}  Dmax={ref_Dmax*1000:.3f}mm")

    # scratch fork：从 step54 全状态恢复，跑候选 B H=10
    restore_full(scratch, layer_scratch, snapshot)
    obs_s = scratch.env._get_observations(force_update=True)
    scratch_first_l6 = None
    scratch_Dmax = 0.0
    nominal_s = snap_nom.copy()
    for t in range(10):
        step = 54 + t
        if step % REPLAN_STEPS == 0:
            nominal_s = generate_block(pol, obs_s, prompt, step // REPLAN_STEPS)
        nom_a_s = nominal_s[step % REPLAN_STEPS]
        act_s, _ = layer_scratch.filter(nom_a_s, obs_s)
        obs_s, _r, _d, _i, ca, cl, _ = step_tracked(scratch, act_s, rg, og)
        scratch_Dmax = max(scratch_Dmax, disp_from_p0(scratch, p0))
        if cl and scratch_first_l6 is None:
            scratch_first_l6 = step

    print(f"[scratch fork]  首次 l6mug @ {scratch_first_l6}  Dmax={scratch_Dmax*1000:.3f}mm")
    print(f"\n对比: 首次接触 step {'一致' if scratch_first_l6 == ref_first_l6 else f'不一致({scratch_first_l6} vs {ref_first_l6})'}")
    print(f"      Dmax {'一致(<1%)' if abs(scratch_Dmax - ref_Dmax) < 0.01 else f'偏差({abs(scratch_Dmax-ref_Dmax)*1000:.3f}mm)'}")
    print(f"      接触判决(是否接触) {'一致' if (scratch_first_l6 is not None) == (ref_first_l6 is not None) else '不一致'}")

    pol.close()
    live.close()
    scratch.close()


if __name__ == "__main__":
    main()
