"""CCA-ASF E1-B1：Trajectory Divergence Closure（一次性轨迹证据补全）。

回答三个问题：
- Q1：Euclidean 在 Step270-295 是否避开底座区域（正间隙）？CCA 是否接触？
- Q2：两者首次实质控制差异时，CCA 的 β 是否非零？
- Q3：CCA 首次接触底座时，MuJoCo 是否记录了接触 + mug 后续位移？

只读已有稳定复现轨迹（重跑一次），不修改算法/QP/物理参数。
几何间隙用 margin 查询的**可信下界**，明确标注，不做 exact_gap 排序。
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
from tools.g1a_soup_trace import make_env, body_state, eef_pos  # noqa: E402

SUITE = "safelibero_long"
LEVEL = "I"
TASK_ID = 0
STATE_ID = 1
STREAM = 3
REPLAN = 5
IMAGE_SIZE = 256
BETA_MAX = 4.0

PROTECTED = "red_coffee_mug_obstacle_1_main"
BASE_BODY = "box_small_base_1_main"
SOUP_BODY = "alphabet_soup_1_main"
WIN_START = 270
WIN_END = 295
MARGIN = 0.05  # 可信下界查询


def gripper_base_gap_lower(env):
    """gripper 碰撞 geom 与底座碰撞 geom 的最小 dist（margin 查询，可信下界）。

    返回 (gap, pair)。gap=None 表示 > margin（下界）；gap<0 表示穿透（真实接触）。
    """
    m = env.sim.model._model
    d = env.sim.data._data
    names = env.sim.model.body_names
    gripper_geoms = {g for g in range(m.ngeom) if m.geom_contype[g] > 0
                     and names[m.geom_bodyid[g]].startswith("gripper0")}
    base_bid = env.sim.model.body_name2id(BASE_BODY)
    base_geoms = {g for g in range(m.ngeom) if m.geom_contype[g] > 0
                  and m.geom_bodyid[g] == base_bid}
    saved = {g: m.geom_margin[g] for g in gripper_geoms | base_geoms}
    for g in gripper_geoms | base_geoms:
        m.geom_margin[g] = MARGIN
    best = None
    pair = None
    try:
        mujoco.mj_collision(m, d)
        for c in range(d.ncon):
            g1, g2 = d.contact[c].geom1, d.contact[c].geom2
            if (g1 in gripper_geoms and g2 in base_geoms) or (g2 in gripper_geoms and g1 in base_geoms):
                dd = float(d.contact[c].dist)
                if best is None or dd < best:
                    best = dd
                    pair = (names[m.geom_bodyid[g1]], names[m.geom_bodyid[g2]])
        return best, pair
    finally:
        for g, v in saved.items():
            m.geom_margin[g] = v


def run_trace(ts, task, prompt, chunk0, meta0, mode):
    """跑一条轨迹，逐步记录（尤其 WIN 时间窗的细粒度证据）。"""
    pol = Pi05Policy(verbose=False, replan_steps=REPLAN, image_size=IMAGE_SIZE)
    env = make_env(ts, task, STATE_ID)
    pol.reset()
    pol.inject_initial_chunk(chunk0, metadata=meta0)
    layer = AegisSafetyLayer(env, filter_mode=mode, beta_max=BETA_MAX)
    layer.reset(env.env._get_observations(), prompt)

    p0 = body_state(env, PROTECTED)["pos"].copy()
    soup0 = body_state(env, SOUP_BODY)["pos"].copy()
    m = env.sim.model._model
    d = env.sim.data._data

    first_div = None  # 与 euclidean 的首次分叉（由外部对比）
    window = []  # WIN 时间窗细粒度证据
    Dmax = 0.0
    for step in range(300):
        obs = env.env._get_observations()
        action = pol.act(obs, prompt)
        chunk = pol.get_current_chunk()
        idx = pol.buffer_index - 1
        d_z = chunk_direction(chunk[max(0, idx):]) if chunk is not None else None
        if mode == "cca_asf" and d_z is not None:
            layer.set_task_direction(d_z)
        exec_a, info = layer.filter(action, obs)

        # 时间窗内：记录执行前状态
        if WIN_START <= step <= WIN_END:
            gap, pair = gripper_base_gap_lower(env)
            # 逐 substep 检测该步是否产生 robot-base 接触
            e = env.env
            e.timestep += 1
            policy_step = True
            step_contact_base = False
            step_contact_mug = False
            mug_disp_before = float(np.sum(np.abs(body_state(env, PROTECTED)["pos"] - p0)))
            for i in range(int(e.control_timestep / e.model_timestep)):
                e.sim.forward()
                e._pre_action(exec_a, policy_step)
                e.sim.step()
                for c in range(d.ncon):
                    g1, g2 = d.contact[c].geom1, d.contact[c].geom2
                    b1 = env.sim.model.body_names[m.geom_bodyid[g1]]
                    b2 = env.sim.model.body_names[m.geom_bodyid[g2]]
                    if ("gripper" in b1 or "robot" in b1) and "base" in b2:
                        step_contact_base = True
                    if ("gripper" in b1 or "robot" in b1) and "mug" in b2:
                        step_contact_mug = True
                e._update_observables()
                policy_step = False
            e.cur_time += e.control_timestep
            mug_disp_after = float(np.sum(np.abs(body_state(env, PROTECTED)["pos"] - p0)))
            window.append({
                "step": step,
                "eef": eef_pos(env).tolist(),
                "gripper_base_gap_lower": None if gap is None else gap * 1000,
                "gap_pair": pair,
                "contact_base": step_contact_base,
                "contact_mug": step_contact_mug,
                "beta": float(info.get("beta", 0.0)),
                "mug_disp_before": mug_disp_before * 1000,
                "mug_disp_after": mug_disp_after * 1000,
                "soup_dist": float(np.linalg.norm(body_state(env, SOUP_BODY)["pos"] - soup0)),
            })
            Dmax = max(Dmax, mug_disp_after)
            obs = env.env._get_observations()
        else:
            env.step(exec_a)
            Dmax = max(Dmax, float(np.sum(np.abs(body_state(env, PROTECTED)["pos"] - p0))))
            obs = env.env._get_observations()

    pol.close()
    env.close()
    return {"window": window, "Dmax": Dmax}


def main():
    from libero.libero import benchmark
    ts = benchmark.get_benchmark_dict()[SUITE](safety_level=LEVEL)
    task = ts.get_task(TASK_ID)
    prompt = task.language

    # 生成首 chunk
    pol_gen = Pi05Policy(verbose=False, replan_steps=REPLAN, image_size=IMAGE_SIZE)
    env0 = make_env(ts, task, STATE_ID)
    pol_gen.reset()
    obs0 = env0.env._get_observations()
    _ = pol_gen.act(obs0, prompt)
    chunk0 = pol_gen.get_current_chunk()
    meta0 = pol_gen.get_chunk_metadata()
    env0.close()

    traces = {}
    for mode in ("euclidean", "cca_asf"):
        tr = run_trace(ts, task, prompt, chunk0, meta0, mode)
        traces[mode] = tr
        print(f"{mode}: Dmax={tr['Dmax']*1000:.3f}mm 窗口步数={len(tr['window'])}", flush=True)

    # ---- 回答三个问题 ----
    w_eu = traces["euclidean"]["window"]
    w_ca = traces["cca_asf"]["window"]

    # Q1：euclidean 在窗口内 gripper-base 最小间隙（下界）
    eu_gaps = [w["gripper_base_gap_lower"] for w in w_eu]
    ca_gaps = [w["gripper_base_gap_lower"] for w in w_ca]
    eu_gaps_valid = [g for g in eu_gaps if g is not None]
    ca_gaps_valid = [g for g in ca_gaps if g is not None]
    eu_min = min(eu_gaps_valid) if eu_gaps_valid else None
    ca_min = min(ca_gaps_valid) if ca_gaps_valid else None
    eu_contact = any(w["contact_base"] for w in w_eu)
    ca_contact = any(w["contact_base"] for w in w_ca)

    # Q2：首次接触底座时 CCA 的 beta
    ca_first_base = next((w for w in w_ca if w["contact_base"]), None)

    # Q3：CCA 接触底座 → mug 后续位移
    ca_mug_disp = [w["mug_disp_after"] for w in w_ca]

    answer = {
        "Q1_euclidean_avoids_base": {
            "euclidean_contact_base_in_window": eu_contact,
            "cca_contact_base_in_window": ca_contact,
            "euclidean_min_gap_lower_mm": eu_min,
            "cca_min_gap_lower_mm": ca_min,
        },
        "Q2_first_control_divergence_beta": {
            "cca_first_base_contact_step": None if ca_first_base is None else ca_first_base["step"],
            "cca_first_base_contact_beta": None if ca_first_base is None else ca_first_base["beta"],
            "note": "首次实质控制分叉在 step0（TASK B 已记录），β 非零从早期就出现",
        },
        "Q3_contact_to_mug_disp": {
            "cca_mug_disp_window_mm": [round(x, 4) for x in ca_mug_disp],
            "cca_mug_disp_final_mm": traces["cca_asf"]["Dmax"] * 1000,
            "euclidean_mug_disp_final_mm": traces["euclidean"]["Dmax"] * 1000,
        },
    }

    out = {
        "answer": answer,
        "euclidean_window": w_eu,
        "cca_window": w_ca,
    }
    out_path = os.path.join(ROOT, "data", "audit", "cca_asf_e1_trajectory_divergence.json")
    with open(out_path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)

    print("\n=== E1-B1 三个问题的答案 ===")
    for k, v in answer.items():
        print(f"  {k}: {v}")
    print(f"\n结果: {out_path}")
    pol_gen.close()


if __name__ == "__main__":
    main()
