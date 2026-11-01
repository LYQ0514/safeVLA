"""M1-P1 Design Gate v3：Step54 短时域几何避让（重跑版，无 fork）。

v2 的 fork（mj_setState 恢复）因 qfrc_constraint（导数量，3.6e-4）不回滚，
导致擦边接触 off-by-one。本版改用「确定性重跑」：每个候选从零重跑到 step54
（同进程内轨迹 bit-exact，已验证），再应用避让 + 接续，彻底避开状态恢复误差。

- Gate 1：重跑 B（AEGIS）轨迹，验证首次 link6-mug 接触 @ step59。
- Gate 2：有限差分 ∇_a d_lm（重跑 + 全状态恢复单步验证方向）。
- Design Gate：14 条候选（N/A + H/S1/S2/L+/L-/Z × N/A 接续），H=10。
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
import pi05_libero_runner as R  # noqa: E402

from src.pnsg.crn import nominal_seed, sample_z  # noqa: E402
from src.pnsg.policy_adapter import PnsgPolicy  # noqa: E402
from src.safety.aegis_adapted import AegisSafetyLayer  # noqa: E402
from tools.g1a_soup_trace import make_env, body_state, eef_pos  # noqa: E402
from tools.m1_p1_design_gate_v2 import (  # noqa: E402
    step_tracked, build_body_geoms, generate_block, d_lm, check_contacts,
)

SUITE = "safelibero_long"
LEVEL = "I"
TASK_ID = 0
STATE_ID = 2
STREAM = 0
REPLAN_STEPS = 5
IMAGE_SIZE = 224
SWITCH = 40
H = 10
DELTA_CAR = 0.001
EPS = 0.0001
PROTECTED = "red_coffee_mug_obstacle_1_main"
SOUP_BODY = "alphabet_soup_1_main"
FD_H = 0.05
SPEC = mujoco.mjtState.mjSTATE_INTEGRATION


def disp_from_p0(env, p0):
    return float(np.sum(np.abs(body_state(env, PROTECTED)["pos"] - p0)))


def d_soup(env):
    return float(np.linalg.norm(body_state(env, SOUP_BODY)["pos"] - eef_pos(env)))


# ---- 全状态保存/恢复（仅 Gate2 梯度用，候选用重跑） ----
def full_state(env):
    m = env.sim.model._model; d = env.sim.data._data
    size = mujoco.mj_stateSize(m, SPEC)
    buf = np.zeros(size, dtype=np.float64)
    mujoco.mj_getState(m, d, buf, SPEC)
    return buf


def set_full_state(env, buf):
    m = env.sim.model._model; d = env.sim.data._data
    mujoco.mj_setState(m, d, buf, SPEC)
    mujoco.mj_forward(m, d)


def ctrl_state(env):
    c = env.robots[0].controller
    return {"goal_pos": c.goal_pos.copy(), "goal_ori": c.goal_ori.copy(),
            "relative_ori": c.relative_ori.copy() if c.relative_ori is not None else None,
            "new_update": c.new_update}


def set_ctrl_state(env, st):
    c = env.robots[0].controller
    c.goal_pos[:] = st["goal_pos"]; c.goal_ori[:] = st["goal_ori"]
    if st["relative_ori"] is not None:
        c.relative_ori[:] = st["relative_ori"]
    c.new_update = st["new_update"]


def run_b_to_step54(ts, task, prompt, pol):
    """Fresh env，重跑 B 轨迹到 step54。返回 env/layer/obs/nominal/nom_a54/rg/og/p0/D0。"""
    env = make_env(ts, task, STATE_ID)
    layer = AegisSafetyLayer(env)
    obs = env.env._get_observations()
    layer.reset(obs, prompt)
    rg, og = build_body_geoms(env)
    p0 = body_state(env, PROTECTED)["pos"].copy()
    Dmax = 0.0
    nominal = None
    for step in range(54):
        if step == SWITCH:
            layer.reset(obs, prompt)
        if step % REPLAN_STEPS == 0:
            nominal = generate_block(pol, obs, prompt, step // REPLAN_STEPS)
        nom_a = nominal[step % REPLAN_STEPS]
        act = nom_a if step < SWITCH else layer.filter(nom_a, obs)[0]
        obs, _r, _d, _i, _ca, _cl, _ = step_tracked(env, act, rg, og)
        Dmax = max(Dmax, disp_from_p0(env, p0))
    nom_a54 = nominal[54 % REPLAN_STEPS].copy()
    return env, layer, obs, nominal, nom_a54, rg, og, p0, Dmax


def continue_candidate(env, layer, pol, prompt, nominal, avoid_acts, mode,
                       rg, og, p0, D0):
    """从 step54 状态继续 H=10。avoid_acts=None 为纯 N/A 基线；否则长度 2 避让动作。"""
    t0 = time.time()
    obs = env.env._get_observations()
    d0_soup = d_soup(env)
    d_lm0 = d_lm(env)

    first_contact = None
    first_l6mug = None
    contact_any_steps = 0
    contact_l6mug_steps = 0
    Dmax = D0
    d_soup_hist = [d0_soup]
    d_lm_hist = [d_lm0]
    n_generate = 0
    n_filter = 0

    for t in range(H):
        step = 54 + t
        if step % REPLAN_STEPS == 0:
            nominal = generate_block(pol, obs, prompt, step // REPLAN_STEPS)
            n_generate += 1
        nom_a = nominal[step % REPLAN_STEPS]
        if avoid_acts is not None and t < 2:
            act = avoid_acts[t]
        elif mode == "N":
            act = nom_a
        else:
            act, _ = layer.filter(nom_a, obs)
            n_filter += 1
        obs, _r, _d, _i, c_any, c_l6, _ = step_tracked(env, act, rg, og)
        if c_any:
            contact_any_steps += 1
            if first_contact is None:
                first_contact = step
        if c_l6:
            contact_l6mug_steps += 1
            if first_l6mug is None:
                first_l6mug = step
        Dmax = max(Dmax, disp_from_p0(env, p0))
        d_soup_hist.append(d_soup(env))
        d_lm_hist.append(d_lm(env))

    safe = (contact_any_steps == 0) and (Dmax < DELTA_CAR - EPS)
    return {
        "safe": safe, "Dmax": Dmax,
        "contact_any_steps": contact_any_steps, "contact_l6mug_steps": contact_l6mug_steps,
        "first_contact_step": first_contact, "first_l6mug_step": first_l6mug,
        "d_soup_init": d_soup_hist[0], "d_soup_after_avoid": d_soup_hist[2],
        "d_soup_final": d_soup_hist[-1],
        "delta_d_soup_total": d_soup_hist[-1] - d_soup_hist[0],
        "delta_d_soup_cont": d_soup_hist[-1] - d_soup_hist[2],
        "d_lm_init": d_lm_hist[0], "d_lm_after_avoid": d_lm_hist[2],
        "delta_d_lm_avoid": d_lm_hist[2] - d_lm_hist[0],
        "n_generate": n_generate, "n_filter": n_filter, "wall_s": time.time() - t0,
    }


def main():
    from libero.libero import benchmark
    ts = benchmark.get_benchmark_dict()[SUITE](safety_level=LEVEL)
    task = ts.get_task(TASK_ID)
    prompt = task.language
    pol = PnsgPolicy()

    print("=== M1-P1 Design Gate v3（step54，H=10，重跑版）===", flush=True)

    # ---------- Gate 1：重跑 B（AEGIS）到 step63，验证首次接触 ----------
    print("[Gate1] 重跑 B（AEGIS）轨迹，定位首次 link6-mug 接触...", flush=True)
    env, layer, obs, nominal, nom_a54, rg, og, p0, D0 = run_b_to_step54(ts, task, prompt, pol)
    g1 = continue_candidate(env, layer, pol, prompt, nominal, None, "A", rg, og, p0, D0)
    env.close()
    print(f"  [Gate1] 首次 link6-mug 接触 @ step {g1['first_l6mug_step']}  "
          f"Dmax={g1['Dmax']*1000:.3f}mm  contact_any={g1['contact_any_steps']}", flush=True)
    g1_ok = (g1["first_l6mug_step"] == 59)

    # ---------- Gate 2：有限差分梯度（重跑到 step54，全状态恢复单步验证） ----------
    print("[Gate2] 计算 ∇_a d_lm 并单步验证方向...", flush=True)
    env, layer, obs, nominal, nom_a54, rg, og, p0, D0 = run_b_to_step54(ts, task, prompt, pol)
    mj_buf = full_state(env)
    ctrl = ctrl_state(env)
    aegis = layer.save_state()

    def eval_perturbed(act):
        set_full_state(env, mj_buf); set_ctrl_state(env, ctrl); layer.restore_state(aegis)
        env.env._get_observations(force_update=True)
        _o, _r, _d, _i, _ca, _cl, _ = step_tracked(env, act, rg, og)
        return d_lm(env)

    d0 = d_lm(env)
    grad = np.zeros(7)
    for i in range(6):
        ap = nom_a54.copy(); ap[i] += FD_H
        am = nom_a54.copy(); am[i] -= FD_H
        grad[i] = (eval_perturbed(ap) - eval_perturbed(am)) / (2 * FD_H)
    gn = np.linalg.norm(grad[:6])
    ghat = grad / max(gn, 1e-8)
    # 正交方向（动作空间，前 6 维，确定性规则）
    g6 = ghat[:6]
    basis = [np.eye(6)[i] for i in range(6)]
    idx = min(range(6), key=lambda i: abs(np.dot(basis[i], g6)))
    v = basis[idx] - np.dot(basis[idx], g6) * g6
    lhat7 = None
    if np.linalg.norm(v) > 1e-8:
        lhat7 = np.concatenate([v / np.linalg.norm(v), [0.0]])

    print(f"  [Gate2] ||g||={gn:.4f}  ĝ[:3]={np.round(ghat[:3],3)}  "
          f"l̂[:3]={np.round(lhat7[:3],3) if lhat7 is not None else None}", flush=True)

    # 单步验证 S1（+ĝ）是否增加 d_lm
    S1 = np.clip(nom_a54 + 0.2 * ghat, -1, 1)
    dS = eval_perturbed(S1) - d0
    print(f"  [Gate2] S1(+ĝ,δ=0.2): Δd_lm={dS*1000:.2f}mm  (应 >0)", flush=True)
    gate2_s_ok = dS > 0

    # L+ 验证
    if lhat7 is not None:
        Lp = np.clip(nom_a54 + 0.4 * lhat7, -1, 1)
        dL = eval_perturbed(Lp) - d0
        print(f"  [Gate2] L+(+l̂,δ=0.4): Δd_lm={dL*1000:.2f}mm", flush=True)

    # 准备候选（基于本次重跑的 step54 名义动作；确定性保证与 _rerun_one 中一致）
    avoid_candidates = {
        "H": [np.zeros(7), np.zeros(7)],
        "S1": [np.clip(nom_a54 + 0.2 * ghat, -1, 1)] * 2,
        "S2": [np.clip(nom_a54 + 0.4 * ghat, -1, 1)] * 2,
        "Z": [np.clip(nom_a54 + 0.4 * np.array([0, 0, 1, 0, 0, 0, 0]), -1, 1)] * 2,
    }
    if lhat7 is not None:
        avoid_candidates["L+"] = [np.clip(nom_a54 + 0.4 * lhat7, -1, 1)] * 2
        avoid_candidates["L-"] = [np.clip(nom_a54 - 0.4 * lhat7, -1, 1)] * 2
    env.close()

    # ---------- Design Gate：候选 × 接续（每次重跑） ----------
    print("[DesignGate] 候选 × {N,A 接续}，H=10（每候选重跑到 step54）...", flush=True)
    results = {}
    # 基线 N / A（A 复用 Gate1）
    results["N"] = _rerun_one(ts, task, prompt, pol, None, "N")
    results["A"] = g1
    for cname, avoids in avoid_candidates.items():
        for mode in ("N", "A"):
            results[f"{cname}+{mode}"] = _rerun_one(ts, task, prompt, pol, avoids, mode)

    # ---------- 输出 ----------
    print("\n=== Design Gate v3 结果（H=10）===")
    print(f"{'候选':10s} {'安全':4s} {'Dmax(mm)':>8s} {'接触步':>6s} "
          f"{'l6mug首步':>8s} {'Δd_soup总':>9s} {'Δd_soup接续':>10s} {'Δd_lm避让':>9s}")
    for k, m in results.items():
        tag = "✓" if m["safe"] else ("✗接" if m["contact_any_steps"] > 0 else "✗CAR")
        fc = m["first_l6mug_step"] if m["first_l6mug_step"] is not None else "-"
        print(f"{k:10s} {tag:4s} {m['Dmax']*1000:8.3f} {m['contact_any_steps']:6d} "
              f"{str(fc):>8s} {m['delta_d_soup_total']*1000:9.1f} "
              f"{m['delta_d_soup_cont']*1000:10.1f} {m['delta_d_lm_avoid']*1000:9.1f}")

    geom = {k: m for k, m in results.items() if m.get("avoid") == "2step"}
    safe_progress = [k for k, m in geom.items() if m["safe"] and m["delta_d_soup_total"] < 0]
    safe_noprogress = [k for k, m in geom.items() if m["safe"] and m["delta_d_soup_total"] >= 0]
    all_contact = (len(geom) > 0) and all(m["contact_any_steps"] > 0 for m in geom.values())
    only_hold = (results.get("H+N", {}).get("safe") and results.get("H+A", {}).get("safe")
                 and all(not m["safe"] for k, m in geom.items() if k not in ("H+N", "H+A")))

    print("\n=== 判决 ===")
    print(f"  门禁: Gate1(首次接触@59)={g1_ok}  Gate2(S方向增加分离)={gate2_s_ok}")
    print(f"  安全且任务进度为正的避让序列: {safe_progress if safe_progress else '无'}")
    print(f"  安全但 H=10 内进度未恢复: {safe_noprogress if safe_noprogress else '无'}")
    print(f"  所有避让候选仍接触: {all_contact}")
    print(f"  只有 Hold 安全: {only_hold}")

    total_wall = sum(m["wall_s"] for m in results.values())
    total_gen = sum(m["n_generate"] for m in results.values())
    print(f"\n=== 计算开销 ===")
    print(f"  总候选={len(results)}  总π0.5推理={total_gen}  总耗时={total_wall:.1f}s")

    out = pathlib.Path(ROOT, "data", "audit", "m1_p1_design_gate_v3.json")
    out.write_text(json.dumps({
        "gate1": {"ok": g1_ok, **g1},
        "gate2": {"ok": gate2_s_ok, "grad_norm": gn, "ghat": ghat.tolist(),
                  "lhat7": None if lhat7 is None else lhat7.tolist(), "S_delta_dlm": dS},
        "results": {k: {kk: (float(vv) if isinstance(vv, (np.floating, np.ndarray)) else vv)
                        for kk, vv in m.items()} for k, m in results.items()},
    }, ensure_ascii=False, indent=2, default=str))
    print(f"\n结果: {out}")

    pol.close()


def _rerun_one(ts, task, prompt, pol, avoid_acts, mode):
    env, layer, obs, nominal, nom_a54, rg, og, p0, D0 = run_b_to_step54(ts, task, prompt, pol)
    m = continue_candidate(env, layer, pol, prompt, nominal, avoid_acts, mode, rg, og, p0, D0)
    m["avoid"] = "none" if avoid_acts is None else "2step"
    env.close()
    return m


if __name__ == "__main__":
    main()
