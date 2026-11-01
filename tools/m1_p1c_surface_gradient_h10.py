"""M1-P1c G3：Surface-Gradient 候选的 H=10 闭环验证。

回答决定性问题的答案：
  真实碰撞表面生成的动作，能否在 Step54–55 提前改变机器人运动，使冻结 π0.5 后续
  接续时不再发生 Link6–mug 接触，同时保持任务进度？

两阶段闭环避让（真正的闭环，非 open-loop）：
  - step54：用 step50 的剩余名义动作，计算 g54，执行 G54 = Π(a_nom54 + 0.2 ĝ54)
  - step55：用分叉后新观测重规划（replan_steps=5 原始时序），计算 g55，执行 G55
  - step56–59：用 step55 的动作块接续；step60 再次重规划；step61–63 接续

分支（共享前缀 = 原始 B 0→54）：
  B（AEGIS） / N（nominal） / H（hold→nominal） / G-N / G-A /
  Z+A（参考） / L-+A（参考，v3 固定方向）

全用确定性重跑（fork 不可 bit-exact，见 v3 审计）。
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

from src.pnsg.policy_adapter import PnsgPolicy  # noqa: E402
from tools.m1_p1_design_gate_v2 import (  # noqa: E402
    step_tracked, build_body_geoms, generate_block, check_contacts,
)
from tools.m1_p1_design_gate_v3 import run_b_to_step54, disp_from_p0, d_soup  # noqa: E402
from tools.m1_p1c_surface_gradient_gate import (  # noqa: E402
    make_query_model, surface_gap, link6_mug_contact_raw,
    QUERY_MARGIN, FD_H, DELTA,
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
PROTECTED = "red_coffee_mug_obstacle_1_main"
SOUP_BODY = "alphabet_soup_1_main"
DELTA_CAR = 0.001
EPS = 0.0001

# v3 固定参考方向（冻结，不重算幅度）
Z_DIR = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
L_NEG_DIR = np.array([-0.024, -0.992, 0.099, 0.0, 0.0, 0.0, 0.0])  # v3 L- 方向


def exec_one_step(env, act):
    """执行 1 个完整控制步（25 substep），返回 obs。"""
    e = env.env
    e.timestep += 1
    policy_step = True
    for _ in range(int(e.control_timestep / e.model_timestep)):
        e.sim.forward()
        e._pre_action(act, policy_step)
        e.sim.step()
        e._update_observables()
        policy_step = False
    e.cur_time += e.control_timestep
    return e._get_observations()


def reach_step54(ts, task, prompt, pol):
    """确定性重跑到 step54（原始 B 前缀）。"""
    return run_b_to_step54(ts, task, prompt, pol)


def reach_step55(ts, task, prompt, pol, G54):
    """确定性重跑到 step54 + 执行 G54，再在 step55 重规划。返回 s55 状态。"""
    env, layer, obs, nominal, nom_a54, rg, og, p0, D0 = reach_step54(ts, task, prompt, pol)
    obs, *_ = step_tracked(env, G54, rg, og)
    block = generate_block(pol, obs, prompt, 55 // REPLAN_STEPS)
    nom_a55 = block[0].copy()
    return env, layer, obs, block, nom_a55, rg, og, p0, D0


def eval_gap_after(env, act, qmodel, qdata):
    """从 env 当前状态执行 act（1 步），返回末端表面间隙。会 mutate env。"""
    exec_one_step(env, act)
    gap, pair, _ = surface_gap(env, qmodel, qdata)
    return gap, pair


def finite_diff_gap(ts, task, prompt, pol, qmodel, qdata, reach_fn, base_act):
    """在 reach_fn() 到达的状态，对 base_act 做中心差分，返回 grad（7 维）。"""
    grad = np.zeros(7)
    denom = np.zeros(6)
    for i in range(6):
        ei = np.zeros(7); ei[i] = 1.0
        ap = np.clip(base_act + FD_H * ei, -1, 1)
        am = np.clip(base_act - FD_H * ei, -1, 1)
        actual_h = ap[i] - am[i]
        if actual_h < 1e-6:
            denom[i] = 0.0
            continue
        env, *_ = reach_fn()
        fp, _ = eval_gap_after(env, ap, qmodel, qdata)
        env.close()
        env, *_ = reach_fn()
        fm, _ = eval_gap_after(env, am, qmodel, qdata)
        env.close()
        fp = QUERY_MARGIN if fp is None else fp
        fm = QUERY_MARGIN if fm is None else fm
        grad[i] = (fp - fm) / actual_h
        denom[i] = actual_h
    return grad, denom


def run_branch(ts, task, prompt, pol, qmodel, qdata, branch, G54=None, G55=None):
    """共享前缀 0→54，然后按 branch 执行 step54–63。返回完整指标。"""
    t0 = time.time()
    env, layer, obs, nominal, nom_a54, rg, og, p0, D0 = reach_step54(ts, task, prompt, pol)

    first_contact = None
    first_l6mug = None
    contact_steps = 0
    l6mug_steps = 0
    Dmax = D0
    d_soup_init = d_soup(env)
    d_soup_hist = [d_soup_init]
    min_gap = None
    min_gap_step = None
    min_gap_pair = None
    n_generate = 0
    n_filter = 0

    for t in range(H):
        step = 54 + t
        if step % REPLAN_STEPS == 0:
            nominal = generate_block(pol, obs, prompt, step // REPLAN_STEPS)
            n_generate += 1
        nom_a = nominal[step % REPLAN_STEPS]

        # 选择动作
        if branch == "B":
            act = layer.filter(nom_a, obs)[0]; n_filter += 1
        elif branch == "N":
            act = nom_a
        elif branch == "H":
            act = np.zeros(7) if t < 2 else nom_a
        elif branch == "G-N":
            act = G54 if t == 0 else (G55 if t == 1 else nom_a)
        elif branch == "G-A":
            if t < 2:
                act = G54 if t == 0 else G55
            else:
                act = layer.filter(nom_a, obs)[0]; n_filter += 1
        elif branch == "Z+A":
            act = np.clip(nom_a + 0.4 * Z_DIR, -1, 1) if t < 2 else layer.filter(nom_a, obs)[0]
            if t >= 2:
                n_filter += 1
        elif branch == "L-+A":
            act = np.clip(nom_a + 0.4 * L_NEG_DIR, -1, 1) if t < 2 else layer.filter(nom_a, obs)[0]
            if t >= 2:
                n_filter += 1
        else:
            raise ValueError(branch)

        obs, _r, _d, _i, c_any, c_l6, _ = step_tracked(env, act, rg, og)
        if c_any:
            contact_steps += 1
            if first_contact is None:
                first_contact = step
        if c_l6:
            l6mug_steps += 1
            if first_l6mug is None:
                first_l6mug = step
        Dmax = max(Dmax, disp_from_p0(env, p0))
        d_soup_hist.append(d_soup(env))
        g, pr, _ = surface_gap(env, qmodel, qdata)
        if g is not None and (min_gap is None or g < min_gap):
            min_gap = g
            min_gap_step = step
            min_gap_pair = pr

    eef_pose_final = None
    try:
        eef_pose_final = np.asarray(env.sim.data._data.site_xpos[
            env.robots[0].eef_site_id]).round(4).tolist()
    except Exception:
        pass

    safe = (contact_steps == 0) and (Dmax < DELTA_CAR - EPS)
    metrics = {
        "branch": branch, "safe": safe,
        "first_contact": first_contact, "first_l6mug": first_l6mug,
        "contact_steps": contact_steps, "l6mug_steps": l6mug_steps,
        "Dmax": Dmax,
        "d_soup_init": d_soup_init, "d_soup_final": d_soup_hist[-1],
        "delta_d_soup_total": d_soup_hist[-1] - d_soup_init,
        "d_soup_after_avoid": d_soup_hist[2],
        "min_gap": min_gap, "min_gap_step": min_gap_step, "min_gap_pair": min_gap_pair,
        "eef_pose_final": eef_pose_final,
        "n_generate": n_generate, "n_filter": n_filter, "wall_s": time.time() - t0,
    }
    env.close()
    return metrics


def main():
    from libero.libero import benchmark
    ts = benchmark.get_benchmark_dict()[SUITE](safety_level=LEVEL)
    task = ts.get_task(TASK_ID)
    prompt = task.language
    pol = PnsgPolicy()

    print("=== G3：Surface-Gradient H=10 闭环验证 ===", flush=True)

    # 构造查询模型 + 获取 step54 名义动作
    env, layer, obs, nominal, nom_a54, rg, og, p0, D0 = reach_step54(ts, task, prompt, pol)
    qmodel, qdata = make_query_model(env)
    env.close()

    # ---- g54（step54 的梯度）----
    print("[g54] 有限差分（step54 状态，12 次重跑）...", flush=True)
    t_g = time.time()
    reach54 = lambda: reach_step54(ts, task, prompt, pol)
    g54, denom54 = finite_diff_gap(ts, task, prompt, pol, qmodel, qdata, reach54, nom_a54)
    gn54 = np.linalg.norm(g54[:6])
    if gn54 < 1e-8:
        print("  GRADIENT_UNRELIABLE @ step54：梯度退化")
        pol.close(); return
    ghat54 = g54 / gn54
    G54 = np.clip(nom_a54 + DELTA * ghat54, -1, 1)
    print(f"  ||g54||={gn54:.4f}  ĝ54[:3]={np.round(ghat54[:3],4)}  "
          f"G54 实际幅度={np.linalg.norm(G54-nom_a54):.4f}", flush=True)

    # ---- g55（step55 的梯度，需先执行 G54）----
    print("[g55] 有限差分（step55 状态，需先执行 G54，12 次重跑）...", flush=True)
    # 先取 step55 的名义动作
    env2, layer2, obs2, block2, nom_a55, rg2, og2, p02, D02 = reach_step55(ts, task, prompt, pol, G54)
    env2.close()
    reach55 = lambda: reach_step55(ts, task, prompt, pol, G54)
    g55, denom55 = finite_diff_gap(ts, task, prompt, pol, qmodel, qdata, reach55, nom_a55)
    gn55 = np.linalg.norm(g55[:6])
    g55_ok = True
    if gn55 < 1e-8:
        print("  GRADIENT_UNRELIABLE @ step55：梯度退化（不调参，记录）", flush=True)
        g55_ok = False
        G55 = nom_a55  # 记录为退化，不作为有效避让
    else:
        ghat55 = g55 / gn55
        G55 = np.clip(nom_a55 + DELTA * ghat55, -1, 1)
        print(f"  ||g55||={gn55:.4f}  ĝ55[:3]={np.round(ghat55[:3],4)}  "
              f"G55 实际幅度={np.linalg.norm(G55-nom_a55):.4f}", flush=True)
    print(f"  [梯度耗时 {time.time()-t_g:.1f}s]", flush=True)

    # ---- 运行全分支 ----
    print("[分支] 运行 B/N/H/G-N/G-A/Z+A/L-+A ...", flush=True)
    branches = {}
    branches["B"] = run_branch(ts, task, prompt, pol, qmodel, qdata, "B")
    branches["N"] = run_branch(ts, task, prompt, pol, qmodel, qdata, "N")
    branches["H"] = run_branch(ts, task, prompt, pol, qmodel, qdata, "H")
    branches["G-N"] = run_branch(ts, task, prompt, pol, qmodel, qdata, "G-N", G54, G55)
    branches["G-A"] = run_branch(ts, task, prompt, pol, qmodel, qdata, "G-A", G54, G55)
    branches["Z+A"] = run_branch(ts, task, prompt, pol, qmodel, qdata, "Z+A")
    branches["L-+A"] = run_branch(ts, task, prompt, pol, qmodel, qdata, "L-+A")

    # ---- 输出 ----
    print("\n=== G3 结果（H=10 闭环）===")
    hdr = (f"{'分支':6s} {'安全':4s} {'首接触':>6s} {'首l6mug':>7s} {'Dmax(mm)':>8s} "
           f"{'d_soup54':>9s} {'d_soup64':>9s} {'Δd_soup':>8s} {'min_gap':>8s}")
    print(hdr)
    for k in ["B", "N", "H", "G-N", "G-A", "Z+A", "L-+A"]:
        m = branches[k]
        tag = "✓" if m["safe"] else ("✗接" if m["contact_steps"] > 0 else "✗CAR")
        fc = m["first_contact"] if m["first_contact"] is not None else "-"
        fl = m["first_l6mug"] if m["first_l6mug"] is not None else "-"
        mg = "—" if m["min_gap"] is None else f"{m['min_gap']*1000:.1f}"
        print(f"{k:6s} {tag:4s} {str(fc):>6s} {str(fl):>7s} {m['Dmax']*1000:8.3f} "
              f"{m['d_soup_init']*1000:9.1f} {m['d_soup_final']*1000:9.1f} "
              f"{m['delta_d_soup_total']*1000:8.1f} {mg:>8s}")

    print(f"\n  G-N 与 H 的 d_soup64 对比: "
          f"G-N={branches['G-N']['d_soup_final']*1000:.1f}mm  "
          f"H={branches['H']['d_soup_final']*1000:.1f}mm")
    print(f"  G-N 与 N/B 的首接触: G-N={branches['G-N']['first_contact']}  "
          f"N={branches['N']['first_contact']}  B={branches['B']['first_contact']}")
    print(f"  G-A 与 B 的首接触: G-A={branches['G-A']['first_contact']}  "
          f"B={branches['B']['first_contact']}")

    # 判决
    print("\n=== G3 判决 ===")
    gn_safe = branches["G-N"]["safe"]
    ga_safe = branches["G-A"]["safe"]
    gn_task = branches["G-N"]["d_soup_final"] < branches["H"]["d_soup_final"]
    ga_task = branches["G-A"]["d_soup_final"] < branches["H"]["d_soup_final"]
    print(f"  G-N: 安全={gn_safe}  任务优于Hold={gn_task}")
    print(f"  G-A: 安全={ga_safe}  任务优于Hold={ga_task}")

    if (gn_safe or ga_safe) and (gn_task or ga_task):
        print("  → 强正结果：安全且任务进度优于 Hold，进入完整闭环原型")
    elif (gn_safe or ga_safe) and not (gn_task or ga_task):
        print("  → 有效安全避让，但任务增益尚未证明")
    elif branches["G-N"]["contact_steps"] > 0 or branches["G-A"]["contact_steps"] > 0:
        print("  → G2 间隙改善，但 G3 后续接触：局部方向有效，预测安全性不足")
    if not g55_ok:
        print("  → 注意：step55 梯度退化（GRADIENT_UNRELIABLE），G55 未构成有效避让")

    out = pathlib.Path(ROOT, "data", "audit", "m1_p1c_g3_h10.json")
    out.write_text(json.dumps({
        "g54": {"grad": g54.tolist(), "norm": gn54, "denom": denom54.tolist(),
                "G54_delta": float(np.linalg.norm(G54 - nom_a54))},
        "g55": {"ok": g55_ok, "grad": g55.tolist(), "norm": gn55, "denom": denom55.tolist(),
                "G55_delta": float(np.linalg.norm(G55 - nom_a55))},
        "branches": {k: {kk: (float(vv) if isinstance(vv, (np.floating, np.ndarray)) else vv)
                         for kk, vv in m.items()} for k, m in branches.items()},
    }, ensure_ascii=False, indent=2, default=str))
    print(f"\n结果: {out}")
    pol.close()


if __name__ == "__main__":
    main()
