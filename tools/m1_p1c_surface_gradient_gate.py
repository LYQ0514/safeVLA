"""M1-P1c G2：Surface-Gap 梯度方向响应验证。

用 MuJoCo 原生碰撞几何间隙（凸 mesh ↔ box，经独立查询模型 + geom_margin）构造
动作空间有限差分梯度，验收 f(a_new) > f(a_nom)，并检查最小 substep 间隙/原始接触/历史 CAR。

工程约束（严格遵循）：
1. 独立查询模型（deepcopy），margin 只设在查询模型，真实 model 的 margin 恒为 0。
2. d_surf 覆盖 mug 全部 5 个碰撞盒（g202-g206），记录最近 geom ID。
3. 差分用实际执行值（clip 后）作分母，不用名义 ±h。
4. f(a) 用「执行完整 25 substep 后」的积分后 qpos 评估，口径一致。
5. 确定性重跑（fork 不可 bit-exact），每个差分点从零重跑到 step54。
"""

from __future__ import annotations

import copy
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
from tools.m1_p1_design_gate_v2 import step_tracked, build_body_geoms, generate_block  # noqa: E402
from tools.m1_p1_design_gate_v3 import run_b_to_step54, disp_from_p0, d_soup  # noqa: E402

SUITE = "safelibero_long"
LEVEL = "I"
TASK_ID = 0
STATE_ID = 2
STREAM = 0
REPLAN_STEPS = 5
IMAGE_SIZE = 224
SWITCH = 40
PROTECTED = "red_coffee_mug_obstacle_1_main"

LINK6_GEOM = 64
MUG_GEOMS = [202, 203, 204, 205, 206]
QUERY_MARGIN = 0.3      # 独立查询模型 margin（仅查询，不参与动力学）
FD_H = 0.05             # 有限差分步长（预注册，不调）
DELTA = 0.2             # 候选修正幅度（预注册，S1 档，不调）
DELTA_CAR = 0.001
EPS = 0.0001


def make_query_model(env):
    """deepcopy 真实 model，设 link6+mug 的 margin，返回独立 (qmodel, qdata)。"""
    m = env.sim.model._model
    qmodel = copy.deepcopy(m)
    for g in [LINK6_GEOM] + MUG_GEOMS:
        qmodel.geom_margin[g] = QUERY_MARGIN
    qdata = mujoco.MjData(qmodel)
    return qmodel, qdata


def surface_gap(env, qmodel, qdata):
    """在当前 env 状态测 link6-mug 最小表面间隙。

    返回 (gap, pair, contacts_list)。gap=None 表示 OUT_OF_QUERY_RANGE（> margin）。
    """
    d = env.sim.data._data
    qdata.qpos[:] = d.qpos
    qdata.qvel[:] = d.qvel
    mujoco.mj_forward(qmodel, qdata)
    mujoco.mj_collision(qmodel, qdata)
    best = None
    best_pair = None
    pairs = []
    for c in range(qdata.ncon):
        g1, g2 = qdata.contact[c].geom1, qdata.contact[c].geom2
        if (g1 == LINK6_GEOM and g2 in MUG_GEOMS) or (g2 == LINK6_GEOM and g1 in MUG_GEOMS):
            dd = float(qdata.contact[c].dist)
            pairs.append((g1, g2, dd))
            if best is None or dd < best:
                best = dd
                best_pair = (g1, g2)
    return best, best_pair, pairs


def link6_mug_contact_raw(env, rg, og):
    """原始 MuJoCo 接触记录（真实动力学 margin=0），是否 link6-mug 接触。"""
    m = env.sim.model._model
    dd = env.sim.data._data
    for c in range(dd.ncon):
        g1, g2 = dd.contact[c].geom1, dd.contact[c].geom2
        if (g1 in rg and g2 in og) or (g2 in rg and g1 in og):
            b1 = env.sim.model.body_names[m.geom_bodyid[g1]]
            b2 = env.sim.model.body_names[m.geom_bodyid[g2]]
            if "link6" in b1 or "link6" in b2:
                return True
    return False


def evaluate_one(ts, task, prompt, pol, qmodel, qdata, act):
    """重跑到 step54，执行 act（1 完整控制步），返回完整指标。"""
    env, layer, obs, nominal, nom_a54, rg, og, p0, D0 = run_b_to_step54(ts, task, prompt, pol)
    # 执行 act：手工 substep 循环，逐 substep 测表面间隙
    e = env.env
    e.timestep += 1
    policy_step = True
    min_gap = None
    min_gap_substep = None
    min_gap_pair = None
    contact_raw = False
    n_sub = int(e.control_timestep / e.model_timestep)
    for i in range(n_sub):
        e.sim.forward()
        e._pre_action(act, policy_step)
        e.sim.step()
        if link6_mug_contact_raw(env, rg, og):
            contact_raw = True
        g, p, _ = surface_gap(env, qmodel, qdata)
        if g is not None and (min_gap is None or g < min_gap):
            min_gap = g
            min_gap_substep = i
            min_gap_pair = p
        e._update_observables()
        policy_step = False
    e.cur_time += e.control_timestep
    # 末端间隙 = 25 substep 后的 surface_gap
    end_gap, end_pair, _ = surface_gap(env, qmodel, qdata)
    Dmax = disp_from_p0(env, p0)
    env.close()
    return {
        "end_gap": end_gap, "end_pair": end_pair,
        "min_gap": min_gap, "min_gap_substep": min_gap_substep, "min_gap_pair": min_gap_pair,
        "contact_raw": contact_raw, "Dmax": Dmax,
    }


def main():
    from libero.libero import benchmark
    ts = benchmark.get_benchmark_dict()[SUITE](safety_level=LEVEL)
    task = ts.get_task(TASK_ID)
    prompt = task.language
    pol = PnsgPolicy()

    # 构造查询模型 + 验证 deepcopy collision 与真实 model 一致
    env, layer, obs, nominal, nom_a54, rg, og, p0, D0 = run_b_to_step54(ts, task, prompt, pol)
    qmodel, qdata = make_query_model(env)
    # deepcopy 一致性：真实 model（margin=0）与 qmodel（margin=0）在同一 qpos 下一致
    m = env.sim.model._model
    d = env.sim.data._data
    qmodel.geom_margin[LINK6_GEOM] = 0.0
    for g in MUG_GEOMS:
        qmodel.geom_margin[g] = 0.0
    qdata.qpos[:] = d.qpos; qdata.qvel[:] = d.qvel
    mujoco.mj_forward(qmodel, qdata); mujoco.mj_collision(qmodel, qdata)
    mujoco.mj_collision(m, d)
    # 对比 link6-mug 对
    def collect(model, data):
        out = []
        for c in range(data.ncon):
            g1, g2 = data.contact[c].geom1, data.contact[c].geom2
            if (g1 == LINK6_GEOM and g2 in MUG_GEOMS) or (g2 == LINK6_GEOM and g1 in MUG_GEOMS):
                out.append((g1, g2, round(float(data.contact[c].dist) * 1000, 4)))
        return out
    real_contacts = collect(m, d)
    query_contacts = collect(qmodel, qdata)
    # 恢复 qmodel margin 到查询值
    for g in [LINK6_GEOM] + MUG_GEOMS:
        qmodel.geom_margin[g] = QUERY_MARGIN
    print("=== G2：Surface-Gap 梯度方向响应 ===")
    print(f"[deepcopy 一致性] step54 无接触（预期空）: real={real_contacts} query={query_contacts}")
    env.close()

    # 名义动作末端间隙
    t0 = time.time()
    nom_res = evaluate_one(ts, task, prompt, pol, qmodel, qdata, nom_a54)
    f_nom = nom_res["end_gap"]
    print(f"\n[名义] f(a_nom) 末端间隙 = {_fmt(f_nom)}  最小substep间隙 = {_fmt(nom_res['min_gap'])} "
          f"@substep{nom_res['min_gap_substep']}  接触={nom_res['contact_raw']}  Dmax={nom_res['Dmax']*1000:.3f}mm")

    # 有限差分梯度（6 个 pose 维，gripper 维不差分）
    grad = np.zeros(7)
    fd_denom = np.zeros(6)
    clipped_dim = []
    for i in range(6):
        ap = np.clip(nom_a54 + FD_H * np.eye(7)[i], -1, 1)
        am = np.clip(nom_a54 - FD_H * np.eye(7)[i], -1, 1)
        actual_h = ap[i] - am[i]
        if actual_h < 1e-6:
            fd_denom[i] = 0.0
            clipped_dim.append(i)
            continue
        fp = evaluate_one(ts, task, prompt, pol, qmodel, qdata, ap)["end_gap"]
        fm = evaluate_one(ts, task, prompt, pol, qmodel, qdata, am)["end_gap"]
        if fp is None or fm is None:
            # OUT_OF_QUERY_RANGE：视为 +margin（保守下界）
            fp = QUERY_MARGIN if fp is None else fp
            fm = QUERY_MARGIN if fm is None else fm
        grad[i] = (fp - fm) / actual_h
        fd_denom[i] = actual_h

    gn = np.linalg.norm(grad[:6])
    print(f"\n[梯度] ||g||={gn:.4f}  g[:3]={np.round(grad[:3],4)}")
    print(f"[梯度] 差分分母(实际)={np.round(fd_denom,4)}  裁剪维={clipped_dim if clipped_dim else '无'}")

    if gn < 1e-8:
        print("GRADIENT_UNRELIABLE：梯度退化")
        pol.close()
        return

    ghat = grad / gn
    a_new = np.clip(nom_a54 + DELTA * ghat, -1, 1)
    actual_delta = np.linalg.norm(a_new - nom_a54)
    new_res = evaluate_one(ts, task, prompt, pol, qmodel, qdata, a_new)
    f_new = new_res["end_gap"]

    print(f"\n[候选] a_new = a_nom + {DELTA}*ĝ")
    print(f"  实际执行幅度 ||a_new - a_nom|| = {actual_delta:.4f}（名义 0.2，clip 后实际）")
    print(f"  f(a_nom) = {_fmt(f_nom)}  f(a_new) = {_fmt(f_new)}  Δf = {_fmt((f_new if f_new is not None else QUERY_MARGIN) - (f_nom if f_nom is not None else QUERY_MARGIN))}")
    print(f"  最小substep间隙: 名义={_fmt(nom_res['min_gap'])}@substep{nom_res['min_gap_substep']}  "
          f"新={_fmt(new_res['min_gap'])}@substep{new_res['min_gap_substep']}")
    print(f"  最近geom对: 名义={nom_res['end_pair']} 新={new_res['end_pair']} "
          f"（切换={nom_res['end_pair'] != new_res['end_pair']}）")
    print(f"  原始接触: 名义={nom_res['contact_raw']} 新={new_res['contact_raw']}")
    print(f"  mug历史Dmax: 名义={nom_res['Dmax']*1000:.3f}mm 新={new_res['Dmax']*1000:.3f}mm")

    # 判决
    f_nom_v = QUERY_MARGIN if f_nom is None else f_nom
    f_new_v = QUERY_MARGIN if f_new is None else f_new
    df = f_new_v - f_nom_v
    print(f"\n=== G2 判决 ===")
    if df > 1e-6 and not new_res["contact_raw"] and new_res["Dmax"] < DELTA_CAR - EPS:
        print(f"  G2 PASS：Δf>0、无接触、CAR 合格 → 进入 G3")
    elif df > 1e-6 and new_res["contact_raw"]:
        print(f"  G2 部分：Δf>0 但中间发生接触（方向局部有效，安全候选失败）")
    elif df <= 1e-6:
        print(f"  G2 FAIL：Δf<=0（方向未改善表面间隙）")
    if nom_res["end_pair"] != new_res["end_pair"]:
        print(f"  提示：最近 geom 对切换（非光滑），需报告")

    print(f"\n  总耗时 {time.time()-t0:.1f}s")

    out = pathlib.Path(ROOT, "data", "audit", "m1_p1c_g2_surface_gradient.json")
    out.write_text(json.dumps({
        "f_nom": f_nom, "f_new": f_new, "df": df,
        "grad": grad.tolist(), "grad_norm": gn, "fd_denom": fd_denom.tolist(),
        "clipped_dim": clipped_dim, "a_new_delta_actual": actual_delta,
        "nom": nom_res, "new": new_res,
    }, ensure_ascii=False, indent=2, default=str))
    print(f"\n结果: {out}")
    pol.close()


def _fmt(v):
    return "OUT_OF_RANGE(>300mm)" if v is None else f"{v*1000:.3f}mm"


if __name__ == "__main__":
    main()
