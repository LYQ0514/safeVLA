"""M1-P1c G2 前置：同状态数值一致性核对 + margin 语义 + 独立查询模型可行性。

三个目标：
1. mjModel 拷贝可行性（独立查询模型，绝不让 margin 污染真实动力学）
2. margin 与 contact.dist 的数值关系（dist = f(真实表面间隙, margin, gap)）
3. 同状态核对：margin=0 的 mj_collision 与 mj_step 内部接触记录（G0b）是否一致
"""

from __future__ import annotations

import copy
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

SUITE = "safelibero_long"
LEVEL = "I"
TASK_ID = 0
STATE_ID = 2
STREAM = 0
REPLAN_STEPS = 5
IMAGE_SIZE = 224
SWITCH = 40
LINK6_GEOM = 64
MUG_GEOMS = [202, 203, 204, 205, 206]


def coll_dist(model, data, margin):
    """在给定 margin 下跑 mj_collision，返回 (link6-mug 最小 dist, 对应 geom 对)。"""
    for g in [LINK6_GEOM] + MUG_GEOMS:
        model.geom_margin[g] = margin
    mujoco.mj_collision(model, data)
    best = None
    pair = None
    for c in range(data.ncon):
        g1, g2 = data.contact[c].geom1, data.contact[c].geom2
        if (g1 == LINK6_GEOM and g2 in MUG_GEOMS) or (g2 == LINK6_GEOM and g1 in MUG_GEOMS):
            dd = float(data.contact[c].dist)
            if best is None or dd < best:
                best = dd
                pair = (g1, g2)
    return best, pair


def main():
    from libero.libero import benchmark
    ts = benchmark.get_benchmark_dict()[SUITE](safety_level=LEVEL)
    task = ts.get_task(TASK_ID)
    prompt = task.language

    env = make_env(ts, task, STATE_ID)
    layer = AegisSafetyLayer(env)
    obs = env.env._get_observations()
    layer.reset(obs, prompt)
    rg, og = build_body_geoms(env)
    pol = PnsgPolicy()
    m = env.sim.model._model
    d = env.sim.data._data

    # ---- 目标 1：model 拷贝可行性 ----
    print("=== 目标1：mjModel 拷贝 ===")
    try:
        m2 = copy.deepcopy(m)
        print(f"  deepcopy 成功, 类型={type(m2).__name__}, ngeom={m2.ngeom}")
    except Exception as e:
        print(f"  deepcopy 失败: {e}")
        m2 = None
    if m2 is None:
        try:
            m2 = mujoco.MjModel.from_xml_string(m.to_xml_string())
            print(f"  from_xml_string 成功, ngeom={m2.ngeom}")
        except Exception as e:
            print(f"  from_xml_string 失败: {e}")

    # ---- 跑到 step59 执行后，取 qpos/qvel 做同状态核对 ----
    print("\n=== 跑到 step59 执行后 ===")
    step59_qpos = None
    step59_qvel = None
    substep24_contact = None
    for step in range(60):
        if step == SWITCH:
            layer.reset(obs, prompt)
        if step % REPLAN_STEPS == 0:
            nominal = generate_block(pol, obs, prompt, step // REPLAN_STEPS)
        nom_a = nominal[step % REPLAN_STEPS]
        act = nom_a if step < SWITCH else layer.filter(nom_a, obs)[0]
        if step == 59:
            e = env.env
            e.timestep += 1
            policy_step = True
            for i in range(int(e.control_timestep / e.model_timestep)):
                e.sim.forward()
                e._pre_action(act, policy_step)
                e.sim.step()
                if i == 24:
                    # 记录 mj_step 内部真实接触（G0b 口径）
                    for c in range(d.ncon):
                        g1, g2 = d.contact[c].geom1, d.contact[c].geom2
                        if (g1 == LINK6_GEOM and g2 in MUG_GEOMS) or (g2 == LINK6_GEOM and g1 in MUG_GEOMS):
                            substep24_contact = (g1, g2, float(d.contact[c].dist))
                    step59_qpos = np.asarray(d.qpos).copy()
                    step59_qvel = np.asarray(d.qvel).copy()
                e._update_observables()
                policy_step = False
            e.cur_time += e.control_timestep
            obs = e._get_observations()
        else:
            obs, _r, _d, _i, _ca, _cl, _ = step_tracked(env, act, rg, og)

    print(f"  step59 substep24 mj_step 内部接触: {substep24_contact}")

    # ---- 目标 3：同状态核对（margin=0 的 collision vs mj_step 内部） ----
    print("\n=== 目标3：同状态核对（同一 qpos/qvel，margin=0 collision）===")
    # 用独立 data 跑，避免污染真实 data
    qd = mujoco.MjData(m)
    qd.qpos[:] = step59_qpos
    qd.qvel[:] = step59_qvel
    mujoco.mj_forward(m, qd)
    for g in [LINK6_GEOM] + MUG_GEOMS:
        m.geom_margin[g] = 0.0
    mujoco.mj_collision(m, qd)
    for c in range(qd.ncon):
        g1, g2 = qd.contact[c].geom1, qd.contact[c].geom2
        if (g1 == LINK6_GEOM and g2 in MUG_GEOMS) or (g2 == LINK6_GEOM and g1 in MUG_GEOMS):
            print(f"  margin=0 collision: g{g1}-g{g2} dist={qd.contact[c].dist*1000:.4f}mm")
    print(f"  对照 mj_step 内部: g{substep24_contact[0]}-g{substep24_contact[1]} dist={substep24_contact[2]*1000:.4f}mm")

    # ---- 目标 2：margin 语义（dist 随 margin 的变化） ----
    print("\n=== 目标2：margin 与 dist 关系（同一 qpos）===")
    for mg in [0.0, 0.01, 0.02, 0.05, 0.10]:
        qd2 = mujoco.MjData(m)
        qd2.qpos[:] = step59_qpos
        qd2.qvel[:] = step59_qvel
        mujoco.mj_forward(m, qd2)
        for g in [LINK6_GEOM] + MUG_GEOMS:
            m.geom_margin[g] = mg
        mujoco.mj_collision(m, qd2)
        best, pair = coll_dist(m, qd2, mg)
        print(f"  margin={mg*1000:5.1f}mm → dist={('None' if best is None else f'{best*1000:.4f}mm')}  pair={pair}")

    # ---- 恢复：确认真实 model 的 margin 已被改回（前面循环最后改到 0.10） ----
    print("\n=== 恢复检查 ===")
    print(f"  真实 model geom64 margin 当前={m.geom_margin[LINK6_GEOM]}（应恢复为 0.0）")

    pol.close()
    env.close()


if __name__ == "__main__":
    main()
