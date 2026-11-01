"""M1-P1c G1-de-risk：验证 MuJoCo geom_margin 能否报告正表面间隙。

若 link6 凸 mesh ↔ mug 盒在非穿透时，设置正 margin 后 mj_collision 能给出
正的 contact.dist，即可用 MuJoCo 自身碰撞检测做精确表面距离（优于手写 GJK）。
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


def min_dist_link6_mug(env, margin=0.05):
    """设置正 margin，跑 mj_collision，返回 link6-mug 最小 dist（正=分离，负=穿透）。"""
    m = env.sim.model._model
    d = env.sim.data._data
    saved = {g: m.geom_margin[g] for g in [LINK6_GEOM] + MUG_GEOMS}
    for g in [LINK6_GEOM] + MUG_GEOMS:
        m.geom_margin[g] = margin
    try:
        mujoco.mj_collision(m, d)
        best = None
        for c in range(d.ncon):
            g1, g2 = d.contact[c].geom1, d.contact[c].geom2
            if (g1 == LINK6_GEOM and g2 in MUG_GEOMS) or (g2 == LINK6_GEOM and g1 in MUG_GEOMS):
                dist = float(d.contact[c].dist)
                best = dist if best is None else min(best, dist)
        return best  # None 表示 gap > margin
    finally:
        for g, v in saved.items():
            m.geom_margin[g] = v


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

    print("=== G1-de-risk：geom_margin 表面间隙验证 ===")
    print(f"默认 link6 geom64 margin={m.geom_margin[LINK6_GEOM]}, "
          f"mug geom202 margin={m.geom_margin[202]}")

    results = []
    for step in range(60):
        if step == SWITCH:
            layer.reset(obs, prompt)
        if step % REPLAN_STEPS == 0:
            nominal = generate_block(pol, obs, prompt, step // REPLAN_STEPS)
        nom_a = nominal[step % REPLAN_STEPS]
        act = nom_a if step < SWITCH else layer.filter(nom_a, obs)[0]

        # 在 step54 和 step59 各 substep 测表面距离
        if step in (54, 59):
            e = env.env
            if step == 54:
                d54 = min_dist_link6_mug(env)
                print(f"  step54 执行前: link6-mug 表面间隙 = "
                      f"{('>50mm' if d54 is None else f'{d54*1000:.3f}mm')}")
            # 跑 substep 并在每个 substep 测距
            e.timestep += 1
            policy_step = True
            for i in range(int(e.control_timestep / e.model_timestep)):
                e.sim.forward()
                e._pre_action(act, policy_step)
                e.sim.step()
                if step == 59 and i >= 18:
                    di = min_dist_link6_mug(env)
                    results.append((step, i, di))
                e._update_observables()
                policy_step = False
            e.cur_time += e.control_timestep
            obs = e._get_observations()
        else:
            obs, _r, _d, _i, _ca, _cl, _ = step_tracked(env, act, rg, og)

    print("  step59 substep 18-24 表面间隙（正=分离，负=穿透）:")
    for step, i, di in results:
        print(f"    substep{i}: {('>50mm' if di is None else f'{di*1000:.3f}mm')}")

    pol.close()
    env.close()


if __name__ == "__main__":
    main()
