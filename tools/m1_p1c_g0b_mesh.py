"""M1-P1c G0b：link6 mesh 数据 + step59 substep 内 link6-mug 精确接触对。

1. link6 碰撞 mesh（geom 64）的顶点/面数量、包围球 rbound、凸性。
2. mug 碰撞盒（202-206）的完整几何。
3. 在 step59 的 25 个 substep 中，捕获 link6-mug 接触的精确 geom 对与 substep 索引。
"""

from __future__ import annotations

import json
import os
import pathlib
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

    # ---- 1. link6 mesh 数据 ----
    link6_mesh_geom = 64  # robot0_g47_vis
    mesh_id = int(m.geom_dataid[link6_mesh_geom])
    nvert = m.mesh_vertadr[mesh_id + 1] - m.mesh_vertadr[mesh_id] if mesh_id + 1 < len(m.mesh_vertadr) else 0
    nface = m.mesh_faceadr[mesh_id + 1] - m.mesh_faceadr[mesh_id] if mesh_id + 1 < len(m.mesh_faceadr) else 0
    rbound = float(m.geom_rbound[link6_mesh_geom])
    print("=== link6 碰撞 mesh（geom 64 = robot0_g47_vis）===")
    print(f"  mesh_id={mesh_id}  顶点数={nvert}  面数={nface}")
    print(f"  rbound（包围球半径）={rbound*1000:.1f}mm")
    print(f"  geom_size={np.asarray(m.geom_size[link6_mesh_geom]).round(5).tolist()}")
    print(f"  geom_type={m.geom_type[link6_mesh_geom]}（7=mesh）")

    # 凸性：MuJoCo 碰撞 mesh 必须是凸的；取顶点算 bounding box
    if nvert > 0 and nvert < 200:
        v0 = m.mesh_vertadr[mesh_id]
        verts = np.asarray(m.mesh_vert[v0:v0 + nvert])
        print(f"  顶点范围: min={verts.min(axis=0).round(4).tolist()}  max={verts.max(axis=0).round(4).tolist()}")

    # ---- 2. mug 碰撞盒完整信息 ----
    print("\n=== mug 碰撞盒（body=red_coffee_mug_obstacle_1_main）===")
    mug_box_geoms = [202, 203, 204, 205, 206]
    for g in mug_box_geoms:
        print(f"  geom{g} ({env.sim.model.geom_names[g]}): box size={np.asarray(m.geom_size[g]).round(5).tolist()} "
              f"pos={np.asarray(m.geom_pos[g]).round(5).tolist()}")

    # ---- 3. step59 substep 内 link6-mug 接触对 ----
    print("\n=== step59 substep 内 link6-mug 接触对 ===")
    mug_geom_ids = set()
    for g in range(m.ngeom):
        bname = env.sim.model.body_names[m.geom_bodyid[g]]
        if bname == "red_coffee_mug_obstacle_1_main" and m.geom_contype[g] > 0:
            mug_geom_ids.add(g)
    link6_geom_ids = {g for g in range(m.ngeom)
                      if env.sim.model.body_names[m.geom_bodyid[g]] == "robot0_link6"
                      and m.geom_contype[g] > 0}
    print(f"  link6 碰撞 geom={link6_geom_ids}")
    print(f"  mug 碰撞 geom={mug_geom_ids}")

    contact_log = []
    for step in range(60):
        if step == SWITCH:
            layer.reset(obs, prompt)
        if step % REPLAN_STEPS == 0:
            nominal = generate_block(pol, obs, prompt, step // REPLAN_STEPS)
        nom_a = nominal[step % REPLAN_STEPS]
        act = nom_a if step < SWITCH else layer.filter(nom_a, obs)[0]
        if step == 59:
            # 手工 substep 循环，捕获 link6-mug 接触
            e = env.env
            e.timestep += 1
            policy_step = True
            for i in range(int(e.control_timestep / e.model_timestep)):
                e.sim.forward()
                e._pre_action(act, policy_step)
                e.sim.step()
                d = env.sim.data._data
                for c in range(d.ncon):
                    g1, g2 = d.contact[c].geom1, d.contact[c].geom2
                    if (g1 in link6_geom_ids and g2 in mug_geom_ids) or (g2 in link6_geom_ids and g1 in mug_geom_ids):
                        contact_log.append({
                            "substep": i,
                            "geom1": g1, "geom2": g2,
                            "name1": env.sim.model.geom_names[g1],
                            "name2": env.sim.model.geom_names[g2],
                            "dist": float(d.contact[c].dist),
                            "pos": np.asarray(d.contact[c].pos).round(5).tolist(),
                        })
                e._update_observables()
                policy_step = False
            e.cur_time += e.control_timestep
            obs = e._get_observations()
        else:
            obs, _r, _d, _i, _ca, _cl, _ = step_tracked(env, act, rg, og)

    print(f"  step59 substep 内 link6-mug 接触次数: {len(contact_log)}")
    for c in contact_log:
        print(f"    substep{c['substep']}: {c['name1']} ↔ {c['name2']}  dist={c['dist']*1000:.4f}mm  pos={c['pos']}")

    out = pathlib.Path(ROOT, "data", "audit", "m1_p1c_g0b_mesh_substep.json")
    out.write_text(json.dumps({
        "link6_mesh": {"geom_id": link6_mesh_geom, "mesh_id": mesh_id, "nvert": nvert,
                       "nface": nface, "rbound": rbound,
                       "geom_size": np.asarray(m.geom_size[link6_mesh_geom]).round(5).tolist()},
        "mug_box_geoms": mug_box_geoms,
        "link6_geom_ids": sorted(link6_geom_ids),
        "mug_geom_ids": sorted(mug_geom_ids),
        "step59_link6_mug_contacts": contact_log,
    }, ensure_ascii=False, indent=2, default=str))
    print(f"\n结果: {out}")

    pol.close()
    env.close()


if __name__ == "__main__":
    main()
