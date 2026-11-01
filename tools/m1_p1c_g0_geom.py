"""M1-P1c G0：link6 与 mug 的 collision geom 清单。

在 B 轨迹 step54（接触前）与 step59（首次接触）两个状态，导出：
- link6 / mug 所属 body 下的全部 geom：ID、name、type、size、pos、quat、contype/conaffinity
- step59 的接触对（sim.data.contact 中的 geom 对，含 dist/pos/frame）
- 区分 collision geom（contype>0）与 visual geom

用途：为 G1 表面间隙查询确定几何（胶囊/盒/球/圆柱/网格）。
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

from src.pnsg.policy_adapter import PnsgPolicy  # noqa: E402
from src.safety.aegis_adapted import AegisSafetyLayer  # noqa: E402
from tools.g1a_soup_trace import make_env, body_state  # noqa: E402
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

GEOM_TYPE = {0: "plane", 1: "hfield", 2: "sphere", 3: "capsule", 4: "ellipsoid",
             5: "cylinder", 6: "box", 7: "mesh"}


def geom_inventory(env, body_name):
    """返回 body 名下所有 geom 的清单。"""
    m = env.sim.model._model
    d = env.sim.data._data
    out = []
    for g in range(m.ngeom):
        bname = env.sim.model.body_names[m.geom_bodyid[g]]
        if bname != body_name:
            continue
        gname = env.sim.model.geom_names[g] if g < len(env.sim.model.geom_names) else f"geom{g}"
        out.append({
            "geom_id": g,
            "geom_name": gname,
            "type": GEOM_TYPE.get(int(m.geom_type[g]), str(int(m.geom_type[g]))),
            "size": np.asarray(m.geom_size[g]).round(5).tolist(),
            "pos": np.asarray(m.geom_pos[g]).round(5).tolist(),
            "quat": np.asarray(m.geom_quat[g]).round(5).tolist(),
            "contype": int(m.geom_contype[g]),
            "conaffinity": int(m.geom_conaffinity[g]),
            "group": int(m.geom_group[g]),
            "is_collision": int(m.geom_contype[g]) > 0 and int(m.geom_conaffinity[g]) > 0,
            "world_xpos": np.asarray(d.geom_xpos[g]).round(5).tolist(),
            "world_xmat": np.asarray(d.geom_xmat[g]).round(4).tolist(),
        })
    return out


def contact_pairs(env):
    """当前状态所有接触对（geom 对 + 名称 + dist）。"""
    m = env.sim.model._model
    d = env.sim.data._data
    out = []
    gnames = env.sim.model.geom_names
    ngn = len(gnames)
    for c in range(d.ncon):
        g1, g2 = d.contact[c].geom1, d.contact[c].geom2
        out.append({
            "geom1": g1, "geom2": g2,
            "body1": env.sim.model.body_names[m.geom_bodyid[g1]],
            "body2": env.sim.model.body_names[m.geom_bodyid[g2]],
            "name1": gnames[g1] if g1 < ngn else f"geom{g1}",
            "name2": gnames[g2] if g2 < ngn else f"geom{g2}",
            "dist": float(d.contact[c].dist),
            "pos": np.asarray(d.contact[c].pos).round(5).tolist(),
        })
    return out


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

    # 找到 mug 与 link6 的 body 名（mug 可能有多个 body，link6 是 robot0_link6）
    m = env.sim.model._model
    mug_bodies = [b for b in env.sim.model.body_names if "mug" in b]
    link6_bodies = [b for b in env.sim.model.body_names if b == "robot0_link6"]
    print("=== G0：碰撞 geom 清单 ===")
    print(f"mug bodies: {mug_bodies}")
    print(f"link6 bodies: {link6_bodies}")

    snapshots = {}
    for step in range(60):  # 跑到 step59 接触
        if step == SWITCH:
            layer.reset(obs, prompt)
        if step % REPLAN_STEPS == 0:
            nominal = generate_block(pol, obs, prompt, step // REPLAN_STEPS)
        nom_a = nominal[step % REPLAN_STEPS]
        if step in (54, 59):
            snapshots[step] = {
                "link6": {b: geom_inventory(env, b) for b in link6_bodies},
                "mug": {b: geom_inventory(env, b) for b in mug_bodies},
                "contacts": contact_pairs(env),
            }
        act = nom_a if step < SWITCH else layer.filter(nom_a, obs)[0]
        obs, _r, _d, _i, _ca, _cl, _ = step_tracked(env, act, rg, og)

    for step, snap in snapshots.items():
        print(f"\n--- step {step} ---")
        for b in link6_bodies:
            for g in snap["link6"][b]:
                mark = "✓碰撞" if g["is_collision"] else "  视觉"
                print(f"  [link6] {mark} {g['geom_name']}: {g['type']} size={g['size']} "
                      f"pos={g['pos']}")
        for b in mug_bodies:
            for g in snap["mug"][b]:
                mark = "✓碰撞" if g["is_collision"] else "  视觉"
                print(f"  [mug]   {mark} {g['geom_name']} (body={b}): {g['type']} "
                      f"size={g['size']} pos={g['pos']}")

    print("\n--- step59 接触对（涉及 link6 或 mug）---")
    s59 = snapshots[59]["contacts"]
    for c in s59:
        if "link6" in c["body1"] or "link6" in c["body2"] or "mug" in c["body1"] or "mug" in c["body2"]:
            print(f"  {c['body1']}/{c['name1']} ↔ {c['body2']}/{c['name2']}  dist={c['dist']*1000:.3f}mm")

    out = pathlib.Path(ROOT, "data", "audit", "m1_p1c_g0_geom_inventory.json")
    out.write_text(json.dumps(snapshots, ensure_ascii=False, indent=2, default=str))
    print(f"\n结果: {out}")

    pol.close()
    env.close()


if __name__ == "__main__":
    main()
