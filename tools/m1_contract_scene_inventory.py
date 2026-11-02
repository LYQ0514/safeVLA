"""M1-Contract-Audit 侦察：完整枚举场景 body/geom 结构与关节类型。

目标：厘清 mug 底座（box_small_base_1_main）的真实归属、所有 FREE joint 环境物体、
以及 geom 名与 body 归属的对应关系，为显式安全对象清单提供事实基础。
"""

from __future__ import annotations
import os, sys, warnings, json
warnings.filterwarnings("ignore")
ROOT = "/data/safeVLA"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "dataset", "SafeLIBERO", "safelibero"))
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import mujoco
from libero.libero import benchmark
from tools.g1a_soup_trace import make_env

JNT = {0: "FREE", 1: "BALL", 2: "SLIDE", 3: "HINGE"}

def main():
    ts = benchmark.get_benchmark_dict()["safelibero_long"](safety_level="I")
    task = ts.get_task(0)
    env = make_env(ts, task, 1)
    m = env.sim.model._model
    d = env.sim.data._data

    # 1. 所有 body：id, name, parent, joint type, 是否有碰撞 geom
    print("=== 全部 body（含 joint 类型与碰撞 geom）===")
    rows = []
    for bid in range(m.nbody):
        name = env.sim.model.body_names[bid]
        jntadr = m.body_jntadr[bid]
        jt = int(m.jnt_type[jntadr]) if jntadr >= 0 else -1
        jtn = JNT.get(jt, str(jt))
        parent = env.sim.model.body_names[m.body_parentid[bid]] if m.body_parentid[bid] < m.nbody else "?"
        ncol = sum(1 for g in range(m.ngeom) if m.geom_bodyid[g] == bid and m.geom_contype[g] > 0)
        nvis = sum(1 for g in range(m.ngeom) if m.geom_bodyid[g] == bid and m.geom_contype[g] == 0)
        qa = m.jnt_qposadr[jntadr] if jntadr >= 0 else -1
        xyz = np.round(d.xpos[bid], 3) if bid < len(d.xpos) else None
        rows.append({
            "bid": bid, "name": name, "parent": parent, "jnt": jtn,
            "n_collision_geom": ncol, "n_visual_geom": nvis,
            "qpos_adr": qa, "xpos": None if xyz is None else xyz.tolist(),
        })
    # 只打印有碰撞 geom 或 FREE/可动的 body
    interesting = [r for r in rows if r["n_collision_geom"] > 0 or r["jnt"] in ("FREE", "SLIDE", "BALL")]
    for r in interesting:
        print(f"  body[{r['bid']:3d}] {r['name']:38s} jnt={r['jnt']:5s} "
              f"col={r['n_collision_geom']:2d} vis={r['n_visual_geom']:2d} "
              f"parent={r['parent']:30s} xpos={r['xpos']}")

    # 2. mug 底座 body 与 mug body 的 geom 归属
    print("\n=== mug 与 box_small_base 的 body/geom 归属 ===")
    for bname in ["red_coffee_mug_obstacle_1_main", "box_small_base_1_main"]:
        bid = env.sim.model.body_name2id(bname)
        gnames = env.sim.model.geom_names
        ngn = len(gnames)
        geoms = [(g, gnames[g] if g < ngn else f"geom{g}") for g in range(m.ngeom) if m.geom_bodyid[g] == bid]
        print(f"  body '{bname}' (bid={bid}):")
        for g, gn in geoms:
            contype = m.geom_contype[g]
            print(f"    geom[{g}] {gn:35s} contype={contype} type={int(m.geom_type[g])}")

    # 3. 完整 obstacle joint 列表 vs 所有 FREE joint body
    print("\n=== 所有 FREE joint 环境物体（可动）===")
    free_bodies = []
    for bid in range(m.nbody):
        jntadr = m.body_jntadr[bid]
        if jntadr >= 0 and int(m.jnt_type[jntadr]) == 0:
            name = env.sim.model.body_names[bid]
            if "robot" not in name and "gripper" not in name:
                free_bodies.append(name)
    print("  ", free_bodies)

    out = json.dumps({"bodies": rows, "free_env_bodies": free_bodies},
                     ensure_ascii=False, indent=2, default=str)
    with open(os.path.join(ROOT, "data", "audit", "m1_contract_scene_inventory.json"), "w") as f:
        f.write(out)
    print("\n结果: data/audit/m1_contract_scene_inventory.json")
    env.close()

if __name__ == "__main__":
    main()
