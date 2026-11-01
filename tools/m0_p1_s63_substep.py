"""S63 局部事件审计：定位 step63 首次越界的精确 substep + 接触。

从 B 分支 step62 末状态出发，执行 step63 的动作（AEGIS 过滤后），
逐 substep 记录 mug 位移、接触对、速度，定位跨过 1mm 的精确 substep。
"""

from __future__ import annotations

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
import pi05_libero_runner as R  # noqa: E402

from src.pnsg.crn import nominal_seed, sample_z  # noqa: E402
from src.pnsg.policy_adapter import PnsgPolicy  # noqa: E402
from src.safety.aegis_adapted import AegisSafetyLayer  # noqa: E402
from src.eval_safelibero import active_obstacle  # noqa: E402
from tools.g1a_soup_trace import make_env, body_state, eef_pos  # noqa: E402

SUITE = "safelibero_long"
LEVEL = "I"
TASK_ID = 0
REPLAN_STEPS = 5
IMAGE_SIZE = 224
SWITCH = 40


def main():
    from libero.libero import benchmark
    ts = benchmark.get_benchmark_dict()[SUITE](safety_level=LEVEL)
    task = ts.get_task(TASK_ID)
    prompt = task.language
    state_id, stream = 2, 0

    env = make_env(ts, task, state_id)
    layer = AegisSafetyLayer(env)
    obs = env.env._get_observations()
    name = active_obstacle(env, obs)
    init_mug = body_state(env, name + "_main")["pos"].copy()
    layer.reset(obs, prompt)  # B 前40不filter，但step40后filter

    m = env.sim.model._model
    robot_geoms = {g for g in range(m.ngeom) if m.geom_contype[g] > 0 and
                   (env.sim.model.body_names[m.geom_bodyid[g]].startswith("robot0") or
                    env.sim.model.body_names[m.geom_bodyid[g]].startswith("gripper0"))}
    obstacle_names = set(n.replace("_joint0", "") for n in env.sim.model.joint_names if "obstacle" in n)
    obstacle_geoms = {g for g in range(m.ngeom) if m.geom_contype[g] > 0 and
                      any(env.sim.model.body_names[m.geom_bodyid[g]].startswith(o) for o in obstacle_names)}

    pol = PnsgPolicy()
    step63_snapshot = None
    step63_act = None
    step63_nom = None

    # 跑 B 到 step63 前
    for step in range(550):
        use_aegis = step >= SWITCH
        if step == SWITCH:
            layer.reset(obs, prompt)
        qi = step // REPLAN_STEPS
        if step % REPLAN_STEPS == 0:
            z0, _ = sample_z(nominal_seed(SUITE, TASK_ID, state_id, stream, qi))
            po = R.obs_to_policy_obs(obs, prompt, image_size=IMAGE_SIZE)
            nominal = np.clip(pol.generate(z0, po), -1, 1)
        nom_a = nominal[step % REPLAN_STEPS]
        if step == 63:
            # 记录 step63 前的快照和动作
            step63_snapshot = env.sim.get_state()
            step63_nom = nom_a.copy()
            if use_aegis:
                step63_act, _ = layer.filter(nom_a, obs)
            else:
                step63_act = nom_a.copy()
        act_a = step63_act if step == 63 else (layer.filter(nom_a, obs)[0] if use_aegis else nom_a)
        obs, _, done, _ = env.step(act_a)
        if step == 63:
            break

    # 从 step63 快照逐 substep 重放
    env.sim.set_state(step63_snapshot)
    env.sim.forward()
    mug_before = body_state(env, name + "_main")["pos"].copy()
    D_before = float(np.sum(np.abs(mug_before - init_mug)))
    print(f"=== step63 逐 substep（D_before={D_before*1000:.3f}mm）===")
    print(f"  名义动作 a_nom[:3]={np.round(step63_nom[:3],3)} 过滤动作 a_act[:3]={np.round(step63_act[:3],3)}")

    for i in range(int(env.env.control_timestep / env.env.model_timestep)):
        env.sim.forward()
        env.env._pre_action(step63_act, (i == 0))
        env.sim.step()
        dd = env.sim.data._data
        contact_pairs = []
        for c in range(dd.ncon):
            g1, g2 = dd.contact[c].geom1, dd.contact[c].geom2
            if (g1 in robot_geoms and g2 in obstacle_geoms) or (g2 in robot_geoms and g1 in obstacle_geoms):
                b1 = env.sim.model.body_names[m.geom_bodyid[g1]]
                b2 = env.sim.model.body_names[m.geom_bodyid[g2]]
                contact_pairs.append((b1, b2, round(float(dd.contact[c].dist), 4)))
        mug = body_state(env, name + "_main")["pos"]
        D = float(np.sum(np.abs(mug - init_mug)))
        if D > 0.0005 or contact_pairs:
            print(f"  substep{i:>2}: D={D*1000:.3f}mm 接触={contact_pairs if contact_pairs else '无'}")

    pol.close()
    env.close()


if __name__ == "__main__":
    main()
