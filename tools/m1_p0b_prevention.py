"""M1-P0b：接触前预防测试。

沿一条连续轨迹（B 分支：前40名义，后AEGIS），定位首次 Link6-mug 接触的
控制步/substep。然后在接触前最后一个完整无接触状态，测试 M1 候选集能否
选择避免进入接触状态的动作。

不新增候选、不调参数、不向前无限搜索。
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
from tools.g1a_soup_trace import make_env, body_state  # noqa: E402

SUITE = "safelibero_long"
LEVEL = "I"
TASK_ID = 0
REPLAN_STEPS = 5
IMAGE_SIZE = 224
SWITCH = 40
DELTA_CAR = 0.001
EPS_MARGIN = 0.0001
ALPHAS = [0.25, 0.5, 0.75]


def build_body_geoms(env):
    m = env.sim.model._model
    robot_geoms = {g for g in range(m.ngeom) if m.geom_contype[g] > 0 and
                   (env.sim.model.body_names[m.geom_bodyid[g]].startswith("robot0") or
                    env.sim.model.body_names[m.geom_bodyid[g]].startswith("gripper0"))}
    obstacle_names = set(n.replace("_joint0", "") for n in env.sim.model.joint_names if "obstacle" in n)
    obstacle_geoms = {g for g in range(m.ngeom) if m.geom_contype[g] > 0 and
                      any(env.sim.model.body_names[m.geom_bodyid[g]].startswith(o) for o in obstacle_names)}
    return robot_geoms, obstacle_geoms


def link6_mug_contact(env, robot_geoms, obstacle_geoms):
    """当前状态是否有 link6-mug 接触。"""
    m = env.sim.model._model
    dd = env.sim.data._data
    for c in range(dd.ncon):
        g1, g2 = dd.contact[c].geom1, dd.contact[c].geom2
        if (g1 in robot_geoms and g2 in obstacle_geoms) or (g2 in robot_geoms and g1 in obstacle_geoms):
            b1 = env.sim.model.body_names[m.geom_bodyid[g1]]
            b2 = env.sim.model.body_names[m.geom_bodyid[g2]]
            if "link6" in b1 or "link6" in b2:
                return True
    return False


def main():
    from libero.libero import benchmark
    ts = benchmark.get_benchmark_dict()[SUITE](safety_level=LEVEL)
    task = ts.get_task(TASK_ID)
    prompt = task.language
    state_id, stream = 2, 0

    env = make_env(ts, task, state_id)
    layer = AegisSafetyLayer(env)
    obs = env.env._get_observations()
    layer.reset(obs, prompt)
    robot_geoms, obstacle_geoms = build_body_geoms(env)
    p0 = {"red_coffee_mug_obstacle_1_main": body_state(env, "red_coffee_mug_obstacle_1_main")["pos"].copy()}
    pol = PnsgPolicy()

    # 第一步：沿 B 轨迹定位首次 link6-mug 接触
    first_contact_step = None
    pre_contact_snapshot = None
    pre_contact_step = None
    last_no_contact_snapshot = None

    for step in range(MAX_STEPS := 550):
        use_aegis = step >= SWITCH
        if step == SWITCH:
            layer.reset(obs, prompt)
        qi = step // REPLAN_STEPS
        if step % REPLAN_STEPS == 0:
            z0, _ = sample_z(nominal_seed(SUITE, TASK_ID, state_id, stream, qi))
            po = R.obs_to_policy_obs(obs, prompt, image_size=IMAGE_SIZE)
            nominal = np.clip(pol.generate(z0, po), -1, 1)
        nom_a = nominal[step % REPLAN_STEPS]

        # 记录执行前是否有接触（用于定位入口）
        contact_before = link6_mug_contact(env, robot_geoms, obstacle_geoms)
        if not contact_before:
            last_no_contact_snapshot = (env.sim.get_state(), step, nom_a.copy())

        act_a = nom_a if not use_aegis else layer.filter(nom_a, obs)[0]
        obs, _, done, _ = env.step(act_a)

        if first_contact_step is None and link6_mug_contact(env, robot_geoms, obstacle_geoms):
            first_contact_step = step
            pre_contact_snapshot, pre_contact_step, pre_contact_nom = last_no_contact_snapshot
            break

    print(f"=== M1-P0b：接触前预防测试 ===")
    print(f"首次 link6-mug 接触 @ step {first_contact_step}")
    print(f"接触前最后一个无接触状态 @ step {pre_contact_step}")

    if pre_contact_snapshot is None:
        print("未找到无接触前置状态")
        pol.close(); env.close()
        return

    # 第二步：在接触前状态，测试候选集能否避免进入接触
    env.sim.set_state(pre_contact_snapshot)
    env.sim.forward()
    nom_a = pre_contact_nom
    aeg_a, _ = layer.filter(nom_a, obs) if pre_contact_step >= SWITCH else (nom_a.copy(), {})

    cands = {"nom": nom_a, "aegis": aeg_a, "hold": np.zeros(7)}
    for al in ALPHAS:
        cands[f"a{al}"] = al * nom_a + (1 - al) * aeg_a

    print(f"\n在 step{pre_contact_step}（无接触）测试候选能否避免接触:")
    for cname, cact in cands.items():
        env.sim.set_state(pre_contact_snapshot)
        env.sim.forward()
        # 执行 1 步，看是否产生 link6-mug 接触
        contact = False
        for k in range(int(env.env.control_timestep / env.env.model_timestep)):
            env.sim.forward()
            env.env._pre_action(cact, (k == 0))
            env.sim.step()
            if link6_mug_contact(env, robot_geoms, obstacle_geoms):
                contact = True
                break
        print(f"  {cname:6s}: {'✗产生接触' if contact else '✓避免接触'}")

    pol.close()
    env.close()


if __name__ == "__main__":
    main()
