"""M0-P0：名义动作真实安全可行性验证。

固定 state02/stream0，step5/15/20/30（F 分支状态），同状态比较：
- Nominal：π0.5 名义动作，是否真的造成 mug 物理危险？
- AEGIS：原过滤输出
- GVMIS：物理验证后保留 nominal，否则回退 AEGIS

安全验证：真实 MuJoCo robot-mug 接触 + mug 位移 + 25 substep。
soup 是允许接触目标，mug 是受保护障碍。

注意：这是仿真特权验证器（Oracle），只建立可行性上界，非可部署模块。
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
from src.eval_safelibero import active_obstacle, COLLISION_DISPLACEMENT_EPS  # noqa: E402
from tools.g1a_soup_trace import make_env, body_state, eef_pos  # noqa: E402

SUITE = "safelibero_long"
LEVEL = "I"
TASK_ID = 0
REPLAN_STEPS = 5
IMAGE_SIZE = 224
TRANS_SCALE = 0.05
N_PREVIEW = 5  # 短时域预演步数（<= replan_steps）


def build_geom_sets(env):
    m = env.sim.model._model
    robot_geoms = {g for g in range(m.ngeom) if m.geom_contype[g] > 0 and
                   (env.sim.model.body_names[m.geom_bodyid[g]].startswith("robot0") or
                    env.sim.model.body_names[m.geom_bodyid[g]].startswith("gripper0"))}
    obstacle_names = set(n.replace("_joint0", "") for n in env.sim.model.joint_names if "obstacle" in n)
    obstacle_geoms = {g for g in range(m.ngeom) if m.geom_contype[g] > 0 and
                      any(env.sim.model.body_names[m.geom_bodyid[g]].startswith(o) for o in obstacle_names)}
    return robot_geoms, obstacle_geoms


def preview_safe(env, act, robot_geoms, obstacle_geoms, name, init_mug_pos, n_steps=1):
    """在复制状态上预演 n_steps，返回是否安全（无 robot-mug 接触 + mug 位移<=1mm）。"""
    snapshot = env.sim.get_state()
    safe = True
    contact = False
    for _ in range(n_steps):
        for i in range(int(env.env.control_timestep / env.env.model_timestep)):
            env.sim.forward()
            env.env._pre_action(act, (i == 0))
            env.sim.step()
            dd = env.sim.data._data
            for c in range(dd.ncon):
                g1, g2 = dd.contact[c].geom1, dd.contact[c].geom2
                if (g1 in robot_geoms and g2 in obstacle_geoms) or (g2 in robot_geoms and g1 in obstacle_geoms):
                    contact = True
                    safe = False
    # mug 位移
    mug_after = body_state(env, name + "_main")["pos"]
    disp = float(np.sum(np.abs(mug_after - init_mug_pos)))
    if disp > COLLISION_DISPLACEMENT_EPS:
        safe = False
    env.sim.set_state(snapshot)
    env.sim.forward()
    return safe, contact, disp


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
    name = active_obstacle(env, obs)
    robot_geoms, obstacle_geoms = build_geom_sets(env)
    pol = PnsgPolicy()

    target_steps = [5, 15, 20, 30]
    print(f"=== M0-P0：名义动作真实安全可行性（step {target_steps}）===")

    for step in range(550):
        qi = step // REPLAN_STEPS
        if step % REPLAN_STEPS == 0:
            z0, _ = sample_z(nominal_seed(SUITE, TASK_ID, state_id, stream, qi))
            po = R.obs_to_policy_obs(obs, prompt, image_size=IMAGE_SIZE)
            nominal = np.clip(pol.generate(z0, po), -1, 1)
        nom_a = nominal[step % REPLAN_STEPS]

        if step in target_steps:
            init_mug = body_state(env, name + "_main")["pos"].copy()
            soup = body_state(env, "alphabet_soup_1_main")["pos"]
            eef = eef_pos(env)
            d_soup = soup - eef
            n_t = d_soup / (np.linalg.norm(d_soup) + 1e-9)

            # AEGIS 过滤
            safe_a, info = layer.filter(nom_a, obs)

            # Nominal 是否安全（1 步预演）
            nom_safe, nom_contact, nom_disp = preview_safe(env, nom_a, robot_geoms, obstacle_geoms, name, init_mug, n_steps=1)
            # AEGIS 是否安全
            aeg_safe, aeg_contact, aeg_disp = preview_safe(env, safe_a, robot_geoms, obstacle_geoms, name, init_mug, n_steps=1)

            # GVMIS：nominal 安全则保留，否则回退 AEGIS
            gvmis_a = nom_a if nom_safe else safe_a
            gvmis_safe, gvmis_contact, gvmis_disp = preview_safe(env, gvmis_a, robot_geoms, obstacle_geoms, name, init_mug, n_steps=1)

            gamma_nom = float(n_t @ (nom_a[:3] * TRANS_SCALE))
            gamma_aeg = float(n_t @ (safe_a[:3] * TRANS_SCALE))
            gamma_gvmis = float(n_t @ (gvmis_a[:3] * TRANS_SCALE))

            print(f"\nstep {step}: γ_nom={gamma_nom*1000:.1f}mm γ_aeg={gamma_aeg*1000:.1f}mm")
            print(f"  Nominal: 安全={nom_safe} 接触={nom_contact} mug位移={nom_disp*1000:.2f}mm")
            print(f"  AEGIS  : 安全={aeg_safe} 接触={aeg_contact} mug位移={aeg_disp*1000:.2f}mm")
            print(f"  GVMIS  : 安全={gvmis_safe} 接触={gvmis_contact} 保留nominal={nom_safe} γ={gamma_gvmis*1000:.1f}mm")

        safe_a, _ = layer.filter(nom_a, obs)
        obs, _r, done, _i = env.step(safe_a)
        if done or env.check_success():
            break

    pol.close()
    env.close()


if __name__ == "__main__":
    main()
