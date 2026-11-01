"""M1-P0：全身历史契约 Oracle 验证器（固定状态审计）。

对 state02/stream0 的 step15/20/62/63，构造 6 候选（nom/aegis/3插值/hold），
每个候选做全身（含 link6）+ 历史 CAR 契约验证，选择最小干预的安全候选。

验收：
- step15/20：应能保留安全名义动作；
- step62/63：应能识别 link6-mug 接触 / CAR 余量不足 / 无候选。
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
DELTA_CAR = 0.001  # 1mm
EPS_MARGIN = 0.0001  # 安全裕度 0.1mm
ALPHAS = [0.25, 0.5, 0.75]


def build_body_geom_sets(env):
    """全身 robot geoms（含 link6 等所有 link）+ obstacle geoms。"""
    m = env.sim.model._model
    robot_geoms = {g for g in range(m.ngeom) if m.geom_contype[g] > 0 and
                   (env.sim.model.body_names[m.geom_bodyid[g]].startswith("robot0") or
                    env.sim.model.body_names[m.geom_bodyid[g]].startswith("gripper0"))}
    obstacle_names = set(n.replace("_joint0", "") for n in env.sim.model.joint_names if "obstacle" in n)
    obstacle_geoms = {g for g in range(m.ngeom) if m.geom_contype[g] > 0 and
                      any(env.sim.model.body_names[m.geom_bodyid[g]].startswith(o) for o in obstacle_names)}
    return robot_geoms, obstacle_geoms


def oracle_verify(env, act, robot_geoms, obstacle_geoms, p0, D_t):
    """全身 + 历史 CAR 契约验证。返回 (safe, contact_pairs, D_pred)。"""
    snapshot = env.sim.get_state()
    contact = False
    contact_pairs = []
    D_pred = D_t
    m = env.sim.model._model
    for k in range(int(env.env.control_timestep / env.env.model_timestep)):
        env.sim.forward()
        env.env._pre_action(act, (k == 0))
        env.sim.step()
        dd = env.sim.data._data
        for c in range(dd.ncon):
            g1, g2 = dd.contact[c].geom1, dd.contact[c].geom2
            if (g1 in robot_geoms and g2 in obstacle_geoms) or (g2 in robot_geoms and g1 in obstacle_geoms):
                contact = True
                b1 = env.sim.model.body_names[m.geom_bodyid[g1]]
                b2 = env.sim.model.body_names[m.geom_bodyid[g2]]
                contact_pairs.append((b1, b2))
        # 历史位移（相对 episode 初始 p0）
        for o in ["red_coffee_mug_obstacle_1_main", "box_small_base_1_main"]:
            try:
                pj = body_state(env, o)["pos"]
                d = float(np.sum(np.abs(pj - p0[o])))
                D_pred = max(D_pred, d)
            except Exception:
                pass
    env.sim.set_state(snapshot)
    env.sim.forward()
    safe = (not contact) and (D_pred < DELTA_CAR - EPS_MARGIN)
    return safe, contact_pairs, D_pred


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
    robot_geoms, obstacle_geoms = build_body_geom_sets(env)

    # episode 初始受保护物体位置
    p0 = {
        "red_coffee_mug_obstacle_1_main": body_state(env, "red_coffee_mug_obstacle_1_main")["pos"].copy(),
        "box_small_base_1_main": body_state(env, "box_small_base_1_main")["pos"].copy(),
    }

    pol = PnsgPolicy()
    target_steps = [15, 20, 62, 63]
    print(f"=== M1-P0：全身历史契约 Oracle（step {target_steps}）===")

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

        if step in target_steps:
            # 当前 D_t（历史最大，相对 p0）
            D_t = 0.0
            for o in p0:
                D_t = max(D_t, float(np.sum(np.abs(body_state(env, o)["pos"] - p0[o]))))
            # AEGIS 过滤
            aeg_a, _ = layer.filter(nom_a, obs) if use_aegis else (nom_a.copy(), {})

            # 候选集合
            cands = {"nom": nom_a, "aegis": aeg_a, "hold": np.zeros(7)}
            for al in ALPHAS:
                cands[f"a{al}"] = al * nom_a + (1 - al) * aeg_a

            print(f"\nstep {step} (D_t={D_t*1000:.3f}mm):")
            verified = {}
            for cname, cact in cands.items():
                safe, pairs, D_pred = oracle_verify(env, cact, robot_geoms, obstacle_geoms, p0, D_t)
                if safe:
                    verified[cname] = cact
                tag = "✓安全" if safe else ("✗接触" if pairs else "✗CAR")
                print(f"  {cname:6s}: {tag} D_pred={D_pred*1000:.3f}mm 接触={set(pairs) if pairs else '无'}")

            if verified:
                # 最小干预：最接近 nominal 的 verified 候选
                best = min(verified, key=lambda c: np.linalg.norm(verified[c] - nom_a))
                print(f"  → 选择 {best}（最小干预）")
            else:
                print(f"  → NO_VERIFIED_ACTION（无安全候选）")

        # 正式推进（B 分支语义：前40名义，后AEGIS）
        act = nom_a if not use_aegis else layer.filter(nom_a, obs)[0]
        obs, _, done, _ = env.step(act)
        if done or env.check_success():
            break

    pol.close()
    env.close()


if __name__ == "__main__":
    main()
