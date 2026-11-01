"""M0-P1：选择性名义保持的闭环验证（V/F/B/M 四分支）。

固定 state02/stream0，同 CRN，同 worker。

M（GVMIS）：前 40 步，V_1 物理验证名义动作（25 substep，robot-mug 接触 + mug 位移 <=1mm），
安全则保留 nominal，否则回退 AEGIS。step40 后统一 AEGIS。

门禁：
G1 Oracle 不污染：验证用状态副本，恢复后不影响真实轨迹。
G2 AEGIS 状态切换：M 保留 nominal 时不调 filter；step40 后用与 B 一致的 reset。
G3 真实物理评价：25 substep 接触 + mug 历史最大位移。
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
from src.eval_safelibero import active_obstacle, COLLISION_DISPLACEMENT_EPS  # noqa: E402
from tools.g1a_soup_trace import make_env, body_state, eef_pos, subgoal_values  # noqa: E402

SUITE = "safelibero_long"
LEVEL = "I"
TASK_ID = 0
REPLAN_STEPS = 5
IMAGE_SIZE = 224
TRANS_SCALE = 0.05
SWITCH_STEP = 40
MAX_STEPS = 550


def build_geom_sets(env):
    m = env.sim.model._model
    robot_geoms = {g for g in range(m.ngeom) if m.geom_contype[g] > 0 and
                   (env.sim.model.body_names[m.geom_bodyid[g]].startswith("robot0") or
                    env.sim.model.body_names[m.geom_bodyid[g]].startswith("gripper0"))}
    obstacle_names = set(n.replace("_joint0", "") for n in env.sim.model.joint_names if "obstacle" in n)
    obstacle_geoms = {g for g in range(m.ngeom) if m.geom_contype[g] > 0 and
                      any(env.sim.model.body_names[m.geom_bodyid[g]].startswith(o) for o in obstacle_names)}
    return robot_geoms, obstacle_geoms


def oracle_verify(env, act, robot_geoms, obstacle_geoms, name, init_mug_pos):
    """V_1：状态副本上预演 1 个控制步（25 substep），返回是否安全。"""
    snapshot = env.sim.get_state()
    safe = True
    for i in range(int(env.env.control_timestep / env.env.model_timestep)):
        env.sim.forward()
        env.env._pre_action(act, (i == 0))
        env.sim.step()
        dd = env.sim.data._data
        for c in range(dd.ncon):
            g1, g2 = dd.contact[c].geom1, dd.contact[c].geom2
            if (g1 in robot_geoms and g2 in obstacle_geoms) or (g2 in robot_geoms and g1 in obstacle_geoms):
                safe = False
    mug_after = body_state(env, name + "_main")["pos"]
    disp = float(np.sum(np.abs(mug_after - init_mug_pos)))
    if disp > COLLISION_DISPLACEMENT_EPS:
        safe = False
    env.sim.set_state(snapshot)
    env.sim.forward()
    return safe


def run_branch(mode):
    from libero.libero import benchmark
    ts = benchmark.get_benchmark_dict()[SUITE](safety_level=LEVEL)
    task = ts.get_task(TASK_ID)
    prompt = task.language
    state_id, stream = 2, 0

    env = make_env(ts, task, state_id)
    layer = AegisSafetyLayer(env)
    obs = env.env._get_observations()
    name = active_obstacle(env, obs)
    robot_geoms, obstacle_geoms = build_geom_sets(env)
    init_mug = body_state(env, name + "_main")["pos"].copy()

    aegis_active = False
    if mode in ("F", "M"):
        layer.reset(obs, prompt)
        aegis_active = True
    elif mode == "B":
        # B：前40 π0.5，后 AEGIS
        aegis_active = False
    # V: 全程 π0.5

    pol = PnsgPolicy()
    success = False
    n_intervene = 0
    n_keep_nominal = 0
    n_reject = 0
    max_disp = 0.0
    steps = 0
    trace = []

    for step in range(MAX_STEPS):
        use_aegis = (mode == "F") or (mode == "M" and step >= SWITCH_STEP) or \
                    (mode == "B" and step >= SWITCH_STEP)
        use_gvmis = (mode == "M" and step < SWITCH_STEP)

        # B/M 在 step40 冷启动 AEGIS
        if step == SWITCH_STEP and mode in ("B", "M"):
            layer.reset(obs, prompt)
            aegis_active = True

        qi = step // REPLAN_STEPS
        if step % REPLAN_STEPS == 0:
            z0, _ = sample_z(nominal_seed(SUITE, TASK_ID, state_id, stream, qi))
            po = R.obs_to_policy_obs(obs, prompt, image_size=IMAGE_SIZE)
            nominal = np.clip(pol.generate(z0, po), -1, 1)
        nom_a = nominal[step % REPLAN_STEPS]

        act_a = nom_a
        if use_gvmis:
            # GVMIS：物理验证名义，安全则保留，否则回退 AEGIS
            mug_now = body_state(env, name + "_main")["pos"].copy()
            if oracle_verify(env, nom_a, robot_geoms, obstacle_geoms, name, mug_now):
                act_a = nom_a
                n_keep_nominal += 1
            else:
                act_a, info = layer.filter(nom_a, obs)
                n_reject += 1
                if info.get("intervened"):
                    n_intervene += 1
        elif use_aegis and aegis_active:
            act_a, info = layer.filter(nom_a, obs)
            if info.get("intervened"):
                n_intervene += 1

        soup_before = body_state(env, "alphabet_soup_1_main")["pos"].copy()
        eef_before = eef_pos(env).copy()
        obs, _r, done, _i = env.step(act_a)
        steps += 1

        disp = float(np.sum(np.abs(body_state(env, name + "_main")["pos"] - init_mug)))
        max_disp = max(max_disp, disp)

        if step % 10 == 0 or step == SWITCH_STEP:
            sg = subgoal_values(env)
            trace.append({"step": step, "disp": disp,
                          "soup_pred": bool(sg.get("alphabet_soup_1")),
                          "cream_pred": bool(sg.get("cream_cheese_1"))})

        if done or env.check_success():
            success = True
            break

    d_soup = float(np.linalg.norm(body_state(env, "alphabet_soup_1_main")["pos"] - eef_pos(env)))
    pol.close()
    env.close()
    return {
        "success": success, "collided": max_disp > COLLISION_DISPLACEMENT_EPS,
        "max_displacement": max_disp, "steps": steps,
        "n_intervene": n_intervene, "n_keep_nominal": n_keep_nominal, "n_reject": n_reject,
        "d_soup_final": d_soup,
        "final_subgoals": {"soup": None, "cream": None},  # 占位，subgoal 在 trace 里
        "trace": trace,
    }


def main():
    print(f"=== M0-P1：四分支闭环验证 ===")
    results = {}
    for mode in ["V", "F", "B", "M"]:
        r = run_branch(mode)
        results[mode] = r
        soup_ever = any(t["soup_pred"] for t in r["trace"])
        cream_ever = any(t["cream_pred"] for t in r["trace"])
        print(f"  {mode}: success={r['success']} coll={r['collided']} "
              f"Dmax={r['max_displacement']*1000:.2f}mm steps={r['steps']} "
              f"soup曾满足={soup_ever} cream曾满足={cream_ever} "
              f"keep={r['n_keep_nominal']} reject={r['n_reject']} int={r['n_intervene']}",
              flush=True)

    out = pathlib.Path(ROOT, "data", "audit", "m0_p1.json")
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str))
    print(f"\n结果: {out}")

    # 判决
    V, F, B, M = results["V"], results["F"], results["B"], results["M"]
    print("\n=== 判决 ===")
    if M["success"] and not M["collided"]:
        if M["n_reject"] > 0:
            print("M 安全成功且部分动作被拒绝 → 初步支持选择性保留名义的闭环价值")
        else:
            print("M 安全成功且 40/40 全保留 → 等价于 B，尚未证明选择性过滤能力")
    elif M["success"] and M["collided"]:
        print("M 成功但 CAR 违规 → 定位验证器预测与真实执行失配")
    else:
        print("M 失败 → 见明细")


if __name__ == "__main__":
    main()
