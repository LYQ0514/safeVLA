"""C1：闭环反事实 —— 早期/后期 AEGIS 干预对 soup 任务恢复的影响。

固定 Long-I/task0/state02/stream0，同一 CRN 随机流，同一 worker。

四分支 2×2 时间干预设计：
  V: 全程 π0.5
  A: 前40步 AEGIS，之后 π0.5（检验早期干预持续影响）
  B: 前40步 π0.5，之后 AEGIS（检验后期干预影响）
  F: 全程 AEGIS（参照）

门禁：
  G1 前缀一致性：A/F 前40步物理状态一致；V/B 前40步一致
  G2 切换不重置：step40 不重置 MuJoCo/控制器/随机流

记录：d_eef_soup(t), γ_nom(t), t_grasp, P_soup(t), P_cream(t), D_max(t),
      N_intervene, 最终 TSR/CAR/SSR。
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

SUITE = "safelibero_long"
LEVEL = "I"
TASK_ID = 0
REPLAN_STEPS = 5
NUM_STEPS_WAIT = 10
MAX_STEPS = 550
IMAGE_SIZE = 224
SWITCH_STEP = 40  # 0-39 前段，40+ 后段
TRANS_SCALE = 0.05


def make_env(ts, task, state_id):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256,
                             controller="OSC_POSE")
    env.seed(7)
    env.reset()
    env.set_init_state(ts.get_task_init_states(TASK_ID)[state_id])
    dummy = np.zeros(7, dtype=np.float32); dummy[-1] = -1.0
    for _ in range(NUM_STEPS_WAIT):
        env.step(dummy)
    return env


def body_pos(env, name):
    bid = env.sim.model.body_name2id(name)
    return np.asarray(env.sim.data._data.xpos[bid]).copy()


def subgoal_values(env):
    inner = env.env
    gs = inner.parsed_problem["goal_state"]
    return {st[1]: bool(inner._eval_predicate(st)) for st in gs}


def eef_pos(env):
    return np.asarray(env.sim.data._data.site_xpos[env.robots[0].eef_site_id]).copy()


def run_c1_branch(policy, env, layer, state_id, stream, prompt, early_aegis, late_aegis):
    """跑一个分支。early_aegis/late_aegis 控制前段/后段是否用 AEGIS。"""
    obs = env.env._get_observations()
    name = active_obstacle(env, obs)
    init_pos = body_pos(env, name + "_main")
    # layer 只在需要时 reset
    layer_active = False
    if early_aegis and layer is not None:
        layer.reset(obs, prompt)
        layer_active = True

    success = False
    n_intervene = 0
    max_disp = 0.0
    steps = 0
    trace = []

    for step in range(MAX_STEPS):
        use_aegis = (early_aegis if step < SWITCH_STEP else late_aegis)
        # 切换点：B 分支在 step40 冷启动 AEGIS
        if step == SWITCH_STEP and late_aegis and not early_aegis and layer is not None:
            layer.reset(obs, prompt)
            layer_active = True

        qi = step // REPLAN_STEPS
        if step % REPLAN_STEPS == 0:
            z0, _ = sample_z(nominal_seed(SUITE, TASK_ID, state_id, stream, qi))
            po = R.obs_to_policy_obs(obs, prompt, image_size=IMAGE_SIZE)
            nominal = np.clip(policy.generate(z0, po), -1, 1)

        nom_a = nominal[step % REPLAN_STEPS]
        act_a = nom_a
        intervened = False
        if use_aegis and layer_active and layer is not None:
            act_a, info = layer.filter(nom_a, obs)
            intervened = bool(info.get("intervened"))
            if intervened:
                n_intervene += 1

        # 记录（执行前）
        soup_pos = body_pos(env, "alphabet_soup_1_main")
        eef = eef_pos(env)
        d_eef_soup = float(np.linalg.norm(eef - soup_pos))
        u = soup_pos - eef
        u = u / (np.linalg.norm(u) + 1e-9)
        g_nom = float(u @ (nom_a[:3] * TRANS_SCALE))

        obs, _r, done, _i = env.step(act_a)
        steps += 1

        disp = float(np.sum(np.abs(body_pos(env, name + "_main") - init_pos)))
        max_disp = max(max_disp, disp)
        sg = subgoal_values(env)

        if step % 10 == 0 or step < SWITCH_STEP + 10:
            trace.append({
                "step": step, "d_eef_soup": d_eef_soup, "g_nom": g_nom,
                "intervened": intervened, "disp": disp,
                "soup_pred": bool(sg.get("alphabet_soup_1")),
                "cream_pred": bool(sg.get("cream_cheese_1")),
            })

        if done or env.check_success():
            success = True
            break

    return {
        "success": success,
        "collided": max_disp > COLLISION_DISPLACEMENT_EPS,
        "max_displacement": max_disp,
        "steps": steps,
        "n_intervene": n_intervene,
        "final_subgoals": subgoal_values(env),
        "trace": trace,
    }


def main():
    from libero.libero import benchmark
    ts = benchmark.get_benchmark_dict()[SUITE](safety_level=LEVEL)
    task = ts.get_task(TASK_ID)
    prompt = task.language

    state_id, stream = 2, 0
    policy = PnsgPolicy()

    branches = {
        "V": (False, False),
        "A": (True, False),
        "B": (False, True),
        "F": (True, True),
    }

    print(f"=== C1：闭环反事实 state02/stream0（SWITCH_STEP={SWITCH_STEP}）===")
    results = {}
    try:
        for bname, (early, late) in branches.items():
            env = make_env(ts, task, state_id)
            layer = AegisSafetyLayer(env) if (early or late) else None
            r = run_c1_branch(policy, env, layer, state_id, stream, prompt, early, late)
            env.close()
            results[bname] = r
            print(f"  {bname}: success={r['success']} coll={r['collided']} "
                  f"Dmax={r['max_displacement']*1000:.2f}mm steps={r['steps']} "
                  f"int={r['n_intervene']} soup={r['final_subgoals'].get('alphabet_soup_1')} "
                  f"cream={r['final_subgoals'].get('cream_cheese_1')}", flush=True)
    finally:
        policy.close()

    out = pathlib.Path(ROOT, "data", "audit", "g1a_c1_counterfactual.json")
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str))
    print(f"\n结果: {out}")

    # 判决
    V = results["V"]; A = results["A"]; B = results["B"]; F = results["F"]
    print("\n=== 判决 ===")
    if V["success"] and not A["success"] and not F["success"]:
        print("V成功 A失败 F失败 → 早期安全干预历史足以造成该案例的闭环任务损失")
    elif V["success"] and A["success"] and not F["success"]:
        print("V成功 A成功 F失败 → 早期干预后任务可恢复；持续/后期过滤是关键")
    elif V["success"] and B["success"] and not F["success"]:
        print("V成功 B成功 F失败 → 后期单独启用AEGIS未必破坏任务，早期干预更重要")
    elif V["success"] and not B["success"]:
        print("V成功 B失败 → 后期干预也破坏任务，不能归因于唯一早期窗口")
    else:
        print("V失败或模式不匹配，见明细")

    # A 分支 step40 后是否恢复接近
    a_trace = A["trace"]
    post_switch = [t for t in a_trace if t["step"] >= SWITCH_STEP]
    early_d = [t["d_eef_soup"] for t in a_trace if t["step"] < SWITCH_STEP]
    late_d = [t["d_eef_soup"] for t in post_switch]
    if early_d and late_d:
        print(f"\nA分支 d_eef_soup: 前40步末={early_d[-1]:.3f} → 后段末={late_d[-1]:.3f}")
        if late_d[-1] < early_d[-1] - 0.02:
            print("A分支 step40 后恢复接近")
        else:
            print("A分支 step40 后未恢复接近")


if __name__ == "__main__":
    main()
