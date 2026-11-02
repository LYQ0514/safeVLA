"""M3-P1a-R：阶段感知的恢复负担与任务价值重分析（细粒度日志重采集）。

只重跑 M3-P1a 的 8 对配对（不移动检查点、不改变候选、不改变配对），但补全日志：
1. N_H（未来干预次数）与 D_H（未来累计修正负担）分离；
2. 每步 CBF 激活状态（h 值、interven、连续激活长度）→ 判断 Entry vs Persistent；
3. 任务事件（soup 抬升 / check_success）→ 判断任务是否可判，否则 INCONCLUSIVE。

复用 m3_p1a_burden_transfer.py 的核心函数。
"""

from __future__ import annotations

import json
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
import cvxpy as cp  # noqa: E402

from src.policies import Pi05Policy  # noqa: E402
from src.safety.aegis_adapted import AegisSafetyLayer  # noqa: E402
from src.safety.contract_objects import monitor_contacts  # noqa: E402
from src.safety.cca_asf.intention import chunk_direction  # noqa: E402
from src.safety.cca_asf.metric import map_dz_to_dy  # noqa: E402
from tools.g1a_soup_trace import make_env, body_state, eef_pos  # noqa: E402
from tools.m1_p2_p1a_pilot import save_full, restore_full  # noqa: E402
from tools.m3_p1a_burden_transfer import (  # noqa: E402
    solve_u, u_to_action, _u_nom, _constraint,
    SUITE, LEVEL, TASK_ID, REPLAN, IMAGE_SIZE, H, GAMMA, W,
    PROTECTED, SOUP_BODY, CAR_EPS, STATES, STREAMS,
)

EPS_INTERV = 1e-4  # 干预判定阈值（W 范数）


def run_branch_detailed(ts, task, prompt, pol, env, layer, parent_state, u_first_9,
                        action7, R1, p0, n_steps, use_u_first):
    """从父状态分叉，跑 n_steps，返回细粒度日志。"""
    restore_full(env, layer, parent_state)
    obs = env.env._get_observations(force_update=True)
    pol.reset()

    D_H = 0.0
    N_H = 0
    n_forbidden = 0
    Dmax = 0.0
    success = False
    h_list = []
    interv_list = []
    step_burden_list = []
    soup_lift_step = None
    soup_z0 = body_state(env, SOUP_BODY)["pos"][2]

    for step in range(n_steps):
        action = pol.act(obs, prompt)
        if step == 0 and use_u_first:
            exec_a = u_to_action(u_first_9, action, R1)
            u_nom = _u_nom(action, obs, layer)
            a_full, r = _constraint(action, obs, layer)
            # 第一步用 u_first，其「修正」是 u_first - u_nom（A 的修正）
            u_safe = u_first_9
            interv = False
            h_val = layer._h_last if hasattr(layer, '_h_last') else None
        else:
            exec_a, info = layer.filter(action, obs)
            u_nom = _u_nom(action, obs, layer)
            a_full, r = _constraint(action, obs, layer)
            u_safe = solve_u(W, u_nom, a_full, r) if a_full is not None else u_nom
            interv = info.get("intervened") is True
            h_val = info.get("h")

        # 单步负担（仅后续 AEGIS 步计入 D_H；第一步不算，因为是「当前选择」）
        if u_safe is not None and (step > 0 or not use_u_first):
            dev = u_safe - u_nom
            b = float(dev @ W @ dev)
            step_burden_list.append(b)
            D_H += (GAMMA ** (step - 1)) * b
            if b > EPS_INTERV:
                N_H += 1
        else:
            step_burden_list.append(0.0)

        h_list.append(h_val)
        interv_list.append(interv)

        obs, _r, _d, _i = env.step(exec_a)
        mon = monitor_contacts(env)
        n_forbidden += len(mon["forbidden"]) + len(mon["unclassified"])
        Dmax = max(Dmax, float(np.sum(np.abs(body_state(env, PROTECTED)["pos"] - p0))))

        # soup 抬升（合法抓取代理事件）
        if soup_lift_step is None and (body_state(env, SOUP_BODY)["pos"][2] - soup_z0) > 0.02:
            soup_lift_step = step

        if env.check_success():
            success = True
            break

    S_H = (Dmax < CAR_EPS) and (n_forbidden == 0)
    return {
        "D_H": D_H, "N_H": N_H, "S_H": S_H, "success": success,
        "h_list": h_list, "interv_list": interv_list,
        "step_burden_list": step_burden_list,
        "soup_lift_step": soup_lift_step, "n_forbidden": n_forbidden, "Dmax": Dmax,
    }


def main():
    from libero.libero import benchmark
    ts = benchmark.get_benchmark_dict()[SUITE](safety_level=LEVEL)
    task = ts.get_task(TASK_ID)
    prompt = task.language

    all_pairs = []

    for state_id in STATES:
        for stream in STREAMS:
            pol = Pi05Policy(verbose=False, replan_steps=REPLAN, image_size=IMAGE_SIZE)
            env = make_env(ts, task, state_id)
            layer = AegisSafetyLayer(env, filter_mode="euclidean")
            layer.reset(env.env._get_observations(), prompt)
            pol.reset()

            p0 = body_state(env, PROTECTED)["pos"].copy()

            # 跑 AEGIS 轨迹，收集干预步 + 连续激活长度
            intervention_states = []
            run_interv_seq = []  # 每步是否 interven（判断连续激活）
            for step in range(300):
                obs = env.env._get_observations()
                action = pol.act(obs, prompt)
                exec_a, info = layer.filter(action, obs)
                run_interv_seq.append(info.get("intervened") is True)
                if info.get("intervened") is True:
                    from src.safety.aegis_adapted import _quat_wxyz_to_rot
                    R1 = _quat_wxyz_to_rot(np.asarray(obs["robot0_eef_quat"], dtype=np.float64))
                    u_nom = _u_nom(action, obs, layer)
                    a_full, r = _constraint(action, obs, layer)
                    u_A = solve_u(W, u_nom, a_full, r)
                    if u_A is not None:
                        d_z = chunk_direction(pol.get_current_chunk()[max(0, pol.buffer_index-1):]) \
                            if pol.get_current_chunk() is not None else None
                        if d_z is not None:
                            d_y = map_dz_to_dy(d_z, R1)
                            if d_y is not None:
                                a_norm_sq = float(a_full @ a_full)
                                v_null = d_y - (d_y @ a_full) / max(a_norm_sq, 1e-12) * a_full
                                vn = float(np.linalg.norm(v_null))
                                if vn > 1e-6:
                                    v_null = v_null / vn
                                    dev_A = float(np.linalg.norm(u_A - u_nom))
                                    u_B = None
                                    for beta in [0.5, 1.0, 1.5, 2.0]:
                                        cand = u_A + beta * v_null
                                        if a_full @ cand >= r - 1e-9:
                                            if float(np.linalg.norm(cand - u_nom)) > dev_A + 1e-6:
                                                u_B = cand
                                                break
                                    if u_B is not None:
                                        # 连续激活长度：当前步往前数连续 interven 的次数
                                        run_len = 0
                                        for k in range(len(run_interv_seq) - 1, -1, -1):
                                            if run_interv_seq[k]:
                                                run_len += 1
                                            else:
                                                break
                                        intervention_states.append({
                                            "step": step,
                                            "run_len": run_len,  # 连续激活长度（≥1 首次，>1 持续）
                                            "parent": save_full(env, layer),
                                            "u_A": u_A, "u_B": u_B,
                                            "action": action.copy(), "R1": R1.copy(),
                                        })
                env.step(exec_a)

            n_int = len(intervention_states)
            if n_int == 0:
                print(f"S{state_id:02d}/st{stream}: 无干预", flush=True)
                pol.close(); env.close()
                continue

            for off in [0, 5]:
                if off >= n_int:
                    continue
                cs = intervention_states[off]
                res_A = run_branch_detailed(ts, task, prompt, pol, env, layer, cs["parent"],
                                            cs["u_A"], cs["action"], cs["R1"], p0, H * REPLAN, True)
                res_B = run_branch_detailed(ts, task, prompt, pol, env, layer, cs["parent"],
                                            cs["u_B"], cs["action"], cs["R1"], p0, H * REPLAN, True)

                # 任务进度判断：用 soup 抬升事件 + success（而非 eef 距离）
                task_A = "INCONCLUSIVE"
                task_B = "INCONCLUSIVE"
                if res_A["soup_lift_step"] is not None or res_A["success"]:
                    task_A = "PROGRESS"
                if res_B["soup_lift_step"] is not None or res_B["success"]:
                    task_B = "PROGRESS"

                reversal_D = res_A["D_H"] > res_B["D_H"] + 1e-9
                reversal_N = res_A["N_H"] > res_B["N_H"]

                pair = {
                    "state_id": state_id, "stream": stream,
                    "checkpoint_step": cs["step"], "run_len": cs["run_len"],
                    "D_A": res_A["D_H"], "D_B": res_B["D_H"],
                    "N_A": res_A["N_H"], "N_B": res_B["N_H"],
                    "S_A": res_A["S_H"], "S_B": res_B["S_H"],
                    "task_A": task_A, "task_B": task_B,
                    "soup_lift_A": res_A["soup_lift_step"], "soup_lift_B": res_B["soup_lift_step"],
                    "reversal_D": reversal_D, "reversal_N": reversal_N,
                    "h_A": res_A["h_list"][:H * REPLAN],
                    "h_B": res_B["h_list"][:H * REPLAN],
                }
                all_pairs.append(pair)
                print(f"S{state_id:02d}/st{stream} @{cs['step']}(run_len={cs['run_len']}): "
                      f"D_A={res_A['D_H']:.4f} D_B={res_B['D_H']:.4f} "
                      f"N_A={res_A['N_H']} N_B={res_B['N_H']} "
                      f"反转D={reversal_D} 反转N={reversal_N} "
                      f"task={task_A}/{task_B}", flush=True)

            pol.close()
            env.close()

    # 汇总：按 run_len 分组看反转
    entry = [p for p in all_pairs if p["run_len"] <= 1]
    persistent = [p for p in all_pairs if p["run_len"] > 1]
    print(f"\n=== M3-P1a-R 重分析 ===")
    print(f"Entry（首次干预，run_len<=1）: {len(entry)} 对")
    print(f"  反转D: {sum(1 for p in entry if p['reversal_D'])}/{len(entry)}")
    print(f"  反转N: {sum(1 for p in entry if p['reversal_N'])}/{len(entry)}")
    print(f"Persistent（持续干预，run_len>1）: {len(persistent)} 对")
    print(f"  反转D: {sum(1 for p in persistent if p['reversal_D'])}/{len(persistent)}")
    print(f"  反转N: {sum(1 for p in persistent if p['reversal_N'])}/{len(persistent)}")

    out = {"pairs": all_pairs}
    out_path = os.path.join(ROOT, "data", "audit", "m3_p1a_stage_reanalysis.json")
    with open(out_path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果: {out_path}")


if __name__ == "__main__":
    main()
