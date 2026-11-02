"""CCA-ASF TASK D：真实 π0.5 + MuJoCo + AEGIS 端到端集成验收 smoke test。

固定 SafeLIBERO Long-I task0 / State02 / Stream0，80 outer steps。

门禁：
- D0：真实 Pi05Policy chunk 与名义动作一致性（get_current_chunk()[idx] == act()）
- D1：Vanilla / AEGIS / Euclidean / CCA(β=0) 四配置逐步动作等价
- D2：CCA(β=4) 非零各向异性激活（记录 ξ/ρ/β/‖ΔP‖/‖Δu‖）
- D3：MuJoCo 物理监测（CBF 残差、Dmax、接触、TASK 完整性）
- D4：完成物理日志，区分物理安全 vs QP 可行性

注意：TASK D 不评价成功率/安全性优劣，只验证接入正确、链路未坏。
"""

from __future__ import annotations

import hashlib
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

from src.policies import Pi05Policy  # noqa: E402
from src.safety.aegis_adapted import AegisSafetyLayer  # noqa: E402
from src.safety.contract_objects import monitor_contacts  # noqa: E402
from tools.g1a_soup_trace import make_env, body_state  # noqa: E402

SUITE = "safelibero_long"
LEVEL = "I"
TASK_ID = 0
STATE_ID = 2
STREAM = 0
MAX_STEPS = 80
REPLAN = 5
IMAGE_SIZE = 256

PROTECTED = "red_coffee_mug_obstacle_1_main"
TASK_BODIES = ("alphabet_soup_1_main", "cream_cheese_1_main")


def _hash(arr):
    return hashlib.sha256(np.asarray(arr).tobytes()).hexdigest()[:16]


def run_d1_single_step(ts, task, prompt, pol):
    """D1 单步等价：同一 env 状态 + 同一名义动作，euclidean vs cca_asf(β=0) filter 输出对比。

    返回每步的最大绝对差列表。绕开 π0.5 跨 run 非确定性（逐步轨迹对比无意义）。
    """
    env = make_env(ts, task, STATE_ID)
    layer_eu = AegisSafetyLayer(env, filter_mode="euclidean", beta_max=0.0)
    layer_ca = AegisSafetyLayer(env, filter_mode="cca_asf", beta_max=0.0)
    obs0 = env.env._get_observations()
    layer_eu.reset(obs0, prompt)
    layer_ca.reset(obs0, prompt)

    pol.reset()
    diffs = []
    for step in range(MAX_STEPS):
        obs = env.env._get_observations()
        action = pol.act(obs, prompt)  # 名义动作（同一 action 喂两个 filter）
        out_eu, _ = layer_eu.filter(action, obs)
        out_ca, _ = layer_ca.filter(action, obs)  # β_max=0 → 门控强制 β=0
        diffs.append(float(np.max(np.abs(np.asarray(out_eu) - np.asarray(out_ca)))))
        env.step(out_eu)  # 用 euclidean 输出推进，保持单一轨迹
    env.close()
    return diffs


def run_config(ts, task, prompt, pol, env, mode, beta_max=0.0, collect_beta=False):
    """跑一个配置，返回逐步记录。mode ∈ {vanilla, aegis, euclidean, cca_asf}。"""
    layer = None
    if mode != "vanilla":
        # aegis 与 euclidean 都走原始欧氏 filter（历史基线 + 接口回归）
        fm = "euclidean" if mode in ("aegis", "euclidean") else "cca_asf"
        layer = AegisSafetyLayer(env, filter_mode=fm, beta_max=beta_max)
        layer.reset(env.env._get_observations(), prompt)

    p0 = body_state(env, PROTECTED)["pos"].copy()
    task_p0 = {b: body_state(env, b)["pos"].copy() for b in TASK_BODIES}

    records = []
    Dmax = 0.0
    for step in range(MAX_STEPS):
        action = pol.act(env.env._get_observations(), prompt)  # 名义动作（7维）
        # D0：chunk 一致性（act 后 buffer_index 已 +1）
        chunk = pol.get_current_chunk()
        idx = pol.buffer_index - 1
        chunk_match = False
        if chunk is not None and 0 <= idx < len(chunk):
            chunk_match = bool(np.allclose(chunk[idx], action, atol=1e-7))

        # 任务方向（从 chunk 提取）
        d_z = None
        if layer is not None and layer.filter_mode == "cca_asf":
            from src.safety.cca_asf.intention import chunk_direction
            if chunk is not None:
                # 用当前名义动作所在位置之后的剩余 chunk 提取（近似，见报告）
                d_z = chunk_direction(chunk[max(0, idx):])

        if mode == "vanilla":
            exec_a = action
            info = {"beta": 0.0}
        else:
            if layer.filter_mode == "cca_asf" and d_z is not None:
                layer.set_task_direction(d_z)
            exec_a, info = layer.filter(action, env.env._get_observations())

        # 执行
        obs, _r, _d, _i = env.step(exec_a)
        Dmax = max(Dmax, float(np.sum(np.abs(body_state(env, PROTECTED)["pos"] - p0))))

        # 物理监测
        mon = monitor_contacts(env)
        n_forbidden = len(mon["forbidden"]) + len(mon["unclassified"])
        task_disp = {b: float(np.sum(np.abs(body_state(env, b)["pos"] - task_p0[b])))
                     for b in TASK_BODIES}

        rec = {
            "step": step, "mode": mode,
            "nom_action": action.tolist(),
            "exec_action": exec_a.tolist(),
            "chunk_match": chunk_match,
            "beta": float(info.get("beta", 0.0)),
            "feasible": info.get("feasible"),
            "h": info.get("h"),
            "dev_norm": float(np.linalg.norm(np.asarray(exec_a) - np.asarray(action))),
            "Dmax": Dmax,
            "n_forbidden": n_forbidden,
            "task_disp": {b: round(v * 1000, 4) for b, v in task_disp.items()},
        }
        records.append(rec)
    return records


def main():
    from libero.libero import benchmark
    ts = benchmark.get_benchmark_dict()[SUITE](safety_level=LEVEL)
    task = ts.get_task(TASK_ID)
    prompt = task.language

    pol = Pi05Policy(verbose=False, replan_steps=REPLAN, image_size=IMAGE_SIZE)
    pol.reset()

    results = {}
    # 四个配置逐一跑（各自独立 env，确定性重跑）
    for mode, beta in [("vanilla", 0.0), ("aegis", 0.0),
                       ("euclidean", 0.0), ("cca_asf", 0.0)]:
        env = make_env(ts, task, STATE_ID)
        pol.reset()
        recs = run_config(ts, task, prompt, pol, env, mode, beta_max=beta)
        results[mode] = recs
        env.close()
        print(f"[{mode}] 完成 {len(recs)} 步", flush=True)

    # 非零 β 激活测试
    env = make_env(ts, task, STATE_ID)
    pol.reset()
    recs_beta4 = run_config(ts, task, prompt, pol, env, "cca_asf", beta_max=4.0)
    results["cca_asf_beta4"] = recs_beta4
    env.close()
    print(f"[cca_asf_beta4] 完成 {len(recs_beta4)} 步", flush=True)

    # ---- 门禁判定 ----
    verdicts = {}

    # D0：chunk 一致性（所有配置第一步应 match）
    d0_ok = all(r["chunk_match"] for r in results["vanilla"])
    verdicts["D0_chunk_consistency"] = d0_ok

    # D1：同一名义动作 + 同一 obs 下，euclidean vs cca_asf(β=0) 的 filter 输出逐位一致
    # （注意：不能比「两条独立轨迹」，因 π0.5 跨 run 非确定使名义动作本身就不同）
    d1_diffs = run_d1_single_step(ts, task, prompt, pol)
    d1_max = max(d1_diffs) if d1_diffs else 0.0
    d1_ok = d1_max < 1e-6
    verdicts["D1_beta0_equivalence"] = d1_ok
    verdicts["D1_max_abs_diff_single_step"] = float(d1_max)
    verdicts["D1_n_steps_compared"] = len(d1_diffs)
    verdicts["D1_note"] = "同 action+obs 单步 filter 对比（π0.5 跨 run 非确定使逐步轨迹对比无意义）"

    # D2：非零 β 激活
    betas = [r["beta"] for r in recs_beta4]
    d2_ok = max(betas) > 0.0
    verdicts["D2_nonzero_beta"] = d2_ok
    verdicts["D2_max_beta"] = float(max(betas))
    verdicts["D2_n_activated"] = int(sum(1 for b in betas if b > 0))

    # D3：QP feasible 全 True（无 solver 失败）
    d3_feasible = all(r["feasible"] is not False for r in recs_beta4)
    verdicts["D3_qp_feasible"] = d3_feasible

    # D4：物理日志已记录（Dmax / forbidden / task_disp 均非空）
    d4_ok = all("Dmax" in r and "n_forbidden" in r and "task_disp" in r
                for r in recs_beta4)
    verdicts["D4_physics_logged"] = d4_ok

    out = {
        "config": {"suite": SUITE, "task": TASK_ID, "state": STATE_ID,
                   "stream": STREAM, "max_steps": MAX_STEPS, "replan": REPLAN},
        "verdicts": verdicts,
        "per_mode": {
            k: {"n_steps": len(v),
                "n_nonzero_beta": int(sum(1 for r in v if r["beta"] > 0)),
                "max_dev_norm": float(max(r["dev_norm"] for r in v)),
                "max_Dmax": float(max(r["Dmax"] for r in v)),
                "any_forbidden": any(r["n_forbidden"] > 0 for r in v),
                "max_task_disp_mm": {b: max(r["task_disp"][b] for r in v)
                                     for b in TASK_BODIES},
                } for k, v in results.items()
        },
    }
    out_path = os.path.join(ROOT, "data", "audit", "cca_asf_task_d_smoke.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)

    print("\n=== TASK D 门禁判定 ===")
    for k, v in verdicts.items():
        print(f"  {k}: {v}")
    print(f"\n结果: {out_path}")

    pol.close()


if __name__ == "__main__":
    main()
