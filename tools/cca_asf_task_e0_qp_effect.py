"""CCA-ASF TASK E0：共同 QP 快照对照（局部各向异性机制验证）。

从真实 π0.5 + AEGIS 运行中采集「发生安全干预」的 QP 快照（u_nom/A_row/r/h/d_y），
然后在同一快照上分别解欧氏（W0）与 CCA（W0+ΔP）QP，对比：

  D_∥ = |d_y^T B (u* - u_nom)|     （任务方向偏移）
  D_⊥ = ||(I - d_y d_y^T) B (u* - u_nom)||_2  （正交扰动）

统计：β>0 比例、Δu≠0 比例、ΔD_∥ 的符号分布。
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
from src.safety.cca_asf.metric import B_DIAG, compute_conflict_beta, map_dz_to_dy  # noqa: E402
from src.safety.cca_asf.intention import chunk_direction  # noqa: E402
from tools.g1a_soup_trace import make_env  # noqa: E402

SUITE = "safelibero_long"
LEVEL = "I"
TASK_ID = 0
STATE_ID = 2
STREAM = 0
MAX_STEPS = 120
REPLAN = 5
IMAGE_SIZE = 256
BETA_MAX = 4.0

W0 = np.diag([1.0 / 25] * 6 + [1.0, 1.0, 1.0])


def solve_qp(W, u_nom, A_row, r):
    """解 min (u-u_nom)^T W (u-u_nom) s.t. A_row @ u >= r。返回 u* 或 None。"""
    u = cp.Variable(9)
    obj = cp.Minimize(cp.quad_form(u - u_nom, W))
    con = [A_row @ u >= r]
    prob = cp.Problem(obj, con)
    try:
        prob.solve(solver=cp.OSQP)
    except Exception:
        return None
    if u.value is None:
        return None
    return np.asarray(u.value, dtype=float)


def dev_components(u_sol, u_nom, d_y):
    """计算 D_∥ 与 D_⊥（白化空间）。d_y 为 9D 单位白化方向。"""
    diff = B_DIAG * (u_sol - u_nom)  # 白化偏移
    dpar = abs(float(d_y @ diff))
    dperp = float(np.linalg.norm(diff - (d_y @ diff) * d_y))
    return dpar, dperp


def main():
    from libero.libero import benchmark
    ts = benchmark.get_benchmark_dict()[SUITE](safety_level=LEVEL)
    task = ts.get_task(TASK_ID)
    prompt = task.language

    pol = Pi05Policy(verbose=False, replan_steps=REPLAN, image_size=IMAGE_SIZE)
    env = make_env(ts, task, STATE_ID)
    # 用 cca_asf 模式跑，才能得到 d_y / beta 快照；离线对比欧氏 vs CCA
    layer = AegisSafetyLayer(env, filter_mode="cca_asf", beta_max=BETA_MAX)
    layer.record_qp_snapshot = True
    layer.reset(env.env._get_observations(), prompt)
    pol.reset()

    snapshots = []  # 收集有干预的 QP 快照
    for step in range(MAX_STEPS):
        obs = env.env._get_observations()
        action = pol.act(obs, prompt)
        chunk = pol.get_current_chunk()
        idx = pol.buffer_index - 1
        # 任务方向（从 chunk 提取）
        d_z = chunk_direction(chunk[max(0, idx):]) if chunk is not None else None
        layer.set_task_direction(d_z)  # euclidean 模式会忽略，但保留 d_z 供快照

        exec_a, info = layer.filter(action, obs)
        # 采集快照（只要 QP 活跃且 d_y 存在）
        snap = info.get("qp_snapshot")
        if snap is not None and snap.get("d_y") is not None:
            # 用当前 action 重建 u_nom 参考（快照里的 u_nom 已含 z 方向）
            u_nom = snap["u_nom"]
            A_row = snap["A_row"]
            r = snap["r"]
            d_y = snap["d_y"]
            # 欧氏解
            u_eu = solve_qp(W0, u_nom, A_row, r)
            # CCA 解（用快照的 beta；若无 beta 重算）
            beta = snap.get("beta", 0.0)
            if beta <= 0.0:
                beta = compute_conflict_beta(A_row[None, :], np.array([r]), u_nom, d_y,
                                             beta_max=BETA_MAX)
            if beta > 0.0:
                Bd = B_DIAG * d_y
                Wcca = W0 + beta * np.outer(Bd, Bd)
                u_cca = solve_qp(Wcca, u_nom, A_row, r)
                if u_eu is not None and u_cca is not None:
                    dpar_eu, dperp_eu = dev_components(u_eu, u_nom, d_y)
                    dpar_cca, dperp_cca = dev_components(u_cca, u_nom, d_y)
                    snapshots.append({
                        "step": step,
                        "beta": beta,
                        "dpar_eu": dpar_eu, "dperp_eu": dperp_eu,
                        "dpar_cca": dpar_cca, "dperp_cca": dperp_cca,
                        "ddpar": dpar_cca - dpar_eu,
                        "du_diff": float(np.linalg.norm(u_cca - u_eu)),
                    })
        env.step(exec_a)

    pol.close()
    env.close()

    # ---- 统计 ----
    print(f"=== E0：共同 QP 快照对照（{len(snapshots)} 个有干预快照）===")
    if not snapshots:
        print("无快照（本段无安全干预）")
        return

    n_beta_gt0 = sum(1 for s in snapshots if s["beta"] > 0)
    n_du_gt0 = sum(1 for s in snapshots if s["du_diff"] > 1e-9)
    n_ddpar_lt0 = sum(1 for s in snapshots if s["ddpar"] < -1e-9)
    n_ddpar_gt0 = sum(1 for s in snapshots if s["ddpar"] > 1e-9)
    n_ddpar_eq0 = sum(1 for s in snapshots if abs(s["ddpar"]) <= 1e-9)

    print(f"  β>0 快照数: {n_beta_gt0}/{len(snapshots)}")
    print(f"  Δu≠0 快照数: {n_du_gt0}/{len(snapshots)}")
    print(f"  ΔD_∥ < 0（CCA 减少任务方向偏移）: {n_ddpar_lt0}")
    print(f"  ΔD_∥ > 0: {n_ddpar_gt0}")
    print(f"  ΔD_∥ = 0: {n_ddpar_eq0}")
    if snapshots:
        ddpars = [s["ddpar"] for s in snapshots if s["du_diff"] > 1e-9]
        if ddpars:
            print(f"  ΔD_∥（Δu≠0 快照）: min={min(ddpars):.6f} max={max(ddpars):.6f} "
                  f"mean={np.mean(ddpars):.6f}")

    out = {
        "n_snapshots": len(snapshots),
        "n_beta_gt0": n_beta_gt0,
        "n_du_gt0": n_du_gt0,
        "n_ddpar_lt0": n_ddpar_lt0,
        "n_ddpar_gt0": n_ddpar_gt0,
        "n_ddpar_eq0": n_ddpar_eq0,
        "snapshots": snapshots,
    }
    out_path = os.path.join(ROOT, "data", "audit", "cca_asf_task_e0_qp_effect.json")
    with open(out_path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果: {out_path}")


if __name__ == "__main__":
    main()
