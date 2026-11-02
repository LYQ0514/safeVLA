"""M3-P1a Gate 1 探测：真实 AEGIS 干预状态下的候选构造可行性。

目标：确认
1. 真实 AEGIS 在哪些 step 发生干预（intervened=True），干预时约束几何；
2. 能否构造「当前修正更大」的 9D 安全候选 u_B（替代 AEGIS QP 解 u_A）。

候选构造（预注册规则，非特权 Oracle）：
- u_A = AEGIS QP 解（贴 CBF 边界）；
- u_B = u_A + β · v_null，其中 v_null = 任务方向 d_y 在约束零空间的分量
  （= d_y - (d_y·a_full)/(||a_full||²)·a_full，即「不改变约束满足度的切向」）；
- β 取正值，使 u_B 偏离名义 u_ref 更远（当前修正更大）。
"""

from __future__ import annotations

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
from src.safety.cca_asf.intention import chunk_direction  # noqa: E402
from src.safety.cca_asf.metric import map_dz_to_dy  # noqa: E402
from tools.g1a_soup_trace import make_env  # noqa: E402

SUITE = "safelibero_long"
LEVEL = "I"
TASK_ID = 0
STATE_ID = 1
STREAM = 3
REPLAN = 5
IMAGE_SIZE = 256

W = np.diag([1.0 / 25] * 6 + [1.0, 1.0, 1.0])


def _R1_from_obs(obs):
    from src.safety.aegis_adapted import _quat_wxyz_to_rot
    return _quat_wxyz_to_rot(np.asarray(obs["robot0_eef_quat"], dtype=np.float64))


def solve_u(Wm, u_ref, a_full, r):
    """解 min (u-u_ref)^T Wm (u-u_ref) s.t. a_full @ u >= r。返回 u* 或 None。"""
    u = cp.Variable(9)
    obj = cp.Minimize(cp.quad_form(u - u_ref, Wm))
    con = [a_full @ u >= r]
    prob = cp.Problem(obj, con)
    try:
        prob.solve(solver=cp.OSQP)
    except Exception:
        return None
    if u.value is None:
        return None
    return np.asarray(u.value, dtype=float)


def main():
    from libero.libero import benchmark
    ts = benchmark.get_benchmark_dict()[SUITE](safety_level=LEVEL)
    task = ts.get_task(TASK_ID)
    prompt = task.language

    pol = Pi05Policy(verbose=False, replan_steps=REPLAN, image_size=IMAGE_SIZE)
    env = make_env(ts, task, STATE_ID)
    layer = AegisSafetyLayer(env, filter_mode="euclidean")
    layer.record_qp_snapshot = True
    layer.reset(env.env._get_observations(), prompt)
    pol.reset()

    n_intervene = 0
    n_constructible = 0
    n_cur_cost_larger = 0
    print(f"=== Gate 1 探测（S{STATE_ID:02d}/st{STREAM}）===", flush=True)

    for step in range(300):
        obs = env.env._get_observations()
        action = pol.act(obs, prompt)
        chunk = pol.get_current_chunk()
        idx = pol.buffer_index - 1
        d_z = chunk_direction(chunk[max(0, idx):]) if chunk is not None else None

        # 用 euclidean filter（AEGIS 原版），手动算 QP 解 + 构造候选
        exec_a, info = layer.filter(action, obs)
        snap = info.get("qp_snapshot")
        intervened = info.get("intervened") is True

        # 手动算 d_y（不依赖 filter_mode，从 chunk 提取任务方向）
        if intervened and snap is not None:
            # 需要 d_y：从 chunk 提取 d_z，映射到 9D 白化方向
            if d_z is None:
                continue
            d_y = map_dz_to_dy(d_z, _R1_from_obs(obs))
            if d_y is None:
                continue
            n_intervene += 1
            u_ref = snap["u_nom"]
            a_full = snap["A_row"]
            r = snap["r"]

            # u_A = AEGIS QP 解（W 目标）
            u_A = solve_u(W, u_ref, a_full, r)
            if u_A is None:
                continue

            # 任务方向在约束零空间的分量
            a_norm_sq = float(a_full @ a_full)
            if a_norm_sq < 1e-12:
                continue
            v_null = d_y - (d_y @ a_full) / a_norm_sq * a_full
            vn = float(np.linalg.norm(v_null))
            if vn < 1e-6:
                continue  # 任务方向完全与约束法向对齐，无零空间分量

            v_null = v_null / vn  # 单位切向

            # u_B = u_A + β·v_null，β 使当前修正更大（偏离 u_ref 更远）
            dev_A = float(np.linalg.norm(u_A - u_ref))
            # 扫描 β，找「当前修正更大」且「仍满足约束」的 u_B
            found = False
            for beta in [0.5, 1.0, 1.5, 2.0]:
                u_B = u_A + beta * v_null
                if a_full @ u_B >= r - 1e-9:  # 仍满足约束
                    dev_B = float(np.linalg.norm(u_B - u_ref))
                    if dev_B > dev_A + 1e-6:
                        found = True
                        n_constructible += 1
                        n_cur_cost_larger += 1
                        break
            if found and n_constructible <= 3:
                print(f"  step{step}: dev_A={dev_A:.4f} dev_B={dev_B:.4f} "
                      f"a^T u_A={a_full@u_A:.3f}(r={r:.3f}) a^T u_B={a_full@u_B:.3f} "
                      f"v_null与d_y夹角cos={v_null@d_y:.3f}", flush=True)

        env.step(exec_a)

    print(f"\n干预步数: {n_intervene}")
    print(f"可构造「当前修正更大」候选的步数: {n_constructible}/{n_intervene}")
    pol.close()
    env.close()


if __name__ == "__main__":
    main()
