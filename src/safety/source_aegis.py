"""Source steering + AEGIS 结合策略（最简版，冒烟用）。

流程（propose-then-filter）：
- 每个重规划点：z0 = nominal_seed 采样，nominal = generate(z0)
- 检测危险（球体代理 selected_constraints 非空）
- 危险时：解 min‖Δz‖ s.t. C·J·Δz ≥ b（复用 PNSG 的源空间求解器），
  z_cand = z0 + Δz，steered = generate(z_cand)
- 取当前步动作，经 AEGIS CBF-QP 过滤后执行

这是「我的方法（源空间 steering）+ AEGIS」的最简串接，仅用于冒烟验证链路，
不作为方法性能证据。
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

import numpy as np  # noqa: E402
import pi05_libero_runner as R  # noqa: E402

from src.pnsg.crn import nominal_seed, sample_z  # noqa: E402
from src.pnsg.geometry_proxy import (  # noqa: E402
    build_control_points, build_obstacles, selected_constraints,
)
from src.pnsg.source_solver import minimum_norm_halfspaces  # noqa: E402
from src.safety.aegis_adapted import AegisSafetyLayer  # noqa: E402

H_FLOOR = 0.005
MAX_CONSTRAINTS = 3
MARGIN = 0.01


class SourceAegisPolicy:
    """源空间 steering 提案 + AEGIS CBF-QP 过滤。"""

    def __init__(self, env, suite, task_id, image_size=224, replan_steps=5,
                 replicate_id=0):
        from src.pnsg.policy_adapter import PnsgPolicy

        self.env = env
        self.suite = suite
        self.task_id = task_id
        self.image_size = image_size
        self.replan_steps = replan_steps
        self.replicate_id = replicate_id

        self.policy = PnsgPolicy()
        self.layer = AegisSafetyLayer(env)

        self._queue = []
        self._episode_idx = 0
        self._query = 0

        self.steer_count = 0
        self.steer_infeasible = 0
        self.danger_detect_count = 0

    def reset(self, obs, prompt="", episode_idx=0):
        self.layer.reset(obs, prompt)
        self._queue = []
        self._episode_idx = episode_idx
        self._query = 0

    def _danger_rows(self, obs, actions_pose):
        """球体代理检测：只返回真正违反的约束（b = h_floor - g > 0，即净空 < h_floor）。

        selected_constraints 用 margin 会额外选入「接近但未违反」的约束（b ≤ 0），
        那些不需要修正；只有 b > 0 才是真正危险。
        """
        cps = build_control_points(self.env, obs)
        obstacles = build_obstacles(self.env, obs)
        rows = selected_constraints(cps, obstacles, actions_pose,
                                    h_floor=H_FLOOR,
                                    max_constraints=MAX_CONSTRAINTS,
                                    margin=MARGIN)
        return [(c, bb) for c, bb in rows if bb > 1e-9]

    def _replan(self, obs, prompt):
        """重规划：生成 nominal，危险时做源空间 steering，返回 chunk。"""
        z0, _ = sample_z(nominal_seed(self.suite, self.task_id,
                                      self._episode_idx, self.replicate_id, self._query))
        po = R.obs_to_policy_obs(obs, prompt, image_size=self.image_size)
        nominal = np.clip(self.policy.generate(z0, po), -1, 1)

        chunk = nominal
        rows = self._danger_rows(obs, nominal[:, :6])
        if rows:
            self.danger_detect_count += 1
            C = np.vstack([c for c, _ in rows])
            b = np.array([bb for _, bb in rows])
            # 源空间 Jacobian（仅危险时计算，jacfwd ~7s）
            _, _, _, ext_J = self.policy.generate_with_jacobian(z0, po)
            A_src = C @ ext_J
            dz, _ = minimum_norm_halfspaces(A_src, b)
            if self.danger_detect_count == 1:
                # 第一次危险时打印诊断，确认不是维度/符号 bug
                print(f"[source_aegis] 首次危险: {len(rows)} 约束, "
                      f"C shape={C.shape}, ext_J shape={ext_J.shape}, "
                      f"A_src shape={A_src.shape}, b={np.round(b, 4)}, "
                      f"A_src 行范数={np.round(np.linalg.norm(A_src, axis=1), 3)}")
            if dz is not None and float(np.linalg.norm(dz)) > 1e-9:
                z_cand = z0 + dz.reshape(z0.shape)
                steered = np.clip(self.policy.generate(z_cand, po), -1, 1)
                chunk = steered
                self.steer_count += 1
            else:
                self.steer_infeasible += 1
                if self.steer_infeasible == 1:
                    print(f"[source_aegis] 首次 infeasible: b={np.round(b, 4)}, "
                          f"b>0 的约束数={int(np.sum(b > 1e-9))}")

        self._query += 1
        return chunk

    def act(self, obs, prompt):
        if not self._queue:
            chunk = self._replan(obs, prompt)
            self._queue = [a for a in chunk[:self.replan_steps]]
        action = self._queue.pop(0)
        safe, _info = self.layer.filter(action, obs)
        return safe

    def stats(self):
        s = self.layer.stats()
        s["steer_count"] = self.steer_count
        s["steer_infeasible"] = self.steer_infeasible
        s["danger_detect_count"] = self.danger_detect_count
        return s

    def close(self):
        self.policy.close()
