"""PNSG 策略（接入评测框架的 policy 接口）。

核心：每个 action chunk 决策点，走完整 PNSG 流程：
1. 采样 z0，求名义动作 + Jacobian（jacfwd ~7s）
2. 几何代理选约束，解 Δz
3. 候选池 + C_2 认证，选第一个通过的候选
4. 返回候选的前 execute_steps 个动作执行

与设计文档第 15 节主流程对应。P_H 未实现（记 NA）。
"""

from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from src.pnsg.geometry_proxy import (  # noqa: E402
    build_control_points, build_obstacles, selected_constraints, compute_clearance,
)
from src.pnsg.policy_adapter import PnsgPolicy  # noqa: E402
from src.pnsg.source_solver import minimum_norm_halfspaces  # noqa: E402
from src.pnsg.full_state import save_full_state, restore_full_state  # noqa: E402
from src.pnsg.certifier import Certifier  # noqa: E402


class PnsgSteeringPolicy:
    """PNSG 源空间 steering 策略。

    act(obs, prompt) 返回 7 维动作（每次 chunk 决策点重新 steering）。
    """

    def __init__(self, env, image_size=224, execute_steps=2, r_budget=12.0,
                 alphas=(0.25, 0.5, 0.75, 1.0), verbose=False):
        self.env = env
        self.image_size = image_size
        self.execute_steps = execute_steps
        self.r_budget = r_budget
        self.alphas = alphas
        self.verbose = verbose

        self.policy = PnsgPolicy()
        self.certifier = Certifier(env)
        self._queue = []
        self._z0 = None
        self._prompt = None
        self._obs_cache = None

        # 统计
        self.n_decisions = 0
        self.n_repairs = 0
        self.n_c2_fail = 0
        self.n_linear_infeasible = 0

    def _policy_obs(self, obs, prompt):
        # runner 在共享区
        shared = "/data/2608.SDL/MojocoProj/src/pi05"
        if shared not in sys.path:
            sys.path.insert(0, shared)
        import pi05_libero_runner as R
        return R.obs_to_policy_obs(obs, prompt, image_size=self.image_size)

    def reset(self, obs=None, prompt=""):
        self._queue = []
        self._z0 = None
        self._prompt = prompt
        self._obs_cache = obs
        self.n_decisions = 0
        self.n_repairs = 0
        self.n_c2_fail = 0
        self.n_linear_infeasible = 0

    def _decide(self, obs, prompt):
        """一次决策：返回 (actions_to_execute, info)。"""
        p_obs = self._policy_obs(obs, prompt)
        z0 = self.policy.sample_noise(seed=self.n_decisions)
        _, _, ext_actions, ext_J = self.policy.generate_with_jacobian(z0, p_obs)
        nominal = np.clip(ext_actions, -1, 1)

        cps = build_control_points(self.env, obs)
        obstacles = build_obstacles(self.env, obs)

        # 无工作空间障碍物 → 无约束，直接执行名义动作
        if len(obstacles) == 0:
            self.n_decisions += 1
            return nominal, {"label": "no-obstacle"}

        # 几何代理选约束
        rows = selected_constraints(cps, obstacles, nominal[:, :6],
                                    h_floor=0.005, max_constraints=3, margin=0.01)

        candidates = [(z0, nominal, 0.0, "nominal")]
        if rows:
            C = np.vstack([c for c, _ in rows])
            b = np.array([bb for _, bb in rows])
            A = C @ ext_J
            dz, rcrit = minimum_norm_halfspaces(A, b)
            if dz is not None:
                self.n_repairs += 1
                dz_2d = dz.reshape(self.policy.H, self.policy.D)
                for alpha in self.alphas:
                    z_c = z0 + alpha * dz_2d
                    a_c = np.clip(self.policy.generate(z_c, p_obs), -1, 1)
                    candidates.append((z_c, a_c, alpha * float(np.linalg.norm(dz)), f"α={alpha}"))
            else:
                self.n_linear_infeasible += 1
        else:
            if self.verbose:
                print(f"[pnsg] 名义动作安全，直接执行")

        # 候选 + C_2 认证（从同一快照）
        snapshot = save_full_state(self.env)
        chosen = None
        for z_c, a_c, c_src, label in candidates:
            # S_wide 代理（用更精确的障碍球，不放大）
            mc = compute_clearance(cps, obstacles, a_c[:, :6]).min()
            S_wide = mc >= 0.005
            # C_2 真实认证前 2 步
            restore_full_state(self.env, snapshot)
            c2 = self.certifier.certify_prefix(a_c[:2])
            if self.verbose:
                print(f"  [{label}] S_wide={S_wide} C2={c2.pass_} 净空={mc*1000:.1f}mm")
            if c2.pass_:
                chosen = (a_c, label)
                break

        if chosen is None:
            # 全都不通过，退回 nominal（已恢复状态由上层重跑）
            self.n_c2_fail += 1
            restore_full_state(self.env, snapshot)
            chosen = (nominal, "fallback-nominal")

        self.n_decisions += 1
        return chosen[0], {"label": chosen[1]}

    def act(self, obs, prompt):
        if not self._queue:
            actions, info = self._decide(obs, prompt)
            self._queue = list(actions[: self.execute_steps])
        return self._queue.pop(0)

    def stats(self):
        return {
            "n_decisions": self.n_decisions,
            "n_repairs": self.n_repairs,
            "n_c2_fail": self.n_c2_fail,
            "n_linear_infeasible": self.n_linear_infeasible,
        }

    def close(self):
        self.policy.close()
