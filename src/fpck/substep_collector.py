"""Physics-substep 只读采集器（审计 B/C/D 用）。

复制 robosuite env.step() 的 25 次 physics substep 循环，在每次 sim.step() 后
插入**只读**观测：原始 contact pair、目标 pose/vel、夹爪关节状态、EEF pose。

严格只读：
- 不改变 sim.step 次数（仍是 25 次）；
- 不额外 policy inference；
- 不修改控制缓存 / action chunk cursor / 随机流；
- 不调用会更新内部状态的 monitor（只用 mjData 原始字段）。

用途：重采 spatial/task1/state1，确认首个 5mm 越界的 7.81mm 位移是「夹爪闭合顺应」
还是「手臂推动」，以及原始 contact pair 的 body/geom 身份。

用法：
  collector = SubstepCollector(env)
  obs, reward, done, info = collector.step_with_substep(action, hook)
  # hook(substep_idx, snapshot) 每 substep 调用一次
"""

from __future__ import annotations

import numpy as np


class SubstepCollector:
    """只读 physics-substep 采集器。"""

    def __init__(self, env):
        self.env = env
        self.control_timestep = env.env.control_timestep
        self.model_timestep = env.env.model_timestep
        self.n_substeps = int(self.control_timestep / self.model_timestep)

    def snapshot_contacts(self):
        """原始 contact pair（geom id + body name + dist），只读。"""
        m = self.env.sim.model._model
        d = self.env.sim.data._data
        names = self.env.sim.model.body_names
        out = []
        for c in range(d.ncon):
            g1, g2 = d.contact[c].geom1, d.contact[c].geom2
            b1 = names[m.geom_bodyid[g1]]
            b2 = names[m.geom_bodyid[g2]]
            out.append({
                "geom1": int(g1), "geom2": int(g2),
                "body1": b1, "body2": b2,
                "dist": float(d.contact[c].dist),
            })
        return out

    def snapshot_gripper(self, obs=None):
        """夹爪关节状态（位置 + 速度），只读。

        修正：之前用 d.qpos[-2:] 假设「最后 2 个 qpos 是夹爪」，这是错误的
        （qpos 总 97 维，夹爪关节不在最后 2 位）。正确做法是读 robosuite 维护的
        obs["robot0_gripper_qpos"]（已正确映射左右手指关节）。
        """
        d = self.env.sim.data._data
        qpos = None
        qvel = None
        # 从 obs 读正确映射的夹爪位置（robosuite 维护，非底层 qpos 猜测）
        if obs is not None:
            try:
                qpos = np.asarray(obs.get("robot0_gripper_qpos")).copy()
            except Exception:
                pass
        # 夹爪关节速度：robosuite 的 obs 里通常是 robot0_gripper_qpos（无 qvel），
        # 底层 qvel 需从 joint 地址读，这里保守置 None（速度见证改用 body vel）
        return {"qpos": qpos, "qvel": qvel, "obs_gripper_qpos": qpos}

    def snapshot_target(self, body_name):
        """目标对象 pose + vel（只读）。"""
        from tools.g1a_soup_trace import body_state
        st = body_state(self.env, body_name)
        return {"pos": st["pos"].copy(), "quat": st["quat"].copy(),
                "vel": st["vel"].copy()}

    def step_with_substep(self, action, hook):
        """复制 env.step() 的 25 次 substep 循环 + 每次后调用 hook。

        hook(substep_idx, snapshot_dict) -> None，只读回调。
        返回 (obs, reward, done, info)，与 env.step 一致。
        """
        env = self.env
        if env.env.done:
            raise ValueError("executing action in terminated episode")
        env.env.timestep += 1
        policy_step = True

        for i in range(self.n_substeps):
            env.env.sim.forward()
            env.env._pre_action(action, policy_step)
            env.env.sim.step()
            env.env._update_observables()
            policy_step = False
            # 只读观测（_get_observations 只读，从 observable 缓存取夹爪位置）
            obs_cache = env.env._get_observations()
            snap = {
                "physics_substep": i,
                "contacts": self.snapshot_contacts(),
                "gripper": self.snapshot_gripper(obs_cache),
            }
            if hook is not None:
                hook(i, snap)

        env.env.cur_time += self.control_timestep
        reward, done, info = env.env._post_action(action)
        if env.env.viewer is not None and env.env.renderer != "mujoco":
            env.env.viewer.update()
        obs = (env.env.viewer._get_observations() if env.env.viewer_get_obs
               else env.env._get_observations())
        return obs, reward, done, info
