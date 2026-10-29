"""π0.5 策略适配：把 openpi 的 Policy 包成 safeVLA 的 Policy 接口。

为什么要单独一层：`/data/2608.SDL/MojocoProj` 里的 runner 面向 notebook 教学，
默认 cache 路径写在模块常量里（root 只读目录）；这里集中处理跨版本 IPC、
缓存重定向、观测转换，让评测代码只依赖一个 `act(obs) -> action` 接口。

父进程 py3.8（robosuite 环境）↔ 子进程 py3.11（/opt/venv/openpi, JAX），
两者 numpy / torch 版本不兼容，故只能走文件 IPC，不能同进程 import。
"""

from __future__ import annotations

import os
import sys

import numpy as np

# 共享区（只读）。可由环境变量覆盖，便于换权重。
PI05_SHARED_ROOT = os.environ.get("PI05_SHARED_ROOT", "/data/2608.SDL/MojocoProj")
PI05_CKPT = os.path.join(PI05_SHARED_ROOT, "checkpoints", "pi05", "pi05_libero")

_RUNNER_INSERTED = False


def _ensure_runner_importable():
    """把共享区的 runner 加进 sys.path（只读目录，只 import 不修改）。"""
    global _RUNNER_INSERTED
    if _RUNNER_INSERTED:
        return
    src = os.path.join(PI05_SHARED_ROOT, "src", "pi05")
    if src not in sys.path:
        sys.path.insert(0, src)
    _RUNNER_INSERTED = True


class Pi05Policy:
    """π0.5 baseline 策略。

    `act(obs)` 返回 env.step 可用的 7 维动作；内部维护 action chunk 队列
    （π0.5 一次推理输出 action_horizon 步，逐帧消费，队列空才重新推理）。
    """

    def __init__(self, ckpt_dir=None, num_infer_steps=10, image_size=256,
                 openpi_data_home=None, verbose=True, replan_steps=None):
        _ensure_runner_importable()
        import pi05_libero_runner as R  # noqa: PLC0415

        self._runner = R
        self.image_size = image_size
        self.num_infer_steps = num_infer_steps
        # replan_steps: 每次推理后只执行前 N 个动作就重新推理（AEGIS 用 5）。
        # None = 执行完整个 action chunk（默认，π0.5 官方 open-loop 方式）。
        self.replan_steps = replan_steps

        # runner 的 OPENPI_DATA_HOME 是模块常量且指向 root 只读目录，
        # openpi 启动时会 chmod 它并抛 PermissionError，必须显式传到可写目录。
        if openpi_data_home is None:
            root = os.environ.get("SAFEVLA_ROOT") or os.getcwd()
            openpi_data_home = os.path.join(root, ".cache-local", "openpi")

        self.policy = R.load_policy(
            "pi05_libero",
            ckpt_dir or PI05_CKPT,
            num_infer_steps=num_infer_steps,
            verbose=verbose,
            openpi_data_home=openpi_data_home,
        )
        self._queue: list[np.ndarray] = []
        self.infer_count = 0

    @property
    def ready(self):
        return bool(getattr(self.policy, "ready", False))

    def reset(self):
        """每个 episode 开始时丢弃上一段的动作队列。"""
        self._queue.clear()

    def act(self, obs, prompt):
        """obs = robosuite 原始观测 dict；返回 7 维动作（已 clip 到 [-1,1]）。"""
        if not self._queue:
            policy_obs = self._runner.obs_to_policy_obs(obs, prompt, self.image_size)
            chunk = self.policy.infer(policy_obs)
            self.infer_count += 1
            # 模型配置 action_dim=32（多机器人统一），后 25 维是 padding，
            # LiberoOutputs 已截到前 7 维；这里再兜一次底。
            self._queue = [
                np.clip(np.asarray(a, dtype=np.float32).reshape(-1)[:7], -1.0, 1.0)
                for a in chunk
            ]
            # closed-loop 重规划：只保留前 replan_steps 个，执行完立即重新推理
            if self.replan_steps:
                self._queue = self._queue[: self.replan_steps]
        return self._queue.pop(0)

    def close(self):
        try:
            self.policy.close()
        except Exception:
            pass
