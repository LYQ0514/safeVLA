"""PNSG 地基验证：π0.5 源噪声注入 + JAX 求 Jacobian。

验证两个前置条件：
1. 能注入固定源噪声 z0，得到确定性动作（对应设计文档 generate(z, observation)）
2. 能用 jax.jvp 对 z 求动作的方向导数（Jacobian 的列）

这是 PNSG 方法的命门：如果 jvp 不可行，整个源空间 steering 就无从谈起。
"""

import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

from openpi.policies import libero_policy
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


def main():
    ckpt = "/data/2608.SDL/MojocoProj/checkpoints/pi05/pi05_libero"
    print("[1] 加载配置与模型…")
    t0 = time.time()
    cfg = _config.get_config("pi05_libero")
    policy = _policy_config.create_trained_policy(
        cfg, ckpt, sample_kwargs={"num_steps": 10}
    )
    print(f"    加载耗时 {time.time()-t0:.1f}s")

    model = policy._model
    sample_actions = model.sample_actions  # 未 jit 的原始方法

    # 构造 observation（用 libero 的 example，随机）
    obs = libero_policy.make_libero_example()
    # 用 policy 的 input transform 得到模型输入
    inputs = policy._input_transform({k: v for k, v in obs.items()})
    inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
    observation = __import__("openpi.models.model", fromlist=["Observation"]).Observation.from_dict(inputs)

    print("[2] 固定源噪声 z0，验证确定性…")
    H = model.action_horizon   # 10
    D = model.action_dim        # 32
    print(f"    action_horizon={H}, action_dim={D}, 源空间维数={H*D}")
    z0 = jax.random.normal(jax.random.key(42), (1, H, D))
    a1 = sample_actions(jax.random.key(0), observation, num_steps=10, noise=z0)
    a2 = sample_actions(jax.random.key(1), observation, num_steps=10, noise=z0)
    diff = float(jnp.abs(a1 - a2).max())
    print(f"    同 z0 两次前向 max|diff| = {diff:.2e}  (应为 0)")

    print("[3] 对 z 求方向导数（jvp）…")
    # 设计文档 Jacobian 对应 pose 通道（前 6 维），action_dim=32 含 padding
    # 先对完整 action 求 jvp，再取 pose 通道
    def f(z):
        return sample_actions(jax.random.key(0), observation, num_steps=10, noise=z)

    v = jax.random.normal(jax.random.key(7), (1, H, D))
    t0 = time.time()
    primals, tangents = jax.jvp(f, (z0,), (v,))
    t_jvp = time.time() - t0
    print(f"    jvp 耗时 {t_jvp:.2f}s")
    print(f"    action shape: {primals.shape}, tangent shape: {tangents.shape}")
    print(f"    tangent 范数: {float(jnp.linalg.norm(tangents)):.4f}")

    # 有限差分自检
    eps = 1e-3
    f_p = f(z0 + eps * v)
    f_m = f(z0 - eps * v)
    fd = (f_p - f_m) / (2 * eps)
    cos = float(jnp.sum(fd * tangents) / (jnp.linalg.norm(fd) * jnp.linalg.norm(tangents) + 1e-12))
    print(f"    jvp vs 有限差分 cosine similarity = {cos:.6f}  (应接近 1)")

    print("[4] 完整 Jacobian 维度确认…")
    # pose 通道 = 前 6 维，H×6 = 60 行；源 320 维 = 320 列
    # 用 vjp 求一行，用 jvp 求一列，确认可微性
    print(f"    J 目标形状: ({H*6}, {H*D}) = ({H*6}, {H*D})")
    print("\n✅ 若 jvp vs 有限差分 cosine 接近 1，则 PNSG 地基成立")


if __name__ == "__main__":
    main()
