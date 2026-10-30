"""诊断 jvp 正确性：标量梯度 + 精度对比。"""

import os
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

from openpi.policies import libero_policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


def main():
    ckpt = "/data/2608.SDL/MojocoProj/checkpoints/pi05/pi05_libero"
    cfg = _config.get_config("pi05_libero")
    policy = _policy_config.create_trained_policy(cfg, ckpt, sample_kwargs={"num_steps": 10})
    model = policy._model
    sample_actions = model.sample_actions

    obs = libero_policy.make_libero_example()
    inputs = policy._input_transform({k: v for k, v in obs.items()})
    inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
    observation = __import__("openpi.models.model", fromlist=["Observation"]).Observation.from_dict(inputs)

    H, D = model.action_horizon, model.action_dim
    z0 = jax.random.normal(jax.random.key(42), (1, H, D))

    def f(z):
        return sample_actions(jax.random.key(0), observation, num_steps=10, noise=z)

    # 取一个标量输出 action[0,0,0]
    def g(z):
        return f(z)[0, 0, 0]

    # 反向模式梯度（jax.grad）
    grad_g = jax.grad(g)(z0)
    print(f"jax.grad 梯度范数: {float(jnp.linalg.norm(grad_g)):.6f}")

    # 前向模式 jvp：v 方向的方向导数
    v = jax.random.normal(jax.random.key(7), (1, H, D))
    v = v / jnp.linalg.norm(v)
    _, t = jax.jvp(g, (z0,), (v,))
    jvp_directional = float(t)
    grad_directional = float(jnp.sum(grad_g * v))
    print(f"jvp 方向导数: {jvp_directional:.6f}")
    print(f"grad·v 方向导数: {grad_directional:.6f}")
    print(f"两者一致(应接近): {jvp_directional:.6f} vs {grad_directional:.6f}")

    # FD 对比标量
    for eps in [1e-1, 1e-2, 1e-3]:
        fd = float((g(z0 + eps * v) - g(z0 - eps * v)) / (2 * eps))
        print(f"FD(eps={eps:.0e}): {fd:.6f}")

    # 检查权重精度
    leaves = jax.tree.leaves(model)
    dtypes = set(str(x.dtype) for x in leaves if hasattr(x, "dtype"))
    print(f"\n模型权重 dtype 分布: {dtypes}")


if __name__ == "__main__":
    main()
