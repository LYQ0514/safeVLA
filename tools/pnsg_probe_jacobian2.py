"""诊断 jvp vs 有限差分 cosine 偏低的原因。"""

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
    v = jax.random.normal(jax.random.key(7), (1, H, D))
    v = v / jnp.linalg.norm(v)

    def f(z):
        return sample_actions(jax.random.key(0), observation, num_steps=10, noise=z)

    primals, tangents = jax.jvp(f, (z0,), (v,))

    print(f"{'eps':>10} {'cosine':>10} {'|fd|/|jvp|':>12}")
    for eps in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
        f_p = f(z0 + eps * v)
        f_m = f(z0 - eps * v)
        fd = (f_p - f_m) / (2 * eps)
        cos = float(jnp.sum(fd * tangents) / (jnp.linalg.norm(fd) * jnp.linalg.norm(tangents) + 1e-12))
        ratio = float(jnp.linalg.norm(fd) / (jnp.linalg.norm(tangents) + 1e-12))
        print(f"{eps:>10.0e} {cos:>10.5f} {ratio:>12.4f}")

    # 额外：检查 tangents 是否本身就是噪声（如果是，说明 jvp 没真正传播）
    print(f"\njvp tangent 范数: {float(jnp.linalg.norm(tangents)):.4f}")
    print(f"输入 v 范数: {float(jnp.linalg.norm(v)):.4f}")
    # 相关性：tangents 与 v 的关系（展平后）
    tv = float(jnp.sum(tangents.ravel() * v.ravel()))
    print(f"tangent·v (展平点积): {tv:.4f}")


if __name__ == "__main__":
    main()
