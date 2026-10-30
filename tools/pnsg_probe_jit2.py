"""测原始 while_loop 的 jacfwd 编译后执行耗时（用上游 sample_actions）。"""

import os
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
    cfg = _config.get_config("pi05_libero")
    policy = _policy_config.create_trained_policy(cfg, "/data/2608.SDL/MojocoProj/checkpoints/pi05/pi05_libero",
                                                  sample_kwargs={"num_steps": 10})
    model = policy._model
    sample = model.sample_actions  # 上游 while_loop 版

    obs = libero_policy.make_libero_example()
    inputs = policy._input_transform({k: v for k, v in obs.items()})
    inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
    observation = __import__("openpi.models.model", fromlist=["Observation"]).Observation.from_dict(inputs)

    H, D = model.action_horizon, model.action_dim
    z0 = jax.random.normal(jax.random.key(42), (1, H, D))

    def f(z):
        return sample(jax.random.key(0), observation, num_steps=10, noise=z)[0, :, :6]

    # jacfwd 三次，看编译 vs 执行
    for i in range(3):
        t0 = time.time()
        J = jax.jacfwd(f)(z0)
        jax.block_until_ready(J)
        t1 = time.time()
        print(f"jacfwd 第{i+1}次: {t1-t0:.1f}s")


if __name__ == "__main__":
    main()
