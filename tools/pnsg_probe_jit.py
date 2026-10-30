"""测 jacfwd 编译后执行耗时（判断 28s 是否主要是编译）。"""

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

import sys
sys.path.insert(0, "/data/safeVLA/src/pnsg")
from sample_fori import make_sample_fori  # noqa: E402


def main():
    cfg = _config.get_config("pi05_libero")
    policy = _policy_config.create_trained_policy(cfg, "/data/2608.SDL/MojocoProj/checkpoints/pi05/pi05_libero",
                                                  sample_kwargs={"num_steps": 10})
    model = policy._model
    sample = make_sample_fori(model, num_steps=10)

    obs = libero_policy.make_libero_example()
    inputs = policy._input_transform({k: v for k, v in obs.items()})
    inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
    observation = __import__("openpi.models.model", fromlist=["Observation"]).Observation.from_dict(inputs)

    H, D = model.action_horizon, model.action_dim
    z0 = jax.random.normal(jax.random.key(42), (1, H, D))

    # jacfwd（前向模式，用 fori 版也能跑）
    def f(z):
        return sample(jax.random.key(0), observation, noise=z)[0, :, :6]

    print("首次 jacfwd（含编译）…")
    t0 = time.time()
    J1 = jax.jacfwd(f)(z0)
    jax.block_until_ready(J1)
    t1 = time.time()
    print(f"  首次 {t1-t0:.1f}s")

    print("第二次 jacfwd（编译缓存）…")
    t0 = time.time()
    J2 = jax.jacfwd(f)(z0)
    jax.block_until_ready(J2)
    t1 = time.time()
    print(f"  二次 {t1-t0:.1f}s")

    # 注意：z0 变了 shape 不变，但 jacfwd 用 jvp，编译缓存 key 是函数+shape，
    # 换不同 z0 应该复用编译
    z1 = jax.random.normal(jax.random.key(99), (1, H, D))
    print("第三次 jacfwd（不同 z0）…")
    t0 = time.time()
    J3 = jax.jacfwd(f)(z1)
    jax.block_until_ready(J3)
    t1 = time.time()
    print(f"  三次 {t1-t0:.1f}s")


if __name__ == "__main__":
    main()
