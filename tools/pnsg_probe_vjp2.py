"""验证 fori_loop 版 sample_actions：数值一致 + vjp 可行 + 性能。"""

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
sys.path.insert(0, "/data/safeVLA")
sys.path.insert(0, "/data/safeVLA/src/pnsg")


def main():
    from sample_fori import make_sample_fori

    cfg = _config.get_config("pi05_libero")
    policy = _policy_config.create_trained_policy(cfg, "/data/2608.SDL/MojocoProj/checkpoints/pi05/pi05_libero",
                                                  sample_kwargs={"num_steps": 10})
    model = policy._model
    sample_fori = make_sample_fori(model, num_steps=10)
    sample_orig = model.sample_actions

    obs = libero_policy.make_libero_example()
    inputs = policy._input_transform({k: v for k, v in obs.items()})
    inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
    observation = __import__("openpi.models.model", fromlist=["Observation"]).Observation.from_dict(inputs)

    H, D = model.action_horizon, model.action_dim
    z0 = jax.random.normal(jax.random.key(42), (1, H, D))

    # 1. 数值一致性
    a_orig = sample_orig(jax.random.key(0), observation, num_steps=10, noise=z0)
    a_fori = sample_fori(jax.random.key(0), observation, noise=z0)
    diff = float(jnp.abs(a_orig - a_fori).max())
    print(f"[1] fori vs while 数值差: {diff:.2e} (应~0)")

    # 2. vjp 可行（3 行：对 pose 的 3 个线性组合）
    print("[2] vjp 测试（3 行线性组合）…")
    def f_pose(z):
        return sample_fori(jax.random.key(0), observation, noise=z)[0, :, :6]  # (H,6)

    t0 = time.time()
    C = jax.random.normal(jax.random.key(1), (3, H * 6))  # 3 个约束行
    def g(z):
        return (C @ f_pose(z).reshape(-1))  # (3,)
    # vjp：对 g 的 3 个输出求 z 的梯度
    val, vjp_fn = jax.vjp(g, z0)
    J_vjp = vjp_fn(jnp.eye(3))[0]  # (1,H,D,3) -> 取
    t1 = time.time()
    print(f"    vjp 3 行耗时 {t1-t0:.1f}s")

    # 3. vjp 结果 vs jacfwd 结果
    print("[3] vjp vs jacfwd 一致性…")
    t0 = time.time()
    J_fwd = jax.jacfwd(f_pose)(z0)  # (H,6,1,H,D)
    t1 = time.time()
    print(f"    jacfwd 耗时 {t1-t0:.1f}s")
    J_fwd_mat = J_fwd[:, :, 0, :, :].reshape(H * 6, H * D)  # (60,320)
    # vjp 得到的是 C @ J，即 (3, 320)
    J_vjp_mat = J_vjp.reshape(3, H * D)  # (3,320)
    J_expected = C @ J_fwd_mat  # (3,320)
    cos = float(jnp.sum(J_vjp_mat * J_expected) / (jnp.linalg.norm(J_vjp_mat) * jnp.linalg.norm(J_expected) + 1e-12))
    rel = float(jnp.linalg.norm(J_vjp_mat - J_expected) / (jnp.linalg.norm(J_expected) + 1e-12))
    print(f"    vjp 与 jacfwd 相对误差: {rel:.2e}, cosine={cos:.6f}")

    print(f"\n✅ 若 vjp 3 行 < 10s 且与 jacfwd 一致，PNSG 性能问题解决")


if __name__ == "__main__":
    main()
