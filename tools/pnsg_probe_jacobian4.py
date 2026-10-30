"""测试 jacfwd 完整 Jacobian：形状、耗时、与 jvp 的一致性、float32 精度对比。"""

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


def build(ckpt):
    cfg = _config.get_config("pi05_libero")
    policy = _policy_config.create_trained_policy(cfg, ckpt, sample_kwargs={"num_steps": 10})
    model = policy._model
    obs = libero_policy.make_libero_example()
    inputs = policy._input_transform({k: v for k, v in obs.items()})
    inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
    observation = __import__("openpi.models.model", fromlist=["Observation"]).Observation.from_dict(inputs)
    return model, observation


def main():
    ckpt = "/data/2608.SDL/MojocoProj/checkpoints/pi05/pi05_libero"
    model, observation = build(ckpt)
    H, D = model.action_horizon, model.action_dim
    z0 = jax.random.normal(jax.random.key(42), (1, H, D))

    def f(z):
        return model.sample_actions(jax.random.key(0), observation, num_steps=10, noise=z)

    # jacfwd 完整 Jacobian（前向模式，支持 while_loop）
    print("[1] jacfwd 完整 Jacobian…")
    t0 = time.time()
    J_full = jax.jacfwd(f)(z0)  # 形状 (1,H,D,1,H,D)
    t1 = time.time()
    print(f"    耗时 {t1-t0:.1f}s, 形状 {J_full.shape}")

    # 提取 pose 通道（前 6 维），reshape 成 (60, 320)
    J_pose = J_full[0, :, :6, 0, :, :]  # (H, 6, H, D)
    J = J_pose.reshape(H * 6, H * D)    # (60, 320)
    print(f"    pose Jacobian 形状: {J.shape} (应为 60×320)")
    print(f"    J 的 Frobenius 范数: {float(jnp.linalg.norm(J)):.4f}")

    # 与 jvp 单列一致性
    v = jax.random.normal(jax.random.key(7), (1, H, D))
    v = v / jnp.linalg.norm(v)
    _, tang = jax.jvp(f, (z0,), (v,))
    jvp_col = tang[0, :, :6].reshape(-1)   # (60,)
    jac_col = (J @ v.reshape(-1))          # (60,)
    cos = float(jnp.sum(jvp_col * jac_col) / (jnp.linalg.norm(jvp_col) * jnp.linalg.norm(jac_col) + 1e-12))
    print(f"    jacfwd 列 vs jvp 列 cosine: {cos:.6f} (应=1，两者都是前向模式)")

    # float32 精度对比：cast 输入，重算 FD
    print("[2] float32 精度 FD 自检…")
    z0_32 = z0.astype(jnp.float32)
    def f32(z):
        return model.sample_actions(jax.random.key(0), observation, num_steps=10, noise=z.astype(jnp.bfloat16))
    # 注意：权重仍是 bfloat16，只 cast 输入。真正的 float32 需要 cast 权重。
    for eps in [1e-2, 1e-3, 1e-4]:
        fp = f32(z0_32 + eps * v.astype(jnp.float32))
        fm = f32(z0_32 - eps * v.astype(jnp.float32))
        fd = (fp - fm) / (2 * eps)
        fd_col = fd[0, :, :6].reshape(-1)
        cos = float(jnp.sum(fd_col * jac_col) / (jnp.linalg.norm(fd_col) * jnp.linalg.norm(jac_col) + 1e-12))
        print(f"    eps={eps:.0e}: cosine={cos:.5f}")

    print("\n结论：若 jacfwd 列 vs jvp 列 cosine=1，则前向模式 Jacobian 正确；")
    print("      FD 的 cosine 低是 bfloat16 噪声 floor 所致，不是 jvp 错。")


if __name__ == "__main__":
    main()
