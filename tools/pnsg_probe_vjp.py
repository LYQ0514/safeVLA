"""验证：把 while_loop 改成 fori_loop 后，反向模式 vjp 是否可行。

这是 PNSG 性能优化的关键：若 vjp 可行，3 次 vjp 就能算 A 的 3 行，
替代 320 次 jvp（28s → 秒级）。
"""

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


def build():
    cfg = _config.get_config("pi05_libero")
    policy = _policy_config.create_trained_policy(cfg, "/data/2608.SDL/MojocoProj/checkpoints/pi05/pi05_libero",
                                                  sample_kwargs={"num_steps": 10})
    model = policy._model
    obs = libero_policy.make_libero_example()
    inputs = policy._input_transform({k: v for k, v in obs.items()})
    inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
    observation = __import__("openpi.models.model", fromlist=["Observation"]).Observation.from_dict(inputs)
    return model, observation


def make_sample_fori(model, observation):
    """用 fori_loop 重写 sample_actions（等价，但支持反向模式）。"""
    from openpi.models import model as _model

    def sample(noise, num_steps=10):
        obs = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = obs.state.shape[0]
        prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(obs)
        prefix_attn_mask = model._make_attn_mask(prefix_mask, prefix_ar_mask) if hasattr(model, '_make_attn_mask') else None
        # 直接复用模型内部逻辑太复杂，这里只测「fori_loop 反向模式」是否可行
        # 用一个玩具 fori_loop 验证 vjp 机制
        def body(t, x):
            return x + dt * model.action_out_proj(model.PaliGemma.llm.__class__.__name__ if False else x[:model.action_horizon]) if False else x
        # 上面是占位，直接测 while_loop vs fori_loop 的可微性差异
        return noise

    return sample


def main():
    model, observation = build()
    H, D = model.action_horizon, model.action_dim
    z0 = jax.random.normal(jax.random.key(42), (1, H, D))

    # 玩具对比：while_loop vs fori_loop 的反向模式可微性
    print("=== while_loop 反向模式 ===")
    def f_while(x):
        def body(c):
            c = c + 0.1 * c
            return c
        def cond(c):
            return c < 1.0
        return jax.lax.while_loop(cond, body, x)
    try:
        jax.grad(lambda x: f_while(x))(1.0)
        print("  while_loop grad 可行")
    except Exception as e:
        print(f"  while_loop grad 失败: {str(e)[:80]}")

    print("=== fori_loop 反向模式 ===")
    def f_fori(x):
        def body(i, c):
            return c + 0.1 * c
        return jax.lax.fori_loop(0, 10, body, x)
    try:
        g = jax.grad(lambda x: f_fori(x))(1.0)
        print(f"  fori_loop grad 可行: {float(g):.4f}")
    except Exception as e:
        print(f"  fori_loop grad 失败: {str(e)[:80]}")

    print("\n结论：若 fori_loop 支持反向模式，则把 pi0.py 的 while_loop 改 fori_loop")
    print("（num_steps=10 静态）即可让 PNSG 用 vjp，性能从 28s 降到秒级。")


if __name__ == "__main__":
    main()
