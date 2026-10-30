"""fori_loop 版 sample_actions（支持反向模式 vjp）。

上游 openpi 的 pi0.sample_actions 用 while_loop，JAX 反向模式（vjp/grad）
不支持动态 while_loop。num_steps 固定（10），用 fori_loop 等价重写后支持 vjp，
PNSG 的 A = C J 可用 vjp 计算（3 次 vjp 替代 320 次 jvp，28s → 秒级）。

数值上与上游 while_loop 版完全一致（相同 step 函数，仅循环结构不同）。
独立成模块，避免依赖 openpi_worker 的 IPC 环境变量。
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def make_sample_fori(model, num_steps=10):
    """返回 sample_actions(rng, observation, *, noise=None)，fori_loop 版。"""
    import einops
    from openpi.models import model as _model
    from openpi.models.pi0 import make_attn_mask

    def sample_actions(rng, observation, *, noise=None):
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, model.action_horizon, model.action_dim))

        prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = model.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def body(i, carry):
            x_t, time, kv_cache = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            (prefix_out, suffix_out), kv_cache = model.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            v_t = model.action_out_proj(suffix_out[:, -model.action_horizon:])
            return x_t + dt * v_t, time + dt, kv_cache

        x_0, _, _ = jax.lax.fori_loop(0, num_steps, body, (noise, 1.0, kv_cache))
        return x_0

    return sample_actions
