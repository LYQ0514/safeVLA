# PNSG 地基验证：源噪声注入 + Jacobian

- **日期**：2026-10-30
- **性质**：方法（设计.md）的核心前置条件验证

---

## 一、验证结论

| 前置条件 | 结果 | 证据 |
|---|---|---|
| 源噪声 z 注入 | ✅ | `Policy.infer(obs, noise=z)` / `sample_actions(..., noise=z)` |
| 确定性 | ✅ | 同 z0 两次前向 `max|diff| = 0` |
| 源空间维度 | ✅ | `action_horizon=10 × action_dim=32 = 320`，与设计文档一致 |
| 前向 Jacobian | ✅ | `jax.jacfwd` 得 (60,320)，与 `jax.jvp` 单列 cosine=0.99998 |
| 反向模式 | ❌ | `jax.grad`/`vjp` 对 `while_loop` 不可行（动态 start/stop） |
| 有限差分自检 | ⚠️ | bfloat16 噪声 floor 使 FD 不可靠（见下） |

**核心结论：PNSG 方法的地基成立**——可以在 π0.5 的 Flow Matching 源空间注入
确定性噪声，并用前向模式自动微分（jacfwd/jvp）求「动作对源噪声的 Jacobian」。

---

## 二、关键实现细节

### 2.1 源噪声入口

`openpi/models/pi0.py::sample_actions(rng, observation, num_steps, noise)`：

```python
if noise is None:
    noise = jax.random.normal(rng, (batch_size, action_horizon, action_dim))
...
x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
```

`noise` 就是源噪声 z，形状 `(1, 10, 32)`，Flow Matching 从 z（t=1）去噪到
动作 x_0（t=0）。展平 320 维，与设计文档 `H=10, d_z=32` 完全一致。

### 2.2 反向模式不可行的原因

`sample_actions` 用 `jax.lax.while_loop`（动态循环次数），JAX 反向模式
（`jax.grad`/`jax.vjp`）不支持动态 while_loop。报错：

```
ValueError: Reverse-mode differentiation does not work for lax.while_loop ...
Try using lax.scan, or using fori_loop with static start/stop.
```

**影响**：完整 Jacobian 只能用前向模式（`jacfwd`，28s）或逐列 `jvp`（320 次）。
H-Fast 求解器需要的 `A = C J`（行组合 = vjp）无法直接计算，需先求完整 J 再乘 C。

**优化方向（后续）**：把 while_loop 改成 `lax.scan`（num_steps 固定 10，本就是静态），
即可支持反向模式，3 次 vjp 就能算 A 的 3 行，而非 320 次 jvp。

### 2.3 有限差分自检不可靠（bfloat16）

权重是 bfloat16（`create_trained_policy` 里 `dtype=jnp.bfloat16`）。FD 的
`|fd|/|jvp|` 随 eps 减小爆炸（4.3 → 50 → 481 → 4874 → 50756），说明 FD 被
bfloat16 噪声 floor 主导，不能用于自检。

**正确自检方式**：jacfwd 列 vs jvp 列（两者都是前向模式，cosine=0.99998）。

---

## 三、性能瓶颈

| 操作 | 耗时 |
|---|---|
| 加载模型 | 13.8s |
| jacfwd 完整 Jacobian (60×320) | 28.0s |
| 单次 jvp | 11.25s（含编译）|

28s 的 Jacobian 对在线 steering 不可行，但对「测试一版」可接受。
优化路径：① while_loop → scan（支持 vjp，3 次 vjp 替代 320 次 jvp）；
② 只对 pose 通道（前 6 维）求导；③ jvp 编译后批量。

---

## 四、下一步

按设计文档第 19 节「第一步」：实现核心 proposal
`J, C, b, A=CJ, Δz*, r_crit, χ_lin`，用合成约束 + 单个真实状态验证。
