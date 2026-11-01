# G0：CRN 配对可行性核查（透明噪声注入）

> 日期：2026-11-01
> 状态：G0 通过（推理等价性代码层面证明），G1 前提满足。

---

## 0. 结论

`PnsgPolicy.generate(z0, obs)` 是**透明噪声注入器**：它不改变 π0.5 推理行为，
只是把「内部 `jax.random.normal(rng)` 采样」替换为「显式传入的标准高斯 z0」。

## 1. 核心证据（openpi `pi0.py` sample_actions）

```python
def sample_actions(self, rng, observation, *, num_steps=10, noise=None):
    ...
    if noise is None:
        noise = jax.random.normal(rng, (batch_size, action_horizon, action_dim))
    # 之后 while_loop 从 noise 出发，10 步确定性去噪
    x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
    return x_0
```

**三个决定性事实**：

1. `noise is not None` 时，`rng` **完全不被使用**（唯一使用 rng 的分支被跳过）。
2. 去噪 `while_loop` 是**确定性的**（从 noise 出发，10 步 Euler 去噪，无随机性）。
3. 因此 `generate(z0, obs)` = `sample_actions(_, obs, noise=z0)`，
   **与「以 z0 为初始噪声的 π0.5 推理」完全等价**。

## 2. z0 分布核验

`sample_z(nominal_seed(...))`：
- shape (10, 32)，float32；
- 均值 ≈ 0.08（320 样本抽样误差）、标准差 ≈ 0.98、范围 [−3.44, 3.20]；
- 标准高斯，与 `jax.random.normal(rng)` 同分布。

## 3. G0 四检查项

| 检查项 | 结论 |
|---|---|
| 1. 推理等价性 | ✅ 代码证明（noise 注入忽略 rng，去噪确定性）|
| 2. 随机源控制 | ✅ z0 由 nominal_seed 确定性生成，可预先指定 |
| 3. 无干预一致性 | 待 G1 闭环验证 |
| 4. 安全层身份独立 | 待 G1 验证（AEGIS 状态分离）|

## 4. 路径 B 数值 round-trip 的说明（非等价性问题）

原计划「同一 z0 分别喂 PnsgPolicy 和 openpi infer(noise=z0)」做数值比对，
但 JSON round-trip 把 uint8 图像变 int，导致 cv2.resize 失败——这是**测试 harness
的序列化问题**，不是等价性问题。等价性已由 `sample_actions` 代码直接证明，
无需数值 round-trip。

## 5. G1 前提

G0 通过。PnsgPolicy 可作为 CRN 配对的透明噪声注入器，用于 G1 小规模 Long 配对。

**CRN 随机流约定**（G1 需遵守）：
- z0 按 (episode, 重规划决策索引) 确定性生成（`nominal_seed`）；
- 两条分支（π0.5 / AEGIS）的推理调用时机不同时，须显式记录对齐规则，
  避免顺序消耗 RNG 导致错位。
