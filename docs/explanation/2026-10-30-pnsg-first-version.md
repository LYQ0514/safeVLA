# PNSG 第一版验证：源空间 steering 核心闭环

- **日期**：2026-10-30
- **性质**：方法（docs/设计.md）的第一版实现与验证

---

## 一、已实现（按设计文档第 19 节「第一步」）

| 模块 | 文件 | 内容 |
|---|---|---|
| 策略适配 | `src/pnsg/openpi_worker.py` + `policy_adapter.py` | 源噪声注入 + 前向 Jacobian |
| 源空间求解器 | `src/pnsg/source_solver.py` | H-Fast 最小范数 + Farkas + χ_lin |
| 几何代理 | `src/pnsg/geometry_proxy.py` | 控制点球覆盖 + 障碍球 + 线性约束 |

---

## 二、核心验证结果

### 2.1 求解器正确性（合成约束）

1/2/3 约束与 QP 解一致，320 维 3 约束满足 `A Δz ≥ b`，不可行时返回 None。
详见 `tools/pnsg_test_solver.py`。

### 2.2 端到端：源空间 Δz 真实改善净空（关键）

固定真实状态（spatial task0），人工放大障碍半径 1.6x 逼出危险场景，
名义动作净空 **-22.12 mm**（穿越障碍）。解 Δz 后，候选真实前向：

| α | 真实最小净空 | 源代价 C_src |
|---|---|---|
| 0.00 | -22.12 mm | 0 |
| 0.25 | -4.49 mm | 5.34 |
| 0.50 | -4.08 mm | 21.35 |
| 0.75 | -3.26 mm | 48.04 |
| 1.00 | **-2.27 mm** | 85.41 |

**结论：源空间 steering 有效** —— 净空随 α 单调改善 20mm，
证明「在 Flow Matching 源空间找最小修正」能在保留策略结构的同时改善安全。
这正是区别于 AEGIS/VLPSA（动作空间）的根本点。

### 2.3 线性近似误差（符合预期）

线性预测 `A Δz ≥ b` 应使净空 ≥ 5mm，但真实只到 -2.27mm。
原因是 Flow Matching 的 320→60 维非线性映射，线性 Jacobian 在远离名义点处失真。

**这印证设计文档第 20 节**："χ_lin 是线性局部难度，不是安全证书"。
因此非线性认证（C_2，真实前向执行）是必须的，不能省略。

---

## 三、关键工程细节

1. **源噪声入口**：`sample_actions(..., noise=z)`，z ∈ R^{10×32}
2. **Jacobian 只能前向模式**：`jax.jacfwd`（28s），反向模式对 while_loop 不可行
3. **Jacobian 坐标**：external = raw × std（Unnormalize affine scaling），
   std 前 6 维 = [0.331, 0.372, 0.452, 0.0395, 0.0628, 0.0732]
4. **action 尺度**（B_k 定义）：平移 0.01039 m/步、旋转 0.09197 rad/步（实测 OSC_POSE）
5. **Unnormalize 需 state 键**：output transform strict=True，需同时传 state + actions

---

## 四、尚未实现（设计文档第 18 节明确列出）

```
I_full / S_full          NOT IMPLEMENTED
P_H model progress       NOT IMPLEMENTED
E_acc = C_2 ∧ P_H        NOT COMPUTABLE
完整在线候选路由         NOT IMPLEMENTED
```

当前最准确命名：「PNSG H-Fast source steering with candidate-dependent wide
proxy checking」，非「完整 PNSG v1.3」。

---

## 五、下一步

1. **接非线性认证 C_2**（真实前向执行 2 步，检查接触）—— 设计文档第三步
2. **完整评测**：与 π0.5 / AEGIS / VLPSA 对比 TSR/CAR/SSR
3. **性能优化**：while_loop → lax.scan（支持反向模式，3 次 vjp 替代 320 次 jvp）
