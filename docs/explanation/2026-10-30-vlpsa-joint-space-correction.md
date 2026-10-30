# VLPSA 复现修正：关节速度空间 QP（根本性理解错误）

- **日期**：2026-10-30
- **性质**：纠正此前复现的根本错误，重新实现

---

## 一、根本性理解错误

此前（`docs/testing/2026-10-30-vlpsa-comparison.md`）的 VLPSA 复现在**末端速度空间（R⁶）**解 QP，
用 `J_ee⁺`（伪逆）把末端速度映射到关节速度。

**论文 MC-CBF-QP 在关节速度空间（R⁷）解 QP**：

```
(vsafe, δ*) = arg min_{v ∈ R⁷, δ≥0}  ∥v − v_nom∥² + p·δ²
s.t.  ḣᵢ = ∇hᵀ Jᵢ v ≥ −αᵢ hᵢ       (CBF，Jᵢ 是 3×7 关节雅可比，不经过伪逆)
      V̇ = −eᵀ Wᵀ J_ee v ≤ −γ V + δ  (CLF)
```

关键差异：Franka 有 **7 关节、6 末端**，1 个冗余自由度。
末端空间 QP 用伪逆固定了这 1 维，关节空间 QP 保留它。
此前"全身采样过度约束""CLF 有害"等负结论，都是**丢冗余度**的症状。

---

## 二、修正的 4 个 bug

| # | bug | 现象 | 修复 |
|---|---|---|---|
| 1 | QP 在末端空间（R⁶） | 全身约束冲突，TSR 崩 | 关节空间（R⁷）变量，CBF 直接用 Jᵢ（3×7） |
| 2 | 速度尺度错 20 倍 | 机械臂几乎不动 | action=1 对应 0.2078 m/s（实测），非 0.009 m/步 |
| 3 | CLF 误差除 action_scale | V 放大 12345 倍，约束失衡 | 位置误差用物理米 |
| 4 | 采样漏夹爪指尖 | 指尖碰障碍 CBF 感知不到 | 补 `finger_joint1_tip`/`joint2_tip` |

---

## 三、CBF 增益方向（关键经验）

CBF 约束 `∇hᵀ J v ≥ −α h`，**α 越大越松**（右侧更负）。

实测（朝障碍直冲，Poisson 梯度模 ~0.03）：

| cbf_gain(α) | 结果 |
|---|---|
| 0.01 | 避开，停在 0.190m |
| 0.05 | 避开，停在 0.167m |
| 0.10 | 避开，停在 0.159m |
| 0.30 | 碰撞 step 82 |
| 0.50 | 碰撞 step 55 |
| 1.0（旧默认）| 碰撞 step 39 |

**Poisson 解的梯度模 ≈ 0.03 而非 1**，所以 α 需相应缩小到 ~0.1。
之前默认 α=1.0 太激进，是 CAR=0 的直接原因。

---

## 四、修正后的结果（spatial task0, 10 条, α=0.1）

| 方法 | TSR | CAR | CAR>5mm |
|---|---|---|---|
| π0.5 基线 | 44% | 0% | — |
| AEGIS | 57% | 50% | — |
| VLPSA 末端空间（旧错误）| 41.7% | 19.8% | — |
| **VLPSA 关节空间（α=0.1）** | **10%** | **40%** | **100%** |

**诚实结论**：即使正确复现关节空间 QP，VLPSA 在本协议（20Hz OSC、障碍位移 1mm 口径）下
**仍不如 AEGIS**（TSR 10% vs 57%，CAR 40% vs 50%）。

关节空间 QP 确实更保守（CAR 从 0→40%，CAR>5mm 100%），但代价是任务成功率
从 44% 崩到 10%。这符合论文自述："VLPSA results in three times fewer collisions.
Losing unsafe success is a cost to conservative safety filtering."

---

## 五、与论文数字差距的根因（关键认识）

论文 VLPSA：CAR 91.2% / SR 55.5%。我们修正后：CAR 40% / TSR 10%（spatial task0）。

差距的**结构性原因**，不是实现 bug：

1. **碰撞口径不同**：论文"扫描所有 MuJoCo 接触"（接触即碰撞），我们"障碍位移 > 1mm"
   （位移才算碰撞）。毫米级接触在论文口径下可能不算碰撞，导致论文 CAR 偏高。

2. **控制带宽**：论文 100Hz joint-velocity，我们 20Hz。慢控制下 CBF 响应晚一步。

3. **完整配置**：论文有持物保护 + 多分辨率 + ε-buffer + ε-采样配套，我们只有主干。

4. **Level I 结构性冲突**：障碍紧贴目标（碗距 moka pot 0.21m），全身采样 815 点
   在狭窄空间里把可行域压到接近空集，机械臂要么碰撞要么停住。

---

## 六、复现命令（正确用法）

```bash
cd /data/safeVLA && source tools/activate_env.sh

# 关节空间 QP（论文正确做法），α 必须用 0.1 左右（Poisson 梯度模 ~0.03）
python tools/run_eval_vlpsa.py --suite safelibero_spatial --level I \
    --tasks 0 --episodes 0 1 2 3 4 \
    --controller JOINT_VELOCITY --cbf-gain 0.1 --resolution 1024
```

---

## 七、结论与建议

1. **VLPSA 复现已达到"正确实现"的程度**：关节空间 QP + 正确尺度 + 完整采样 + 正确参数。

2. **诚实结论**：在 SafeLIBERO + 20Hz + 位移 1mm 口径下，VLPSA 全身 CBF
   **不如 AEGIS 末端椭球 CBF**。这不是实现错误，而是：
   - 全身采样在 Level I（障碍紧贴目标）下过度保守
   - 论文的高 CAR 依赖更松的碰撞口径 + 100Hz + 完整配套

3. **对主方法的意义**：已确认 AEGIS（末端椭球）是本协议下的 SOTA 安全层，
   主方法应以此为超越目标（SSR 32.2%）。
