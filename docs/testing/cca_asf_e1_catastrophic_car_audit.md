# CCA-ASF E1-B0：Catastrophic CAR Violation Audit

> 日期：2026-11-02
> 状态：290mm→27mm mug 位移定向归因完成。
> 前置：`cca_asf_e1_metric_contract.md`（A0 口径冻结）。

---

## 0. 结论

CCA-ASF 在 S01/st3 的 mug 严重位移（可复现，Dmax≈27.5mm）是**真实的闭环安全
负结果**，因果链为：

1. CCA 从 **step0** 起产生与 euclidean 不同的执行动作（各向异性方向重分配，
   E0 已定量 ΔD_⊥>0、ΔD_total>0）；
2. 300 步闭环分叉累积，使 CCA 的夹爪在 **step289** 处于碰底座的位置；
3. step289 夹爪碰底座（`gripper0_right_gripper↔box_small_base_1_main`，
   dist=−0.06mm）→ 底座带 mug → mug 位移 1.01mm 起，最终累积 27.5mm；
4. **碰撞那一步 β=0**（名义动作可行，CCA 未介入）——说明碰撞不是「单步各向
   异性直接撞」，而是「早期各向异性改变轨迹 → 后期名义动作本身进入危险区」。

## 1. 复现证据（稳定）

| 模式 | 重复 | Dmax | CAR | forbid（substep）| interv |
|---|---|---|---|---|---|
| euclidean | 2 次 | 0.000mm | ✅ | 350/501 | ~54 |
| cca_asf | 2 次 | **27.497mm**（逐位一致）| ❌ | 0 | ~384 |

cca_asf 的 forbid=0 但 Dmax=27.5mm：撞飞是「夹爪碰底座 → 底座带 mug」的**间接
传递**，robot-mug 无直接持久接触（瞬态接触在控制步边界消失），故 `monitor_contacts`
在控制步边界漏检。

## 2. 首次 CAR 越界证据（step289/substep23）

```
box_small_base_1_main ↔ gripper0_right_gripper (dist=-0.060mm)  ← 夹爪碰底座
red_coffee_mug ↔ box_small_base_1_main (dist=-0.046mm)          ← 底座带 mug
名义动作 = 执行动作（β=0，CCA 未介入该步）
```

## 3. 因果链（不是单步直接碰撞）

- 首次执行动作分叉 @ **step0**（diff=0.19）——CCA 与 euclidean 从第一步执行动作
  就不同（首 chunk 共享的是名义动作，filter 输出从 step0 分叉）；
- 这是 E0 已定量的「正交扰动增大」在闭环中的累积表现；
- 到 step289 时，CCA 的夹爪位置与 euclidean 不同，恰好进入碰底座区域。

## 4. 判决（按 B 退出标准）

命中：

> **「CCA 的动作变化可追踪到后续 mug 碰撞 → 支持各向异性造成的闭环安全风险」**

具体：
- CCA 从 step0 起的各向异性方向重分配（ΔD_⊥>0）改变闭环轨迹；
- 轨迹分叉在 step289 累积成夹爪碰底座 → mug 撞飞；
- 这是「D_∥↓ + D_⊥↑ → CAR↓」的真实反例，印证 E0 的 trade-off 预言。

## 5. 关键科学含义

$$\boxed{D_\parallel\downarrow \quad\not\Rightarrow\quad \mathrm{CAR}\uparrow}$$

任务方向上的动作保留 ≠ 真实机器人安全。CCA 的低秩加权 QP 保留了 CBF 约束，
但**单条 CBF 约束无法刻画「夹爪碰底座 → 底座带 mug」的间接传递链**（底座不在
CBF 椭球内，它只是 PROTECTED_SUPPORT）。正交方向的大幅修正（ΔD_⊥>0）在长时域
闭环中累积，最终进入 CBF 未覆盖的间接危险区。

## 6. 诚实边界

1. 27.5mm（非最初的 290mm）是稳定复现值；290mm 是更早 smoke 的偶发更大值，
   两者同因（夹爪碰底座），数值差异来自更长的分叉累积；
2. 未证明「每个状态都会撞」，只证明 S01 这个 mug 紧贴抓取路径的困难状态会撞；
3. β=0 的那一步说明「碰撞不是各向异性直接撞」，但「各向异性改变轨迹」是
   必要条件（euclidean 同样名义动作不会到碰底座的位置）。

## 7. 工程产物

- `tools/cca_asf_e1_catastrophic_audit.py`：首次 CAR 越界捕获。
- `data/audit/cca_asf_e1_catastrophic_audit.json`（B 脚本 6 次 300 步未复现，需用
  run_episode 550 步才能复现——撞飞在 step289）。
