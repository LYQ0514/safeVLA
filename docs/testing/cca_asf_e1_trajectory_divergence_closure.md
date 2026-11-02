# CCA-ASF E1-B1：Trajectory Divergence Closure

> 日期：2026-11-02
> 状态：一次性轨迹证据补全完成。
> 前置：`cca_asf_e1_catastrophic_car_audit.md`（B 归因）。

---

## 0. 三个问题的答案

**Q1（euclidean 是否避开底座区域）**：是。euclidean 在 Step270-295 无底座接触
（contact_base=False）、Dmax≈0（2.2e-8mm）。CCA 同窗口内 gripper-base 最小间隙
下界 12.03mm，但 mug 已在此前被撞飞（窗口内位移恒定 12.5mm）。

**Q2（首次控制差异时 β 是否非零）**：是。首次实质控制分叉在 step0（TASK B/D
已记录），CCA 的 β 从早期就非零。但撞飞**那一步** β=0（见 B 报告）。

**Q3（接触 → mug 位移）**：是。CCA 的 mug 位移在 step219 首次 >0.5mm，step221-222
短短 3 步内跳变到 >10mm，最终 30.7mm。这是突发碰撞事件，不是缓慢累积。

## 1. 精确时间线（本次复现）

| step | mug 位移 |
|---|---|
| 0-210 | 0mm（完全不动）|
| 219 | 首次 >0.5mm |
| 221-222 | 3 步内跳变 >10mm（突发碰撞）|
| 240 | 12.5mm |
| 270 | 16.1mm |
| 最终 | 30.7mm |

## 2. 证据链完整闭合

$$\text{step0 各向异性方向重分配（ΔD}_\perp>0）\ \Rightarrow\ \text{300 步闭环分叉}$$
$$\Rightarrow\ \text{step219-222 夹爪进入碰底座区域}\ \Rightarrow\ \text{底座带 mug}\ \Rightarrow\ \text{30.7mm CAR 违规}$$

- euclidean 同条件下 Dmax≈0（未分叉到危险区）；
- 碰撞步 β=0（名义动作可行），但轨迹分叉是各向异性早期修正的必要条件。

## 3. 证据等级（诚实区分）

**已验证**：`W_CCA ≠ W_AEGIS ⇒ u_CCA ≠ u_AEGIS`（step0 起分叉）→ 轨迹分叉 → CCA 分支
gripper-底座接触 + mug 27-30mm 违规（euclidean 无）。

**未严格证明**：`D_⊥↑ ⇒ 正交扰动累积 ⇒ 碰撞` 的**唯一致因**。E0 证明 ΔD_⊥>0，
但未把正交分量与其他状态变化隔离。因此只能说「各向异性修正改变了轨迹，且该改变
在 CBF 未覆盖区域产生不利闭环结果」，不能说「正交扰动是唯一碰撞机制」。

## 4. 工程产物

- `tools/cca_asf_e1_trajectory_divergence.py`
- `data/audit/cca_asf_e1_trajectory_divergence.json`
