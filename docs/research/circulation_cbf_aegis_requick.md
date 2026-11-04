# Circulation CBF 重验证（四元数修复后）

> 日期：2026-11-04
> 代码基线：HEAD `cc7d39a`（四元数 p1 修复 + 几何 MVEE 修复）
> 方法：Control Barrier Functions With Circulation Inequalities（Gonçalves 等，T-CST 2024）
> 协议：与 bug 时代相同——确定性 PnsgPolicy（CRN 固定 seed），5 init × 3 配置（baseline / ccw / cw）
> 前作：`docs/research/circulation_cbf_aegis.md`（已作废，本文件为重验证结果）

---

## 1. 结果（修复后，确定性 CRN）

| init | baseline | circ_ccw | circ_cw |
|---|---|---|---|
| 0 | **成功**，mug 11.1mm，soup 1.1cm | 失败，mug **175.0mm**，soup 2.1cm | 失败，mug 0mm，soup 13.2cm |
| 1 | **成功**，mug 7.9mm，soup 0.9cm | 失败，mug 9.1mm，soup 2.2cm | 失败，mug 26.3mm，soup 13.6cm |
| 2 | **成功**，mug 27.1mm，soup 0.2cm | 失败，mug 5.5mm，soup 2.5cm | 失败，mug 0.9mm，soup 2.1cm |
| 3 | 失败，mug 0mm，soup 20.4cm | 失败，mug 10.5mm，soup 0.8cm | **成功**，mug 0mm，soup 0.5cm |
| 4 | 失败，mug 5.5mm，soup 1.9cm | 失败，mug 3.6mm，soup 0.6cm | 失败，mug 0mm，soup 0.3cm |

**汇总**：

| 配置 | 成功率 | mug 位移（mm） | 末端到 soup 最近（cm） |
|---|---|---|---|
| A_baseline | **3/5** | [11.1, 7.9, 27.1, 0, 5.5] | [1.1, 0.9, 0.2, 20.4, 1.9] |
| B_circ_ccw | **0/5** | [175, 9.1, 5.5, 10.5, 3.6] | [2.1, 2.2, 2.5, 0.8, 0.6] |
| C_circ_cw | **1/5** | [0, 26.3, 0.9, 0, 0] | [13.2, 13.6, 2.1, 0.5, 0.3] |

## 2. 与 bug 时代对比

| 指标 | bug 时代（作废） | 修复后 |
|---|---|---|
| baseline 成功率 | 0/5 | **3/5** |
| baseline 末端到 soup | 21cm 卡死（够不着） | **0.2-1.9cm**（成功时抓到） |
| baseline mug 位移 | 20.5mm（init0） | 11.1mm（init0，更小） |
| circulation ccw 作用 | 「缓解」平衡点（soup 21→7.4cm） | **有害**：0/5 成功，init0 撞飞 mug 175mm |
| circulation cw 作用 | 大多无效 | 1/5 成功，其余把末端带偏（soup 13cm） |

## 3. 核心结论：修复后 circulation 从「缓解」变为「有害」

### 3.1 之前的「平衡点卡死」是四元数 bug 假象

bug 时代的核心观察是「AEGIS 在 long task0 卡死在稳定虚假平衡点（h 恒为负，末端被
mug 径向推离、无法绕行）」。修复后证明：

- **p1 修复前**：末端椭球中心偏移 16cm，导致 z_fixed（末端→障碍物方向）错误，
  末端被「错误方向」的 CBF 推离，卡在 21cm；
- **p1 修复后**：末端椭球位置正确，baseline 直接能完成任务（3/5 成功，末端到 soup
  0.2-1.9cm = 真抓到）。

**所以「平衡点问题」本身是 bug 假象，不是 AEGIS 的真实缺陷。**

### 3.2 修复后 circulation 变成「画蛇添足」

circulation 的切向强制约束，在 baseline 已经能完成任务的情况下：

- **ccw（逆时针）**：强制末端绕障碍物逆时针走，破坏了 baseline 的抓取路径。
  - 0/5 成功（baseline 的 3 个成功 init 全被 ccw 破坏）；
  - init0 撞飞 mug 175mm（baseline 仅 11.1mm），灾难性恶化；
  - 介入次数暴涨（490-512 vs baseline 118-149），说明切向约束与 CBF 约束持续冲突。
- **cw（顺时针）**：大部分把末端带偏（soup 距离 13cm），仅 init3 成功（baseline 失败
  的 init，cw 恰好绕对了）。

### 3.3 之前的「7.4cm 禁区」「末端椭球覆盖任务物体」根因也是 bug 假象

bug 时代结论「末端椭球 y 半轴 12cm + mug 椭球 7.3cm ≈ 19.3cm 禁区，末端永远无法到达
soup 正上方」——这是建立在**错误的 p1（偏移 16cm）**和**错误的 mug 椭球（AABB 而非
MVEE）**之上的。

修复后：末端到 soup 最近 0.2cm（init2），证明「末端椭球覆盖任务物体」的禁区在正确
几何下**不成立**——末端椭球 CBF 能正确区分「接近 soup 抓取」与「碰撞 mug」。

## 4. 最终裁决

| 问题 | bug 时代结论 | 修复后结论 |
|---|---|---|
| 平衡点卡死 | 存在，circulation 缓解 | **不存在**（p1 bug 假象） |
| circulation 价值 | 对症缓解 | **有害**（画蛇添足，破坏 baseline 抓取） |
| 末端椭球覆盖任务物体 | 根因 | **不成立**（几何 bug 假象） |
| AEGIS 在 long task0 的能力 | TSR 0% | **3/5 成功**（正确几何下能完成） |

**circulation 约束应归档为「基于 bug 假象的无效扩展」**：它针对的「平衡点问题」在
正确几何下不存在，反而破坏了 baseline 已经能完成的抓取任务。

这与 CCA-ASF 的结局一致：两个基于 bug 时代 AEGIS 的扩展（CCA-ASF 改目标、circulation
加约束），在四元数修复后都被证明「有害」或「无效」。**核心 AEGIS 修复才是真正解决
问题的，扩展都是建立在 bug 假象上的画蛇添足。**

## 5. 诚实边界

- n=5 init 是最小验证（与 bug 时代同规模），不能下强统计结论；
- 但 baseline 3/5 vs ccw 0/5、init0 撞飞 mug 175mm 是明确的方向性信号；
- 若要发表级结论，建议扩大到更多 init state + 更多 stream。
