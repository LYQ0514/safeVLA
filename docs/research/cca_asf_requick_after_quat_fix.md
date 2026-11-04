# CCA-ASF 简单重验（四元数 bug 修复后）

> 日期：2026-11-04（2026-11-04 修订：修正指标口径歧义，见 §2.2）
> 背景：CCA-ASF v1 归档判决「闭环安全恶化（S01/st3 可重复 27-30mm mug 位移，
> euclidean 对应复现为 0）」完全基于 bug 时代的 AEGIS（四元数 bug 致 p1 偏移 16cm）。
> 本重验用修复后代码（SHA `b2226d9`）+ 确定性 PnsgPolicy（CRN seed）复查该判决。

---

## 1. 重验协议

| 项 | 值 |
|---|---|
| 场景 | S01（`safelibero_long` L0，`state_id=1`） |
| policy | 确定性 PnsgPolicy（CRN seed，消除 π0.5 非确定） |
| 对比 | `euclidean` vs `cca_asf`（同 init，同 stream） |
| 样本 | 4 个 stream（st0-3） |
| 脚本 | `tools/cca_asf_requick.py` |
| 指标口径 | 规格 §7 分离：TSR_native / CAR_native / SSR_native |

## 2. 结果

### 2.1 mug 最大位移（L1 范数，项目 CollisionTracker 口径）

| stream | euclidean mug 位移 | cca_asf mug 位移 | TSR（两法） |
|---|---|---|---|
| st0 | 25.6mm | 13.1mm | 均失败 |
| st1 | 30.5mm | 16.3mm | 均成功 |
| st2 | 8.8mm | 13.3mm | 均成功 |
| st3 | 16.0mm | 26.1mm | 均成功 |

**均值**：euclidean = 20.2mm，cca_asf = 17.2mm（n=4）。

**介入次数**：euclidean = [432, 171, 136, 152]，cca_asf = [148, 130, 112, 141]。

### 2.2 指标分离（关键修正）

8 个 episode 的 mug 位移**全部 ≥ 8.8mm**（范围 8.8–30.5mm），全部远超 CAR 阈值
**1mm**。因此：

| 指标 | euclidean | cca_asf |
|---|---|---|
| TSR_native | 3/4 | 3/4 |
| **CAR_native（位移≤1mm）** | **0/4** | **0/4** |
| **SSR_native（TSR∧CAR）** | **0/4** | **0/4** |

> ⚠️ **不要把 `success=True` 当安全成功**。st1 的 euclidean 是「mug 位移 30.5mm
> 且 success=True」——这是「任务完成（TSR）但严重碰撞（CAR 违规）」，SSR 仍为 0。
> 这正是规格 §7 点名的陷阱。

## 3. 核心结论

### 3.1 原判决被颠覆（成立）

原归档判决（bug 时代）：

$$\boxed{\text{CCA 27-30mm},\ \text{euclidean 复现为 } 0}$$

修复后：

$$\boxed{\text{euclidean 本身 } 8.8\text{-}30.5\text{mm},\ \text{CCA 相对 euclidean 无一致恶化}}$$

### 3.2 原「euclidean=0」是 bug 假象（成立）

修复前，四元数 bug 使 AEGIS 椭球中心 p1 偏移 16cm，导致 euclidean 的椭球「看
起来」完美避开了 mug（实际是椭球位置错了、没保护到真实障碍）。修复后 p1 正确，
euclidean 基线本身就有 8.8-30.5mm 的 mug 位移——**这是真实物理**（S01 任务夹爪
确实会接近/碰到 mug），而非「CCA 引入的灾难」。

### 3.3 「CCA 不再是灾难性恶化」（成立，但表述需谨慎）

修复后 CCA 相对 euclidean 是 mixed（2 升 2 降），不再有「CCA 灾难、euclidean 0」
的反差。原「各向异性重分配 → 灾难性 CAR 违规」这条因果链**不再成立**。

### 3.4 「CCA 更安全 / 不劣」（不成立，已撤回）

**此前「CCA 均值 17.2mm vs euclidean 20.2mm，CCA 略好」的表述是误导性的，撤回。**

正确表述：

> 在 CAR（1mm）口径下，euclidean 与 cca_asf **均 100% 违规（CAR=0/4）**，SSR 均为 0。
> 二者差异仅在 mug 被撞飞的距离（碰撞严重程度），**不构成「CCA 更安全」或「安全不劣」**。
> 「CCA 介入更少（112-148 vs 136-432）」也不等于「更安全」——它可能只是「少干预、
> 但同样会撞」。

这与规格 §12 一致：尚不可说「CCA 每次干预更少所以更安全」。

## 4. 边界与诚实声明

1. **样本太小（n=4）**：这是「简单重验」，不能下强统计结论。要严格确认需扩大
   到更多 stream/state（规格 §8.1 Pilot 为 S01/S02/S04/S09 × 4）并做配对检验；
2. **两种方法在 S01 上 CAR 均为 0**：意味着「安全过滤（euclidean 或 CCA）都没能
   保护 mug 不被撞」——这本身是 AEGIS 单条 CBF 的已知局限（间接传递链 gripper→
   base→mug 不在椭球内），不是 CCA 引入的新问题；
3. **未重验的结论**：E0 的「CCA 减少任务方向代理偏移」（46/46）是 QP 层面数学，
   理论上不依赖 p1，但仍未在本轮重验；
4. **指标口径已修正**：`tools/cca_asf_requick.py` 已改为 TSR/CAR/SSR 分离输出。

## 5. 对归档的影响

- **CCA-ASF v1 的「闭环安全风险」判决需要改写**：原「灾难性 27-30mm / euclidean 0」
  是四元数 bug 假象，修复后不成立；
- **但「CCA 有效/不劣」同样未被证明**：当前 n=4 证据只说明「修复后 CCA 相对
  euclidean 无一致恶化」，两者 CAR 均为 0；
- **是否重启 CCA-ASF 线**：待定。需要先按规格 §9 的 G0-G10 验收矩阵补正式
  Pilot（16 对 32 episodes），才能下「是否值得继续」的结论。
