# FPCK/CKG 跨任务正向搜索：中期结论（goal task0 已耗尽）

> 日期：2026-11-06。修复 3 个工程 bug 后，goal task0 正向搜索无完整 SAFE_REENTRY。

## 修复的 3 个工程 bug（本次排查）

| # | bug | 影响 | 修复 commit |
|---|---|---|---|
| 1 | task_contact 用 startswith("robot0") 漏 gripper0 | long 的抓取见证永不成立 | f90f2cc |
| 2 | TASK_OBJECTS 硬编码 long 的 soup/cream/basket | goal 的 bowl 归 unclassified | 9008d79 |
| 3 | forbidden 把 unclassified 当 forbidden | goal 任何接触误判 forbidden | 9008d79 |

修复后，bridge 改用 `_dynamic_contacts`（动态 target_bodies/obstacle_body 扫描
mjData.contact），不依赖 contract_objects 硬编码角色清单。

## goal task0 搜索结果

baseline 扫描 10 个 init，**只有 state4 是 native_success=True**：

| state | native_success | 结果 |
|---|---|---|
| 0-3 | False | 失败 |
| **4** | **True** | 接近阶段碰 milk → FORBIDDEN_CONTACT → q_bad |
| 5-9 | False（扫描在 state5 后中断） | 失败 |

### state4 的机制（修复后澄清）

```
step 13: π0.5 接近 bowl 时轻碰 milk（~1mm）→ FORBIDDEN_CONTACT → q_bad
step 127: task_contact（bowl 被接触）
step 131: gripper 闭合
step 141+: bowl 抬升 44mm、双指接触、相对漂移<1mm（合法抓取见证）
         但 step 13 已 q_bad，后续见证在吸收态被忽略
```

**关键**：state4 不是「碰撞推动假成功」，而是「接近阶段碰障碍物 → 之后合法抓取
被否决」。这证明 FPCK 在 goal 套件正确工作（识别 milk 障碍物 + 严格 q_bad 优先）。

## 当前结论

1. **goal task0 已耗尽**：唯一 success 候选 state4 失败于接近阶段碰 milk；
2. **还没有找到完整 SAFE_REENTRY**：Gate 5 仍是 INCONCLUSIVE；
3. **但发现了一个重要的机制区分**：
   - long task0：π0.5 碰撞推动 soup（不抓取）→ 假成功；
   - goal task0：π0.5 合法抓 bowl 但接近阶段碰 milk → 严格失败。

## 剩余预算

预注册规则：最多 2 个新任务 × 每任务 3 个 success 候选。
- 任务 1（goal task0）：已耗尽（1 success 候选，失败）；
- 任务 2（spatial task1）：尚未开始。

## 建议

继续 spatial task1（第二个预注册任务），看是否有「接近不碰障碍物」的 success 候选。
若 spatial task1 也找不到，则按裁决正式报告「SAFE_REENTRY support not established」。
