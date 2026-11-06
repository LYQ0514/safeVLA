# FPCK/CKG Positive Control state3 验收报告（Gate 5 候选 2/3）

> 日期：2026-11-06。判定：state3 也不是完整 SAFE_REENTRY，而是「障碍物碰撞假成功」。

## 结果

| 项 | 值 |
|---|---|
| case | positive_control state3（候选 2/3） |
| final_e1_label | **TASK_INTEGRITY_FAIL** |
| violation_code | STATIC_OBSTACLE_MOVED（step 18） |
| native_success | true（step 357） |
| first_task_contact | None |
| first_controlled_witness | None |
| final_lifecycle | soup=RESTING, cream=RESTING |
| full_lifecycle_witness | false |

## 事件链

```
step 18: STATIC_OBSTACLE_MOVED（mug 位移 >1mm）→ q_bad
step 357: native_success=true（soup+cream 进 basket）
→ 但无 TASK_CONTACT / CONTROLLED 见证
→ 又是「碰撞推动假成功」
```

## 两个候选的共同模式

| 候选 | 违规码 | 违规步 | native_success | 见证 |
|---|---|---|---|---|
| state2 | FORBIDDEN_CONTACT | 55 | true@277 | 无 |
| state3 | STATIC_OBSTACLE_MOVED | 18 | true@357 | 无 |

**共同点**：π0.5 在 long task0（soup 紧邻 mug）上完成任务的路径，几乎都是「碰撞推动
soup / 碰障碍物」，很少有「合法抓取→运输→释放」的完整生命周期。

这印证了之前 STG 实验的发现：π0.5 在 soup-mug 紧邻场景的抓取能力不足，native success
大量是「碰撞推动」造成的假成功。

## 对 Gate 5 的裁决

- state3 **未通过** Gate 5（STATIC_OBSTACLE_MOVED 假成功）；
- 按裁决，继续候选 3/3（state5）。

## 重要观察（供总裁决参考）

前两个候选都是假成功，说明：

1. Gate 5（完整生命周期 SAFE_REENTRY）在 long task0 上**可能很难找到正例**——因为
   π0.5 在该场景的合法抓取本身稀少；
2. 这本身是一个**重要的负结果**：native TSR 高（12/20），但「合法完成」的比例可能
   远低于 native success，因为大量 success 是碰撞推动的假成功；
3. 需要 state5 跑完后综合判断 Gate 5 是否可 PASS（若 3 个候选都无完整生命周期，
   Gate 5 应判「未找到正例」，需扩大搜索或转向其他 task/suite）。
