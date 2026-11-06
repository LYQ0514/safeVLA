# FPCK/CKG 跨任务正向搜索的工程排查（角色清单硬编码问题）

> 日期：2026-11-06。排查 goal 套件 positive search 为何 bowl 接触被误判。

## 结论：发现 contract_objects.py 角色清单硬编码为 long 套件，跨套件不可用

跨任务 positive search（goal 套件）暴露了 `contract_objects.py` 的**架构性问题**：
角色清单（PROTECTED_OBJECTS / TASK_OBJECTS / DISTRACTOR_OBJECTS）全部硬编码为
long 套件的对象名，跨套件不通用。

## 诊断证据（goal task0 state4）

```
step=13: unclassified=[('milk_obstacle_1_main', 'gripper0_right_gripper')]
```

- goal 套件的障碍物叫 `milk_obstacle_1_main`，不在 `PROTECTED_OBJECTS`（硬编码为
  long 的 `red_coffee_mug_obstacle_1_main` 等）里 → 归 unclassified；
- goal 套件的 task 物体叫 `akita_black_bowl_1_main`，不在 `TASK_OBJECTS`（硬编码为
  long 的 `alphabet_soup_1_main` 等）里 → 归 unclassified；
- bridge 的 `forbidden = len(mon["forbidden"])>0 or len(mon["unclassified"])>0`
  把 unclassified 当 forbidden → goal 套件任何接触都误判为 forbidden。

## 三个叠加的 bug

| # | bug | 影响 |
|---|---|---|
| 1 | `task_contact` 用 `startswith("robot0")` 漏 gripper0 | 已修复（f90f2cc） |
| 2 | `TASK_OBJECTS` 硬编码 long 的 soup/cream/basket | goal/spatial 的 bowl 归 unclassified |
| 3 | `PROTECTED_OBJECTS` 硬编码 long 的障碍物 | goal 的 milk 等障碍物归 unclassified |
| 4 | `forbidden = ... or len(unclassified)>0` | unclassified 全被当 forbidden |

## 正确架构

`role_manifest.py` 已经能从 BDDL 动态生成每个 task 的角色清单（MANIPULATED_TARGET /
GOAL_RECEPTACLE / STATIC_SAFETY_OBSTACLE / DISTRACTOR）。bridge 应该用动态 manifest
角色，而不是 `contract_objects.py` 的硬编码清单。

## 影响范围

1. **long 套件的 E1a 结果不受影响**：long task0 的对象恰好在硬编码清单里
   （soup/cream/mug），所以 State02/S01/S04/positive control 的结论正确；
2. **跨任务 positive search 受影响**：goal/spatial 的 bowl/milk 接触被误判，
   之前的 goal task0 state4「假成功」结论不可信（bowl 接触被误判 forbidden）；
3. **需要修复后才能做跨任务搜索**：改用动态 manifest 角色。

## 下一步

修复 bridge 用动态 manifest 角色（而非 contract_objects 硬编码），再重新做
跨任务 positive search。这是架构性修复，需谨慎。
