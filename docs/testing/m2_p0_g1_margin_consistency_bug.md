# M2-P0 G1 前置：碰撞检测 margin 一致性缺陷

> 日期：2026-11-02
> 状态：发现评测基础设施第三个缺陷（margin=0 碰撞检测漏检）。
> 前置：`m1_contract_coverage_root_cause_audit.md`（底座漏检）、
> `m1_t0_task_integrity_audit.md`（TASK 漏检）。

---

## 0. 结论

> `classify_ik` 的「SAFE_PREGRASP」判定用 `mj_collision(m, d)` 的**默认 margin=0**
> 碰撞检测，而 `min_gap_full` 用 margin=0.05 的查询。两者不一致：margin=0 的
> 宽相位检测**漏掉了部分真实穿透**（凸网格接触检测在零 margin 下的数值问题），
> 导致 P0r2/P0e-r 的「安全预抓取解」里混入了约 14% 的真碰撞解。

## 1. 证据

修复（正 margin + 排除相邻 body 自碰撞）前后 S01 的 `classify_ik` 判定对比：

| 判定 | 修复前（margin=0）| 修复后（margin=0.05）|
|---|---|---|
| SAFE_PREGRASP | 21 | **5** |
| PROTECTED_OBJECT_COLLISION | 0（漏检）| 7 |
| SELF_COLLISION（非相邻）| 0（漏检）| 3 |
| IK_NONCONVERGED | 其余 | 63 |

**实际影响比最初估计严重**：21 个「SAFE」里只有 5 个是真的，**约 76% 是假安全**
（7 个 robot-mug 碰撞 + 3 个真自碰撞被 margin=0 漏检，其余为 IK 未收敛误判）。

## 2. 根因（两处）

1. **margin=0 宽相位漏检**：`mj_collision` 默认 margin=0，凸网格（convex mesh）
   之间的浅穿透 / 接近因宽相位不触发而漏检，`contact.dist` 不报告；
2. **修复后自碰撞误报**：正 margin 下，相邻连杆（父-子 body）的正常关节连接接触
   被误判为 SELF_COLLISION。修复需排除相邻 body（`body_parentid` 判断），且只保留
   真实穿透（dist < -1mm）。

## 3. 影响范围（重大）

- P0r2 的「安全预抓取解」数量被**严重高估约 76%**（S01 的 20 安全解实际仅约 5 真）；
- P0e-r 的「安全路径」主路径选取（gap 排序）同样受影响——它选的「gap=50mm 最大
  裕度」路径可能实际是碰撞的；
- **但 P0e-r 的最终验收不受影响**：P0e-r 的 E2 验收（Dmax=0、forbidden=0）用的是
  `monitor_contacts` 读 `mj_step` 约束求解器的真实物理接触，比几何查询可靠。
  P0e-r 报告的「真实执行安全」仍是真值；受影响的是「候选搜索 + 路径选取」环节
  的 SAFE 判定，它可能选到了并非真正无碰撞的路径。

## 4. 修复

`classify_ik` 的碰撞判定改为：
1. 正 margin（0.05）查询，与 `min_gap_full` 一致；
2. 自碰撞判定排除相邻 body（`body_parentid` 判断父-子），只保留非相邻 + dist<-1mm
   的真自碰撞。

## 5. 对 M2-P0 的含义

G1 候选注册表的「物理安全」判定必须用修复后的 `classify_ik`。修复后 S01 的安全
解从 21 降到 5，候选注册表的候选池大幅缩小——这是**更诚实**的结果，避免了假安全
候选污染 M2-P0 的接管状态区分实验。

## 6. 与之前两个缺陷的关系

这是评测基础设施的**第三个**缺陷，三者共同指向一个根本问题：

> 碰撞检测的「margin / 宽相位 / 角色清单」三者必须一致且完整，否则「安全」判定
> 会在不同层面漏检。M1-Contract-Audit 修了角色清单（底座），M1-T0 修了 TASK，
> 本缺陷修的是 margin 一致性。
