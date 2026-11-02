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

S01 的 21 个 `classify_ik` 判 SAFE 的解中，3 个经 `min_gap_full`（margin=0.05）
查询发现 gap < 0（真实穿透 −2.7 至 −14mm）。

| 状态 | classify_ik SAFE | min_gap_full<0 真碰撞 |
|---|---|---|
| S01 | 21 | **3（14%）** |

## 2. 根因

`classify_ik` 里 `mj_collision(m, d)` 未设置 `geom_margin`，用默认 margin=0。
`min_gap_full` 先设 `geom_margin=0.05` 再 `mj_collision`。

margin 扩大宽相位检测范围，使 `contact.dist` 报告「接近 + 穿透」的完整几何关系；
margin=0 时凸网格（convex mesh）之间的浅穿透可能因宽相位不触发而漏检。

## 3. 影响范围

- P0r2 的「安全预抓取解」数量被高估约 14%；
- P0e-r 的「安全路径」里可能有同样比例的假安全（因选路径时用了同样的 `classify_ik`）；
- **但这不推翻 P0e-r 的核心结论**：P0e-r 的最终验收（Dmax=0、forbidden=0）用的是
  `monitor_contacts`（margin=0 但读 `mjData.contact`），而执行阶段的真实物理接触
  由 `mj_step` 的约束求解器产生，比几何查询更可靠。P0e-r 的「真实执行安全」结论
  不受影响；受影响的是「候选搜索阶段」的 SAFE 判定。

## 4. 修复

`classify_ik` 的碰撞判定必须与 `min_gap_full` 一致：用正 margin 查询确认无碰撞。
即：SAFE_PREGRASP 的判定条件是「margin=0.05 查询下 robot↔forbidden 无接触」。

## 5. 对 M2-P0 的含义

G1 候选注册表的「物理安全」判定必须用修复后的 `classify_ik`（正 margin），
否则候选注册表会混入假安全候选，污染 M2-P0 的接管状态区分实验。

## 6. 与之前两个缺陷的关系

这是评测基础设施的**第三个**缺陷，三者共同指向一个根本问题：

> 碰撞检测的「margin / 宽相位 / 角色清单」三者必须一致且完整，否则「安全」判定
> 会在不同层面漏检。M1-Contract-Audit 修了角色清单（底座），M1-T0 修了 TASK，
> 本缺陷修的是 margin 一致性。
