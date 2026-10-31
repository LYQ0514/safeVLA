# 碰撞事件身份审计：substep 接触 vs step 后接触

- **日期**：2026-10-31
- **性质**：按 docs/想法v3.md 的优先级，做碰撞评测基础修正

---

## 一、决定性发现：旧数据严重漏检瞬时接触

在 task0（必碰撞任务）跑名义轨迹，逐步对比两种接触检测：

| 检测方式 | 累计接触次数 |
|---|---|
| **substep 级**（复刻 env.step 循环，25 substep/step 逐一检查） | **547** |
| **step 后**（env.step 返回后看 data.contact，旧数据方式） | **24** |

**相差 23 倍。**

示例（step 34~59）：

| step | substep 接触 | step 后接触 | 障碍位移 L2 |
|---|---|---|---|
| 34 | 8 | 1 | 44.2mm |
| 44 | 25 | 1 | 64.2mm |
| 59 | 25 | 1 | 87.6mm |

机械臂在每个 physics substep 都在擦碰障碍物，但 step 结束时接触已断开，
只有少数时刻接触残留到 step 结束。

---

## 二、结论：触发 v3 第 6 条「必须重跑」

| v3 重跑条件 | 是否触发 |
|---|---|
| 旧数据没有 substep contact 记录 | ✅ 是 |
| 历史 action 无法确定性复现 | ✅ 是（baseline 用随机 z） |
| 三种方法误报率不同 | 待验证 |
| 新口径改变排序 | 很可能 |
| AEGIS 效果集中在 1-2mm | 待验证 |
| SCENE_PHYSICS_UNSTABLE 分布不均 | 待验证 |
| target 与 robot 混淆 | 是（旧位移法不区分） |

**因此，旧三方 CAR（π0.5 13.8% / AEGIS 60.4% / VLPSA 19.8%）不能作为
真实 robot-obstacle collision avoidance，只能标记为：**

```
LEGACY_DISPLACEMENT_METRIC_PENDING_CONTACT_RECONCILIATION
```

---

## 三、新的三层指标（已实现基础设施）

`src/pnsg/contact_audit.py`：

1. **ContactEventRecorder**：substep 级 forbidden contact 记录器，
   区分 robot-obstacle 与 target-obstacle，记录 geom/body pair、dist、substep
2. **DisplacementRecorder**：同轨迹 L1/L2 位移（含数学不变量断言）
3. **zero_action_stability**：场景物理稳定性检查

---

## 四、下一步

按 v3 的 relabel audit 流程：

1. 分层抽样旧数据（L1 1-1.5mm / 1.5-3mm / 3-5mm / 5-50mm / >50mm + 对照组）
2. 每条 replay：substep contact + L1/L2 + 稳定性 + first contact pair
3. 计算三种方法各自的「旧 L1 标签 vs robot contact」混淆矩阵
4. 判断旧三方实验是否需要正式重跑（大概率需要）
5. 从确认过的真实 robot/target-obstacle 接触中选 PNSG 正对照

**重要**：PNSG 大规模评测继续暂停。steering 机制验证（源空间 Δz 改善净空
20mm）仍然有效，不受此影响。
