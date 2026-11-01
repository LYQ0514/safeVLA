# 研究全景：从复现到 PNSG 收口（2026-10-28 ~ 11-01）

> 日期：2026-11-01
> 性质：整理文档，串联近期全部数据与研究结论，供一次性回顾与后续引用。
> 各阶段细节以对应 testing / explanation 文档为准，本文只做索引与结论汇总。

---

## 0. 一句话全景

以 SafeLIBERO 为基准，先**复现**两个推理期安全层（AEGIS / VLPSA）并建立统一口径；
再**审计**碰撞机制（PNSG 接触契约）；随后验证**源空间 steering 是否优于动作空间修正**
（负结果，代价匹配解释）；最后探索**源空间候选能否改善 AEGIS 的 long 推进**
（动机未获支持，暂停）。四条线全部收口，未建立源空间 steering 相对动作空间修正的
系统性优势。

---

## 1. 复现阶段：三方安全方法对比

### 1.1 口径建立（`2026-10-29-pi05-full-baseline-calibration.md`）

冻结五个指标，且**两个碰撞口径必须一起报**（实测存在「CAR 与 CAR>5mm 结论相反」）：

| 指标 | 定义 |
|---|---|
| TSR | 任务成功率 |
| CAR（主）| 障碍物位移 ≤ 1mm（官方口径）|
| CAR>5mm（辅）| 位移 ≤ 5mm（无严重碰撞率）|
| SSR | 成功且无碰撞（联合指标，抗注水）|
| ETS | 平均执行步数（双口径：含超时 / 仅成功）|

### 1.2 三方对比（各 1600 条，32 场景 × 50）

| 方法 | TSR | CAR | CAR>5mm | SSR | ETS |
|---|---|---|---|---|---|
| π0.5 基线 | 59.8% | 13.8% | 87.7% | 12.4% | 259.7 |
| **AEGIS** | 44.9% | **60.4%** | 96.6% | **32.2%** | 302.1 |
| VLPSA（复现）| 41.7% | 19.8% | 91.2% | 10.4% | 293.0 |

### 1.3 关键结论

- **AEGIS**（`2026-10-29-aegis-comparison.md`）：避障有效（CAR 4.4 倍、SSR 2.6 倍），
  但**牺牲任务**（TSR −14.9pp）。根因：**long 套件「安全但无用」**——CAR 升到 68% 但
  TSR 崩到 14%，SSR 零收益。分界线是「障碍物是否挡住必经路径」。
  消融推翻「大椭球是主因」，证明是**长程多物体任务与末端 CBF 的结构性冲突**。
- **VLPSA**（`2026-10-30-vlpsa-comparison.md`）：全身 CBF 在本协议下全面落后 AEGIS
  （CAR 19.8%、SSR 10.4% 三方最差）。唯一亮点是 long 套件 TSR 不崩（62%/51% vs
  基线 62%/55%），但这是「温和约束」的代价，不是优势。
- 数字只能表述为「控制层本机表现」，不能与论文直接对比（感知 oracle vs 云 VLM、
  环境差异、上游缺陷）。

---

## 2. PNSG 碰撞机制审计（`docs/想法v10.md` + Step 1-5）

这是「碰撞发生了什么」的机制层，不替代官方 CAR。

- **对象角色门禁**（ROLE_MANIFEST_GATE PASS）：S-expression AST 解析 17/17 任务。
- **接触分类契约 v2.1**（CONTACT_CLASSIFICATION_CONTRACT PASS 20/20）：语义分类与
  allowance 分离、角色顺序无关、初始/新增接触分离。
- **时间身份**：substep 级去重 + 连续事件合并（修复了「聚合错误」）。
- **task0 五级机制**：robot–obstacle 444 native 记录、new object–environment 1916、
  preexisting support 113799，全部分类。
- **关键修正**：撤回「step-end 接触漏检 23 倍」旧判断（实为 control-step 层只漏
  1/67）；确认官方 CAR 触发与 robot–obstacle 接触机制时间一致（单条 rollout 观察）。

---

## 3. PNSG 源空间 steering 正对照（负结果，收口）

### 3.1 研究问题

源空间 steering（噪声空间 z 的最小范数修正）是否比动作空间修正提供**真实安全上的
额外价值**？

### 3.2 关键方法与协议修正（`pnsg_steering_final_report.md`）

- **obs 身份缺陷**（10-31）：机器人 eef sensor 读 `site_xpos[grip_site]` 非 body；
  `env.step` 返回带采样相位的缓存；复现必须加载保存 obs 而非从 snapshot 重建。
  force 刷新属另一种身份（同步诊断），不能当修复。→ 冻结 deployment / synchronous 两身份。
- **完整状态**：snapshot = time+qpos+qvel；AEGIS 等有状态控制器须连内部状态一起保存。

### 3.3 主结果：12 状态固定批次（deployment 身份）

状态级四分类：**both 5 / source_only 2 / action_only 1 / neither 4**。
修复率 Source 7/12 vs Action 6/12；both 状态的 C_act 配对 Action 更省 4/5。

### 3.4 代价匹配诊断（`pnsg_cost_match.md`）

两个 source_only 状态，沿 Δa* 缩放匹配 Source 的 C_act 后，**Action 均 5/5 修复**。

### 3.5 最终结论（冻结）

> 在当前 π0.5、任务切片、代理约束、两步认证和候选生成规则下，实验未建立源空间
> steering 相对动作空间修正的额外价值，因此停止扩大当前方案。已验证的局部修复能力
> 仍成立；未成立的是「更优干预方法」的依据。

**核心洞察**：源空间小步长经非线性前向会放大成动作域大修正（ep06：β=0.25 → C_act=1.44
vs 动作 β=1.0 → 0.145），源空间的表观优势主要是**幅度效应**，不是方向优势。

---

## 4. AEGIS 上游 Source 改造探索（动机未获支持，暂停）

### 4.1 设计（`aegis_source_candidate_design_v1.md`）

原推理：AEGIS 局部投影阻碍推进 → Source 提供更好起点 → AEGIS 处理剩余风险。
三组比较：G1 原 AEGIS / G2 Action+前瞻 / G3 Source+前瞻。

### 4.2 最小采集（6 episode long task0，显式源噪声）

- 6/6 soup 未完成；4/6 cream_cheese 完成且持续保持。
- 时间对齐（`aegis_long_alignment_analysis.md`）：**「CBF 阻碍推进」动机未获支持**。
  - 主导模式是「nominal 本身未接近 soup」（策略提案不足，5/6）；
  - EP2 是「抓取失败」（反复抓但抓不牢，EEF 贴到 0.042m）。
- 安全表：CAR ep0/ep2 碰撞（ep2 严重 197.8mm），ep1/3/4/5 无碰撞。

### 4.3 结论

冻结的 Source 生成器 `min‖Δz‖ s.t. 当前安全约束` 只修当前风险，**不鼓励接近 soup、
改善抓取或完成剩余子目标**，与观察到的失败之间缺少作用机制。→ 暂停，不进入续轨。

---

## 5. Source→AEGIS 串接实现（归档，停止规模扩展）

### 5.1 实现（`source_aegis_concat_archive.md`）

链路：`球体代理检测危险 → 源空间最小范数修正 → AEGIS CBF-QP 过滤`。
**执行边界**：无 C_2 认证，是「AEGIS 过滤后执行」，不能继承「认证后执行」表述。

### 5.2 冒烟结果

- spatial task0 5 episode：TSR 20% / CAR 80% / SSR 20%（小样本）。
- 单案例 ep35：基线碰撞 12.15mm → 组合后 0.002mm（**无同状态基线对照，不归因 Source**）。

### 5.3 成本与结论

- 5 条 37.2 分钟，外推 400 条约 50 小时（小样本估计，非普遍不可行）。
- steering 触发频（ep04 达 63 次），但「是否误报」未量化（全臂覆盖 ≠ 过度保守，
  需区分几何误差 / chunk 时间差异 / 末端 CBF 未覆盖的真实全臂风险）。
- **结论**：触发机制风险辨别能力未量化、成本高、源空间增量价值未建立 → 停止规模扩展。

---

## 6. 贯穿全程的方法论教训

1. **obs 身份陷阱**：传感器读 site 非 body；`env.step` 返回缓存带采样相位；复现必须
   保存并加载实际输入，不能从状态重建。否则「同一 z0 生成不同动作」。
2. **幅度 vs 方向混淆**：比较两种修正方法必须对齐部署动作代价，否则把「改得更多」
   误读成「改得更好」。
3. **代理 ≠ 真实**：球体代理会高估穿透（~200×），真实安全只认 MuJoCo native 几何。
4. **rollout 跨进程发散**：同一 episode 两次运行可能选到不同危险 query，动摇了
   「状态」作为固定样本的定义。
5. **完整状态含控制器内部状态**：AEGIS 等有状态控制器（z_fixed/p2/计数器）须连同
   MuJoCo 状态一起保存/恢复。
6. **止损纪律**：多次「先跑一点再看」导致投入增加而核心证据未增强；最终以「无新的
   可检验机制则不重新立项」收口。

---

## 7. 全部结论一览（按时间）

| 阶段 | 结论 |
|---|---|
| AEGIS 复现 | 避障有效但牺牲任务；long 套件「安全但无用」 |
| VLPSA 复现 | 全身 CBF 落后 AEGIS；协议下不划算 |
| PNSG 机制审计 | 接触分类契约 v2.1、五级机制、CAR 对齐 |
| PNSG 正对照 | 源空间无相对优势；代价匹配解释独占案例 |
| AEGIS 上游改造 | 「CBF 阻碍推进」动机未获支持，暂停 |
| Source→AEGIS 串接 | 链路可跑但成本高、增益未建立，停止扩展 |

## 8. 归档索引

- **git commit**：`23f4cf7`（C_2 审计）→ … → `f7ba810`（12 状态批次）→
  `5f4fded`（代价匹配）→ `30efcc4`（最终报告）→ `f2bae98`（AEGIS 上游收口）→
  `a489c27`（Source→AEGIS 归档）。
- **关键文档**：
  - 最终判断：`docs/testing/pnsg_steering_final_report.md`
  - AEGIS 复现：`docs/testing/2026-10-29-aegis-comparison.md`
  - 机制审计：`docs/想法v10.md`
  - AEGIS 上游收口：`docs/explanation/aegis_long_alignment_analysis.md`
  - 串接归档：`docs/explanation/source_aegis_concat_archive.md`
