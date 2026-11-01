# PNSG 源空间 steering 最终研究报告

> 日期：2026-11-01（实验线收口）
> 性质：本文件固定最终判断，是这条实验线的权威总结；后续引用请以此为准。
> 状态：**实验线结束，不再运行实验。**

---

## 0. 最终结论（一句话）

> 在当前 π0.5、任务切片、代理约束、两步认证和候选生成规则下，实验**未建立**
> 源空间 steering 相对动作空间修正的额外价值，因此**停止扩大当前方案**。
> 已验证的局部修复能力仍然成立；未成立的是「它作为更优干预方法」的依据。

---

## 1. 研究问题与冻结协议

### 研究问题

源空间 steering（在 π0.5 的 Flow Matching 噪声空间 z 上做最小范数修正）是否
比动作空间修正（直接在动作 pose 上做最小范数修正）提供**真实安全上的额外价值**？

### 冻结的协议要素

| 项 | 值 |
|---|---|
| 策略 | π0.5（checkpoint `pi05_libero`），SafeLIBERO spatial task0 |
| 线性问题 | 共用 C,b；源空间 A=CJ（320 维），动作空间 A=C（60 维） |
| 候选族 | Nominal + Source β∈{0.25,0.5,0.75,1.0} + Action β∈{0.25,0.5,0.75,1.0} |
| 源预算 | r_budget = 12，α_j = β_j · min(1, 12/r*) |
| 认证窗口 | 两步（EXECUTE_STEPS=2），完整 C_2（MuJoCo 接触 dist≤0） |
| 真实安全 | 完整 C_2 + native 几何距离；球体代理仅作解释，不作真实净空 |
| 重复 | 5 个独立进程 × 9 候选 = 45 次两步认证 |
| 观测身份 | deployment（env.step 缓存 obs，π0.5 真实部署输入） |
| 停止规则 | 固定预算 12 状态，不根据结果决定纳入/跳过/提前结束 |

### 危险状态入选（两层 5/5 复现门）

初始无 robot-obstacle + nominal 前 2 步 5/5 产生 robot-obstacle 接触且 C_2 拒绝 +
zero-action 分支无同类自发事件。入选完全依据 nominal，不看哪种修正成功。

---

## 2. 主结果（12 状态固定批次，deployment 身份）

### 2.1 状态级四分类

| 结果 | 状态数 | 占比 |
|---|---|---|
| 两组都能修复 | 5 | 41.7% |
| 仅 Source 能修复 | 2 | 16.7% |
| 仅 Action 能修复 | 1 | 8.3% |
| 两组都不能修复 | 4 | 33.3% |

### 2.2 状态级修复率

- **Source 修复 7/12，Action 修复 6/12**；不一致配对 2（source-only）对 1（action-only）。
- 无一方在状态级修复率上占优。

### 2.3 共同成功（both）状态的部署代价

| 状态 | Source 最小通过 C_act | Action 最小通过 C_act | 更省 |
|---|---|---|---|
| ep12 | 0.329 | 0.079 | Action |
| ep15 | 0.737 | 0.526 | Action |
| ep17 | 0.940 | 0.500 | Action |
| ep19 | 1.712 | 0.357 | Action |
| ep20 | 0.101 | 0.157 | Source |

Action 更省 **4/5**。C_act = 实际部署 pose 相对 nominal 的 L2 范数（clipping 后），
5 进程平均。

### 2.4 进度与耗时

- **进度**：两步窗口内所有候选 lift 与 target–goal 均零变化（未抓取、未移动），
  仅 EEF–target 距离有随状态而异的接近幅度。无候选族差异（受 EXECUTE_STEPS=2 限制）。
- **耗时**：Source 组每个 β 需 1 次额外策略前向（4 次）+ Jacobian 求解 + Δz* 求解；
  Action 组复用 nominal，仅 Δa* 求解。Source 计算成本系统性更高。

---

## 3. 探索性诊断（代价匹配，与原批次结果分开）

> 属于结果后提出的解释分析，**不改写主结论**。只分析 2 个 source-only 状态。

对每个 source-only 状态，沿原 Action 方向 Δa* 缩放幅度，使 clipping 后的 C_act
匹配 Source 成功候选的代价（只查代价，不查认证），再按原五次重复 + 同一 C_2 评估。

| 状态 | Source 候选 | target C_act | 匹配 s | 匹配后 C_act | 匹配后 Action |
|---|---|---|---|---|---|
| ep06/q28 | source β=0.25 | 1.4405 | 9.962 | 1.4405 | **5/5 通过** |
| ep07/q18 | source β=1.0 | 0.7576 | 1.073 | 0.7572 | **5/5 通过** |

**诊断结论（准确表述）**：

> 两个 source-only 状态均可由原 Action 方向在匹配部署动作代价后修复。因此，原候选族的
> 幅度覆盖不足足以解释这两个独占案例，无须诉诸源空间方向优势。

### 数值口径（避免量纲混淆）

| 量 | 定义 | ep06 | ep07 |
|---|---|---|---|
| ‖Δa*‖（未加权） | 60 维 L2 | 0.1446 | 0.7057 |
| ‖Δa*‖_S（加权） | 平移 0.01039/步、旋转 0.09197/步 | 0.0189 | 0.0121 |
| 原 action β=1.0 的 C_act | 部署 clip 后 pose 相对 nominal 的 L2 | 0.1446 | 0.7057 |
| Source 成功候选的 C_act | 经非线性前向后，同口径 | 1.4405（β=0.25）| 0.7576（β=1.0）|
| 匹配系数 s | C_act 匹配所需缩放 | 9.962 | 1.073 |

**注意**：`‖Δa*‖` 与 `C_act` 不是同一量——本批次无 clipping 时二者数值上相等（ep06/07
原 Action 无 clip），但这是巧合（action = clip(nom + βΔa*)，nom 未被 clip 时
C_act = ‖βΔa*‖）。**一旦存在 clipping，必须用实际部署动作重算 C_act，不能套
缩放平方关系。** 源空间候选的 C_act 尤其如此：源空间 β 小 ≠ 动作域修正小
（ep06 的 source β=0.25 就产生 C_act=1.44 的动作偏离）。

---

## 4. 影响解释的修正（实验中建立的重要边界）

1. **obs 身份**：机器人 eef sensor 读 `site_xpos[grip_site]`（非 body xpos）；
   `env.step` 返回的 obs 带采样相位。复现必须保存并加载策略实际消费的 obs，
   不能从 snapshot 重建（否则「同一 z0 生成不同动作」）。force 刷新属另一种
   输入协议（同步诊断），改变部署行为。
2. **完整状态与输入保存**：冻结状态 = 物理快照（snapshot：time+qpos+qvel）+
   策略输入 obs，且明确时序关系与身份（deployment / synchronous）。
3. **代理净空 ≠ native 几何**：球体代理会严重高估穿透（历史实测 ~200×），
   真实安全只认 MuJoCo native 接触 dist。二者只作解释对比，不混用。

---

## 5. 最终决定

- **停止扩样**：12 状态固定批次已用完预算，不追加状态、方向或预算。
- **撤回相对优势主张**：不声称「源空间 steering 优于动作空间修正」。
- **保留局部修复能力**：Source steering 能产生通过两步 C_2 认证的修复（7/12），
  其中 2 例是 Action 原候选族未覆盖的（但见 §3，可被幅度放大的 Action 覆盖）。
- **机制观察保留**：候选族间存在双向独占案例（source-only 2 / action-only 1），
  但经代价匹配诊断，source-only 可由幅度覆盖不足解释。

---

## 6. 归档索引

| 项 | 值 |
|---|---|
| 主结果 commit | `f7ba810`（12 状态批次 + 方法主张收缩） |
| 收口 commit | `5f4fded`（代价匹配诊断） |
| 更早 | `86c5db7`（obs 协议冻结 + 三状态身份统一） |
| 碰撞事件契约 | `collision_event_contract_v2.1` |
| CRN 实验版本 | `step5_v1` |
| obs 协议 | `docs/reference/pnsg_obs_protocol.md` v1 |
| 批次汇总 | `data/audit/batch_12state_summary.json` |
| 代价匹配结果 | `data/audit/cost_match_result.json` |
| 批次结果文档 | `docs/testing/pnsg_batch12_result.md` |
| 代价匹配文档 | `docs/testing/pnsg_cost_match.md` |
| deployment 复核 | `docs/testing/pnsg_deployment_recheck.md` |
| 三状态汇总 | `docs/testing/pnsg_deployment_3state_summary.md` |
| 数据身份清单 | `data/audit/README.md`（含 12 状态 snapshot/obs 哈希） |
| 设计文档收缩 | `docs/设计.md` 第 21 节 |

---

## 7. 适用范围声明

本结论仅适用于：当前 π0.5 权重、SafeLIBERO spatial task0、当前代理约束构造
（控制点球 + 障碍物球 + 最小范数半空间）、两步认证窗口（EXECUTE_STEPS=2）、
当前候选生成规则（β∈{0.25,0.5,0.75,1.0} + 预算缩放）与 r_budget=12。

不排除在以下变化下重新评估：更长 horizon 的进度、不同约束构造、代价匹配后的
候选族重设计、或部署目标变化（如优先稳健性而非动作代价）。但**当前方案在此
冻结协议下已无继续投入的依据**。
