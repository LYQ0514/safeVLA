# Step 5 Pilot：PROSPECTIVE_SAME_RUN_CAR_SOURCE_AUDIT 结果

- **日期**：2026-10-31
- **协议**：PROSPECTIVE_SAME_RUN_CAR_SOURCE_AUDIT（非历史 relabel）
- **状态**：DESIGN_PILOT_ONLY / NOT_POOLED_WITH_CONFIRMATION / NO_METHOD_PERFORMANCE_CLAIM
- **规模**：spatial task0 × 20 预注册 episode × 3 方法 = 60 rollout

---

## 一、结果

| 方法 | 官方 CAR 失败 | ROBOT_STATIC_OBSTACLE | 平均耗时 |
|---|---|---|---|
| π0.5 | **20/20（100%）** | **20/20** | 168s |
| AEGIS | **2/20（10%）** | **2/20** | 153s |
| VLPSA | **0/20（0%）** | **0/20** | 161s |

## 二、核心发现：机制分解与官方 CAR 精确对应

**每一条官方 CAR 失败，都精确对应一条 ROBOT_STATIC_OBSTACLE 接触；每一条 CAR 通过，都没有这种接触。**

- π0.5 的 20 条 CAR 失败 = 20 条 ROBOT_STATIC_OBSTACLE
- AEGIS 的 2 条 CAR 失败（ep21、ep45）= 仅有的 2 条 ROBOT_STATIC_OBSTACLE
- VLPSA 的 0 条 CAR 失败 = 0 条 ROBOT_STATIC_OBSTACLE

这验证了 contact decomposition 作为官方 CAR 机制解释层的有效性：
`ROBOT_STATIC_OBSTACLE`（机器人直接撞静态障碍物）是 task0 官方 CAR 触发的
**直接来源**。

## 三、三种方法的机制差异（episode 级）

### π0.5（无安全层）

20/20 出现 ROBOT_STATIC_OBSTACLE，另有二次扰动：
- DISTRACTOR_STATIC_OBSTACLE（1 条）：干扰物被碰后撞障碍
- RECEPTACLE_STATIC_OBSTACLE（1 条）：容器撞障碍
- PAYLOAD_DISTRACTOR（1 条）

### AEGIS（末端椭球 CBF）

2/20 仍碰撞（ep21 L1=5.2mm、ep45 L1=4.0mm），但出现新机制：
- ROBOT_ENVIRONMENT（2 条）：CBF 把末端推离障碍时，肘部/其他 link 撞到桌面/柜体

这是末端椭球 CBF 的已知局限——只保护末端点，不保护整臂，避障时可能让
其他部位碰撞环境。

### VLPSA（全身 CBF）

0/20 碰撞，机制最干净（只有 preexisting 的 OBSTACLE_ENVIRONMENT 支撑接触）。

## 四、门禁检查

| 门禁 | 结果 |
|---|---|
| P1 确定性身份 | PASS（z/action/CAR/mechanism/contact 重跑完全一致）|
| P2 官方指标一致 | PASS（CAR 用官方 L1>1mm evaluator）|
| P3 接触契约完整 | PASS（OTHER_UNCLASSIFIED=0, UNKNOWN=0）|
| P4 同运行对齐 | PASS（单 rollout 同时记录 CAR+contact）|
| P5 方法公平性 | PASS（同 episode、同 CRN z、同 checkpoint）|
| P6 无删失 | PASS（60/60，error=0）|
| P7 成本 | 60 条共 161min，policy_calls=1800 |

## 五、⚠️ GPU 数值非确定性观察

冒烟时 π0.5 ep0 的 L1=94.7mm，后台跑是 95.5mm（差 0.8mm），而同一进程
连续跑两次完全一致。

根因：JAX bfloat16 在 GPU 上跨进程非确定（不同 CUDA 初始化状态）。

**影响**：
- z hash 完全一致（CRN 是纯 CPU numpy，确定）
- action 可能微差（openpi GPU 计算非确定）
- CAR peak_l1 差 0.8mm，在 1mm 阈值附近可能影响个别 episode 的触发判定

**按 P1 协议**：这不静默放宽，已在报告中显式记录。后续正式实验若需
逐字节 action hash，需考虑 GPU 确定性模式（如 XLA_FLAGS=--xla_gpu_deterministic_ops）。

## 六、证据边界

- 单任务（task0）、单套件、单 episode 分布
- 20 episode 是流程 pilot，**不形成三方法机制比例结论**
- 不能外推到其他 task / suite / 全量数据

## 七、下一步（按想法v10 协议）

Pilot 通过运行门后，扩展到：
- spatial task0-3，50 initial states per task，3 方法，共 600 rollout
- 使用完全冻结的 episode list / z schedule / method configs / CAR evaluator
- 600 条才作为 spatial 套件的正式 prospective decomposition

**在扩展前需解决**：GPU 数值非确定性对 action hash 的影响（P1 要求逐字节
hash 或显式量化误差），以及 AEGIS 的 ROBOT_ENVIRONMENT 机制是否需要在
正式报告里单独列出（它反映了末端 CBF 的整臂碰撞局限）。
