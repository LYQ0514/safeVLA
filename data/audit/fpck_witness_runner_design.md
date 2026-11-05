# FPCK/CKG E1a witness 重发现：runner 骨架 + 实现边界

> 日期：2026-11-05。记录 witness 重发现 runner 的设计决策与源复用声明。

## 历史函数复用声明（不复制，不修改）

| 函数 | 源文件 | 用途 |
|---|---|---|
| `reconstruct_nvs` | tools/m1_p4_p0e_joint_execution.py | HVNP 确定性重跑到首次 NVS |
| `plan_main_path` | tools/m1_p4_p1_reentry.py | IK+RRT 主路径规划 |
| `execute_joint_path` | tools/m1_p4_p1_reentry.py | Joint-PD 路径执行器 |
| `reach_and_takeover` | tools/m1_p4_p1_reentry.py | 组合：reconstruct → plan → execute → OSC 接管 |
| `run_branch` | tools/m1_p4_p1_reentry.py | P-V/P-H 分支 rollout |
| `generate_block` | tools/m1_p4_p0e_joint_execution.py | CRN 确定性 π0.5 chunk |

源 commit（历史函数所在）：`85a0d1c`（FPCK 安装前的主线状态）。这些函数
在 FPCK 实验中**未修改**。

## runner 设计（tools/fpck_e1a_witness_rediscovery.py）

- `--case {State02,S01,S04}` + `--run {0,1,2}` + `--dry-run`；
- run_idx 三次：每次独立进程，评估跨进程 GPU 漂移对标签稳定性；
- 历史 RRT seed schedule 固定（不因 run_idx 改变）；
- dry-run：只验证 import + 自动机构建，不跑重型 RRT。

## 当前状态：骨架 + dry-run 通过

- 已实现：历史函数 import、自动机构建（manifest → manipulated_targets + goal clauses）、
  dry-run 验证、结果目录结构、粗粒度标签映射（`classify_from_history`）；
- dry-run 验证 State02：manipulated_targets=['alphabet_soup_1','cream_cheese_1']，
  goal_clauses=2 个 In 谓词，自动机 q_init。

## 待完成（下一步）：事件级自动机接入

裁决核心要求「自动机覆盖两阶段连续，不重新初始化」+「按 v1.2 规则生成标签」，
需要**复制**以下两个循环（裁决允许「若必须复制函数，应记录源文件/commit/函数哈希」）：

1. `execute_joint_path` 的 Joint-PD 循环（逐 substep）：构造 TaskEvent
   - soup 位移 → UNAUTHORIZED_TARGET_MOTION
   - mug 位移 → STATIC_OBSTACLE_MOVED
2. `run_branch` 的 OSC 循环（逐 control step）：构造 TaskEvent
   - monitor_contacts → forbidden_contact
   - goal clause truth + native success
   - 见证检测（CONTROLLED/TRANSPORTED/RELEASED）

历史函数已采集了大部分原始事件（monitor_contacts + body_state + check_success），
复制循环时只需在现有采集点追加 TaskEvent 构造 + 自动机 process()。

## 源函数哈希（复制前记录）

复制 `execute_joint_path` 和 `run_branch` 前，记录其 SHA-256（含源文件路径），
确保「复制不改行为」可审计。

## 诚实边界

当前 `classify_from_history` 是**粗粒度临时映射**（用 success/forbidden/Dmax 字段），
**不是** v1.2 的事件级自动机标签。完整标签需在复制循环 + 事件级采集后实现。
