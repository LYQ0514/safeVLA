# M1-T0：Task-Object Integrity Audit（TASK 物体操作完整性审计）

> 日期：2026-11-02
> 状态：审计完成，发现两处 TASK 完整性缺陷。
> 前置：`m1_p4_p1_frozen_vla_reentry_audit.md`（State02 假成功线索）。

---

## 0. 结论

> 建立 TASK 物体操作完整性监控（5mm 位移阈值，规划/预抓取阶段禁止非授权推动），
> 审计 P4-P1 四条路径执行前缀，发现：
> ① **S04 稳定复现 TASK_INTEGRITY_FAIL**（预抓取路径把 soup 撞飞 247.8mm）；
> ② **State02 的「假成功」是非确定的**——依赖选到哪条 RRT 主路径（某条路径撞飞
> soup 进 basket，另一条不碰 soup）。这证明 `check_success` 的纯位置谓词无法
> 区分「真实抓取」与「路径撞飞」，TASK 完整性监控是必要的基础设施。

## 1. 审计结果

| 状态 | 完整性 | soup 位移 | cream 位移 | 判决 | success 谓词 |
|---|---|---|---|---|---|
| S01 | ✓ | 0.0mm | 0.0mm | PREGRASP_VALID | False |
| **S04** | **✗** | **247.8mm** | 0.0mm | **TASK_INTEGRITY_FAIL** | False |
| S09 | ✓ | 0.0mm | 0.0mm | PREGRASP_VALID | False |
| S02 | ✓ | 0.0mm | 0.006mm | PREGRASP_VALID | True |

## 2. 关键发现 1：S04 稳定复现 TASK 完整性缺陷

S04 的预抓取路径在 substep150 首次推动 soup（位移 8.3mm 超阈值），执行结束时
soup 被撞飞 **247.8mm**（但没有进 basket，所以 success 谓词仍 False）。

这揭示了 P4-P0r2/P0e-r 的一个**未被发现的缺陷**：S04 的「安全预抓取路径」
（Dmax=0、forbidden=0，之前 P0e-r 判定通过）**实际把 soup 撞飞了 248mm**。
因为 soup 是 TASK 物体（非 forbidden），之前的接触监视器没报警。

## 3. 关键发现 2：State02「假成功」是非确定的

对比两个脚本对 S02 的执行结果：

| 脚本 | soup 位置（执行后）| check_success |
|---|---|---|
| P4-P1 reach_and_takeover | [0.011, 0.272, 0.492]（basket 里）| True（假成功）|
| M1-T0 audit_case | 未动（位移 0）| True（?）|

M1-T0 里 S02 soup 位移=0 但 success=True，说明 M1-T0 选到的主路径与 P4-P1 不同。
根因：`reconstruct_nvs` 里 HVNP 的 NVS 步数因 scratch fork 的数值误差**每次 run
波动**，导致 q_start 略不同 → RRT 选出不同主路径 → 一条撞飞 soup、一条不碰。

**结论**：State02 的「假成功」是**路径依赖 + 非确定**的偶发现象。这反而强化了
TASK 完整性监控的必要性——不逐条路径监控，假成功会以非确定方式混入结果。

## 4. 修复内容（`src/safety/contract_objects.py` 新增）

- `TASK_STILL_DISP_THRESHOLD = 0.005`（5mm，规划/预抓取阶段 TASK 物体位移上限）；
- `TaskIntegrityMonitor` 类：阶段感知的 TASK 物体位移跟踪；
- `task_integrity_ok(env, baseline, task_bodies)`：无状态检查；
- 阶段语义：PLANNING/PREGRASP 禁止推动，GRASP 允许夹爪接触，TRANSPORT/PLACE
  允许随机器人移动。

## 5. 关键约束（按裁决）

1. **不修改 native CAR**：mug 1mm 历史位移契约不变；
2. **不修改既有保护对象定义**：PROTECTED/SUPPORT 清单不变；
3. **TSR 分离报告**：
   - `TSR_native = I[check_success]`（原始 benchmark）；
   - `TSR_valid = I[check_success ∧ 操作过程有效]`（新增）。

## 6. 判决

- S04 的 `TASK_INTEGRITY_FAIL` 是**稳定、可复现**的缺陷；
- State02 的假成功是**非确定**的（路径依赖）；
- 两者都证明：**TASK 完整性监控是评测基础设施的必要组成部分**，否则任务谓词
  假阳性会污染任何「安全任务成功」的结论。

## 7. HVNP-P1b 操作过程复核（已完成）

P1b（State02 完整 HVNP 闭环）的操作过程复核结果：

- soup 首次抬升 @ step74（跨 run 略早于原记录的 162，因数值非确定 + stream 修复）；
- 抬升后 5 步：**eef-soup 距离恒定 19.4mm**（相对静止 = 夹持）、**gripper 开合度
  恒定 0.063**（稳定夹持）、**contact=True**、**soup_z 稳定上升**（0.500→0.537）。

**判决：VALID_GRASP**——soup 是通过「夹爪接触 + 相对静止 + 稳定抬升」的**合法
夹持抓取**，不是撞飞。P1b 的「完整安全任务成功」在操作过程层面**仍然成立**。

> 注：最初的判决脚本用 gripper 开合度 < 0.01 作「闭合」判据，误判为 FAIL。
> 实际 robosuite gripper qpos 语义：初始张开 = 0.077（sum|·|），夹持物体 = 0.063，
> 差值 0.014 是夹持宽度。可靠判据是「接触 + eef 相对静止 + 稳定抬升」，而非
> 开合度绝对值。

## 8. 工程产物

- `src/safety/contract_objects.py`：新增 TASK 完整性监控。
- `tools/m1_t0_task_integrity.py`：四状态路径前缀审计。
- `tools/m1_t0_p1b_replay.py`：P1b 操作过程复核。
- `data/audit/m1_t0_task_integrity.json` / `m1_t0_p1b_replay.json`。

## 9. M1-T0 总结

| 结论 | 状态 |
|---|---|
| S04 预抓取路径撞飞 soup 248mm（TASK_INTEGRITY_FAIL）| ✅ 稳定复现，需修复 |
| State02 假成功是非确定的（路径依赖）| ✅ 已定位，需监控 |
| P1b 的操作过程是合法抓取（VALID_GRASP）| ✅ 复核通过，P1b 结论保留 |
| TASK 完整性监控已建立（5mm 阈值 + 阶段语义）| ✅ 已固化 |
