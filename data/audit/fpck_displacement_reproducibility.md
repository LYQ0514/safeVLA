# FPCK/CKG 位移复现性审计 — 7.81mm vs 10.19mm 差异分类

> 日期：2026-11-06。裁决 §三/§四/§五/§九 执行完成。
> 结论：**情况 C（TRUE_TRAJECTORY_DIVERGENCE）**，但分歧根因是 VLA 推理 GPU 非确定性，
> 不是测量口径、不是 MuJoCo 非确定、不是旧 install、不是 harness bug。

## 一、四个先决问题（裁决 §三）的答案

| 问题 | 答案 |
|---|---|
| 1. 目标参考点相同？ | ✅ 相同。两份日志都用 `body_state` = `data.xpos[body_id]`（body COM，world frame） |
| 2. baseline 相同？ | ✅ 相同。都相对 settled baseline（settle 20 步后的 body COM）；4 次 make_env 两两差 0.000000mm |
| 3. 范数相同？ | ✅ 相同。都是 L1（`np.sum(np.abs(...)) * 1000`） |
| 4. 采样时刻相同？ | ⚠️ 需区分：control-step 末值 vs substep 首越界值 |

## 二、数据源梳理

| 源 | 文件 | step37 disp | 说明 |
|---|---|---|---|
| 源1 | `fpck_e1a_positive_search/.../raw_events.jsonl` | 7.8091 | control-step 末值（L1 settled） |
| 源2 | `fpck_substep_reaudit_spatial_t1_s1.jsonl` | 首越界 5.1587 / 末值 10.1941 | 25 substep 全序列 |
| 源3 | `fpck_dual_monitor/.../raw_events.jsonl` | 7.8091 | control-step 末值，与源1 一致 |
| 本审计重跑 | — | 10.1941（多数） | 6 次独立进程 |

**关键**：源1 与源3 的 step37 精确一致（7.80907454015789），说明源3（最新代码）与
源1（旧 commit 578ad4f）在同一轨迹族；而源2 与本审计的多数重跑落在 10.1941 族。

## 三、substep 序列澄清「7.81 vs 10.19」不是测量口径差异

源2 的 step37 完整 25 substep 序列（L1 settled）：

```
substep 3 = 5.1587  ← 首次 >5mm
substep 12 = 7.8051 ← 接近 7.81
substep 24 = 10.1941 ← control-step 末值
```

7.81mm 恰好落在 substep 12（7.8051），10.19mm 是 substep 24（末值）。**这不是同一个
时刻的两种测量，而是同一 trajectory 上不同 substep 的两个采样点**——但它们出现在
**不同进程的不同 trajectory 上**（见下）。

## 四、根因定位实验（裁决 §九）

1. **settled baseline 确定**：4 次 make_env，bowl_pos 两两差 0.000000mm。
2. **substep 采集器透明**：`step_with_substep`（full）与原生 `env.step`（native）
   在同一进程内 step37 都 = 10.1941，完全一致。采集器不引入分歧。
3. **bridge 副作用无**：同一 rollout 内 `env.step` 后立即读 vs `eval_goal_clauses`/
   `native_success` 后读，差 = 0.0000。
4. **xpos 时序无差异**：静止时 forward 前后 xpos/qpos 变化为 0。
5. **决定性证据**：6 次独立进程纯原生 rollout，step37 出现 3 个值：
   `[10.1941, 10.1941, 10.1941, 8.7162, 9.7850, 10.1941]`，step36 也分歧
   （3.4975 / 3.1316 / 5.0778）。
6. **chunk 跨进程不一致**：3 个进程各 8 个 chunk，run#0 的 chunk0=6421...，
   run#1/2 的 chunk0=1f5f...，**bit 级不同**。

## 五、分类：TRUE_TRAJECTORY_DIVERGENCE，根因 = VLA 推理 GPU 非确定性

- **不是** 测量口径差异（情况 A）：同一轨迹的 substep 序列已澄清 7.81 是 substep12、
  10.19 是 substep24；
- **不是** 定义不同（情况 B）：参考点/基线/范数/坐标 frame 全部一致；
- **是** 真实轨迹分歧（情况 C）：不同进程的 policy chunk bit 级不同 → 动作序列微差 →
  接触后的混沌动力学放大为毫米级位移差异。

**第一处分歧位置**：step36（首次 finger–target 接触）。接触发生前（step≤35）位移
0.0002mm（完全一致），接触后立即开始分歧。这符合「接触敏感性 + 微小动作差异」的
物理特征。

**根因机制**：openpi worker 跑 CUDA（`JAX_PLATFORMS=cuda`），`rng_key=jax.random.key(0)`
固定噪声，但 GPU 浮点运算（self-attention / LayerNorm 归约）默认非确定，导致不同进程
policy 输出 bit 级差异。这与裁决 §九预判一致：「GPU 主要影响 VLA 推理；chunk/action
hash 不同 → 后续 MuJoCo 轨迹差异应从 policy 输出寻找」。

## 六、对既有结论的影响

1. **不影响审计 B 铁证**：源2（substep 重采集）在**它自己的轨迹**上，首个 5mm 越界
   （substep3 = 5.16mm）的原始接触对是「left finger + support ramekin，无 right finger /
   robot link」，这结论对**该 trajectory** 成立，与其它进程的 trajectory 无关。
2. **不影响审计 C/D**：审计 C 的在线/离线差分在同一事件流上（同进程同 trajectory），
   零差异结论依然成立；审计 D 的运行时身份结论依然成立。
3. **影响「跨运行复现性」声称**：不能在 VLA 推理未锁定的情况下声称「同 seed 精确复现
   毫米级轨迹」。跨进程 trajectory 有 ±2.5mm 量级的分歧。

## 七、修复方向（裁决 §九 + 后续 harness 规格）

1. **锁定 VLA 推理确定性**：在 openpi worker 中启用
   - `jax.config.update("jax_enable_x64", ...)` 不解决 GPU 归约；正确做法是
   - 固定 CUDA deterministic 算子（`TF32` 关闭、`torch.backends.cudnn.deterministic` 对应
     JAX 的 `XLA_FLAGS=--xla_gpu_deterministic_ops=true` / `--xla_gpu_autotune_level=0`）；
   - 或固定在同一进程内完成「所有需要对比的 trajectory」（单进程内 chunk 已证明确定）。
2. **位移 schema 统一**（裁决 §四）：记录 `disp_l1_settled_control_end_mm`、
   `disp_l1_settled_substep_mm`（含 physics_substep_index）、`disp_l1_contact_*`、
   `target_reference_type`/`target_body_name`/`coordinate_frame`。
3. **baseline 由 hash 绑定**（裁决 §七）：settled_target_position_xyz + quaternion +
   full_state_sha256 + snapshot_sha256。
4. **跨进程确定性门禁**：任何需要「跨运行对比」的评测，必须先过 chunk-hash 一致性
   门禁；不一致则退化到「单进程内对比」或启用 deterministic XLA flags。

## 八、对重跑（③–⑦）的直接影响

重跑必须改为**单进程内完成所有需要对比的 trajectory**，或先在 openpi worker 启用
deterministic XLA flags 并验证 chunk hash 跨进程一致。否则 state3/state4/S04 等
「跨 anchor 对比」会混入 ±2.5mm 的 trajectory 噪声，无法区分「自动机语义差异」与
「VLA 推理噪声」。

**这必须在冻结 v1.3 spec/factory hash 之前解决**（裁决 §七：冻结前必须解释差异）。

## 九、Gate 状态（不变）

| Gate | 状态 |
|---|---|
| 审计 D / C | PASS |
| 位移复现性 | **已定位：VLA GPU 非确定，待 deterministic flags 或单进程策略** |
| Gate 4a / 5 / 6 | 不变（仍 DISPUTED / INVALID / 暂保留） |
