# 实验 6 高速瞬时 goal-truth 反例协议 v1

> 日期：2026-11-07。**在运行前冻结**。protocol_sha256 记录冻结 hash。
> SINGLE_OBJECT：cream_cheese。scene：COUNTEREXAMPLE_SCENE_V1。

## 0. 职责

验证「goal 谓词瞬时为真 + 对象高速运动 ≠ RELEASED_STABLE」。
联合验证终端判据参数 (v_max, ω_max, K, N)，证明所选阈值能拒绝过早接受。

## 1. 场景（COUNTEREXAMPLE_SCENE_V1）

只用 cream_cheese + goal receptacle（basket，In(cream_cheese, basket)）+ Panda/桌面。
scene builder 接受显式角色：
```
manipulated_target / goal_receptacle / goal_support / distractor
```
只移动 `distractor`，goal receptacle 留在工作区。

## 2. 速度初态（直接可审计，非导航/抓取）

1. cream_cheese 放 goal receptacle 上方；
2. 确认无初始穿透；
3. 设预注册线/角速度（写 free joint qvel）；
4. 纯 MuJoCo 动力学；
5. 逐 substep 记录 goal truth、接触、速度、姿态、谓词变化。

速度注入验证：target joint name/id、qvel address、注入前后速度、首 sim step 后速度。

## 3. 速度族（预注册，依据实验 5 正常撞击速度）

实验 5 正常撞击速度 `v_impact_N`（cream_table 撞击峰值，从实验 5 数据取）。
```
v_0 ∈ { v_impact_N, 2*v_impact_N, 4*v_impact_N }
```
至少含：
- 纯垂直高速进入；
- 垂直 + 水平分量；
- 带角速度进入。

## 4. 反例有效性判据

每条反例必须出现 `goal truth = True`，且同时满足至少一个：
```
||v|| > v 或 ||ω|| > ω 或 support contact unstable 或 goal truth 随后消失
```
从未出现 goal truth → 只是高速碰撞，不是「瞬时任务成功反例」。

## 5. 终端判据联合验证

对正常轨迹（实验 5）+ 反例（实验 6），逐组评估候选 (v_max, ω_max, K, N)：
```
Accept_t = GoalTrue_t ∧ Released_t ∧ ContactGone_N ∧ ||v_t||≤v_max ∧ ||ω_t||≤ω_max ∧ StableFor_K
```
合格参数必须：
1. 正常 table release 在 H_post 内被接受；
2. 高速反例在瞬时 goal truth 阶段不被接受；
3. 反例最终真正静止且持续满足 goal 可在之后被接受；
4. 不把短暂 goal truth 消失前标成功。

## 6. 冻结规则

至少一组候选参数同时通过正常轨迹 + 全部反例 → 生成
`terminal_validation_config_v1_single_object`（VERIFIED_WITHIN_SCOPE，
generalization NOT_ESTABLISHED_ACROSS_OBJECTS）。
无候选参数完成分离 → `TERMINAL_CRITERION_NOT_IDENTIFIABLE`（不调参数直到通过）。
