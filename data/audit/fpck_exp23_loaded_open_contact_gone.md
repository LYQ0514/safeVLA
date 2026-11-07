# FPCK/CKG 实验 2/3 — cream_cheese 载物 open + contact gone（单对象表征）

> 日期：2026-11-07。SINGLE_OBJECT_CHARACTERIZATION。
> verification_status: VERIFIED_WITHIN_SCOPE
> generalization_status: NOT_ESTABLISHED_ACROSS_OBJECTS

## 一、实验设置

- 场景：CHARACTERIZATION_SCENE_V2（canonical 姿态来自 frozen manifest）；
- 对象：cream_cheese（唯一正式 PASS，3/3 一致）；
- 抓取：冻结门禁 v3，沿 y 轴（宽面 43.1mm），grasp aperture 42.2mm；
- 抓取成功后发送 open 命令，逐 substep 记录 30 步。

## 二、实验 2：载物 open 响应（3 次 technical replicates 完全一致）

| 指标 | 值 |
|---|---|
| T_open_onset（loaded） | 3 控制步 |
| T_open_stable（loaded） | 11 控制步 |
| grasp aperture | 42.2mm |

对比空载（实验 1）H_open_empty=15 步：
- 载物 open 更快（stable 11 < 15），因为载物时从 42mm 开，空载从 1mm 开；
- 作用域内保守值：`H_open_cream = max(15, ceil(γ*11)+1)`，γ 沿用冻结协议（1.2）
  → max(15, 15) = 15 控制步（空载上界仍是保守绑定）。

## 三、实验 3：contact gone 时序（原始数据，非预先假定窗口）

| 事件 | step |
|---|---|
| 首个 open 命令 | 0 |
| aperture 首次增大（onset） | 3 |
| 首次 finger-object 接触消失 | 4（aperture 43.9mm） |
| 最后一次 finger-object 接触 | 3 |
| 对象开始下落 | 5（位移 15mm） |
| 对象落到桌面 | 6（位移 49.7mm，z 回到 0.4457） |
| aperture 达稳定 | 11 |

**关键**：无重接触（last_contact=3, first_gone=4，之后持续无接触）。
`max_recontact_gap = 0`，故 `N_contact_gone > 0` 即可（最小 1 substep）。

## 四、配置命名与状态

```json
{
  "verification_status": "VERIFIED_WITHIN_SCOPE",
  "scope": {
    "scene": "CHARACTERIZATION_SCENE_V2",
    "object": "cream_cheese",
    "object_count": 1,
    "purpose": ["loaded_open_response", "finger_contact_gone"]
  },
  "generalization_status": "NOT_ESTABLISHED_ACROSS_OBJECTS"
}
```

**不使用**：VERIFIED_GLOBAL / SAFE_LIBERO_WIDE_VALIDATED / OBJECT_INDEPENDENT。

## 五、实验 1/2/3 汇总

| 实验 | 结果 |
|---|---|
| 实验 1 空载 open | H_open_empty=15 控制步 |
| 实验 2 载物 open | onset=3, stable=11 |
| 实验 3 contact gone | 无重接触，N_contact_gone>0 |

## 六、下一步（裁决 §返回 Gate 5 边界）

1. ✅ 实验 2 载物 open；
2. ✅ 实验 3 contact gone；
3. ⏳ 实验 4：静止速度噪声；
4. ⏳ 实验 5：释放后 settling；
5. ⏳ 实验 6：高速瞬时 goal-truth 反例；
6. ⏳ 冻结作用域明确的终端验证配置；
7. ⏳ Gate 5 管线回归；
8. ⏳ 回归通过后正式重跑受影响候选。

实验 4-6 完成前，不因实验 2/3 通过而把 Gate 5 标 PASS（q_acc 依赖释放后稳定性）。
