# FPCK/CKG 实验 4 — cream_cheese 静止速度噪声表征

> 日期：2026-11-07。SINGLE_OBJECT_CHARACTERIZATION。
> verification_status: VERIFIED_WITHIN_SCOPE
> candidate_only: true，final_terminal_thresholds: PENDING_EXPERIMENTS_5_AND_6

## 一、结果

| 条件 | 线速度 p99.9 | 角速度 p99.9 | 状态 |
|---|---|---|---|
| cream_table | 0.0 | 1.0e-5 | PASS |
| cream_plate | — | — | CHARACTERIZATION_INITIALIZATION_INVALID（plate 空心，对象穿透） |
| cream_receptacle | 0.0 | 0.0 | PASS |

噪声底（有效条件）：
```
linear_noise_floor  = 2.8e-11  m/s
angular_noise_floor = 9.6e-6   rad/s
```

## 二、关键发现

1. **cream_plate 标记 INITIALIZATION_INVALID**（正确分类）：plate 是空心浅盘
   （只有 8 个侧壁 box + 1 个薄中心 box，中心无实体底面），cream_cheese 从中心穿透
   落到 plate 下方桌面（z 0.983→0.9196）。按协议 §3 不调位置直到通过。

2. **噪声底 ≈ 0**：cream_table 和 cream_receptacle 完全静止（对象在支撑面稳定后
   速度接近 0）。这符合「完全静止」的物理预期，但**不代表释放后稳定速度**——
   释放瞬间有残余速度，需实验 5 的「正常落置稳定包络」。

## 三、对最终阈值的作用

按裁决，实验 4 只提供噪声**下界** `b_v, b_ω`（≈0），不决定最终阈值。
最终 `v_max/ω_max` 需实验 5（正常落置稳定包络）+ 实验 6（高速反例拒绝区间）联合冻结：

```
max(b_v, v_settled_max) < v_max < v_unsafe_min
```

## 四、配置状态

```json
{
  "verification_status": "VERIFIED_WITHIN_SCOPE",
  "object": "cream_cheese",
  "scene": "CHARACTERIZATION_SCENE_V2",
  "linear_noise_floor": 2.8e-11,
  "angular_noise_floor": 9.6e-6,
  "candidate_only": true,
  "final_terminal_thresholds": "PENDING_EXPERIMENTS_5_AND_6"
}
```

## 五、下一步

实验 5：正常落置稳定包络（脚本化抓取→open→release→settle，测 settled 速度 + T_settle）。
