# FPCK/CKG Step 8 开发回归 — bowl + negative 对象

> 日期：2026-11-06。DEVELOPMENT_ONLY（不进入正式候选选择/统计）。

## 一、sticky 映射精确表征（裁决 §一/§二，已完成）

补齐 9 个 0.25 量化级别，区分 settling transient 与 hold drift：

| current_action | 稳定 aperture（closed 接近 median） | hold drift |
|---|---|---|
| 1.00 | 78.8mm | 1.70mm |
| 0.75 | 68.8mm | 1.70mm |
| 0.50 | 58.8mm | 1.59mm |
| 0.25 | 48.8mm | 1.68mm |
| 0.00 | 38.8mm | 1.67mm |
| -0.25 | 28.8mm | 1.64mm |
| -0.50 | 18.8mm | 1.59mm |
| -0.75 | 8.8mm | 1.69mm |
| -1.00 | 1.2mm | 1.34mm |

关键修正：
- **hold drift 仅 1.25-1.70mm**（之前报告 8-9mm 是混入 settling transient）；
- **迟滞 ~2.4mm**（open 方向接近比 closed 方向大 2.4mm，中间档一致）。

## 二、preshape 档位选择修正（裁决 §三）

`preshape_target_ca` 从「从小到大找第一个满足」（argmin 语义）：
- cream_cheese y 轴（42.6mm）→ ca=1.0（79.5mm，full open，因为 42.6+28=70.6mm 接近 full open）；
- cream_cheese x 轴（17.8mm）→ ca=0.25（49.5mm）。

## 三、bowl（hollow container）开发回归 ✅

- akita_black_bowl 外径 = **94.2mm**（40 个碗壁 box 的圆周分布半径 0.0471 × 2）；
- 夹爪 max aperture = 79.5mm；
- 结果：`PRESHAPE_APERTURE_UNREACHABLE`（物理可解释：外径 94.2 > 79.5）；
- **验证「外径 vs 壁厚不混淆」**：门禁正确用外径 94.2mm 判 FAIL，未错误地夹壁厚（~2mm）。

## 四、negative 对象（cream_cheese 沿 x 窄面）⚠️

- cream_cheese x 轴宽 17.8mm（窄面），closing axis 沿 x；
- 3 次一致失败，但 failure code = `DESCENT_TIMEOUT`；
- 物理根因：窄边夹取时，手指间距（preshape ca=0.25→49.5mm）远大于对象窄边 17.8mm，
  descend 时手指 tip 在对象 x 两侧（间距 52mm >> 17.8mm），tip 内侧距对象表面
  尚远，descend 目标「对象中心 z」的接触语义复杂，未形成干净的双指夹持。

## 五、诚实结论

Step 8 核心目标**部分达成**：

| 目标 | 结果 |
|---|---|
| 空心复合几何正确解释（外径 vs 壁厚） | ✅ bowl 94.2mm 判 FAIL，未混淆壁厚 |
| 复合 geom 分类正确 | ✅ 40 个碗壁 box 正确算出外径 |
| 明确超范围对象物理可解释 FAIL | ✅ bowl `PRESHAPE_APERTURE_UNREACHABLE` |
| negative（夹爪范围内但夹不稳） | ⚠️ cream_cheese x 轴失败，但 code 是 DESCENT_TIMEOUT（非理想 NO_STABLE_BILATERAL_GRASP） |

## 六、剩余问题（正式候选前需解决）

1. **cream_cheese x 轴的 DESCENT_TIMEOUT 语义**：窄边对象的 descend 目标「对象中心 z」
   在手指间距 >> 窄边时，接触语义复杂。需明确「窄边模板」的 descend 完成条件
   （是「tip 到对象中心高度」还是「tip 接触对象即停」）。
2. **这影响 negative 判定的准确性**：`DESCENT_TIMEOUT` 是导航超时，不是「夹不稳」
   的物理资格失败。要么修正 descend 语义让窄边得到 `NO_STABLE_BILATERAL_GRASP`，
   要么明确「窄边模板」本就预期 DESCENT 阶段失败。

## 七、下一步

这些剩余问题属于「descend 语义」的细化，需要裁决：
- 窄边对象的 descend 完成条件应是什么？
- 是否接受「cream_cheese x 轴 DESCENT_TIMEOUT」作为 negative 的合理失败
  （物理上「窄边不可夹」的另一种表现），还是必须优化到 NO_STABLE_BILATERAL_GRASP？

在裁决前，不冻结正式门禁，不运行 10 个正式候选。
