# FPCK/CKG 实验 6 — cream_cheese 高速进入 basket 的瞬时 goal-truth 反例

> 日期：2026-11-07。SINGLE_OBJECT_CHARACTERIZATION。
> scene: COUNTEREXAMPLE_SCENE_V1。

## 一、关键工程发现：free joint qvel 顺序

速度注入用 robosuite `set_joint_qvel`（正确接口），且 **MuJoCo free joint 的 qvel
顺序是 [wx, wy, wz, vx, vy, vz]（角速度前 3 维，线速度后 3 维）**。

- 直接写底层 `d.qvel[vadr+2]` 失败（robosuite 包装层有自己的 qvel 缓存）；
- 把 -v0 放在第 3 维（wz）会注入角速度而非线速度；
- 正确：线速度 vz 在第 6 维。

## 二、反例结果（速度族 {v_N, 2v_N, 4v_N}，v_N=0.0422）

| v0 | goal_seen | 首次 goal_true + 高速线速度 |
|---|---|---|
| 0.0422（正常撞击） | True | step171（撞击回弹，非高速进入） |
| 0.0844（2×） | True | **step1**（v=0.084 > v_N） |
| 0.1688（4×） | True | **step1**（v=0.169 > v_N） |

2×和 4×速度族的反例在 step1 就「goal truth + 高速线速度」，是典型的
「瞬时 goal 为真但对象高速运动」反例。

## 三、关键观察：goal 谓词的 contain region

goal `In(cream_cheese, basket)` 在 step1（对象还在 basket 上方 10cm）就为真，
说明 `basket_1_contain_region` 的判定区域包含 basket 上方垂直空间。这不影响反例
有效性，但说明 goal 谓词的几何定义本身就有「瞬时为真」的空间，进一步支持
「goal truth 不能作为稳定接受的充分条件」。

## 四、实验 6 结论

反例成功构造，验证：
```
goal 谓词瞬时为真 + 对象高速运动 ≠ RELEASED_STABLE
```

终端判据必须联合「goal truth + release + contact gone + 速度 ≤ 阈值 + 持续窗口」，
不能只看瞬时 goal truth。

## 五、下一步：实验 1-6 联合冻结终端验证配置

完成实验 1-6 后，联合评估候选 (v_max, ω_max, K, N)，寻找同时满足：
1. 正常 table release 在 H_post 内被接受（实验 5）；
2. 高速反例在瞬时 goal truth 阶段不被接受（实验 6）；
3. 反例最终真正静止且持续满足 goal 可在之后被接受。

若无候选参数完成分离 → `TERMINAL_CRITERION_NOT_IDENTIFIABLE`。
