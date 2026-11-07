# FPCK/CKG 速度读取来源审计 + 修正

> 日期：2026-11-07。裁决：free-joint qvel 顺序审计。

## 一、六基向量门禁结论

用 robosuite 高层接口 `get_body_xvelp`（线速度）/ `get_body_xvelr`（角速度）实测：

| 注入 | xvelp（线速度） | xvelr（角速度） |
|---|---|---|
| e0 | [1,0,0] | [0,0,0] |
| e1 | [0,1,0] | [0,0,0] |
| e2 | [0,0,1] | [0,0,0] |
| e3 | [0,0,0] | [1,0,0] |
| e4 | [0,0,0] | [0,1,0] |
| e5 | [0,0,0] | [0,0,1] |

**结论**：free-joint `qvel` 前 3 维 = 线速度，后 3 维 = 角速度（与 MuJoCo 官方文档一致）。

## 二、cvel 顺序（错误根源）

`d.cvel[body]` 实测（注入 vz=-1 线速度后）：
```
cvel = [0, 0, 0, 0, 0, -1]   →  前3维=角速度, 后3维=线速度
get_body_xvelp = [0, 0, -1]   →  线速度（正确）
get_body_xvelr = [0, 0, 0]    →  角速度（正确）
```

**`d.cvel` 是 spatial velocity，顺序 [angular, linear]，与 free-qvel 的 [linear, angular] 相反。**

## 三、速度读取来源表

| 数据字段 | 实际来源 | 约定 |
|---|---|---|
| linear velocity | `get_body_xvelp` 或 qvel[:3] | world linear |
| angular velocity | `get_body_xvelr` 或 qvel[3:6] | body angular |
| `d.cvel` | spatial velocity | **前3=角速度，后3=线速度，不能套 qvel 顺序** |
| finite difference | xpos/time | world linear |

## 四、受影响实验

| 实验 | 受影响 | 原因 |
|---|---|---|
| 1-3 | 否 | 只用 aperture/contact，不读 cvel |
| 4 | **是** | `lin=cvel[:3]`（实际角速度），`ang=cvel[3:]`（实际线速度）颠倒 |
| 5 | **是** | 同样颠倒 |
| 6 | **是** | 同样颠倒 + 第二版注入改错（vz 放第 6 维 = wz 角速度） |

## 五、修正方案

1. 所有脚本改用 `get_body_xvelp`（线）/ `get_body_xvelr`（角）；
2. 实验 6 注入恢复 `[0,0,-v0,0,0,0]`（第 3 维 = vz 线速度，正确）；
3. 实验 4/5 的原始轨迹若已存 raw rows（含 cvel 原始值），可离线重算（角/线交换）；
   否则需重跑。

## 六、实验 6 状态

```
ENGINEERING_AUDIT_REQUIRED
scientific_verdict: WITHHELD
```

goal predicate「对象在 basket 上方 10cm 即为真」是独立有效发现；
但「高速线速度反例」尚未证明（速度读取顺序错误）。
