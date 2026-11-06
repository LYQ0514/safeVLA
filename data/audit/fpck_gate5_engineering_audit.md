# FPCK/CKG Gate 5 独立工程审计（审计 A：脚本化合法抓取）

> 日期：2026-11-06。Gate 5 降级为 INVALID PENDING INDEPENDENT ENGINEERING AUDIT。

## 审计 A 核心结论：测量链路存在系统性假阴性

用「脚本化合法抓取」（事件序列人工写死，纯逻辑，不依赖 GPU）验证自动机，
发现 **`UNAUTHORIZED_TARGET_MOTION` 的 5mm 前置违规规则会系统性拒绝真实抓取**。

## 四条脚本化轨迹的结果

| 轨迹 | 结果 | 判定 |
|---|---|---|
| 1. 无接触无位移 | 停在 q_init | ✅ 正确 |
| 2. 理想抓取（双指闭合在前） | **到 q_acc** | ✅ 测量链路无根本 bug |
| 3. 真实抓取（单指→滑移6mm→双指） | **UNAUTHORIZED_TARGET_MOTION** | ❌ **误判** |
| 3b. 滑移4mm→双指闭合 | 到 q_acc | ✅ 阈值分界确认 |

## 核心 bug：5mm 前置违规的语义错误

当前 `_detect_violation`：

```python
if (ev.target_displacement_l1_mm > 5.0  # 相对 settled baseline 的累计位移
        and not self._any_controlled()):
    return "UNAUTHORIZED_TARGET_MOTION"
```

**问题**：`target_displacement_l1_mm` 是「相对 settled baseline 的累计位移」，不是
「当前步增量」。真实抓取中，bowl 在「双指闭合形成 CONTROLLED 见证」之前，通常有
数毫米的自然滑移（手指接触导致的滚动/滑动/顺应），累计位移很容易 >5mm。

于是：
1. 单指接触 bowl（task_contact=True，但无双指）；
2. bowl 自然滑移，累计位移 3mm→6mm；
3. 此时 two_finger 还没稳定成立，`_any_controlled()` = False；
4. `6mm > 5mm AND not controlled` → UNAUTHORIZED_TARGET_MOTION → q_bad。

**这不是 π0.5 能力不足，而是「5mm 阈值 + 累计位移口径 + CONTROLLED 前不豁免」
三者的组合，系统性拒绝了「抓取闭合前有自然滑移」的所有真实抓取。**

## 与观察结果的一致性

spatial task1 三个候选「先 bowl 位移 >5mm，后合法抓取」的统一模式，**很可能就是
这个 bug 的后果**，而不是 π0.5 真的「先推后抓」：
- bowl 在抓取闭合前的自然滑移 >5mm → 被判 UNAUTHORIZED；
- 后续 π0.5 其实完成了合法抓取（lift≥10mm、双指接触、gripper 闭合）；
- 但已被吸收态否决。

## 结论

1. **测量链路本身能到达 q_acc**（轨迹 2 证明），所以不是「自动机完全无法识别成功」；
2. **但 5mm 前置违规规则会系统性误判真实抓取**（轨迹 3 证明）；
3. **Gate 5 的「7/7 假成功」不能归因于 π0.5 能力不足**，而是至少部分（甚至主要）
   由这个测量 bug 造成。

## Gate 5 正式状态

**INVALID PENDING INDEPENDENT ENGINEERING AUDIT**（非 INCONCLUSIVE）。

理由：测量链路本身有合理怀疑（5mm 前置违规的语义错误），当前数据不能用于科学裁决。

## 下一步（按审计清单）

- 审计 B：对 spatial/task1/state1 做逐 substep 真值表，确认 bowl 位移 >5mm 的
  substep 是否真的是「自然滑移」还是「非法推动」；
- 审计 C：事件桥差分验证（独立离线审计器）；
- 审计 D：运行时源码身份（module.__file__ / source_sha256）；
- 修复 5mm 规则的语义（区分「抓取闭合前的自然滑移」vs「非抓取阶段的非法推动」）。
