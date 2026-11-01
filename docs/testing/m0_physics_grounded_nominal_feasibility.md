# M0-P0：名义动作真实安全可行性验证

> 日期：2026-11-01
> 状态：最小方法原型第一步。主线从「解释失败」转向「减少不必要安全干预」。
> 前置：`g1a_r3_g4_closed_loop_execution_attribution.md`。

---

## 0. 结论

> 在 state02/stream0 的 step5/15/20/30，AEGIS 过滤掉的 4 个名义动作
> **全部真实物理安全**（无 robot-mug 接触、mug 位移 0）。
> GVMIS（物理验证后保留名义）在 4/4 步保留了名义动作，γ 从「负向」恢复为
> 「正向」，且仍保持 mug 安全。这初步支持「避免不必要安全修正」的核心命题。

## 1. 结果

| step | γ_nom | γ_aeg | Nominal 安全 | GVMIS γ |
|---|---|---|---|---|
| 5 | +1.2mm | −5.1mm | ✅ | +1.2mm |
| 15 | +15.1mm | −4.7mm | ✅ | +15.1mm |
| 20 | +13.3mm | −5.0mm | ✅ | +13.3mm |
| 30 | +4.6mm | −8.0mm | ✅ | +4.6mm |

## 2. 核心发现

- π0.5 名义动作 γ_nom 全正（朝 soup）；
- AEGIS 反转成 γ_aeg 全负（远离）；
- **但 4 个名义动作物理全安全**（mug 位移 0、无接触）→ AEGIS 的修正在这 4 步是不必要的；
- GVMIS 保留名义后仍安全。

## 3. GVMIS 原型（Oracle）

$$a_t^* = \begin{cases} a_t^{nom}, & \mathcal V_K(s_t, a_t^{nom})=1 \\ a_t^{safe}, & \text{otherwise} \end{cases}$$

$\mathcal V_K$ = 短时域 MuJoCo 预演（robot-mug 接触 + mug 位移 ≤1mm）。

## 4. 边界（不越界）

1. **Oracle 仿真验证器**：只建立可行性上界，非可部署安全模块；
2. **4 步样本**：不能因这 4 步安全就认定所有名义动作安全；
3. mug 位移用 sim.data 真值；
4. 不能拿 V 轨迹的 CAR 反推 F 状态下名义动作安全——本实验在 F 分支相同状态验证。

## 5. 下一步 M0-P1（闭环原型）

P0 通过（4/4 名义安全）。进入闭环对照：
- V / F / B（前40π0.5后AEGIS）/ M（GVMIS 前40，后AEGIS）。
- 验收：CAR 不下降 + soup 接近优于 F + 名义保留率 + 安全验证时延。
