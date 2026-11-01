# W2：跨 Worker 稳定性报告（五对锁定案例）

> 日期：2026-11-01
> 状态：W2 完成；跨 worker 稳定性三层结论。
> 前置：`g1a_scope.md`、`g1a_crn_paired_results.md`。

---

## 0. 结论

> 任务成功翻转 **5/5 跨 worker 稳定**（π0.5 全成功 → AEGIS 全失败）；
> 而碰撞严重度跨 worker 剧烈波动（AEGIS Dmax 从 2mm 到 299mm）。
> **任务失败比碰撞类别更稳定**——这是 R2/R3 的正确研究入口。

## 1. 三层跨 worker 稳定性

| 层次 | 结果 | 判断 |
|---|---|---|
| 任务成功翻转 | 5/5 稳定 | ✅ 结构稳定 |
| 子目标（soup）| 5/5 从未满足 | ✅ 结构稳定（W2 确认）|
| 子目标（cream）| 3/5 完成 | 部分保留 |
| 安全类别/碰撞严重度 | 跨 worker 剧烈波动 | ❌ 不稳定 |

## 2. 五对明细（W2）

| 案例 | π0.5 | AEGIS | 任务翻转 |
|---|---|---|---|
| state02 stream0 | succ, Dmax=0.15mm | fail, Dmax=0.00mm | 1 |
| state05 stream1 | succ, 0.00mm | fail, **27.06mm** | 1 |
| state05 stream2 | succ, 0.00mm | fail, **299.40mm** | 1 |
| state06 stream1 | succ, 0.00mm | fail, 0.00mm | 1 |
| state15 stream0 | succ, 0.93mm | fail, **18.38mm** | 1 |

> 299.40mm / 27.06mm / 18.38mm 均来自 `sim.data.xpos` 物理真值（`obstacle_pos_gt`），
> 非观测缓存。

## 3. 子目标顺序的深刻矛盾（新发现）

π0.5 的任务成功路径是 **soup 先满足**：

| 案例 | π0.5 soup 首满足 | AEGIS soup |
|---|---|---|
| state02 | @150 | 从未 |
| state05 stream1 | @150 | 从未 |
| state05 stream2 | @200 | 从未 |
| state06 | @300 | 从未 |
| state15 | @150 | 从未 |

- π0.5 里 **soup 在 5/5 里都先满足**（@150~300 步）；
- AEGIS 里 **soup 在 5/5 里从未满足**，反而 cream 有时先满足（@150）。

**候选机制**：π0.5 的执行顺序是「soup 优先」，AEGIS 的干预**恰好破坏了 soup 这个
π0.5 首先执行的子目标**。

## 4. 修正记录（相对上一轮口述）

- cream 完成数 = **3/5**（state05 stream1/stream2、state15 stream0），
  上一轮口述「2/5」有误，已更正。
- 299.40mm 已核实为 sim.data.xpos 真值。

## 5. 证据边界

- 5 对是**按 W0 的 SS→* 结果选定**的，非 48 对随机抽样，不能外推总体类别不稳定率。
- 「soup 5/5 从未满足」目前仅 W2 确认；W0/W1 未独立保存子目标谓词，
  不能仅凭 TSR=0 反推，需核实后才升级为跨 worker 子目标稳定性结论。
- 50 步谓词采样粒度粗，cream 可能在采样间隙满足又变化。

## 6. 下一步

进入 soup 子目标阶段审计：区分「未抓取 / 未运输 / 未放置 / 放置后失效」，
需要逐 step 的 soup 位置/高度轨迹 + AEGIS 干预时间线（定向回放）。
