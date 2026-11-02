# CCA-ASF TASK D 运行时 smoke test 报告

> 日期：2026-11-02
> 状态：真实 π0.5 + MuJoCo + AEGIS 端到端集成验收通过。
> 前置：`cca_asf_task_c_integration_report.md`（TASK C）。

---

## 0. 结论

CCA-ASF 在真实执行链路中正确接入：真实 π0.5 worker 的 chunk 与名义动作一致；
在「同一名义动作 + 同一 obs」下，`euclidean` 与 `cca_asf(β=0)` 的 filter 输出
**逐位一致（80 步最大差 0.0）**；`β=4` 时真实 QP 出现非零各向异性激活
（35/80 步，max β=3.98）；QP 全 feasible；物理日志完整。

这证明方法被正确接入、能真正产生作用、基础执行链路未被改坏——**不评价**
CCA-ASF 是否更安全或任务成功（属 TASK E）。

## 1. 门禁判定

| 门禁 | 结果 | 证据 |
|---|---|---|
| D0 chunk 一致性 | ✅ | 80 步全部 `chunk[idx]==act()` |
| D1 β=0 全路径等价 | ✅ | 单步 filter 对比最大差 0.0（80 步）|
| D2 非零各向异性激活 | ✅ | 35/80 步 β>0，max β=3.98 |
| D3 QP 求解可行 | ✅ | 全 feasible |
| D4 物理日志完整 | ✅ | Dmax / forbidden / task_disp 均记录 |

## 2. 关键发现：π0.5 跨 run 非确定 → D1 测试方法修正

最初 D1 设计为「4 配置独立跑 80 步对比逐步轨迹」，结果 aegis vs euclidean 差 0.32。
诊断确认：**两次 vanilla 的名义动作就差 0.06**（第一步就不同）——这是项目文档
已知的「跨 worker GPU 浮点非确定 ~0.85% 动作误差」，不是接入 bug。

修正：D1 改为「同一 env 状态 + 同一名义动作，euclidean vs cca_asf(β=0) 单步 filter
输出对比」。这才是验证「接入是否正确」的正确方式——filter 是确定性函数，等价性
应在其输入输出层面验证，而非对比两条受非确定推理影响的独立轨迹。

## 3. 验证边界（诚实）

- D0 验证了 chunk 一致性，但「读取 get_current_chunk 不消耗 CRN / 不额外推理」是通过
  接口实现（只读 `_chunk`/`_buffer_index`）保证的，未在真实 worker 层面计时验证
  （额外推理会体现在 infer_count 上，本次未显式断言，属 TASK E 可补）；
- D2 的 β 激活来自固定 State02/Stream0 的前 80 步，35/80 步激活说明该段确实存在
  约束冲突，但**不代表**所有状态/任务都会激活；
- 未做任务成功率 / 安全性优劣比较。

## 4. 各配置物理快照（80 步）

| 配置 | max_dev_norm | max_Dmax | 非零β步数 |
|---|---|---|---|
| vanilla | 0 | 1.54mm | 0 |
| aegis/euclidean | ~0.8 | 0 | 0 |
| cca_asf(β=0) | ~0.9 | 0 | 0 |
| cca_asf(β=4) | ~1.0 | 0 | 35 |

（vanilla 的 Dmax=1.54mm 超 CAR，aegis/cca 均 Dmax=0——这是本 smoke test 的
副作用观测，不是方法比较结论；dev_norm 因各配置名义动作本就非确定，不可比。）

## 5. 工程产物

- `tools/cca_asf_task_d_smoke.py`：D0-D4 五门禁 smoke test。
- `data/audit/cca_asf_task_d_smoke.json`：门禁判定 + 各配置汇总。
