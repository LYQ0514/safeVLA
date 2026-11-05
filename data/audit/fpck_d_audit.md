# FPCK/CKG 实验：D 阶段审计记录

> 日期：2026-11-05。SafeLIBERO 接口冒烟（proxy_smoke_only）

## 命令

```bash
cd /data/safeVLA && source tools/activate_env.sh
python tools/run_safelibero_ckg.py \
  --factory src.fpck.example_factory:build \
  --suite safelibero_spatial --level I --tasks 0 --episodes 0 \
  --budget 20 --horizon 40 --no-video --out data/eval/fpck_smoke
```

## 结果

| 项 | 值 |
|---|---|
| evidence_class | `proxy_smoke_only`（已标记，不进主表） |
| TSR | 0.0（proxy 非真控制器，预期未完成） |
| CAR | 1.0（无碰撞） |
| ETS | 300（超时） |
| 决策步数 | 300 |
| status 分布 | certified=37, outside_certified_domain=3, horizon_exhausted=260 |
| runtime_ms | p50=0.006, p95=0.048, p99=0.068 |
| replan 次数（chunk 数） | 60 |

## 接口链路验证（全部通过）

1. `CKGPolicy` 包装 `Pi05Policy`，不改变冻结权重 ✅；
2. `state_builder` 提取 eef + obstacle 状态 ✅；
3. `candidate_builder` 生成 3 候选（nominal/hold/away）✅；
4. `governor.choose` 逐候选评估 `lower_value`，选 worst_value 最小者 ✅；
5. append-only 决策日志：每步记录 status/budget/horizon/chunk_hash/buffer_index/evaluated/runtime_ms ✅；
6. `get_chunk_metadata` / `buffer_index` 从 base policy 正确读取 ✅。

## 发现的问题（如实记录）

### 问题 3：budget_remaining 下溢（-263）

- `CKGPolicy.act()` 第 23 行 `self.budget -= int(dist > eps)` **无条件执行**；
- 当 `horizon_exhausted`（第 40 步后）或 `outside_certified_domain` 时 `d.action is None`，
  action 被设为全零，`dist = ||0 - nominal|| > eps` 恒成立 → 每步扣 budget；
- 结果：300 步后 `budget_remaining = -263`（初始 20）；
- **影响**：D 阶段 proxy smoke 不受影响（接口验证），但 H 主实验的 budget 统计会失真，
  **必须修复后才可进入 H**。

### 问题 4：smoke 结果缺少 README 要求的 FPCK 字段

- README 要求 `native_success`/`valid_success`/`car_1mm`/`car_5mm`/`ssr`/
  `strict_ssr_valid`/`safe_stall`/`budget_exhausted`/`runtime_ms`；
- 当前 smoke 的 `run_episode` 返回 `success`/`collided`/`steps` 等（safeVLA 原生字段）；
- **影响**：D 阶段 smoke 用原生字段（够验证接口），但 H 主实验需按 README 数据契约
  产出 FPCK 字段（E1 阶段的任务）。

## 结论（D PASS，含 2 项待办）

接口链路完整跑通，proxy_smoke_only 正确标记。但发现 **budget 下溢 bug**（问题 3）
和 **字段映射缺口**（问题 4），需在 E1/H 前修复/补齐。这两项均不改变「接口已验证」的
结论，但必须如实记录并作为后续 go/no-go 的前置条件。
