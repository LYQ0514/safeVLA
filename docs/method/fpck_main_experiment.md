# FPCK/CKG 主方法与 SafeLIBERO 首轮实验

本文件只描述主线：冻结 π0.5 的反事实 continuation 数据、经验下界学习、CKG
部署和配对 SafeLIBERO 比较。历史表征脚本不属于本流程。

## 方法闭环

1. `collect_fpck_counterfactual.py` 在每个 task/init/stream 选择一个注册 anchor；
2. nominal、hold、retreat/tangent/damped 候选从完全相同的 MuJoCo、π0.5
   chunk/cursor 与 v1.3 acquisition memory 恢复；
3. 每个 sibling 运行至 q_acc/q_bad/q_unknown 或官方时域结束；
4. `train_fpck_value.py` 按 anchor 分组 bootstrap，拟合候选 continuation 概率，
   并在 calibration split 做单侧 false-optimism 修正；
5. `run_fpck_main.py` 选择经验下界过阈值的最小偏离候选。空集时执行登记的
   emergency hold，并明确记录为 abstention，而非 certificate；
6. `compare_fpck_methods.py` 只在相同 suite/level/task/init/stream 上配对比较。

## 首轮决策实验（服务器）

推荐直接运行单命令入口（要求输出目录为空，防止把不同口径的证据追加到同一运行）：

```bash
cd /data/safeVLA
source tools/activate_env.sh
bash tools/run_fpck_first_round.sh data/eval/fpck_main_first
```

等价的分步命令如下，适合定位失败阶段：

```bash
cd /data/safeVLA
source tools/activate_env.sh
git status --short

RUN=data/eval/fpck_main_first
mkdir -p "$RUN/corpus"

# 训练池：四套件、两级、task0、init 0--4。每 cell 1 anchor × 4 siblings。
for suite in safelibero_spatial safelibero_goal safelibero_object safelibero_long; do
  for level in I II; do
    python tools/collect_fpck_counterfactual.py \
      --suite "$suite" --level "$level" --tasks 0 \
      --episodes 0 1 2 3 4 --streams 0 --candidates 4 \
      --out "$RUN/corpus/${suite}_L${level}"
  done
done

python tools/train_fpck_value.py \
  "$RUN"/corpus/*/samples.jsonl \
  --out "$RUN/continuation_model.json"

# 独立 init 5--9：同一确定性 π0.5，baseline 与 CKG 成对。
for suite in safelibero_spatial safelibero_goal safelibero_object safelibero_long; do
  for level in I II; do
    python tools/run_fpck_main.py --method baseline \
      --suite "$suite" --level "$level" --tasks 0 \
      --episodes 5 6 7 8 9 --streams 0 \
      --out "$RUN/baseline/${suite}_L${level}"
    python tools/run_fpck_main.py --method ckg \
      --model "$RUN/continuation_model.json" \
      --suite "$suite" --level "$level" --tasks 0 \
      --episodes 5 6 7 8 9 --streams 0 \
      --out "$RUN/ckg/${suite}_L${level}"
  done
done

# 合并各目录 episodes.jsonl 后运行配对比较；不得填补缺失 episode。
find "$RUN/baseline" -name episodes.jsonl -print0 | xargs -0 cat > "$RUN/baseline_all.jsonl"
find "$RUN/ckg" -name episodes.jsonl -print0 | xargs -0 cat > "$RUN/ckg_all.jsonl"
python tools/compare_fpck_methods.py "$RUN/baseline_all.jsonl" "$RUN/ckg_all.jsonl" \
  --out "$RUN/comparison.json"
```

## 首轮停止规则

- 若训练脚本报 `NO_SAFE_REENTRY_SUPPORT_OR_SPLIT_COVERAGE`：不训练、不包装结果，
  直接裁决当前池不支持 CKG；
- 若 CKG 的 STS/CAR 改善但 TSR 明显下降：报告安全--任务权衡；
- 若 CKG 与 baseline 无差异且从未干预：检查经验下界覆盖/选择率，不扩大主张；
- 若 CKG 增加任一决定性失败族：停止扩大实验；
- 只有首轮配对结果显示可行，才扩大到完整 E1b/H1。
