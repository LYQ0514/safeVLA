#!/bin/bash
# 串行补跑 AEGIS perception 变体：goal / object / long 三个套件，各 LI + LII。
# 用法：bash tools/run_remaining_aegis.sh
# 串行（避免 GPU 竞争 + perception 并发写临时文件的历史问题）。
set -u
cd /data/safeVLA

PY=.venv/bin/python
EP="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49"

run_one() {
  local suite=$1 level=$2 out=$3 log=$4
  echo "=== $(date '+%F %T') 启动 ${suite} level ${level} ==="
  env PYTHONPATH=/data/safeVLA "$PY" -u tools/run_eval_aegis.py \
    --suite "$suite" --level "$level" --tasks 0 1 2 3 --episodes $EP \
    --resolution 1024 --policy-resize 224 --replan-steps 5 --num-steps-wait 20 \
    --perception --no-video --resume --out "data/eval/$out" > "$log" 2>&1
  echo "=== $(date '+%F %T') 完成 ${suite} level ${level}（exit=$?）==="
}

run_one safelibero_goal   I  aegis_perception_goal_LI   /tmp/aegis_goal_LI.log
run_one safelibero_goal   II aegis_perception_goal_LII  /tmp/aegis_goal_LII.log
run_one safelibero_object I  aegis_perception_object_LI   /tmp/aegis_object_LI.log
run_one safelibero_object II aegis_perception_object_LII  /tmp/aegis_object_LII.log
run_one safelibero_long   I  aegis_perception_long_LI   /tmp/aegis_long_LI.log
run_one safelibero_long   II aegis_perception_long_LII  /tmp/aegis_long_LII.log

echo "=== $(date '+%F %T') 全部 6 个评测完成 ==="
