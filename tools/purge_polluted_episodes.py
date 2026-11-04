"""删除受并发污染影响的 episode，为重跑做准备。

污染机制（已修复于 b2226d9）：LI/LII 并行时共用 /tmp/aegis_obstacle_detection.png，
竞态导致 VLM 读到错误图片 → 识别错误障碍物（静默污染）或图片损坏（LI ep39 OSError）。

污染范围（时间线证据）：
- LI: 最初与 LII 并行（d8bfafd），崩溃在 task0 ep38 后，summary 保存 20 个（ep0-19）
      → 重跑 ep0-19（20 个）。
- LII: 全程 d8bfafd，与 LI 并行期间完成的部分污染；LII 更快，保守重跑 task0 全部 50 个。

用法：python tools/purge_polluted_episodes.py --dry-run   # 先看会删什么
       python tools/purge_polluted_episodes.py           # 执行删除（自动备份）
"""

from __future__ import annotations
import json
import shutil
import sys

ROOT = "/data/safeVLA"

PLAN = {
    "aegis_perception_spatial_LI":  {"task": 0, "max_ep": 19},   # 20 个
    "aegis_perception_spatial_LII": {"task": 0, "max_ep": 49},   # 50 个（保守）
}


def main():
    dry = "--dry-run" in sys.argv
    for fn, spec in PLAN.items():
        p = f"{ROOT}/data/eval/{fn}/summary.json"
        d = json.load(open(p))
        before = len(d["episodes"])
        keep = [e for e in d["episodes"]
                if not (e["task_id"] == spec["task"]
                        and e["episode_idx"] <= spec["max_ep"])]
        removed = before - len(keep)
        print(f"{fn}: 删除 task{spec['task']} ep0-{spec['max_ep']} "
              f"({removed} 个)，剩余 {len(keep)} 个")
        if dry:
            continue
        # 备份
        shutil.copy(p, p + ".bak_polluted")
        d["episodes"] = keep
        json.dump(d, open(p, "w"), ensure_ascii=False, indent=2)
        print(f"  已备份到 {p}.bak_polluted 并写入删除后的 summary")


if __name__ == "__main__":
    main()
