"""G2 病例选择：按预注册 S1-S4 判据离线分析 B0 结果，生成冻结病例表。

判据见 G2_CASE_CRITERIA.md。输出 G2_CASE_SELECTION.csv。
"""

from __future__ import annotations
import csv
import json
import os

import numpy as np

ROOT = "/data/safeVLA"
SRC = os.path.join(ROOT, "experiments", "stg_v1", "outputs", "g2_b0_episodes.jsonl")
OUT = os.path.join(ROOT, "experiments", "stg_v1", "outputs", "G2_CASE_SELECTION.csv")

MAX_STEPS = 550


def classify(ep):
    steps_log = ep["steps_log"]
    n = len(steps_log)
    success = ep["success"]
    intervened = [s["intervened"] for s in steps_log]

    # 后 30% 步
    tail = steps_log[int(n * 0.7):]
    tail_interv_ratio = np.mean([s["intervened"] for s in tail]) if tail else 0.0

    # 末端后 100 步位移
    if len(steps_log) >= 100:
        eef_start = np.array(steps_log[-100]["eef_pos"])
        eef_end = np.array(steps_log[-1]["eef_pos"])
        eef_tail_disp = float(np.sum(np.abs(eef_end - eef_start)))
    else:
        eef_tail_disp = float("inf")

    # 名义动作范数（后 30%）
    tail_nominal = np.mean([
        np.linalg.norm(np.array(s["nominal_7d"][:3])) for s in tail
    ]) if tail else 0.0

    # 修正方向翻转次数
    flips = 0
    prev_dir = None
    for s in steps_log:
        if s["intervened"]:
            delta = np.array(s["exec_7d"][:3]) - np.array(s["nominal_7d"][:3])
            dn = np.linalg.norm(delta)
            if dn > 1e-6:
                d = delta / dn
                if prev_dir is not None and np.dot(prev_dir, d) < 0:
                    flips += 1
                prev_dir = d

    # 初始惯性推走：前 10 步 mug 位移
    mug_early = max((s["mug_disp_mm"] for s in steps_log[:10]), default=0.0)

    labels = []
    # S1 停滞：CBF 长期活跃 + mug 无碰撞 + 名义动作仍在驱动 + 任务未完成
    # （末端被 CBF 卡在虚假平衡点，VLA 仍在尝试但被挡住）
    if (not success and n >= MAX_STEPS - 1 and tail_interv_ratio >= 0.8
            and tail_nominal > 0.05 and ep["max_mug_disp_mm"] < 1.0):
        labels.append("S1")
    # S2 振荡
    if flips >= 3 and (not success or ep["max_mug_disp_mm"] > 1):
        labels.append("S2")
    # S3 抓取失败 / VLA 无能：mug 无碰撞 + CBF 不活跃（tail_interv < 0.5）
    if ep["max_mug_disp_mm"] < 1.0 and tail_interv_ratio < 0.5 and not success:
        labels.append("S3")
    # S4 间接碰撞 / 完整性违规：mug 位移 > 1mm（CAR 碰撞）或任务物体被推
    if ep["max_mug_disp_mm"] > 1.0 or \
       any(v > 5 for v in ep.get("task_final_disp_mm", {}).values()):
        labels.append("S4")

    primary = labels[0] if labels else ("NEG" if success else "UNCLASSIFIED")
    secondary = ",".join(labels[1:]) if len(labels) > 1 else ""
    return primary, secondary, {
        "tail_interv_ratio": round(tail_interv_ratio, 3),
        "eef_tail_disp_m": round(eef_tail_disp, 4),
        "tail_nominal_norm": round(tail_nominal, 4),
        "flips": flips,
        "mug_early_mm": round(mug_early, 2),
    }


def main():
    eps = [json.loads(l) for l in open(SRC)]
    rows = []
    for ep in eps:
        primary, secondary, diag = classify(ep)
        rows.append({
            "task_id": ep["task_id"], "state_id": ep["state_id"],
            "stream": ep["stream"],
            "success": ep["success"], "steps": ep["steps"],
            "max_mug_disp_mm": ep["max_mug_disp_mm"],
            "forbidden_steps": ep["forbidden_steps"],
            "primary_label": primary, "secondary_label": secondary,
            **diag,
        })

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # 汇总
    from collections import Counter
    c = Counter(r["primary_label"] for r in rows)
    print(f"病例表已写 {len(rows)} 行 → {OUT}")
    print(f"分类分布: {dict(c)}")
    for r in rows:
        if r["primary_label"] not in ("NEG",):
            print(f"  {r['primary_label']}: task{r['task_id']}/s{r['state_id']}/st{r['stream']} "
                  f"success={r['success']} steps={r['steps']} mug={r['max_mug_disp_mm']}mm "
                  f"flips={r['flips']} tail_interv={r['tail_interv_ratio']}")


if __name__ == "__main__":
    main()
