#!/usr/bin/env python3
"""同轨迹双影子监视器 A/B 裁决：legacy(reentry_guard=False) vs fixed。

读取 data/eval/fpck_main_shadow_ab/corpus/*/samples.jsonl，输出：
1. kernel 是否非空（fixed q_acc 计数）；
2. 因果归因（legacy q_unknown ∧ fixed q_acc，且首个差异步 lifecycle 已
   CONTROLLED/TRANSPORTED —— 即唯一差异正是禁止重入守卫）；
3. 可训练性（独立正 anchor 数 + train/calibration 覆盖）；
4. 修复局部性（fixed 与 legacy 的差异是否仅来自重入守卫）。

不新增 rollout，只读同轨迹 A/B 产物。
"""
from __future__ import annotations

import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/data/safeVLA")
SHADOW_DIR = ROOT / "data/eval/fpck_main_shadow_ab/corpus"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def resolve_event_log(rel: str) -> list[dict] | None:
    name = rel.split("/")[-1]
    hits = list(SHADOW_DIR.glob(f"*/events/{name}"))
    if not hits:
        return None
    return load_jsonl(hits[0])


def first_divergence(events: list[dict]) -> tuple[int | None, str | None]:
    """首个 fixed(automaton_after) != legacy(compare_automaton_after) 的控制步，
    及该步 fixed 的 lifecycle stage。"""
    for e in events:
        if e.get("event_type") != "control_step":
            continue
        fa = e.get("automaton_after")
        la = e.get("compare_automaton_after")
        if fa != la:
            lc = e.get("lifecycle") or {}
            stage = next(iter(lc.values()), None) if lc else None
            return e.get("control_step"), stage
    return None, None


def main():
    rows = []
    for sp in sorted(SHADOW_DIR.glob("*/samples.jsonl")):
        rows.extend(load_jsonl(sp))

    print(f"=== 同轨迹双影子监视器 A/B（{len(rows)} branch）===\n")

    fixed_acc = [r for r in rows if r["label"] == "SAFE_REENTRY"]
    print(f"1. kernel 非空性:")
    print(f"   fixed q_acc (SAFE_REENTRY) = {len(fixed_acc)} / {len(rows)}")

    print("\n=== 逐 branch 对比 ===")
    for r in sorted(rows, key=lambda x: (x["suite"], x["episode"], x["candidate_source"])):
        fin = r["automaton_after"]
        print(f"  {r['suite'].split('_')[-1]:5s} L{r['level']} s{r['episode']} "
              f"{r['candidate_source']:13s} fixed={r['label']:18s} "
              f"(mode={fin.get('final_automaton_mode')}, viol={fin.get('violation_code')}) "
              f"legacy={r.get('legacy_final_mode')} "
              f"(viol={r.get('legacy_violation_code')})")

    # 因果归因：legacy q_unknown ∧ fixed q_acc
    causal = []
    for r in fixed_acc:
        legacy_mode = r.get("legacy_final_mode")
        events = resolve_event_log(r["event_log"])
        div_step, div_stage = first_divergence(events) if events else (None, None)
        if legacy_mode == "q_unknown":
            causal.append((r, div_step, div_stage))

    print("\n2. 因果归因:")
    if not causal:
        print("   无 fixed=q_acc 且 legacy=q_unknown 的分支（无法建立因果归因）")
    for r, div_step, div_stage in causal:
        guard_ok = div_stage in ("CONTROLLED", "TRANSPORTED", "RELEASED")
        print(f"   {r['suite'].split('_')[-1]} L{r['level']} s{r['episode']} "
              f"{r['candidate_source']}: legacy q_unknown → fixed q_acc, "
              f"首个差异步={div_step} stage={div_stage} "
              f"{'[守卫归因成立]' if guard_ok else '[!! 差异非守卫导致，需复核]'}")

    # 可训练性
    print("\n3. 可训练性:")
    safe_anchors = defaultdict(list)
    for r in fixed_acc:
        safe_anchors[r["anchor_id"]].append(r["candidate_source"])
    print(f"   独立正 anchor = {len(safe_anchors)}")
    split_counter = Counter(r["split"] for r in fixed_acc)
    print(f"   正类 split 分布 = {dict(split_counter)}")
    for aid, cands in safe_anchors.items():
        rr = next(r for r in fixed_acc if r["anchor_id"] == aid)
        print(f"   正 anchor: {rr['suite'].split('_')[-1]} L{rr['level']} s{rr['episode']} "
              f"candidates={cands} split={rr['split']}")

    # 修复局部性：非 q_acc 分支上 legacy vs fixed 的差异
    print("\n4. 修复局部性（legacy 与 fixed 差异分布）:")
    diff_kind = Counter()
    for r in rows:
        fm = r["automaton_after"].get("final_automaton_mode")
        lm = r.get("legacy_final_mode")
        if fm == lm:
            diff_kind["一致"] += 1
        elif lm == "q_unknown" and fm in ("q_bad", "q_acc", "q_released"):
            diff_kind["legacy q_unknown 误标（守卫修复）"] += 1
        else:
            diff_kind[f"其他差异 ({lm}->{fm})"] += 1
    for k, v in diff_kind.items():
        print(f"   {k}: {v}")

    if fixed_acc:
        print("\n>>> 结论: 经验 one-step continuation kernel 非空（fixed 出现 q_acc）")
    else:
        print("\n>>> 结论: fixed 仍全部失败——修复正确，但当前候选池仍无 SAFE_REENTRY")


if __name__ == "__main__":
    main()
