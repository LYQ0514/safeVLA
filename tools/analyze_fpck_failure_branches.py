#!/usr/bin/env python3
"""B：105 个 UNSAFE_RETURN 分支的离线归因分析（以 anchor 为统计单位）。

只读 data/eval/fpck_main_first/samples_effective_v2.jsonl 的 105 个 UNSAFE_RETURN
分支（bug 只影响 transportation 阶段重入→q_unknown，不影响 q_bad，故这些分支
与修复无关、标签稳定），每个干预候选与同 anchor nominal 配对。

预注册分类（优先级从高到低，互斥）：
  RECOVERY                   候选 q_acc（失败 nominal → 候选恢复成功）
  PROGRESS_WITHOUT_COMPLETION 候选进入比 nominal 更高生命周期但未 q_acc
  REDIRECTED_FAILURE          违规码改变（原碰撞消失、转化为另一违规）
  TRANSIENT_AVOIDANCE         Δτ_fail ≥ 5（延迟一个 chunk）但最终仍失败、同违规
  NO_EFFECT                   同违规且 |Δτ_fail| ≤ 1
  MARGINAL_DELAY              同违规、2 ≤ Δτ_fail ≤ 4（既不 NO_EFFECT 也不 TRANSIENT）

进入 C 的门槛：≥3 个独立失败 anchor 出现 TRANSIENT_AVOIDANCE 或
PROGRESS_WITHOUT_COMPLETION，且失败主要发生在单步干预结束、重新交还 nominal 之后。
"""
from __future__ import annotations

import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/data/safeVLA")
SAMPLES = ROOT / "data/eval/fpck_main_first/samples_effective_v2.jsonl"
BASES = [
    ROOT / "data/eval/fpck_main_first/corpus",
    ROOT / "data/eval/fpck_main_first_identity_repair/corpus",
]

MAX_STEPS = {"safelibero_spatial": 300, "safelibero_goal": 300,
             "safelibero_object": 300, "safelibero_long": 550}
LIFE_ORDER = {"RESTING": 0, "TASK_CONTACT": 1, "GRASP_ACQUISITION": 2,
              "CONTROLLED": 3, "TRANSPORTED": 4, "RELEASED": 5}
CHUNK = 5  # replan_steps
INTERVENTIONS = ("hold", "retreat", "tangent_left", "damped_0.5",
                 "tangent_right", "damped_0.25")


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def resolve_log(rel: str) -> Path | None:
    name = rel.split("/")[-1]
    for b in BASES:
        hits = list(b.glob(f"*/events/{name}"))
        if hits:
            return hits[0]
    return None


def max_lifecycle(fin: dict) -> str:
    lc = fin.get("final_lifecycle") or {}
    if not lc:
        return "RESTING"
    return max(lc.values(), key=lambda v: LIFE_ORDER.get(v, -1))


def first_qbad_control_step(events: list[dict]):
    for e in events:
        if e.get("event_type") == "control_step" and e.get("automaton_after") == "q_bad":
            return e.get("control_step")
    return None


def first_divergence(nom_ev: list[dict], cand_ev: list[dict]):
    n = min(len(nom_ev), len(cand_ev))
    for i in range(n):
        ne, ce = nom_ev[i], cand_ev[i]
        if ne.get("event_type") == "control_step" and ce.get("event_type") == "control_step":
            if ne.get("automaton_after") != ce.get("automaton_after"):
                return ce.get("control_step")
    return None


def trace_max_target_disp(events: list[dict]) -> float:
    mx = 0.0
    for e in events:
        td = e.get("target_displacement_mm")
        if isinstance(td, dict):
            vals = [v for v in td.values() if isinstance(v, (int, float))]
            if vals:
                mx = max(mx, max(vals))
        elif isinstance(td, (int, float)):
            mx = max(mx, float(td))
    return mx


def trace_max_obstacle_disp(events: list[dict]) -> float:
    mx = 0.0
    for e in events:
        od = e.get("obstacle_displacement_mm")
        if isinstance(od, (int, float)):
            mx = max(mx, float(od))
    return mx


def trace_has_contact(events: list[dict]) -> bool:
    return any(bool(e.get("contact_detail")) for e in events)


def main():
    rows = load_jsonl(SAMPLES)
    unsafe = [r for r in rows if r["label"] == "UNSAFE_RETURN"]
    by_anchor = defaultdict(dict)
    for r in rows:
        by_anchor[r["anchor_id"]][r["candidate_source"]] = r

    pairs = []  # per (anchor, candidate) result
    for aid, cands in by_anchor.items():
        nom = cands.get("nominal")
        if nom is None:
            continue
        # 只分析该 anchor 内 UNSAFE_RETURN 的干预候选
        for src in INTERVENTIONS:
            cand = cands.get(src)
            if cand is None or cand["label"] != "UNSAFE_RETURN":
                continue
            nom_ev = load_jsonl(resolve_log(nom["event_log"])) if resolve_log(nom["event_log"]) else []
            cand_ev = load_jsonl(resolve_log(cand["event_log"])) if resolve_log(cand["event_log"]) else []

            nom_fail = first_qbad_control_step(nom_ev)
            cand_fail = first_qbad_control_step(cand_ev)
            nom_viol = nom["automaton_after"].get("violation_code")
            cand_viol = cand["automaton_after"].get("violation_code")
            nom_life = max_lifecycle(nom["automaton_after"])
            cand_life = max_lifecycle(cand["automaton_after"])

            delta_tau = (cand_fail - nom_fail) if (nom_fail is not None and cand_fail is not None) else None
            viol_changed = (cand_viol != nom_viol)
            life_advanced = LIFE_ORDER[cand_life] > LIFE_ORDER[nom_life]
            div = first_divergence(nom_ev, cand_ev)

            # 失败是否在交还 nominal 之后（anchor 步 + 候选 1 步 + 至少 1 nominal 步）
            anchor_step = MAX_STEPS[cand["suite"]] - cand["remaining_horizon"]
            fail_after_handoff = (cand_fail is not None and cand_fail > anchor_step + 1)

            # 物理量差异（候选 vs nominal 全程最大值）
            td_delta = trace_max_target_disp(cand_ev) - trace_max_target_disp(nom_ev)
            od_delta = trace_max_obstacle_disp(cand_ev) - trace_max_obstacle_disp(nom_ev)

            # 语义 trace 是否完全一致
            semantic_identical = (div is None and cand_viol == nom_viol
                                  and cand_life == nom_life)

            # 分类（优先级）
            if cand["automaton_after"].get("final_automaton_mode") == "q_acc":
                cls = "RECOVERY"
            elif life_advanced:
                cls = "PROGRESS_WITHOUT_COMPLETION"
            elif viol_changed:
                cls = "REDIRECTED_FAILURE"
            elif delta_tau is not None and delta_tau >= CHUNK:
                cls = "TRANSIENT_AVOIDANCE"
            elif delta_tau is not None and abs(delta_tau) <= 1:
                cls = "NO_EFFECT"
            else:
                cls = "MARGINAL_DELAY"

            pairs.append({
                "suite": cand["suite"], "level": cand["level"],
                "episode": cand["episode"], "anchor_id": aid,
                "anchor_source": cand["anchor_source"],
                "candidate_source": src,
                "nominal_label": nom["label"],
                "delta_tau": delta_tau,
                "nom_viol": nom_viol, "cand_viol": cand_viol,
                "viol_changed": viol_changed,
                "nom_life": nom_life, "cand_life": cand_life,
                "life_advanced": life_advanced,
                "divergence_step": div,
                "anchor_step": anchor_step,
                "cand_fail_step": cand_fail,
                "fail_after_handoff": fail_after_handoff,
                "target_disp_delta_mm": round(td_delta, 2),
                "obstacle_disp_delta_mm": round(od_delta, 2),
                "semantic_identical": semantic_identical,
                "classification": cls,
            })

    out = ROOT / "data/audit/fpck_failure_attribution.jsonl"
    with out.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # ---- 报告 ----
    cls_counter = Counter(p["classification"] for p in pairs)
    print(f"=== B 离线归因：{len(pairs)} 个 (anchor, 候选) 配对，{len(set(p['anchor_id'] for p in pairs))} 个 anchor ===\n")
    print("候选级分类分布:")
    for c in ["RECOVERY", "PROGRESS_WITHOUT_COMPLETION", "REDIRECTED_FAILURE",
              "TRANSIENT_AVOIDANCE", "MARGINAL_DELAY", "NO_EFFECT"]:
        if cls_counter.get(c):
            print(f"  {c:28s}: {cls_counter[c]}")

    # anchor 级：每个 anchor 是否出现某分类
    anchor_cls = defaultdict(set)
    for p in pairs:
        anchor_cls[p["anchor_id"]].add(p["classification"])
    gate_anchor = [aid for aid, cs in anchor_cls.items()
                   if cs & {"TRANSIENT_AVOIDANCE", "PROGRESS_WITHOUT_COMPLETION"}]
    print(f"\n=== anchor 级汇总 ===")
    print(f"出现 TRANSIENT_AVOIDANCE 或 PROGRESS_WITHOUT_COMPLETION 的独立 anchor: {len(gate_anchor)}")
    print(f"出现 REDIRECTED_FAILURE 的独立 anchor: "
          f"{sum(1 for cs in anchor_cls.values() if 'REDIRECTED_FAILURE' in cs)}")
    print(f"纯 NO_EFFECT / MARGINAL_DELAY 的独立 anchor: "
          f"{sum(1 for cs in anchor_cls.values() if cs <= {'NO_EFFECT','MARGINAL_DELAY'})}")

    print("\n=== 各 anchor 分类明细（以 anchor 为统计单位）===")
    for aid in sorted(anchor_cls, key=lambda a: (pairs_by_anchor_meta(a, pairs))):
        cs = anchor_cls[aid]
        meta = next(p for p in pairs if p["anchor_id"] == aid)
        print(f"  {meta['suite'].split('_')[-1]:6s} L{meta['level']} s{meta['episode']} "
              f"({meta['anchor_source']:16s}): {sorted(cs)}")

    # 交还 nominal 后失败的比例（gate 第二条件）
    gate_pairs = [p for p in pairs if p["classification"] in
                  ("TRANSIENT_AVOIDANCE", "PROGRESS_WITHOUT_COMPLETION")]
    handoff_ratio = (sum(1 for p in gate_pairs if p["fail_after_handoff"]) / len(gate_pairs)
                     if gate_pairs else 0.0)
    print(f"\n=== gate 第二条件 ===")
    print(f"TRANSIENT/PROGRESS 配对中「失败在交还 nominal 后」比例: "
          f"{sum(1 for p in gate_pairs if p['fail_after_handoff'])}/{len(gate_pairs)} = {handoff_ratio:.2%}")

    # 语义 trace 完全一致的比例（NO_EFFECT 的强佐证）
    noeffect = [p for p in pairs if p["classification"] == "NO_EFFECT"]
    print(f"NO_EFFECT 配对中 semantic_identical 比例: "
          f"{sum(1 for p in noeffect if p['semantic_identical'])}/{len(noeffect)}")

    # 结论（按用户决策树，而非单一 gate）
    n_noeffect = cls_counter["NO_EFFECT"] + cls_counter["MARGINAL_DELAY"]
    n_redirect = cls_counter["REDIRECTED_FAILURE"]
    n_transient_progress = cls_counter["TRANSIENT_AVOIDANCE"] + cls_counter["PROGRESS_WITHOUT_COMPLETION"]
    total = len(pairs)
    print("\n=== 进入 C 的门槛裁决（决策树）===")
    print(f"候选级占比: NO_EFFECT/MARGINAL {n_noeffect}/{total} ({n_noeffect/total:.0%}) | "
          f"REDIRECTED {n_redirect}/{total} ({n_redirect/total:.0%}) | "
          f"TRANSIENT/PROGRESS {n_transient_progress}/{total} ({n_transient_progress/total:.0%})")
    print(f"anchor 级: 纯 NO_EFFECT {sum(1 for cs in anchor_cls.values() if cs <= {'NO_EFFECT','MARGINAL_DELAY'})}/28 | "
          f"REDIRECTED {sum(1 for cs in anchor_cls.values() if 'REDIRECTED_FAILURE' in cs)}/28 | "
          f"TRANSIENT/PROGRESS {len(gate_anchor)}/28")
    print()
    # 关键诊断：NO_EFFECT 中语义 trace 是否仍分歧
    print(f"诊断：NO_EFFECT 配对中 semantic_identical 仅 {sum(1 for p in noeffect if p['semantic_identical'])}/{len(noeffect)}，")
    print(f"即 {len(noeffect) - sum(1 for p in noeffect if p['semantic_identical'])}/{len(noeffect)} 的 NO_EFFECT 轨迹语义有分歧，")
    print(f"但分歧幅度不足以改变失败时间/违规码 —— 说明候选扰动是「亚阈值」的。")

    cond1 = len(gate_anchor) >= 3
    cond2 = handoff_ratio >= 0.5
    majority_noeffect = n_noeffect > 0.5 * total
    print("\n结论:")
    if majority_noeffect:
        print("  → 多数为 NO_EFFECT/MARGINAL（>50%），按裁决「若多数为 NO_EFFECT 则不直接做 C」。")
        print("    应【先重构候选几何与控制权限】，而非进入整 chunk backward reachability：")
        print("    延长一个亚阈值动作不会产生恢复能力。")
        if n_redirect > n_transient_progress:
            print("    同时 REDIRECTED_FAILURE 占比显著，需考虑【联合约束候选】而非单纯扩大动作幅度。")
    elif cond1 and cond2:
        print("  → 候选有控制作用但单步 continuation 定义太短，进入整 chunk + 分层 backward reachability。")
    else:
        print("  → 证据不充分：无明确多数，需进一步核对。")

    print(f"\n结果已写: {out}")


def pairs_by_anchor_meta(aid, pairs):
    return str(aid)


if __name__ == "__main__":
    main()
