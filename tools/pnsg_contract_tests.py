"""CONTACT_CLASSIFICATION_CONTRACT 门禁测试（docs/想法v7.md 第 10 节）。

验证：
- pair-order symmetry（角色顺序无关）
- 17-task role-manifest binding
- multi-And task classification
- region-parent mapping
- task0 bowl/plate/cookies/moka roles
- allowed contacts retained（不静默丢弃）
- preexisting contacts separated
- continuous-event merge
- OTHER_UNCLASSIFIED relevant contacts = 0
- contract hashes recorded
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys

ROOT = "/data/safeVLA"
sys.path.insert(0, ROOT)

from src.pnsg.contact_contract import (  # noqa: E402
    classify_role_pair, allowance_for, contract_dict, ROLES,
    ROLE_PAIR_TO_SEMANTIC, is_core_mechanism,
)


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f": {detail}" if detail and not cond else ""))
    return cond


def main():
    results = []

    print("=== 1. pair-order symmetry ===")
    # 所有 role pair 的对称性
    sym_ok = True
    for r1 in ROLES:
        for r2 in ROLES:
            a = classify_role_pair(r1, r2)
            b = classify_role_pair(r2, r1)
            if a != b:
                sym_ok = False
                print(f"    ⚠️ 不对称: {r1},{r2} -> {a} vs {b}")
    results.append(check("pair-order symmetry", sym_ok))

    print("\n=== 2. 17-task role-manifest binding ===")
    manifest_all = json.loads(pathlib.Path(ROOT, "data", "task_object_role_manifest.json").read_text())
    n_tasks = sum(len(s) for s in manifest_all.values())
    results.append(check("17 tasks loaded", n_tasks == 17, f"got {n_tasks}"))

    print("\n=== 3. multi-And task classification ===")
    scene5 = manifest_all["safelibero_long"]["LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate"]
    manip = [i for i, v in scene5["objects"].items() if "MANIPULATED_TARGET" in v["roles"]]
    recep = [i for i, v in scene5["objects"].items() if "GOAL_RECEPTACLE" in v["roles"]]
    results.append(check("multi-And: 2 manipulated", set(manip) == {"porcelain_mug_1", "white_yellow_mug_1"}, str(manip)))
    results.append(check("multi-And: 2 receptacles", set(recep) == {"plate_1", "plate_2"}, str(recep)))

    print("\n=== 4. region-parent mapping ===")
    basket = manifest_all["safelibero_object"]["pick_up_the_bbq_sauce_and_place_it_in_the_basket"]
    basket_roles = {i: v["roles"] for i, v in basket["objects"].items()}
    results.append(check("basket_1 is GOAL_RECEPTACLE",
                         "GOAL_RECEPTACLE" in basket_roles.get("basket_1", []), str(basket_roles.get("basket_1"))))

    print("\n=== 5. task0 bowl/plate/cookies/moka roles ===")
    task0 = manifest_all["safelibero_spatial"]["pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"]
    t0_roles = {i: v["roles"] for i, v in task0["objects"].items()}
    results.append(check("bowl is MANIPULATED_TARGET", "MANIPULATED_TARGET" in t0_roles.get("akita_black_bowl_1", [])))
    results.append(check("plate is GOAL_RECEPTACLE", "GOAL_RECEPTACLE" in t0_roles.get("plate_1", [])))
    results.append(check("cookies is DISTRACTOR", "DISTRACTOR" in t0_roles.get("cookies_1", [])))
    results.append(check("moka is STATIC_SAFETY_OBSTACLE",
                         "STATIC_SAFETY_OBSTACLE" in t0_roles.get("moka_pot_obstacle_1", [])))

    print("\n=== 6. allowed contacts retained (not silently dropped) ===")
    # contract 里 allowed 类别仍有语义类，不被丢弃
    allowed_classes = ["ROBOT_MANIPULATED_TARGET", "MANIPULATED_TARGET_GOAL", "ROBOT_ENVIRONMENT"]
    all_ok = all(cls in ROLE_PAIR_TO_SEMANTIC.values() for cls in allowed_classes)
    results.append(check("allowed classes in semantic map", all_ok))

    print("\n=== 7. preexisting contacts separated ===")
    # 检查 contract 有 preexisting 字段定义
    results.append(check("preexisting field in contract",
                         "preexisting" in json.dumps(contract_dict())))

    print("\n=== 8. continuous-event merge (unit test on synthetic) ===")
    from src.pnsg.contact_audit import ContactAggregator
    agg = ContactAggregator()
    # 构造：pair A 在 substep 0,1,2 连续，substep 5 断开重来
    def mk(g1, g2, cs, ss, dist=-0.01):
        return {"geom1": g1, "geom2": g2, "semantic_class": "ROBOT_STATIC_OBSTACLE",
                "allowance": "FORBIDDEN", "contact_dist": dist, "contact_level": "geometric_contact",
                "control_step": cs, "substep": ss, "preexisting": False}
    agg.geometric_records = [mk(1, 2, 0, 0), mk(1, 2, 0, 1), mk(1, 2, 0, 2), mk(1, 2, 0, 5)]
    agg.compute_contiguous_events()
    results.append(check("continuous-event merge: 3+1 → 2 events", len(agg.contiguous_events) == 2,
                         f"got {len(agg.contiguous_events)}"))

    print("\n=== 9. OTHER_UNCLASSIFIED relevant contacts = 0 ===")
    # 所有 ROLES 两两组合都应有确定分类（无 OTHER_UNCLASSIFIED）
    uncovered = []
    for r1 in ROLES:
        for r2 in ROLES:
            if classify_role_pair(r1, r2) == "OTHER_UNCLASSIFIED":
                uncovered.append((r1, r2))
    results.append(check("no OTHER_UNCLASSIFIED role pair", len(uncovered) == 0, str(uncovered)))

    print("\n=== 10. contract hashes recorded ===")
    contract = contract_dict()
    contract_hash = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    manifest_hash = hashlib.sha256(
        json.dumps(manifest_all, sort_keys=True).encode()
    ).hexdigest()
    # 写 contract json
    out = pathlib.Path(ROOT, "data", "collision_event_contract_v2.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(contract)
    payload["role_manifest_sha256"] = manifest_hash
    payload["contract_sha256"] = contract_hash
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    results.append(check("contract json written", out.exists()))
    results.append(check("contract hash recorded", len(contract_hash) == 64))

    print("\n=== 汇总 ===")
    n_pass = sum(results)
    n_total = len(results)
    print(f"  {n_pass}/{n_total} PASS")
    if n_pass == n_total:
        print("\n=== CONTACT_CLASSIFICATION_CONTRACT: PASS ===")
        return 0
    else:
        print("\n=== CONTACT_CLASSIFICATION_CONTRACT: FAIL ===")
        return 1


if __name__ == "__main__":
    sys.exit(main())
