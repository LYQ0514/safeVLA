"""对象角色清单生成器（v6 版：AST 解析 + 完整证据链）。

对应 docs/想法v5.md 第四节 + 想法v6.md 的 ROLE_MANIFEST_GATE。

用 S-expression AST 解析 BDDL，正确处理：
- 多重 And goal clause
- region 参数 → 父对象解析
- 多角色支持（同一 object 可出现在多个 goal clause）
- 完整证据链（goal clause + argument position）

角色（允许集合）：
- MANIPULATED_TARGET：goal On/In 第一参数
- GOAL_RECEPTACLE：goal On/In 第二参数（若父对象是 movable object）
- TASK_CONTROLLED_OBJECT：Open/Close/TurnOn 等的操作对象（本 benchmark 未出现，预留）
- DISTRACTOR：objects 里出现、非上述角色
- STATIC_SAFETY_OBSTACLE：名字含 obstacle
- ROBOT：robot0/gripper0

注意：只用于机制审计和认证器，不修改官方 CAR。
"""

from __future__ import annotations

import hashlib
import json
import pathlib

from src.pnsg.sexpr import find_blocks, parse_text, parse_declarations, parse_regions


def _collect_goal_clauses(ast):
    """从 goal AST 收集所有 (On/In/Open/Close/TurnOn/TurnOff a b) 子句。

    返回 [(predicate, args_list)]。
    """
    clauses = []
    if isinstance(ast, dict):
        pred = ast["predicate"]
        if pred in ("On", "In", "Open", "Close", "TurnOn", "TurnOff"):
            args = [a for a in ast["arguments"] if isinstance(a, str)]
            clauses.append((pred, args))
        for a in ast["arguments"]:
            clauses.extend(_collect_goal_clauses(a))
    return clauses


def _resolve_region_to_object(region_name, regions):
    """region 名 → 父对象。region 完整名 = 父对象 + '_' + region_name。"""
    for r in regions:
        if region_name.endswith("_" + r["name"]):
            return r["target"]
    return None


def build_role_manifest(bddl_path):
    """从 bddl 生成角色清单（v6 完整版）。"""
    bddl_path = pathlib.Path(bddl_path)
    bddl_text = bddl_path.read_text()
    bddl_sha = hashlib.sha256(bddl_text.encode()).hexdigest()

    blocks = find_blocks(bddl_text)
    objects = parse_declarations(blocks.get(":objects", "()"))
    fixtures = parse_declarations(blocks.get(":fixtures", "()"))
    regions = parse_regions(blocks.get(":regions", "()"))
    goal_ast = parse_text(blocks.get(":goal", "()")) if ":goal" in blocks else None
    goal_clauses = _collect_goal_clauses(goal_ast) if goal_ast else []

    # obj_of_interest
    ooi_names = []
    if ":obj_of_interest" in blocks:
        ooi_ast = parse_text(blocks[":obj_of_interest"])
        if ooi_ast:
            ooi_names = [a for a in ooi_ast["arguments"] if isinstance(a, str)]

    # 收集每个 object 的角色 + 证据
    roles = {}  # inst -> {"roles": set, "evidence": list}

    for inst, cls in objects.items():
        roles[inst] = {"roles": set(), "evidence": [], "object_class": cls}

    # 从 goal clause 推导角色
    for pred, args in goal_clauses:
        if pred in ("On", "In"):
            # 第一参数：被操作物
            if args and args[0] in roles:
                roles[args[0]]["roles"].add("MANIPULATED_TARGET")
                roles[args[0]]["evidence"].append({
                    "predicate": pred, "argument_position": 0,
                    "goal_clause": f"({pred} {' '.join(args)})",
                })
            # 第二参数：可能是 object 或 region
            if len(args) >= 2:
                second = args[1]
                if second in roles:
                    # 直接是 movable object
                    roles[second]["roles"].add("GOAL_RECEPTACLE")
                    roles[second]["evidence"].append({
                        "predicate": pred, "argument_position": 1,
                        "goal_clause": f"({pred} {' '.join(args)})",
                    })
                elif second.endswith("_region"):
                    parent = _resolve_region_to_object(second, regions)
                    if parent and parent in roles:
                        roles[parent]["roles"].add("GOAL_RECEPTACLE")
                        roles[parent]["evidence"].append({
                            "predicate": pred, "argument_position": 1,
                            "goal_clause": f"({pred} {' '.join(args)})",
                            "region_name": second, "region_parent": parent,
                        })
        elif pred in ("Open", "Close", "TurnOn", "TurnOff"):
            if args and args[0] in roles:
                roles[args[0]]["roles"].add("TASK_CONTROLLED_OBJECT")
                roles[args[0]]["evidence"].append({
                    "predicate": pred, "argument_position": 0,
                    "goal_clause": f"({pred} {' '.join(args)})",
                })

    # 障碍物
    for inst in objects:
        if "obstacle" in inst or "obstacle" in objects[inst]:
            roles[inst]["roles"].add("STATIC_SAFETY_OBSTACLE")

    # 无角色的 → DISTRACTOR
    for inst in roles:
        if not roles[inst]["roles"]:
            roles[inst]["roles"].add("DISTRACTOR")

    manifest = {
        "bddl_path": str(bddl_path),
        "bddl_sha256": bddl_sha,
        "goal_predicate": json.dumps(goal_ast, ensure_ascii=False) if goal_ast else "",
        "goal_clauses": goal_clauses,
        "obj_of_interest": ooi_names,
        "fixtures": fixtures,
        "regions": regions,
        "objects": {},
    }

    for inst in objects:
        manifest["objects"][inst] = {
            "object_class": objects[inst],
            "roles": sorted(roles[inst]["roles"]),
            "evidence": roles[inst]["evidence"],
            "in_obj_of_interest": inst in ooi_names,
        }

    manifest["role_manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    return manifest


def build_all_manifests(bddl_root, out_path):
    """遍历所有套件的 bddl，生成全部角色清单。"""
    bddl_root = pathlib.Path(bddl_root)
    all_manifests = {}
    for suite_dir in sorted(bddl_root.iterdir()):
        if not suite_dir.is_dir():
            continue
        suite = suite_dir.name
        all_manifests[suite] = {}
        for bddl_file in sorted(suite_dir.glob("*.bddl")):
            manifest = build_role_manifest(bddl_file)
            all_manifests[suite][bddl_file.stem] = manifest

    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_manifests, f, ensure_ascii=False, indent=2)
    return all_manifests


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else \
        "/data/safeVLA/dataset/SafeLIBERO/safelibero/libero/libero/bddl_files"
    out = sys.argv[2] if len(sys.argv) > 2 else \
        "/data/safeVLA/data/task_object_role_manifest.json"
    manifests = build_all_manifests(root, out)
    total = sum(len(s) for s in manifests.values())
    print(f"生成 {total} 个任务的角色清单 → {out}")
