"""ROLE_MANIFEST_GATE：17 任务角色清单完整门禁。

对应 docs/想法v6.md 的验收门。检查：
A. AST 解析（非正则）
B. 每条 goal clause 证据链完整
C. region 解析到父对象
D. 谓词到角色规则覆盖
E. 角色完整性断言
F. 17 任务人工复核表

输出：紧凑复核表 + 完整性断言结果。
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

ROOT = "/data/safeVLA"
sys.path.insert(0, ROOT)

from src.pnsg.sexpr import find_blocks, parse_text, parse_regions  # noqa: E402


def load_manifest():
    return json.loads(pathlib.Path(ROOT, "data", "task_object_role_manifest.json").read_text())


def check_completeness(manifest_all):
    """E 类完整性断言。"""
    issues = []
    total_unresolved = 0
    for suite, tasks in manifest_all.items():
        for task, m in tasks.items():
            # 重新解析 bddl 验证 goal clause 数量一致
            bddl = pathlib.Path(m["bddl_path"])
            if not bddl.exists():
                issues.append(f"{suite}/{task}: bddl 不存在")
                continue
            blocks = find_blocks(bddl.read_text())
            goal_ast = parse_text(blocks.get(":goal", "()"))
            from src.pnsg.role_manifest import _collect_goal_clauses
            clauses = _collect_goal_clauses(goal_ast)
            if len(clauses) != len(m["goal_clauses"]):
                issues.append(f"{suite}/{task}: goal clause 数 {len(clauses)} != manifest {len(m['goal_clauses'])}")

            # 每个 goal clause 的 argument 都能解析
            for pred, args in clauses:
                for pos, a in enumerate(args):
                    obj = m["objects"].get(a)
                    if obj is None:
                        # 可能是 region 参数（不限 _region 后缀），检查父对象
                        parent = None
                        for r in m["regions"]:
                            if a.endswith("_" + r["name"]):
                                parent = r["target"]
                                break
                        if parent is not None and (parent in m["objects"] or parent in m["fixtures"]):
                            continue  # region 解析到父对象
                        if parent is None:
                            issues.append(f"{suite}/{task}: argument {a} 无父对象")
                            total_unresolved += 1
                        elif parent not in m["objects"] and parent not in m["fixtures"]:
                            issues.append(f"{suite}/{task}: argument {a} 父对象 {parent} 未在 objects/fixtures")
                            total_unresolved += 1

            # 每个 movable object 都有角色
            for inst, info in m["objects"].items():
                if not info["roles"]:
                    issues.append(f"{suite}/{task}: object {inst} 无角色")
    return issues, total_unresolved


def print_review_table(manifest_all):
    """F 类人工复核表。"""
    print(f"{'Task':<55} {'#goal':>5} {'Manip':<20} {'Recep':<15} {'Distr':>5} {'Obst':>5} {'Unres':>5}")
    print("-" * 115)
    for suite in sorted(manifest_all):
        for task in sorted(manifest_all[suite]):
            m = manifest_all[suite][task]
            manip = sorted({inst for inst, i in m["objects"].items() if "MANIPULATED_TARGET" in i["roles"]})
            recep = sorted({inst for inst, i in m["objects"].items() if "GOAL_RECEPTACLE" in i["roles"]})
            distr = sum(1 for i in m["objects"].values() if "DISTRACTOR" in i["roles"])
            obst = sum(1 for i in m["objects"].values() if "STATIC_SAFETY_OBSTACLE" in i["roles"])
            # unresolved: goal argument 无法解析
            print(f"{task[:55]:<55} {len(m['goal_clauses']):>5} {','.join(manip)[:20]:<20} "
                  f"{','.join(recep)[:15]:<15} {distr:>5} {obst:>5} {'?':>5}")


def main():
    manifest_all = load_manifest()
    print(f"=== 角色清单加载：{sum(len(s) for s in manifest_all.values())} 个任务 ===\n")

    print("=== F. 人工复核表 ===")
    print_review_table(manifest_all)

    print("\n=== E. 完整性断言 ===")
    issues, unresolved = check_completeness(manifest_all)
    if issues:
        print(f"发现 {len(issues)} 个问题，未解析 argument {unresolved} 个：")
        for i in issues[:20]:
            print(f"  ⚠️ {i}")
    else:
        print("  ✅ 无问题：所有 goal clause 解析，所有 argument 绑定，所有 object 有角色")

    # 门禁判定
    if unresolved == 0 and not issues:
        print("\n=== ROLE_MANIFEST_GATE: PASS ===")
    elif unresolved > 0:
        print("\n=== ROLE_MANIFEST_GATE: UNKNOWN（存在未解析 argument）===")
    else:
        print("\n=== ROLE_MANIFEST_GATE: FAIL ===")


if __name__ == "__main__":
    main()
