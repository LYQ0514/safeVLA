"""对象角色清单生成器（docs/想法v5.md Step 1）。

从 BDDL 解析对象角色，生成 task_object_role_manifest.json。

角色定义（想法v5.md 第四节）：
- MANIPULATED_TARGET：goal predicate 中「被搬运」的物体（goal 的 On 第一参数）
- GOAL_RECEPTACLE：goal predicate 中的目标容器（goal 的 On 第二参数）
- DISTRACTOR：objects 里出现、非 obstacle、非 goal 相关的 movable 物体
- STATIC_SAFETY_OBSTACLE：名字含 obstacle 的物体
- ROBOT：robot0/gripper0 相关 body

注意：角色清单只用于机制审计和认证器，**不修改官方 CAR evaluator**。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import pathlib


def _parse_objects(bddl_text):
    """解析 (:objects ...) 块，返回 {instance_name: class_name}。"""
    m = re.search(r'\(:objects(.*?)\)\s*\(:', bddl_text, re.S)
    if not m:
        return {}
    body = m.group(1)
    result = {}
    # 每行是 "inst1 inst2 ... - class_name"
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith(';'):
            continue
        # 去掉行内注释
        line = line.split(';')[0].strip()
        if not line or '-' not in line:
            continue
        # 找最后一个 " - " 分隔
        parts = re.split(r'\s+-\s+', line)
        if len(parts) < 2:
            continue
        instances = parts[0].split()
        cls = parts[-1].strip()
        for inst in instances:
            result[inst] = cls
    return result


def _parse_goal(bddl_text):
    """解析 (:goal ...) 块，返回 goal 文本（用括号深度匹配，非正则截断）。"""
    start = bddl_text.find("(:goal")
    if start == -1:
        return ""
    depth = 0
    i = start
    while i < len(bddl_text):
        ch = bddl_text[i]
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0:
                return bddl_text[start:i + 1]
        i += 1
    return bddl_text[start:]


def _parse_goal_roles(goal_text):
    """从 goal predicate 提取 MANIPULATED_TARGET 和 GOAL_RECEPTACLE。

    goal 形如 (And (On x y)) 或 (And (In x region))。
    每个二元关系 (On/In x y) 中 x 是被搬运物体，y 是目标容器/区域。
    """
    manipulated = set()
    receptacles = set()
    for m in re.finditer(r'\((On|In)\s+([^\s()]+)\s+([^\s()]+)\)', goal_text):
        manipulated.add(m.group(2))
        receptacles.add(m.group(3))
    return manipulated, receptacles


def build_role_manifest(bddl_path):
    """从 bddl 文件生成角色清单，返回 dict。"""
    bddl_path = pathlib.Path(bddl_path)
    bddl_text = bddl_path.read_text()
    bddl_sha = hashlib.sha256(bddl_text.encode()).hexdigest()

    objects = _parse_objects(bddl_text)
    goal_text = _parse_goal(bddl_text)
    manipulated, receptacles = _parse_goal_roles(goal_text)

    manifest = {"objects": {}, "bddl_path": str(bddl_path), "bddl_sha256": bddl_sha,
                "goal_predicate": goal_text.strip()}

    for inst, cls in objects.items():
        if "obstacle" in inst or "obstacle" in cls:
            role = "STATIC_SAFETY_OBSTACLE"
        elif inst in manipulated:
            role = "MANIPULATED_TARGET"
        elif inst in receptacles:
            role = "GOAL_RECEPTACLE"
        else:
            role = "DISTRACTOR"

        # obj_of_interest：BDDL 作者显式标注的任务相关物体（物体名，非 region 名）
        ooi_m = re.search(r'\(:obj_of_interest\s*(.*?)\)\s*\(', bddl_text, re.S)
        ooi_names = ooi_m.group(1).split() if ooi_m else []
        in_ooi = inst in ooi_names

        # receptacle 可能是 region 名（如 basket_1_contain_region），
        # 用 obj_of_interest 里对应的物体名补正 GOAL_RECEPTACLE 角色。
        # 规则：在 ooi 里、且不是 manipulated、且某个 receptacle region 名以它开头 → GOAL_RECEPTACLE
        is_receptacle_by_ooi = False
        if in_ooi and inst not in manipulated:
            for rec in receptacles:
                if rec.startswith(inst):  # basket_1_contain_region startswith basket_1
                    is_receptacle_by_ooi = True
                    break
        if is_receptacle_by_ooi and role == "DISTRACTOR":
            role = "GOAL_RECEPTACLE"

        manifest["objects"][inst] = {
            "object_class": cls,
            "role": role,
            "role_evidence": {
                "in_goal_manipulated": inst in manipulated,
                "in_goal_receptacle": inst in receptacles,
                "in_obj_of_interest": in_ooi,
                "receptacle_by_ooi": is_receptacle_by_ooi,
                "is_obstacle": "obstacle" in inst or "obstacle" in cls,
            },
        }
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
    # 示例
    for suite in list(manifests)[:1]:
        for task in list(manifests[suite])[:1]:
            m = manifests[suite][task]
            print(f"\n示例 {suite}/{task}:")
            for inst, info in m["objects"].items():
                print(f"  {inst:30s} {info['role']}")
