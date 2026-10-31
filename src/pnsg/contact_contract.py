"""Contact classification contract（v2）：语义分类与允许规则分离。

对应 docs/想法v7.md。核心设计：
1. 语义分类（semantic_class）与允许状态（allowance）分离 —— 先纯分类，再赋允许
2. 角色顺序无关（对称性）：{A,B} 与 {B,A} 产生相同分类
3. 底层保存通用 role_pair，上层映射到机制类别
4. 12+ 语义类别，覆盖所有 relevant role pair
5. allowance 规则表（FORBIDDEN / ENVIRONMENT_DISTURBANCE / ...）

角色集合：
- ROBOT
- MANIPULATED_TARGET（payload）
- GOAL_RECEPTACLE
- DISTRACTOR
- STATIC_SAFETY_OBSTACLE
- ENVIRONMENT（fixture：table/floor/cabinet/stove/mount 等）

注意：只用于机制审计，不修改官方 CAR evaluator。
"""

from __future__ import annotations

import json

CONTRACT_VERSION = "collision_event_contract_v2"

# 角色集合
ROLES = [
    "ROBOT",
    "MANIPULATED_TARGET",
    "GOAL_RECEPTACLE",
    "DISTRACTOR",
    "STATIC_SAFETY_OBSTACLE",
    "ENVIRONMENT",
]

# 角色对（frozenset，天然对称）→ 语义类别
ROLE_PAIR_TO_SEMANTIC = {
    # 核心机制（CAR 来源分解）
    frozenset({"ROBOT", "STATIC_SAFETY_OBSTACLE"}): "ROBOT_STATIC_OBSTACLE",
    frozenset({"MANIPULATED_TARGET", "STATIC_SAFETY_OBSTACLE"}): "PAYLOAD_STATIC_OBSTACLE",
    frozenset({"ROBOT", "DISTRACTOR"}): "ROBOT_DISTRACTOR",
    frozenset({"DISTRACTOR", "STATIC_SAFETY_OBSTACLE"}): "DISTRACTOR_STATIC_OBSTACLE",
    # 任务相关（保留记录，不丢弃）
    frozenset({"ROBOT", "MANIPULATED_TARGET"}): "ROBOT_MANIPULATED_TARGET",
    frozenset({"MANIPULATED_TARGET", "GOAL_RECEPTACLE"}): "MANIPULATED_TARGET_GOAL",
    frozenset({"ROBOT", "GOAL_RECEPTACLE"}): "ROBOT_GOAL",
    # 次要机制
    frozenset({"MANIPULATED_TARGET", "DISTRACTOR"}): "PAYLOAD_DISTRACTOR",
    frozenset({"DISTRACTOR", "GOAL_RECEPTACLE"}): "DISTRACTOR_GOAL",
    frozenset({"GOAL_RECEPTACLE", "STATIC_SAFETY_OBSTACLE"}): "RECEPTACLE_STATIC_OBSTACLE",
    # 环境相关
    frozenset({"ROBOT", "ENVIRONMENT"}): "ROBOT_ENVIRONMENT",
    frozenset({"MANIPULATED_TARGET", "ENVIRONMENT"}): "PAYLOAD_ENVIRONMENT",
    frozenset({"GOAL_RECEPTACLE", "ENVIRONMENT"}): "GOAL_ENVIRONMENT",
    frozenset({"DISTRACTOR", "ENVIRONMENT"}): "DISTRACTOR_ENVIRONMENT",
    frozenset({"STATIC_SAFETY_OBSTACLE", "ENVIRONMENT"}): "OBSTACLE_ENVIRONMENT",
    # 同角色
    frozenset({"ROBOT"}): "ROBOT_SELF",
    frozenset({"MANIPULATED_TARGET"}): "PAYLOAD_SELF",
    frozenset({"DISTRACTOR"}): "DISTRACTOR_SELF",
    frozenset({"STATIC_SAFETY_OBSTACLE"}): "OBSTACLE_SELF",
    frozenset({"GOAL_RECEPTACLE"}): "GOAL_SELF",
    frozenset({"ENVIRONMENT"}): "ENVIRONMENT_SELF",
}

# 语义类别 → 默认 allowance
SEMANTIC_TO_ALLOWANCE = {
    "ROBOT_STATIC_OBSTACLE": "FORBIDDEN",
    "PAYLOAD_STATIC_OBSTACLE": "FORBIDDEN",
    "ROBOT_DISTRACTOR": "ENVIRONMENT_DISTURBANCE",
    "DISTRACTOR_STATIC_OBSTACLE": "SECONDARY_DISTURBANCE",
    "ROBOT_MANIPULATED_TARGET": "TASK_CONTACT",
    "MANIPULATED_TARGET_GOAL": "ALLOWED_BY_TASK_STAGE",
    "ROBOT_GOAL": "TASK_DEPENDENT",
    "PAYLOAD_DISTRACTOR": "ENVIRONMENT_DISTURBANCE",
    "DISTRACTOR_GOAL": "TASK_DEPENDENT",
    "RECEPTACLE_STATIC_OBSTACLE": "SECONDARY_DISTURBANCE",
    "ROBOT_ENVIRONMENT": "ALLOWED",
    "PAYLOAD_ENVIRONMENT": "ALLOWED",
    "GOAL_ENVIRONMENT": "ALLOWED",
    "DISTRACTOR_ENVIRONMENT": "ALLOWED",
    "OBSTACLE_ENVIRONMENT": "ALLOWED",
    "ROBOT_SELF": "FORBIDDEN",
    "PAYLOAD_SELF": "UNKNOWN",
    "DISTRACTOR_SELF": "UNKNOWN",
    "OBSTACLE_SELF": "UNKNOWN",
    "OTHER_UNCLASSIFIED": "UNKNOWN",
}

# 前四类是当前 CAR 来源分解的核心机制
CORE_MECHANISM_CLASSES = [
    "ROBOT_STATIC_OBSTACLE",
    "PAYLOAD_STATIC_OBSTACLE",
    "ROBOT_DISTRACTOR",
    "DISTRACTOR_STATIC_OBSTACLE",
]


def classify_role_pair(r1, r2):
    """对称分类：{r1,r2} 与 {r2,r1} 产生相同语义类别（frozenset 保证）。"""
    return ROLE_PAIR_TO_SEMANTIC.get(frozenset({r1, r2}), "OTHER_UNCLASSIFIED")


def allowance_for(semantic_class):
    """语义类别 → 默认 allowance。"""
    return SEMANTIC_TO_ALLOWANCE.get(semantic_class, "UNKNOWN")


def is_core_mechanism(semantic_class):
    """是否属于 CAR 来源分解的核心机制（前四类）。"""
    return semantic_class in CORE_MECHANISM_CLASSES


def contract_dict():
    """返回版本化的 contract dict（用于写 json 和 hash）。"""
    # frozenset 不能 json 序列化，转成 "role1||role2" 字符串
    pair_map = {}
    for fs, cls in sorted(ROLE_PAIR_TO_SEMANTIC.items(), key=lambda x: sorted(x[0])):
        key = "||".join(sorted(fs))
        pair_map[key] = cls
    return {
        "contract_version": CONTRACT_VERSION,
        "roles": ROLES,
        "role_pair_to_semantic": pair_map,
        "semantic_to_allowance": dict(sorted(SEMANTIC_TO_ALLOWANCE.items())),
        "core_mechanism_classes": CORE_MECHANISM_CLASSES,
        "contact_levels": {
            "native_contact_record": "contact 存在（含 margin 内 proximity）",
            "geometric_contact": "contact.dist <= 0（主机制审计用）",
            "proximity_only": "contact.dist > 0",
        },
        "contact_record_fields": [
            "episode_id", "control_step", "physics_substep",
            "geom1", "geom2", "body1", "body2", "instance1", "instance2",
            "role1", "role2", "semantic_class", "contact_dist",
            "contact_level", "preexisting_at_snapshot", "task_stage",
            "allowance", "role_manifest_sha256", "contract_version",
        ],
        "statistical_units": [
            "raw_contact_record_count",
            "contact_positive_substep_count",
            "contact_positive_control_step_count",
            "contiguous_contact_event_count",
            "contact_positive_episode_count",
        ],
    }


def dump_contract(path):
    import pathlib
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(contract_dict(), f, ensure_ascii=False, indent=2)
    return path
