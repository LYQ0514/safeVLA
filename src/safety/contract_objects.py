"""安全契约对象清单 + 接触监视器（M1-Contract-Audit）。

替代 `build_body_geoms` 的字符串推断，建立显式场景角色分类：

- ROBOT：robot0_* / gripper0_* 碰撞 geom
- PROTECTED：CAR 直接评估的障碍物（*_obstacle_*_main）
- PROTECTED_SUPPORT：受保护物体的支撑物（如 mug 底座 box_small_base_1_main），
  禁止 robot 接触——它是「通过物理接触影响受保护物体」的传递通道
- TASK：合法任务物体（soup/cream/basket），按任务阶段允许必要抓取接触
- DISTRACTOR：场景摆设（tomato_sauce/ketchup），FREE 但任务不操作，默认也禁止
  robot 接触（保守），可单独审计
- FIXED_ENV：固定环境（floor/table/table_col），正常支撑，允许接触

原则：
1. 显式 body 名清单，不靠 "obstacle" 子串推断全部保护关系。
2. 独立接触监视器扫描原生 mjData.contact 的全部 robot↔environment 接触对，
   报告未被角色分类覆盖的事件。
3. Native CAR 仍只看受保护物体位移（sim.data 真值）；底座保护是更严格约束，
   不修改 benchmark 的 CAR 定义。
"""

from __future__ import annotations

import warnings

import numpy as np

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 显式角色清单（第一版，基于 m1_contract_scene_inventory.json 的事实枚举）
# ---------------------------------------------------------------------------
PROTECTED_OBJECTS = [
    "red_coffee_mug_obstacle_1_main",
    "moka_pot_small_obstacle_1_main",
    "white_storage_box_obstacle_1_main",
    "milk_small_obstacle_1_main",
    "wine_bottle_small_obstacle_1_main",
    "yellow_book_obstacle_1_main",
]

# 受保护支撑物：不直接是 CAR 对象，但通过接触传递影响受保护物体
PROTECTED_SUPPORTS = [
    "box_small_base_1_main",  # mug 底座
]

TASK_OBJECTS = [
    "alphabet_soup_1_main",
    "cream_cheese_1_main",
    "basket_1_main",
]

# 场景摆设（FREE 但任务不操作）。保守默认禁止 robot 接触，可单独审计。
DISTRACTOR_OBJECTS = [
    "tomato_sauce_1_main",
    "ketchup_1_main",
]

# 固定环境（正常支撑/桌面，允许接触）
FIXED_ENV_PREFIXES = ("floor", "living_room_table", "mount0_base", "robot0_base")


def _geom_body_names(env):
    """geom_id -> body_name 映射（几何查询里 geom_names 可能截断，用 body 名判断）。"""
    m = env.sim.model._model
    names = env.sim.model.body_names
    return {g: names[m.geom_bodyid[g]] for g in range(m.ngeom)}


def _collision_geoms(env):
    """所有 contype>0 的碰撞 geom id 集合。"""
    m = env.sim.model._model
    return {g for g in range(m.ngeom) if m.geom_contype[g] > 0}


def robot_geoms(env):
    """机器人碰撞 geom（robot0_* / gripper0_*）。"""
    g2b = _geom_body_names(env)
    return {g for g in _collision_geoms(env)
            if g2b[g].startswith("robot0") or g2b[g].startswith("gripper0")}


def protected_geoms(env):
    """受保护物体（含支撑物）的碰撞 geom。"""
    g2b = _geom_body_names(env)
    targets = set(PROTECTED_OBJECTS) | set(PROTECTED_SUPPORTS)
    return {g for g in _collision_geoms(env) if g2b[g] in targets}


def forbidden_contact_geoms(env):
    """禁止 robot 接触的几何集合 = 受保护物体 + 支撑物 + 摆设（保守）。"""
    g2b = _geom_body_names(env)
    targets = set(PROTECTED_OBJECTS) | set(PROTECTED_SUPPORTS) | set(DISTRACTOR_OBJECTS)
    return {g for g in _collision_geoms(env) if g2b[g] in targets}


def task_geoms(env):
    """合法任务物体碰撞 geom。"""
    g2b = _geom_body_names(env)
    return {g for g in _collision_geoms(env) if g2b[g] in set(TASK_OBJECTS)}


def is_fixed_env(body_name):
    return any(body_name.startswith(p) for p in FIXED_ENV_PREFIXES)


def classify_body(body_name):
    """body 名 -> 角色。返回 (role, 说明)。"""
    if body_name in PROTECTED_OBJECTS:
        return "PROTECTED", "CAR 直接评估对象"
    if body_name in PROTECTED_SUPPORTS:
        return "PROTECTED_SUPPORT", "受保护物体支撑物"
    if body_name in TASK_OBJECTS:
        return "TASK", "合法任务物体"
    if body_name in DISTRACTOR_OBJECTS:
        return "DISTRACTOR", "场景摆设"
    if is_fixed_env(body_name):
        return "FIXED_ENV", "固定环境"
    if body_name.startswith("robot0") or body_name.startswith("gripper0"):
        return "ROBOT", "机器人本体"
    return "UNCLASSIFIED", "未分类"


def monitor_contacts(env):
    """独立接触监视器：扫描原生 mjData.contact 的全部 robot↔environment 接触。

    返回 {
      "forbidden": [...],      # robot ↔ protected/support/distractor
      "task": [...],           # robot ↔ task object
      "fixed_env": [...],      # robot ↔ floor/table（正常支撑）
      "unclassified": [...],   # robot ↔ 未分类 body（契约覆盖缺口）
      "robot_robot": [...],    # robot 自碰撞
      "env_env": [...],        # 环境物体间接触（含 mug↔base）
    }
    """
    m = env.sim.model._model
    d = env.sim.data._data
    names = env.sim.model.body_names
    rg = robot_geoms(env)

    out = {"forbidden": [], "task": [], "fixed_env": [], "unclassified": [],
           "robot_robot": [], "env_env": []}

    for c in range(d.ncon):
        g1, g2 = d.contact[c].geom1, d.contact[c].geom2
        b1 = names[m.geom_bodyid[g1]]
        b2 = names[m.geom_bodyid[g2]]
        is_r1 = g1 in rg
        is_r2 = g2 in rg
        dist = float(d.contact[c].dist)
        efc = int(d.contact[c].efc_address)
        entry = {"body1": b1, "body2": b2, "geom1": g1, "geom2": g2,
                 "dist": dist, "efc_address": efc}

        if is_r1 and is_r2:
            out["robot_robot"].append(entry)
        elif is_r1 or is_r2:
            # robot ↔ environment
            env_body = b1 if is_r2 else b2
            role, _ = classify_body(env_body)
            if role in ("PROTECTED", "PROTECTED_SUPPORT", "DISTRACTOR"):
                out["forbidden"].append(entry)
            elif role == "TASK":
                out["task"].append(entry)
            elif role == "FIXED_ENV":
                out["fixed_env"].append(entry)
            else:
                out["unclassified"].append(entry)
        else:
            out["env_env"].append(entry)
    return out


def has_forbidden_contact(env):
    """是否存在 robot↔受保护物体/支撑物/摆设 的接触（含未分类，保守）。"""
    m = monitor_contacts(env)
    return len(m["forbidden"]) > 0 or len(m["unclassified"]) > 0
