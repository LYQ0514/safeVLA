"""独立 goal evaluator：从 BDDL parsed_problem goal_state 求值谓词。

规范 v1.2 §5 要求「每个 BDDL goal clause 由独立 predicate evaluator 判真」，
不能只用 env.check_success()。SafeLIBERO 的 _eval_predicate 就是独立求值器
（不经过 check_success 的 done 包装），本模块封装它，并返回逐 clause 的 truth。

输出 canonical_clause -> bool（不可求值的 clause 返回 None = unknown）。
"""

from __future__ import annotations

from typing import Dict, Optional


def normalize_predicate_name(pred) -> str:
    """谓词名统一小写（manifest 用 'In'，_eval_predicate 用 'in'）。"""
    return str(pred).lower()


def canonical_clause(state) -> str:
    """谓词状态 → canonical clause 字符串（统一小写 + 空格连接，如 "in a b"）。

    兼容两种输入：
    - goal_state 里的 list：['in', 'a', 'b']；
    - role_manifest 的 tuple：('In', ['a', 'b'])。
    """
    if isinstance(state, (tuple, list)):
        items = list(state)
        # ('In', ['a','b']) 形式：第二元素是 list
        if len(items) >= 2 and isinstance(items[1], (list, tuple)):
            flat = [items[0]] + list(items[1])
        else:
            flat = items
        pred = normalize_predicate_name(flat[0])
        args = [str(x) for x in flat[1:]]
        return " ".join([pred] + args)
    return str(state)


def goal_state_clauses(env) -> list:
    """从 env 提取 goal_state（谓词列表）。"""
    e = env.env
    return list(e.parsed_problem.get("goal_state", []))


def eval_goal_clauses(env) -> Dict[str, Optional[bool]]:
    """独立求值所有 goal clause，返回 {canonical_clause: bool|None}。"""
    e = env.env
    result = {}
    for state in goal_state_clauses(env):
        key = canonical_clause(state)
        try:
            result[key] = bool(e._eval_predicate(state))
        except Exception:
            result[key] = None  # 不可求值 → unknown
    return result


def native_success(env) -> bool:
    """env.check_success()（native 口径，用于对照）。"""
    return bool(env.check_success())
