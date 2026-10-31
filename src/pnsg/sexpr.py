"""S-expression 解析器（BDDL 用）。

把 (And (On a b) (In c d_region)) 解析为嵌套 dict：
{"predicate": "And", "children": [
    {"predicate": "On", "arguments": ["a", "b"]},
    {"predicate": "In", "arguments": ["c", "d_region"]},
]}
"""

from __future__ import annotations


def tokenize(text):
    """把 s-expression 文本切成 token（去掉注释）。"""
    tokens = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == ';':  # 注释，跳过到行尾
            while i < n and text[i] != '\n':
                i += 1
        elif ch in '()':
            tokens.append(ch)
            i += 1
        elif ch.isspace():
            i += 1
        else:
            start = i
            while i < n and not text[i].isspace() and text[i] not in '();':
                i += 1
            tokens.append(text[start:i])
    return tokens


def parse(tokens):
    """从 token 列表解析 s-expression，返回嵌套 dict 或字符串。"""
    if not tokens:
        return None
    tok = tokens.pop(0)
    if tok != '(':
        return tok  # 原子（字符串）
    # 解析一个列表
    if not tokens:
        return {"predicate": "", "arguments": []}
    pred = tokens.pop(0)
    args = []
    while tokens and tokens[0] != ')':
        args.append(parse(tokens))
    if tokens:
        tokens.pop(0)  # 消费 ')'
    return {"predicate": pred, "arguments": args}


def parse_text(text):
    """解析 s-expression 文本，返回 AST。"""
    tokens = tokenize(text)
    return parse(tokens)


def find_blocks(text):
    """从 BDDL 文本提取所有顶层 (:key ...) 块，返回 {key: block_text}。

    用括号深度匹配，正确处理嵌套。
    """
    blocks = {}
    i = 0
    n = len(text)
    while i < n:
        # 找下一个 '('
        while i < n and text[i] != '(':
            i += 1
        if i >= n:
            break
        # 检查是否是 "(:key"（'(' 后紧跟 ':'）
        if i + 1 < n and text[i + 1] != ':':
            i += 1
            continue
        start = i
        # 读 key（':' 到空白/括号）
        j = i + 2  # 跳过 "(:"
        key_start = j
        while j < n and not text[j].isspace() and text[j] not in '()':
            j += 1
        key = ':' + text[key_start:j]
        # 深度匹配到闭合括号
        depth = 0
        k = start
        while k < n:
            if text[k] == '(':
                depth += 1
            elif text[k] == ')':
                depth -= 1
                if depth == 0:
                    break
            k += 1
        blocks[key] = text[start:k + 1]
        i = k + 1
    return blocks


def parse_declarations(block_text):
    """解析 (:objects ...) 或 (:fixtures ...) 块，返回 {instance: class}。

    形如 "inst1 inst2 - class_name" 多行。
    """
    # 去掉首尾括号和 key
    body = block_text[block_text.find('(') + 1:]
    # 去掉 key
    body = body[body.find(' ') + 1:]
    # 去掉最外层括号
    body = body.strip()
    if body.endswith(')'):
        body = body[:-1]
    result = {}
    for line in body.splitlines():
        line = line.split(';')[0].strip()
        if not line or '-' not in line:
            continue
        parts = line.split(' - ')
        if len(parts) < 2:
            continue
        instances = parts[0].split()
        cls = parts[-1].strip()
        for inst in instances:
            result[inst] = cls
    return result


def parse_regions(block_text):
    """解析 (:regions ...) 块，返回 [{name, target}]。

    用正则匹配 (name (:target obj) 结构。name 不限后缀（_region/_side/
    table_center 等都算 region），只要紧跟 (:target 即可。
    """
    import re
    regions = []
    # region 形如 (name  (:target parent) ...
    pattern = re.compile(r'\(\s*([a-zA-Z_][\w]*)\s*\n?\s*\(:target\s+([\w]+)\)', re.S)
    for m in pattern.finditer(block_text):
        regions.append({"name": m.group(1), "target": m.group(2)})
    return regions
