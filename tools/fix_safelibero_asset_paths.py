#!/usr/bin/env python
"""把 SafeLIBERO 资产 XML 里写死的作者机器绝对路径改成本地相对路径。

背景：SafeLIBERO 官方发布的 assets 里，部分 XML 的 file= 属性写的是作者机器上的绝对路径
（/mnt/workspace/openpi/third_party/... 、/home/yifengz/...），换机器后 MuJoCo 直接报
`resource not found via provider or OS filesystem`。robosuite 的 MujocoXML 加载时会把
file= 相对路径按 XML 所在目录解析成绝对路径（robosuite/models/base.py::resolve_asset_dependency），
所以只要把写死的绝对路径换回「相对该 XML 所在目录」的相对路径即可。

用法（幂等，可重复跑）：
    source tools/activate_env.sh
    python tools/fix_safelibero_asset_paths.py --dry-run   # 只看要改什么
    python tools/fix_safelibero_asset_paths.py             # 真改

注意：dataset/ 不入 git，重新下载数据集后需重跑本脚本。
"""

import argparse
import os
import re
import sys

# 已知的外来绝对路径前缀 -> 换算到本地 assets 下的相对位置
FOREIGN_PREFIXES = {
    "/mnt/workspace/openpi/third_party/safelibero/libero/libero/assets/": "",
    "/mnt/workspace/openpi/third_party/libero/libero/libero/assets/": "",
    "/home/yifengz/workspace/robosuite-master/robosuite/models/assets/": "",
}

FILE_ATTR = re.compile(r'(\bfile=")([^"]+)(")')


def default_assets_root():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(
        root, "dataset", "SafeLIBERO", "safelibero", "libero", "libero", "assets"
    )


def local_from_foreign(path, assets_root):
    """外来绝对路径 -> 本机绝对路径；认不出前缀时返回 None。"""
    for prefix, sub in FOREIGN_PREFIXES.items():
        if path.startswith(prefix):
            return os.path.normpath(os.path.join(assets_root, sub, path[len(prefix) :]))
    return None


def process_file(xml_path, assets_root, dry_run):
    with open(xml_path, "r", encoding="utf-8") as f:
        text = f.read()

    xml_dir = os.path.dirname(xml_path)
    changes = []

    def repl(match):
        raw = match.group(2)
        target = None
        if os.path.isabs(raw):
            target = local_from_foreign(raw, assets_root)
            if target is None:
                changes.append((raw, "SKIP(未知前缀)"))
                return match.group(0)
            if not os.path.exists(target):
                changes.append((raw, "SKIP(本地无此文件: %s)" % target))
                return match.group(0)
        else:
            # 已是相对路径，检查是否真能解析到；能就跳过
            if os.path.exists(os.path.join(xml_dir, raw)):
                return match.group(0)
            target = local_from_foreign(os.path.abspath(os.path.join(xml_dir, raw)), assets_root)
            if target is None:
                return match.group(0)

        new = os.path.relpath(target, xml_dir)
        changes.append((raw, new))
        return match.group(1) + new + match.group(3)

    new_text = FILE_ATTR.sub(repl, text)
    if new_text != text and not dry_run:
        with open(xml_path, "w", encoding="utf-8") as f:
            f.write(new_text)
    return changes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets-root", default=default_assets_root())
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not os.path.isdir(args.assets_root):
        sys.exit("[fix] assets 目录不存在: %s" % args.assets_root)

    touched = 0
    for dirpath, _dirnames, filenames in os.walk(args.assets_root):
        for name in filenames:
            if not name.endswith(".xml"):
                continue
            xml_path = os.path.join(dirpath, name)
            changes = process_file(xml_path, args.assets_root, args.dry_run)
            changes = [c for c in changes if c[0] != c[1]]
            if not changes:
                continue
            touched += 1
            rel = os.path.relpath(xml_path, args.assets_root)
            print("[%s] %s" % ("dry-run" if args.dry_run else "fixed", rel))
            for old, new in changes:
                print("    %s\n -> %s" % (old, new))

    print("\n[fix] 涉及文件数: %d%s" % (touched, "（dry-run，未写入）" if args.dry_run else ""))


if __name__ == "__main__":
    main()
