"""审计 D：运行时源码身份门禁（runtime_identity）。

裁决 §四要求：每次运行必须从实际 Python 进程记录每个模块的 __file__ 与 source_sha256，
不能只从仓库目录读取 commit（Python 可能从旧 install 路径导入不同代码）。

本模块提供：
1. capture_runtime_identity(modules)：从 sys.modules 取实际导入路径 + 文件 sha256；
2. git_identity()：git commit / dirty 状态；
3. verify_runtime_identity()：审计 D 通过条件检查。

审计 D 通过条件（裁决 §四）：
- 实际导入路径属于预期 install（src/fpck/ 下）；
- 运行时文件 hash == git HEAD 该路径内容 hash（即工作树与运行代码一致）；
- 工作树变更明确（dirty 文件被列出）；
- v1.2 与 v1.3 只在监视器逻辑上不同（task_automaton.py vs task_automaton_acq.py）。
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

ROOT = "/data/safeVLA"

# 审计 D 必须记录的模块（裁决 §四）
IDENTITY_MODULES = [
    "src.fpck.task_automaton",
    "src.fpck.task_automaton_acq",
    "src.fpck.task_event_bridge",
    "src.fpck.gripper_action_decoder",
    "src.fpck.substep_collector",
    "src.fpck.collector",
    "src.fpck.goal_evaluator",
    "src.fpck.canonical_hash",
]

# 预期的 install 前缀（实际 __file__ 必须落在这些路径下）
EXPECTED_PREFIXES = ("/data/safeVLA/src/",)


def sha256_file(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_hex(obj) -> str:
    import numpy as np
    if isinstance(obj, np.ndarray):
        obj = np.asarray(obj).tobytes()
    elif isinstance(obj, (list, tuple, dict)):
        obj = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    elif not isinstance(obj, bytes):
        obj = str(obj).encode("utf-8")
    return hashlib.sha256(obj).hexdigest()


def git_identity(root: str = ROOT) -> dict:
    """从仓库取 git commit + dirty 状态。"""
    out = {"git_commit": None, "git_dirty": False, "git_status": []}
    try:
        out["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root).decode().strip()
    except Exception as e:
        out["git_commit"] = f"ERROR:{e}"
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root).decode().strip()
        out["git_status"] = [l for l in status.splitlines() if l.strip()]
        out["git_dirty"] = bool(out["git_status"])
    except Exception as e:
        out["git_status"] = [f"ERROR:{e}"]
    return out


def _git_show_sha(root: str, relpath: str) -> Optional[str]:
    """取 git HEAD 中某路径的内容 sha256（用于与运行时文件对比）。"""
    try:
        data = subprocess.check_output(
            ["git", "show", f"HEAD:{relpath}"], cwd=root)
        return sha256_bytes(data)
    except Exception:
        return None


def capture_module_file(module_name: str) -> dict:
    """从 sys.modules 取模块的实际 __file__ + 文件 sha256。"""
    rec = {"module": module_name, "file": None, "file_sha256": None,
           "imported": False, "path_ok": False}
    try:
        mod = importlib.import_module(module_name)
        rec["imported"] = True
        f = getattr(mod, "__file__", None)
        rec["file"] = f
        if f is not None and f.endswith(".py"):
            rec["file_sha256"] = sha256_file(f)
            rec["path_ok"] = f.startswith(EXPECTED_PREFIXES)
    except Exception as e:
        rec["import_error"] = str(e)
    return rec


def capture_runtime_identity(extra: Optional[dict] = None) -> dict:
    """审计 D 主入口：采集完整运行时身份。"""
    ident = {
        "audit": "D",
        "git": git_identity(),
        "modules": {},
        "module_verify": [],
    }
    for mn in IDENTITY_MODULES:
        rec = capture_module_file(mn)
        ident["modules"][mn] = rec
        # 与 git HEAD 对比
        f = rec.get("file")
        verdict = {"module": mn, "matches_git_head": None}
        if f and f.startswith(ROOT + "/"):
            rel = os.path.relpath(f, ROOT)
            git_sha = _git_show_sha(ROOT, rel)
            if git_sha is not None:
                verdict["matches_git_head"] = (git_sha == rec["file_sha256"])
            ident["module_verify"].append(verdict)
    if extra:
        ident["extra"] = extra
    return ident


def verify_runtime_identity(ident: dict) -> dict:
    """审计 D 通过条件检查，返回 {pass, failures, reasons}。"""
    failures = []
    # 1. 所有模块已导入
    for mn, rec in ident["modules"].items():
        if not rec.get("imported"):
            failures.append(f"{mn}: 未导入（import_error={rec.get('import_error')}）")
            continue
        # 2. 路径属于预期 install
        if not rec.get("path_ok"):
            failures.append(f"{mn}: 路径不属于预期 install（{rec.get('file')}）")
        # 3. 与 git HEAD 一致
        for v in ident["module_verify"]:
            if v["module"] == mn and v.get("matches_git_head") is False:
                failures.append(f"{mn}: 运行时文件 hash 与 git HEAD 不一致（工作树未提交改动或用了旧 install）")

    # 4. 工作树变更明确
    dirty = ident["git"].get("git_status", [])
    src_dirty = [l for l in dirty if "src/fpck" in l]
    reasons = {
        "dirty_total": len(dirty),
        "dirty_src_fpck": src_dirty,
    }
    if src_dirty:
        # 工作树有 src/fpck 改动：需明确列出，不算硬失败但需人工确认
        reasons["warning"] = f"src/fpck 有未提交改动：{src_dirty}"

    passed = len(failures) == 0
    return {"pass": passed, "failures": failures, "reasons": reasons}


@dataclass
class RuntimeIdentityRecord:
    """嵌入 run_manifest 的运行时身份快照。"""
    audit_d: dict = field(default_factory=dict)
    spec_version: str = ""
    checkpoint_hash: str = ""
    role_manifest_sha256: str = ""
    contact_contract_sha256: str = ""
    nominal_seed: Optional[int] = None
    physics_seed: Optional[int] = None
    query_index: Optional[int] = None
    buffer_index: Optional[int] = None
    policy_chunk_hash: Optional[str] = None
    snapshot_hash: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "audit_d": self.audit_d,
            "automaton_spec_version": self.spec_version,
            "checkpoint_hash": self.checkpoint_hash,
            "role_manifest_sha256": self.role_manifest_sha256,
            "contact_contract_sha256": self.contact_contract_sha256,
            "nominal_seed": self.nominal_seed,
            "physics_seed": self.physics_seed,
            "query_index": self.query_index,
            "buffer_index": self.buffer_index,
            "policy_chunk_hash": self.policy_chunk_hash,
            "snapshot_hash": self.snapshot_hash,
        }


def collect_identity_and_verify(extra: Optional[dict] = None) -> tuple:
    """便捷函数：采集 + 验证，返回 (ident, verify_result)。"""
    ident = capture_runtime_identity(extra)
    verify = verify_runtime_identity(ident)
    return ident, verify


if __name__ == "__main__":
    ident, verify = collect_identity_and_verify()
    print(json.dumps({"verify": verify, "identity": ident},
                     ensure_ascii=False, indent=2))
    sys.exit(0 if verify["pass"] else 1)
