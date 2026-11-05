"""E1a 采集器：透明包装 frozen π0.5，记录完整反事实状态。

设计原则（裁决要求）：
- 透明：采集器只记录，不改变动作、不增加推理、不推进随机流、不额外 env.step；
- append-only：原始事件日志只追加，汇总标签另写文件；
- 完整状态：每 anchor 记录 full-state hash / chunk hash / cursor / prompt hash /
  manifest hash / contract hash / automaton spec version 等身份；
- snapshot round-trip：save(env full state + policy query cursor) → 恢复 → 复现同一动作。

确定性机制：用 PnsgPolicy（CRN）+ nominal_seed + sample_z，query_idx 显式决定 chunk，
恢复时只需恢复 query_idx，不依赖内部 _chunk 缓存。
"""

from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np

from src.pnsg.crn import nominal_seed, sample_z
from src.pnsg.full_state import save_full_state, restore_full_state


# ---------------------------------------------------------------------------
# 身份 hash
# ---------------------------------------------------------------------------
def sha256_hex(obj) -> str:
    if isinstance(obj, np.ndarray):
        obj = np.asarray(obj).tobytes()
    elif isinstance(obj, (list, tuple, dict)):
        obj = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    elif not isinstance(obj, bytes):
        obj = str(obj).encode("utf-8")
    return hashlib.sha256(obj).hexdigest()


class AnchorIdentity:
    """E1a anchor 身份（含完整状态 + 策略 cursor + 契约 hash）。"""

    def __init__(self, repo_commit, checkpoint_hash, suite, level, task_id,
                 state_id, stream, prompt, manifest_hash, contract_hash,
                 spec_version, replan_steps):
        self.repo_commit = repo_commit
        self.checkpoint_hash = checkpoint_hash
        self.suite = suite
        self.level = level
        self.task_id = task_id
        self.state_id = state_id
        self.stream = stream
        self.prompt_hash = sha256_hex(prompt)
        self.manifest_hash = manifest_hash
        self.contract_hash = contract_hash
        self.spec_version = spec_version
        self.replan_steps = replan_steps
        self.full_state_hash = None       # 由 save_snapshot 填充
        self.chunk_hash = None
        self.buffer_index = 0

    def as_dict(self):
        return {
            "repo_commit": self.repo_commit,
            "checkpoint_hash": self.checkpoint_hash,
            "suite": self.suite, "level": self.level,
            "task_id": self.task_id, "state_id": self.state_id,
            "stream": self.stream, "prompt_hash": self.prompt_hash,
            "manifest_hash": self.manifest_hash, "contract_hash": self.contract_hash,
            "spec_version": self.spec_version, "replan_steps": self.replan_steps,
            "full_state_hash": self.full_state_hash,
            "chunk_hash": self.chunk_hash, "buffer_index": self.buffer_index,
        }


class PolicyCursor:
    """π0.5 确定性策略的 cursor：query_idx 决定下一个 chunk（CRN 无状态）。"""

    def __init__(self, query_idx=0, buffer_index=0):
        self.query_idx = query_idx
        self.buffer_index = buffer_index

    def snapshot(self):
        return {"query_idx": self.query_idx, "buffer_index": self.buffer_index}

    def restore(self, snap):
        self.query_idx = snap["query_idx"]
        self.buffer_index = snap["buffer_index"]


class Snapshot:
    """完整快照：MuJoCo full state + policy cursor + chunk payload。"""

    def __init__(self, env_snapshot, cursor_snapshot, chunk):
        self.env_snapshot = env_snapshot
        self.cursor_snapshot = cursor_snapshot
        self.chunk = chunk
        self.chunk_hash = sha256_hex(np.asarray(chunk).tobytes())

    def restore_env(self, env):
        restore_full_state(env, self.env_snapshot)


class Collector:
    """透明采集器：包装 PnsgPolicy，记录每步事件，不改变任何动作。"""

    def __init__(self, policy, anchor: AnchorIdentity, suite, task_id, state_id,
                 stream, image_size, replan_steps, log_path):
        self.policy = policy
        self.anchor = anchor
        self.suite = suite
        self.task_id = task_id
        self.state_id = state_id
        self.stream = stream
        self.image_size = image_size
        self.replan_steps = replan_steps
        self.cursor = PolicyCursor()
        self.chunk = None
        self.log_path = log_path
        self.step = 0
        self._events = []  # append-only，逐条 flush

    def _flush(self, row):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def reset(self):
        self.cursor = PolicyCursor()
        self.chunk = None
        self.step = 0

    def generate_chunk(self, obs, prompt):
        """按 CRN 确定性生成 chunk（query_idx 由 cursor 决定）。"""
        import pi05_libero_runner as R
        q = self.cursor.query_idx
        z0, _ = sample_z(nominal_seed(self.suite, self.task_id,
                                      self.state_id, self.stream, q))
        po = R.obs_to_policy_obs(obs, prompt, image_size=self.image_size)
        chunk = np.clip(self.policy.generate(z0, po), -1, 1)[:self.replan_steps]
        self.chunk = chunk
        self.cursor.buffer_index = 0
        self.cursor.query_idx += 1
        return chunk

    def nominal_action(self, obs, prompt):
        """返回当前 chunk 的下一步 nominal 动作（不推进推理，仅推进 buffer index）。"""
        if self.chunk is None or self.cursor.buffer_index >= len(self.chunk):
            self.generate_chunk(obs, prompt)
        act = self.chunk[self.cursor.buffer_index]
        self.cursor.buffer_index += 1
        return act

    def snapshot(self, env, obs):
        """保存当前完整快照（env + cursor + chunk）。"""
        env_snap = save_full_state(env)
        return Snapshot(env_snap, self.cursor.snapshot(), self.chunk)

    def record_step(self, row):
        """追加一条控制步事件。"""
        row["step"] = self.step
        self.step += 1
        self._events.append(row)
        self._flush(row)


def default_anchor(suite, level, task_id, state_id, stream, prompt,
                   manifest_hash, contract_hash, spec_version, replan_steps):
    """构造 anchor 身份（repo commit / checkpoint 由调用方填）。"""
    import subprocess
    repo_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd="/data/safeVLA").decode().strip()
    checkpoint_hash = "b838d8b85a3f317c91950738172ee7f174afe7f2d36d9d8ed80f7110a085e90b"
    return AnchorIdentity(repo_commit, checkpoint_hash, suite, level, task_id,
                          state_id, stream, prompt, manifest_hash, contract_hash,
                          spec_version, replan_steps)
