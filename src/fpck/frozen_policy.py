"""Deterministic frozen π0.5 chunk policy used by FPCK collection/evaluation."""
from __future__ import annotations
import hashlib
import os
import sys
import numpy as np

from src.pnsg.crn import nominal_seed, sample_z


class DeterministicFrozenPolicy:
    def __init__(self, worker, *, suite, task_id, state_id, stream=0,
                 replan_steps=5, image_size=224):
        self.worker = worker; self.suite = suite; self.task_id = int(task_id)
        self.state_id = int(state_id); self.stream = int(stream)
        self.replan_steps = int(replan_steps); self.image_size = int(image_size)
        self.reset()

    @property
    def ready(self):
        return True

    def reset(self, obs=None, prompt=None):
        self.query_idx = 0; self.buffer_index = 0; self.chunk = None

    def _policy_obs(self, obs, prompt):
        shared_root = os.environ.get("PI05_SHARED_ROOT", "/data/2608.SDL/MojocoProj")
        shared = os.path.join(shared_root, "src", "pi05")
        if shared not in sys.path:
            sys.path.insert(0, shared)
        import pi05_libero_runner as R
        return R.obs_to_policy_obs(obs, prompt, image_size=self.image_size)

    def _query(self, obs, prompt):
        self.chunk, query_meta = self.counterfactual_chunk(
            obs, prompt, query_idx=self.query_idx)
        self.buffer_index = 0; self.query_idx += 1
        query_meta["counterfactual"] = False
        self._meta = query_meta

    def counterfactual_chunk(self, obs, prompt, query_idx=None):
        """Query frozen pi0.5 without mutating the deployed chunk cursor.

        Viability planning uses this at a simulated terminal observation to ask
        whether the unchanged VLA can safely resume.  The CRN identity is the
        next deployed query index, while the live policy state remains intact.
        """
        q = self.query_idx if query_idx is None else int(query_idx)
        seed = nominal_seed(self.suite, self.task_id, self.state_id,
                            self.stream, q)
        z, noise_meta = sample_z(seed, shape=(self.worker.H, self.worker.D))
        actions, hashes = self.worker.generate_with_hashes(
            z, self._policy_obs(obs, prompt))
        chunk = np.clip(actions, -1, 1)[:self.replan_steps].astype(np.float32)
        meta = dict(hashes or {})
        meta.update({"seed": int(seed), "query_idx": q,
                     "chunk_hash": hashlib.sha256(chunk.tobytes()).hexdigest(),
                     "chunk_len": int(len(chunk)), "noise": noise_meta,
                     "counterfactual": True})
        return chunk, meta

    def peek(self, obs, prompt):
        if self.chunk is None or self.buffer_index >= len(self.chunk):
            self._query(obs, prompt)
        return self.chunk[self.buffer_index].copy()

    def act(self, obs, prompt):
        action = self.peek(obs, prompt); self.buffer_index += 1; return action

    def get_current_chunk(self):
        return None if self.chunk is None else self.chunk.copy()

    def get_chunk_metadata(self):
        return dict(getattr(self, "_meta", {}))

    def snapshot(self):
        return {"query_idx": self.query_idx, "buffer_index": self.buffer_index,
                "chunk": None if self.chunk is None else self.chunk.copy(),
                "meta": dict(getattr(self, "_meta", {}))}

    def restore(self, snap):
        self.query_idx = int(snap["query_idx"]); self.buffer_index = int(snap["buffer_index"])
        self.chunk = None if snap["chunk"] is None else np.asarray(snap["chunk"], np.float32).copy()
        self._meta = dict(snap.get("meta", {}))

    def close(self):
        # Worker ownership belongs to the runner and may be shared across episodes.
        pass
