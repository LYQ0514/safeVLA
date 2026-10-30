"""PNSG policy adapter：管理 openpi worker 子进程，提供源空间接口。

用法：
    from src.pnsg.policy_adapter import PnsgPolicy
    policy = PnsgPolicy()
    z0 = policy.sample_noise(seed=42)          # (10,32)
    actions, J = policy.generate_with_jacobian(z0, obs)
    candidates = policy.generate(z, obs)       # 注入候选 z
    policy.close()
"""

from __future__ import annotations

import os
import pickle
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

_OPENPI_PY = "/opt/venv/openpi/bin/python"
_WORKER_SRC = str(Path(__file__).parent / "openpi_worker.py")
_CKPT = "/data/2608.SDL/MojocoProj/checkpoints/pi05/pi05_libero"


class PnsgPolicy:
    def __init__(self, ckpt_dir=_CKPT, num_steps=10, openpi_py=_OPENPI_PY):
        self._tmpdir = tempfile.TemporaryDirectory(prefix="pnsg_")
        base = Path(self._tmpdir.name)
        self._ready = base / "ready.pkl"
        self._req = base / "req.pkl"
        self._resp = base / "resp.pkl"
        self._log = base / "worker.log"

        env = dict(
            os.environ,
            PI05_LIBERO_CKPT=ckpt_dir,
            PI05_CONFIG="pi05_libero",
            PI05_NUM_INFER_STEPS=str(num_steps),
            PI05_READY_PATH=str(self._ready),
            PI05_REQ_PATH=str(self._req),
            PI05_RESP_PATH=str(self._resp),
            JAX_PLATFORMS="cuda",
            XLA_PYTHON_CLIENT_PREALLOCATE="false",
        )
        self._log_fh = open(self._log, "wb")
        self._proc = subprocess.Popen(
            [openpi_py, _WORKER_SRC],
            stdin=subprocess.DEVNULL,
            stdout=self._log_fh,
            stderr=subprocess.STDOUT,
            env=env,
            bufsize=0,
        )
        # 等 ready
        t0 = time.time()
        while not self._ready.exists():
            if self._proc.poll() is not None:
                raise RuntimeError(f"worker 退出 code={self._proc.returncode}:\n{self._tail()}")
            if time.time() - t0 > 600:
                raise RuntimeError("worker ready 超时")
            time.sleep(0.2)
        with open(self._ready, "rb") as f:
            self.info = pickle.load(f)["info"]
        self.H = self.info["action_horizon"]
        self.D = self.info["action_dim"]
        self.source_dim = self.H * self.D

    def _tail(self, nbytes=4000):
        try:
            return Path(self._log).read_bytes()[-nbytes:].decode("utf-8", "ignore")
        except Exception:
            return ""

    def _call(self, req, timeout=300):
        tmp = str(self._req) + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(req, f)
        os.replace(tmp, str(self._req))
        t0 = time.time()
        while not self._resp.exists():
            if self._proc.poll() is not None:
                raise RuntimeError(f"worker 退出 code={self._proc.returncode}:\n{self._tail()}")
            if time.time() - t0 > timeout:
                raise RuntimeError(f"worker 调用超时:\n{self._tail()}")
            time.sleep(0.01)
        with open(self._resp, "rb") as f:
            resp = pickle.load(f)
        try:
            os.remove(self._resp)
        except Exception:
            pass
        return resp

    def sample_noise(self, seed=0):
        """采样源噪声 z ∈ R^{H×D}（对应 openpi 的 jax.random.normal）。"""
        rng = np.random.default_rng(seed)
        return rng.standard_normal((self.H, self.D)).astype(np.float32)

    def generate(self, z, obs):
        """注入源噪声 z，返回 external action（未 clip，H×7）。"""
        resp = self._call({"cmd": "generate", "z": np.asarray(z, dtype=np.float32), "obs": obs})
        if resp.get("status") != "ok":
            raise RuntimeError(f"generate 失败: {resp.get('error')}")
        return np.asarray(resp["actions"], dtype=np.float32)

    def generate_with_jacobian(self, z, obs):
        """注入 z，返回 (raw_actions, raw_J, external_actions, external_J)。"""
        resp = self._call({"cmd": "generate_jacobian", "z": np.asarray(z, dtype=np.float32), "obs": obs})
        if resp.get("status") != "ok":
            raise RuntimeError(f"generate_jacobian 失败: {resp.get('error')}")
        return (
            np.asarray(resp["raw_actions"], dtype=np.float32),
            np.asarray(resp["raw_jacobian"], dtype=np.float32),
            np.asarray(resp["actions"], dtype=np.float32),
            np.asarray(resp["external_jacobian"], dtype=np.float32),
        )

    def close(self):
        try:
            self._call({"cmd": "close"})
        except Exception:
            pass
        try:
            self._proc.wait(timeout=5)
        except Exception:
            self._proc.kill()
        self._log_fh.close()
        self._tmpdir.cleanup()
