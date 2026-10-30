"""PNSG openpi worker：源空间 steering 的策略前向与 Jacobian 服务。

在 openpi（py3.11 + JAX）环境里运行，通过文件 pickle IPC 与 .venv（py3.8）通信。

暴露命令：
- generate(z, obs) -> external action（clip 后，7 维）
- generate_jacobian(z, obs) -> raw action + raw Jacobian + external action + external Jacobian

Jacobian 对应「外部 pose 动作坐标」（Unnormalize 之后），包含 affine scaling，
不包含 clipping。参见 docs/设计.md 第 3 节。

用法：由 src/pnsg/policy_adapter.py 启动，或手动：
    PI05_READY_PATH=/tmp/r PI05_REQ_PATH=/tmp/q PI05_RESP_PATH=/tmp/s \
    /opt/venv/openpi/bin/python src/pnsg/openpi_worker.py
"""

import os
import pickle
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

CKPT_DIR = os.environ.get("PI05_LIBERO_CKPT", "/data/2608.SDL/MojocoProj/checkpoints/pi05/pi05_libero")
CONFIG_NAME = os.environ.get("PI05_CONFIG", "pi05_libero")
NUM_STEPS = int(os.environ.get("PI05_NUM_INFER_STEPS", "10"))
_READY_PATH = os.environ["PI05_READY_PATH"]
_REQ_PATH = os.environ["PI05_REQ_PATH"]
_RESP_PATH = os.environ["PI05_RESP_PATH"]

sys.stdout = sys.stderr  # 日志走 stderr，不污染 IPC


def _atomic_dump(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


def build():
    cfg = _config.get_config(CONFIG_NAME)
    policy = _policy_config.create_trained_policy(
        cfg, CKPT_DIR, sample_kwargs={"num_steps": NUM_STEPS}
    )
    model = policy._model
    sample_actions = model.sample_actions  # 未 jit 原始方法（可 jacfwd）
    # Unnormalize 的 affine 参数（前 6 维 pose 的 std）
    std = policy._output_transform  # 组合 transform，含 Unnormalize
    return policy, model, sample_actions


def _obs_to_inputs(policy, obs):
    """把 .venv 传来的 obs dict 转成模型输入，返回 (observation, state_np)。

    state_np 供 output transform 使用（Unnormalize 需要 state 键）。
    """
    inputs = policy._input_transform({k: v for k, v in obs.items()})
    state_np = np.asarray(inputs["state"])
    inputs_jax = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
    from openpi.models import model as _model
    return _model.Observation.from_dict(inputs_jax), state_np


def _norm_stats_std(policy):
    """提取 pose 前 6 维的 Unnormalize std（external = raw * std + mean）。

    直接从 norm stats 加载（比遍历 transform 对象更稳健）。
    """
    import json
    import pathlib
    stats_path = pathlib.Path(CKPT_DIR) / "assets" / "physical-intelligence" / "libero" / "norm_stats.json"
    with open(stats_path) as f:
        d = json.load(f)
    std = np.asarray(d["norm_stats"]["actions"]["std"], dtype=np.float32)
    return std[:6]


def main():
    policy, model, sample_actions = build()
    std_pose = _norm_stats_std(policy)  # (6,)
    H = model.action_horizon
    D = model.action_dim
    rng_key = jax.random.key(0)

    info = {
        "action_horizon": H,
        "action_dim": D,
        "std_pose": std_pose.tolist(),
        "source_dim": H * D,
    }
    _atomic_dump(_READY_PATH, {"status": "ready", "info": info})

    while True:
        try:
            if os.path.exists(_REQ_PATH):
                try:
                    with open(_REQ_PATH, "rb") as f:
                        req = pickle.load(f)
                except (EOFError, pickle.UnpicklingError, OSError):
                    time.sleep(0.01)
                    continue
                try:
                    os.remove(_REQ_PATH)
                except Exception:
                    pass

                cmd = req.get("cmd")
                if cmd == "close":
                    _atomic_dump(_RESP_PATH, {"status": "closed"})
                    break

                if cmd in ("generate", "generate_jacobian"):
                    try:
                        z = np.asarray(req["z"], dtype=np.float32)  # (H, D)
                        observation, state_np = _obs_to_inputs(policy, req["obs"])
                        z_jax = jnp.asarray(z)[np.newaxis, ...]  # (1,H,D)

                        f = lambda zz: sample_actions(rng_key, observation, num_steps=NUM_STEPS, noise=zz)
                        raw = f(z_jax)  # (1,H,D)

                        if cmd == "generate_jacobian":
                            f_pose = lambda zz: f(zz)[0, :, :6]  # (H,6)
                            J_raw = jax.jacfwd(f_pose)(z_jax)     # (H,6,1,H,D)
                            J_raw = J_raw[:, :, 0, :, :].reshape(H * 6, H * D)  # (60,320)
                            # external J = raw J * std（逐 pose 通道）
                            scale = np.tile(std_pose, H)  # (60,)
                            J_ext = J_raw * scale[:, None]

                        raw_np = np.asarray(raw[0], dtype=np.float32)  # (H,32)
                        # output transform 需带 state 键（Unnormalize strict=True）
                        out = policy._output_transform({"state": state_np, "actions": raw_np})
                        actions_ext = np.asarray(out["actions"], dtype=np.float32)  # (H,7)

                        resp = {
                            "status": "ok",
                            "raw_actions": raw_np,          # (H,32) 归一化空间
                            "actions": actions_ext,          # (H,7) external，未 clip
                        }
                        if cmd == "generate_jacobian":
                            resp["raw_jacobian"] = np.asarray(J_raw, dtype=np.float32)   # (60,320)
                            resp["external_jacobian"] = np.asarray(J_ext, dtype=np.float32)
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        resp = {"status": "error", "error": f"{type(e).__name__}: {e}"}
                    _atomic_dump(_RESP_PATH, resp)
                    continue

                _atomic_dump(_RESP_PATH, {"status": "error", "error": f"unknown cmd {cmd!r}"})
        except Exception:
            import traceback
            traceback.print_exc()
            time.sleep(0.01)
        time.sleep(0.002)


if __name__ == "__main__":
    main()
