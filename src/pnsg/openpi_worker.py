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

import hashlib
import json
import os
import pickle
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# XLA_FLAGS 必须在 import jax 之前已存在于 os.environ（由 policy_adapter 的 env 传入）。
# 这里只记录，不改写；若调用方未设，则为 None（默认非确定配置）。
_XLA_FLAGS_AT_IMPORT = os.environ.get("XLA_FLAGS")

import jax
import jax.numpy as jnp
import numpy as np

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


def _hash_array(arr) -> str:
    """规范化数组 hash：C-order 连续 + 固定 dtype，排除设备地址/wall time/元数据。"""
    a = np.ascontiguousarray(np.asarray(arr))
    meta = json.dumps({
        "dtype": str(a.dtype), "shape": list(a.shape),
        "contains_nan": bool(np.isnan(a).any()) if a.dtype.kind == "f" else False,
        "contains_inf": bool(np.isinf(a).any()) if a.dtype.kind == "f" else False,
    }, sort_keys=True)
    return hashlib.sha256(meta.encode() + b"\x00" + a.tobytes(order="C")).hexdigest()


def _hash_inputs(inputs, prefix=""):
    """递归 hash 送入模型的最终输入张量（裁决 §二.2）。

    inputs 可能是嵌套 dict（如 image: {agentview: array, ...}），递归到叶子数组。
    """
    out = {}
    for k, v in inputs.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_hash_inputs(v, prefix=key))
            continue
        arr = np.asarray(jax.device_get(v)) if not isinstance(v, (np.ndarray, list, tuple)) else np.asarray(v)
        out[key] = {
            "sha256": _hash_array(arr),
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "minmax": [float(arr.min()), float(arr.max())] if arr.size else None,
        }
    return out

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
    # 两条路径分开：
    # - generate 用 module_jit 包装（性能：编译后 ~0.08s，而非每次 5s）
    # - generate_jacobian 用原始 while_loop 版（jacfwd 需要，编译后 ~7s）
    # 注意：反向模式（vjp）对 while_loop 不可行；fori_loop 改写因 KV cache
    # shape 动态增长（968→978）也失败。故 Jacobian 只能 jacfwd。
    from openpi.shared import nnx_utils
    sample_actions_jit = nnx_utils.module_jit(model.sample_actions)
    sample_actions_raw = model.sample_actions
    return policy, model, sample_actions_jit, sample_actions_raw


def _obs_to_inputs(policy, obs):
    """把 .venv 传来的 obs dict 转成模型输入，返回 (observation, state_np, inputs)。

    inputs 是 policy._input_transform 的输出 dict（已完成 resize/normalize/tokenize 的
    最终张量，batch 前），供 hash 记录（裁决 §二.2）。
    state_np 供 output transform 使用（Unnormalize 需要 state 键）。
    """
    inputs = policy._input_transform({k: v for k, v in obs.items()})
    state_np = np.asarray(inputs["state"])
    inputs_jax = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
    from openpi.models import model as _model
    return _model.Observation.from_dict(inputs_jax), state_np, inputs


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


def extract_prefix_features(model, observation):
    """提取观测的 prefix hidden states（mean-pooled），供成功价值 probe 使用。

    复用 sample_actions 的前几步（embed_prefix + PaliGemma.llm prefix 前向），
    返回 prefix_out 的 mean pooling，shape (1, width)。
    """
    from openpi.models import model as _model
    from openpi.models.pi0 import make_attn_mask
    observation = _model.preprocess_observation(None, observation, train=False)
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    (prefix_out, _), _ = model.PaliGemma.llm(
        [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
    )
    mask = prefix_mask.astype(jnp.float32)[..., None]  # (1, prefix_len, 1)
    feats = (prefix_out * mask).sum(axis=1) / mask.sum(axis=1)  # (1, width)
    return np.asarray(feats[0], dtype=np.float32)


def main():
    policy, model, sample_actions_jit, sample_actions_raw = build()
    std_pose = _norm_stats_std(policy)  # (6,)
    H = model.action_horizon
    D = model.action_dim
    rng_key = jax.random.key(0)

    info = {
        "action_horizon": H,
        "action_dim": D,
        "std_pose": std_pose.tolist(),
        "source_dim": H * D,
        # 运行时身份（裁决 §二.1：模型/环境身份）
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__ if "jaxlib" in globals() else None,
        "default_backend": str(jax.default_backend()),
        "devices": [str(d) for d in jax.devices()],
        "xla_flags_at_import": _XLA_FLAGS_AT_IMPORT,
        "xla_flags_env": os.environ.get("XLA_FLAGS"),
    }
    try:
        import jaxlib as _jl
        info["jaxlib_version"] = _jl.__version__
    except Exception:
        pass
    try:
        info["xla_client_version"] = jax.lib.xla_client.__version__ \
            if hasattr(jax.lib, "xla_client") else None
    except Exception:
        info["xla_client_version"] = None
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

                if cmd == "extract_features":
                    try:
                        observation, state_np, _ = _obs_to_inputs(policy, req["obs"])
                        feats = extract_prefix_features(model, observation)
                        resp = {"status": "ok", "features": feats}
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        resp = {"status": "error", "error": f"{type(e).__name__}: {e}"}
                    _atomic_dump(_RESP_PATH, resp)
                    continue

                if cmd in ("generate", "generate_jacobian"):
                    try:
                        z = np.asarray(req["z"], dtype=np.float32)  # (H, D)
                        observation, state_np, inputs = _obs_to_inputs(policy, req["obs"])
                        z_jax = jnp.asarray(z)[np.newaxis, ...]  # (1,H,D)

                        # 输入张量 hash（裁决 §二.2：最终送入模型的张量，非原始 obs）
                        input_hashes = _hash_inputs(inputs)
                        z_hash = _hash_array(z)

                        if cmd == "generate_jacobian":
                            # jacfwd 需要原始 while_loop 版（jit 版 jacfwd 不兼容）
                            f = lambda zz: sample_actions_raw(rng_key, observation, num_steps=NUM_STEPS, noise=zz)
                            raw = f(z_jax)
                            f_pose = lambda zz: f(zz)[0, :, :6]  # (H,6)
                            J_raw = jax.jacfwd(f_pose)(z_jax)     # (H,6,1,H,D)
                            J_raw = J_raw[:, :, 0, :, :].reshape(H * 6, H * D)  # (60,320)
                            # external J = raw J * std（逐 pose 通道）
                            scale = np.tile(std_pose, H)  # (60,)
                            J_ext = J_raw * scale[:, None]
                        else:
                            # generate 用 jit 版（性能：编译后 ~0.08s）
                            raw = sample_actions_jit(rng_key, observation, num_steps=NUM_STEPS, noise=z_jax)

                        raw_np = np.asarray(raw[0], dtype=np.float32)  # (H,32)
                        # output transform 需带 state 键（Unnormalize strict=True）
                        out = policy._output_transform({"state": state_np, "actions": raw_np})
                        actions_ext = np.asarray(out["actions"], dtype=np.float32)  # (H,7)

                        resp = {
                            "status": "ok",
                            "raw_actions": raw_np,          # (H,32) 归一化空间
                            "actions": actions_ext,          # (H,7) external，未 clip
                            # 规范化 hash（裁决 §二/§三：区分模型推理 vs 后处理）
                            "input_hashes": input_hashes,
                            "noise_z_sha256": z_hash,
                            "raw_chunk_sha256": _hash_array(raw_np),
                            "deployed_chunk_sha256": _hash_array(actions_ext),
                            "xla_flags_at_import": _XLA_FLAGS_AT_IMPORT,
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
