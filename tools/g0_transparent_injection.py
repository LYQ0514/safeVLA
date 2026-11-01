"""G0：PnsgPolicy 透明噪声注入等价性验证。

验证 PnsgPolicy.generate(z0, obs) 是否等价于「同一 z0 作为 noise 直接传给
openpi policy.infer(obs, noise=z0)」。

若等价，则 PnsgPolicy 是「透明噪声注入器」——它不改变 π0.5 推理行为，
只是把内部随机采样替换为显式注入的标准高斯 z0。

这是 CRN 配对的前提（G0 第 1 条：推理等价性）。
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")

ROOT = "/data/safeVLA"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "dataset", "SafeLIBERO", "safelibero"))
sys.path.insert(0, "/data/2608.SDL/MojocoProj/src/pi05")

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

from src.pnsg.crn import sample_z  # noqa: E402
from src.pnsg.policy_adapter import PnsgPolicy  # noqa: E402


def load_obs_and_z0(ep=0, step=0):
    """从 aegis_long_design 快照加载一个真实 obs 和 z0。"""
    import json as _json
    DESIGN = pathlib.Path(ROOT, "data", "audit", "aegis_long_design")
    with np.load(DESIGN / f"ep{ep:02d}" / f"decision_{step:03d}.npz") as f:
        d = {k: f[k] for k in f.files}
    j = _json.loads((DESIGN / f"ep{ep:02d}" / f"decision_{step:03d}.json").read_text())
    obs = {k: d[k] for k in ("agentview_image", "robot0_eye_in_hand_image",
                             "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")}
    z0 = d["z0"]
    return obs, z0, j["prompt"] if "prompt" in j else ""


def main():
    import subprocess

    obs, z0, prompt = load_obs_and_z0(0, 0)
    print("=== G0：推理等价性验证 ===")
    print(f"z0 shape={z0.shape}, obs keys={list(obs.keys())}")
    print()

    # 转成 openpi 策略输入格式（obs_to_policy_obs）
    import pi05_libero_runner as R  # noqa: E402
    policy_obs = R.obs_to_policy_obs(obs, "pick up the black bowl", image_size=224)
    print(f"policy_obs keys={list(policy_obs.keys())}")
    print()

    # 路径 A：PnsgPolicy.generate(z0, policy_obs)
    pol = PnsgPolicy()
    actions_A = pol.generate(z0, policy_obs)
    pol.close()
    print(f"[A] PnsgPolicy.generate(z0): shape={actions_A.shape}")
    print(f"    actions_A[:2,:6] = {np.round(actions_A[:2,:6], 5)}")
    print()

    # 路径 B：直接 openpi policy.infer(obs, noise=z0)
    # 用 pi05_libero_runner 的 Policy 加载，但需要绕过 infer 不传 noise 的问题
    # 直接调 openpi 的 infer(noise=...)
    script = r'''
import os, sys, json
import numpy as np
sys.path.insert(0, "/data/2608.SDL/MojocoProj/src/pi05")
import pi05_libero_runner as R
from openpi.policies import policy_config as _pc
from openpi.training import config as _cfg

CKPT = "/data/2608.SDL/MojocoProj/checkpoints/pi05/pi05_libero"
cfg = _cfg.get_config("pi05_libero")
policy = _pc.create_trained_policy(cfg, CKPT, sample_kwargs={"num_steps": 10})

# 原始 obs（robosuite 裸观测）和 z0 从 stdin 读
inp = json.loads(sys.stdin.read())
raw_obs = {k: np.asarray(v) for k, v in inp["raw_obs"].items()}
z0 = np.asarray(inp["z0"], dtype=np.float32)
prompt = inp["prompt"]

# 用 obs_to_policy_obs 转 openpi 格式，再 infer(noise=z0)
pobs = R.obs_to_policy_obs(raw_obs, prompt, image_size=224)
out2 = policy.infer(pobs, noise=z0)
print(json.dumps({"actions": np.asarray(out2["actions"]).tolist()}))
'''
    # 用 openpi venv 跑
    def _ser(v):
        return v.tolist() if hasattr(v, "tolist") else v
    payload = json.dumps({"raw_obs": {k: _ser(v) for k, v in obs.items()},
                          "z0": z0.tolist(), "prompt": prompt})
    result = subprocess.run(
        ["/opt/venv/openpi/bin/python", "-c", script],
        input=payload, capture_output=True, text=True, timeout=600,
        env={**os.environ, "JAX_PLATFORMS": "cuda", "XLA_PYTHON_CLIENT_PREALLOCATE": "false"},
    )
    if result.returncode != 0:
        print("路径 B 失败:", result.stderr[-2000:])
        return
    # 从 stdout 提取 json
    out_B = None
    for line in result.stdout.strip().splitlines():
        try:
            out_B = json.loads(line)
            break
        except Exception:
            continue
    if out_B is None:
        print("路径 B 无 JSON 输出，stdout:", result.stdout[-1000:])
        return
    actions_B = np.asarray(out_B["actions"])
    print(f"[B] policy.infer(noise=z0): shape={actions_B.shape}")
    print(f"    actions_B[:2,:6] = {np.round(actions_B[:2,:6], 5)}")
    print()

    # 比较
    diff = np.max(np.abs(actions_A - actions_B))
    print(f"=== 等价性判定 ===")
    print(f"max |A - B| = {diff:.6e}")
    tol = 1e-4
    if diff < tol:
        print(f"PASS：PnsgPolicy.generate(z0) 与 policy.infer(noise=z0) 在 {tol:.0e} 内一致")
    else:
        print(f"FAIL：差异 {diff:.6e} 超过容差 {tol:.0e}")


if __name__ == "__main__":
    main()
