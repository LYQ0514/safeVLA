"""确定性验证矩阵（裁决 §四）。

三组配置 × 多个冷启动进程，对比 input/raw chunk/deployed chunk 的规范化 hash：
  组 A：默认（无 XLA_FLAGS）
  组 B：--xla_gpu_autotune_level=0
  组 C：--xla_gpu_exclude_nondeterministic_ops --xla_gpu_autotune_level=0

每个进程 3 次推理（同 obs + 同 z），记录：
  - input_hashes（image/proprio/lang token/attention mask/state 逐张量 sha256）
  - noise_z_sha256
  - raw_chunk_sha256（模型原始输出，未后处理）
  - deployed_chunk_sha256（后处理 + gripper adapter 后）

门禁 PASS 标准（裁决 §五）：input hash 全一致 + raw chunk hash 唯一 + deployed chunk hash 唯一。

用法：
  python tools/fpck_determinism_matrix.py --group A|B|C --n-proc 3 --n-reps 3
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = "/data/safeVLA"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "dataset", "SafeLIBERO", "safelibero"))
sys.path.insert(0, "/data/2608.SDL/MojocoProj/src/pi05")
sys.path.insert(0, os.path.join(ROOT, "third_party", "aegis"))
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

import pi05_libero_runner as R  # noqa: E402
from src.pnsg.policy_adapter import PnsgPolicy  # noqa: E402
from src.pnsg.crn import nominal_seed, sample_z  # noqa: E402

SUITE = "safelibero_spatial"
TASK_ID = 1
STATE_ID = 1
IMAGE_SIZE = 224
NUM_STEPS_WAIT = 20

GROUPS = {
    "A": None,
    "B": "--xla_gpu_autotune_level=0",
    "C": "--xla_gpu_exclude_nondeterministic_ops --xla_gpu_autotune_level=0",
}


def collect_obs_z():
    """采集一次固定 obs + z，保存到 .cache-local（跨进程复用）。"""
    from libero.libero import benchmark
    import pathlib
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    ts = benchmark.get_benchmark_dict()[SUITE](safety_level="I")
    task = ts.get_task(TASK_ID)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256,
                             controller="OSC_POSE")
    env.seed(7); env.reset()
    env.set_init_state(ts.get_task_init_states(TASK_ID)[STATE_ID])
    dummy = np.zeros(7, dtype=np.float32); dummy[-1] = -1.0
    for _ in range(NUM_STEPS_WAIT):
        env.step(dummy)
    obs = env.env._get_observations()
    po = R.obs_to_policy_obs(obs, task.language, image_size=IMAGE_SIZE)
    z, _ = sample_z(nominal_seed(SUITE, TASK_ID, STATE_ID, 0, 0))
    env.close()
    return po, np.asarray(z, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=["A", "B", "C"], default="A")
    ap.add_argument("--n-proc", type=int, default=3)
    ap.add_argument("--n-reps", type=int, default=3)
    a = ap.parse_args()

    cache_dir = os.path.join(ROOT, ".cache-local", "fpck_determinism")
    os.makedirs(cache_dir, exist_ok=True)
    obs_path = os.path.join(cache_dir, "obs.pkl")
    z_path = os.path.join(cache_dir, "z.npy")

    # 采集一次 obs + z（若缓存不存在）
    if not (os.path.exists(obs_path) and os.path.exists(z_path)):
        print("采集固定 obs + z ...")
        po, z = collect_obs_z()
        with open(obs_path, "wb") as f:
            pickle.dump(po, f)
        np.save(z_path, z)
    else:
        with open(obs_path, "rb") as f:
            po = pickle.load(f)
        z = np.load(z_path)

    xla_flags = GROUPS[a.group]
    print(f"=== 组 {a.group}: XLA_FLAGS = {xla_flags!r} ===")

    results = []
    for pi in range(a.n_proc):
        policy = PnsgPolicy(xla_flags=xla_flags)
        # 记录 worker info（jax/jaxlib 版本、backend、devices）
        info = policy.info
        for ri in range(a.n_reps):
            actions, hashes = policy.generate_with_hashes(z, po)
            hashes["group"] = a.group
            hashes["proc"] = pi
            hashes["rep"] = ri
            hashes["raw_chunk_sha256"] = hashes["raw_chunk_sha256"]
            hashes["deployed_chunk_sha256"] = hashes["deployed_chunk_sha256"]
            results.append(hashes)
        policy.close()
        # 打印该进程的运行时身份
        print(f"  proc{pi}: jax={info.get('jax_version')} backend={info.get('default_backend')} "
              f"xla_flags_at_import={info.get('xla_flags_at_import')!r}")

    # 汇总
    print()
    print("=== 汇总 ===")
    raw_set = set(r["raw_chunk_sha256"] for r in results)
    dep_set = set(r["deployed_chunk_sha256"] for r in results)
    z_set = set(r["noise_z_sha256"] for r in results)
    # input hash 是否跨进程一致（取每个 input 键的第一个 sha 做对比）
    input_keys = list(results[0]["input_hashes"].keys())
    print(f"input 张量键: {input_keys}")
    print(f"noise_z 唯一数: {len(z_set)}（应=1）")
    print(f"raw_chunk 唯一数: {len(raw_set)}（应=1）")
    print(f"deployed_chunk 唯一数: {len(dep_set)}（应=1）")

    # input hash 跨进程对比
    print()
    print("=== input hash 跨进程一致性 ===")
    for k in input_keys:
        vals = [r["input_hashes"][k]["sha256"] for r in results]
        same = all(v == vals[0] for v in vals)
        print(f"  {k}: {'一致' if same else '不一致! ' + str(vals[:4])}")

    # 详细结果
    print()
    print("=== 逐进程 raw/deployed hash ===")
    for r in results:
        print(f"  proc{r['proc']} rep{r['rep']}: raw={r['raw_chunk_sha256'][:16]} "
              f"deployed={r['deployed_chunk_sha256'][:16]}")

    # 写结果文件
    out_path = os.path.join(cache_dir, f"group_{a.group}.json")
    with open(out_path, "w") as f:
        json.dump({"group": a.group, "xla_flags": xla_flags,
                   "results": results,
                   "raw_unique": len(raw_set),
                   "deployed_unique": len(dep_set),
                   "noise_z_unique": len(z_set)}, f, ensure_ascii=False, indent=2)
    print(f"\n→ {out_path}")


if __name__ == "__main__":
    main()
