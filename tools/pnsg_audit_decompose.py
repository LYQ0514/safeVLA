"""Step 3：重算 task0 机制分解，验证 raw contact 按角色正确分类。

对应想法v5.md Step 2/3。跑 task0 名义轨迹，输出 6 类接触事件分解，
验证之前的 547 条 raw contact 被正确归类。
"""

from __future__ import annotations

import os
import sys
import warnings
from collections import Counter

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "dataset", "SafeLIBERO", "safelibero"))
sys.path.insert(0, "/data/2608.SDL/MojocoProj/src/pi05")

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
import pathlib  # noqa: E402

import pi05_libero_runner as R  # noqa: E402


def main():
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    from src.pnsg.contact_audit import ContactEventRecorder, load_role_manifest
    from src.pnsg.policy_adapter import PnsgPolicy

    manifest_all = load_role_manifest(os.path.join(ROOT, "data", "task_object_role_manifest.json"))

    suite, level, task_id, ep = "safelibero_spatial", "I", 0, 0
    ts = benchmark.get_benchmark_dict()[suite](safety_level=level)
    task = ts.get_task(task_id)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256,
                             controller="OSC_POSE")
    env.seed(7)
    obs = env.reset()
    obs = env.set_init_state(ts.get_task_init_states(task_id)[ep])
    dummy = np.array([0.0] * 6 + [-1.0])
    for _ in range(10):
        obs, _r, _d, _i = env.step(dummy)

    prompt = task.language
    task_manifest = manifest_all.get(suite, {}).get(task.name, {})
    print(f"任务: {task.name}")
    print(f"角色清单对象数: {len(task_manifest.get('objects', {}))}")

    policy = PnsgPolicy()
    rec = ContactEventRecorder(env, role_manifest=task_manifest.get("objects", {}))

    # 打印角色 geom 分布
    for role, geoms in rec._role_geoms.items():
        if geoms:
            print(f"  {role}: {len(geoms)} geoms")

    # 跑 100 步名义轨迹，累计分类接触
    type_counter = Counter()
    type_pair_counter = Counter()
    n_contact_steps = 0
    n_steps = 100

    for step in range(n_steps):
        po = R.obs_to_policy_obs(obs, prompt, image_size=224)
        z0 = policy.sample_noise(seed=step // 2)
        ext = policy.generate(z0, po)
        action = np.clip(ext[step % 2], -1, 1)

        obs, summary = rec.substep_step(action)
        for ev in summary["events"]:
            type_counter[ev["type"]] += 1
            type_pair_counter[(ev["type"], ev["body1"], ev["body2"])] += 1
        if summary["any_substep_contact"]:
            n_contact_steps += 1

    print(f"\n=== task0 机制分解（{n_steps} 步，{n_contact_steps} 步有接触）===")
    print(f"{'事件类型':<25} {'raw记录数':>10}")
    for typ, cnt in type_counter.most_common():
        print(f"  {typ:<25} {cnt:>10}")

    print(f"\n=== 按 pair 的 top 分布 ===")
    for (typ, b1, b2), cnt in type_pair_counter.most_common(15):
        print(f"  [{typ}] {b1} <-> {b2}: {cnt}")

    policy.close()
    env.close()


if __name__ == "__main__":
    main()
