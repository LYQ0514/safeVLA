"""canonical pose manifest 生成（离线，一次提取后冻结）。

协议：characterization_scene_v2.md
scene_protocol_v2_sha256=900bcdd4cd231eaca734bf40ac307dbcbe94455f0e6e1d822f8ac3dacf1148fb

裁决 §三/§四：离线提取姿态来源，写入 manifest，冻结 hash，之后不再调 reset 决定姿态。
canonical 姿态来源优先级（v2）：
1. 资产注册默认（对象类 rotation/rotation_axis 字段）；
2. reset 后自然稳定姿态（作为资产注册默认的实际体现）；
3. 无稳定 → CANONICAL_STABLE_POSE_NOT_FOUND。
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = "/data/safeVLA"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "dataset", "SafeLIBERO", "safelibero"))
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

SCENE_PROTOCOL_V2_SHA = "900bcdd4cd231eaca734bf40ac307dbcbe94455f0e6e1d822f8ac3dacf1148fb"

# 对象类名 → object_class（用于从 manifest 反查类）
OBJECT_CLASS_TO_BODY = {
    "cream_cheese": "cream_cheese_1_main",
    "chocolate_pudding": "chocolate_pudding_1_main",
    "bbq_sauce": "bbq_sauce_1_main",
    "milk": "milk_1_main",
    "orange_juice": "orange_juice_1_main",
    "alphabet_soup": "alphabet_soup_1_main",
    "tomato_sauce": "tomato_sauce_1_main",
    "akita_black_bowl": "akita_black_bowl_1_main",
    "porcelain_mug": "porcelain_mug_1_main",
    "white_yellow_mug": "white_yellow_mug_1_main",
}

# 对象 class → 所在 suite + task_name（用于加载场景提取姿态）
OBJECT_TASK = {
    "cream_cheese": ("safelibero_long", "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket"),
    "chocolate_pudding": ("safelibero_long", "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate"),
    "bbq_sauce": ("safelibero_object", "pick_up_the_bbq_sauce_and_place_it_in_the_basket"),
    "milk": ("safelibero_object", "pick_up_the_milk_and_place_it_in_the_basket"),
    "orange_juice": ("safelibero_object", "pick_up_the_orange_juice_and_place_it_in_the_basket"),
    "alphabet_soup": ("safelibero_long", "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket"),
    "tomato_sauce": ("safelibero_long", "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket"),
    "akita_black_bowl": ("safelibero_spatial", "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"),
    "porcelain_mug": ("safelibero_long", "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate"),
    "white_yellow_mug": ("safelibero_long", "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate"),
}


def canonicalize_quat(q):
    """归一化 wxyz，第一个非零分量为正（q 与 -q 同姿态统一符号）。"""
    q = np.asarray(q, dtype=np.float64)
    q = q / (np.linalg.norm(q) + 1e-12)
    for i in range(4):
        if abs(q[i]) > 1e-9:
            if q[i] < 0:
                q = -q
            break
    return q


def extract_canonical_quat(suite, task_name, body_name):
    """离线提取对象的稳定自然姿态（reset 后 settling 稳定）。"""
    from libero.libero import benchmark
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    import pathlib
    ts = benchmark.get_benchmark_dict()[suite](safety_level="I")
    task_id = None
    for i in range(ts.n_tasks):
        if ts.get_task(i).name == task_name:
            task_id = i
            break
    if task_id is None:
        return None, f"task not found"
    task = ts.get_task(task_id)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256,
                             camera_widths=256, controller="OSC_POSE")
    env.seed(7)
    env.reset()
    d = env.sim.data._data
    m = env.sim.model
    bid = m.body_name2id(body_name)
    # settling（对象 reset 后从初始摆放掉落到桌面，需 ~100 步才稳定）
    for _ in range(200):
        env.env.sim.forward()
        env.env.sim.step()
        env.env._update_observables()
    vel = np.asarray(d.cvel[bid])
    quat = canonicalize_quat(np.asarray(d.xquat[bid]))
    env.close()
    if np.linalg.norm(vel) > 0.02:
        return None, f"unstable after settling (vel={np.linalg.norm(vel):.4f})"
    return quat, None


def main():
    manifest = {}
    for cls, body in OBJECT_CLASS_TO_BODY.items():
        suite, task_name = OBJECT_TASK[cls]
        quat, err = extract_canonical_quat(suite, task_name, body)
        if quat is None:
            manifest[cls] = {"status": "CANONICAL_STABLE_POSE_NOT_FOUND",
                             "reason": err}
            print(f"{cls:22s}: FAIL ({err})")
            continue
        manifest[cls] = {
            "object_class": cls, "body": body,
            "orientation_source": "ASSET_REGISTERED_DEFAULT",
            "source_reference": f"{suite}/{task_name} reset settled",
            "quaternion_order": "wxyz",
            "quaternion": quat.tolist(),
            "settling_gate": {"status": "PASS"},
        }
        print(f"{cls:22s}: quat={np.round(quat, 4).tolist()}")

    out_dir = os.path.join(ROOT, ".cache-local", "fpck_characterization")
    os.makedirs(out_dir, exist_ok=True)
    # 冻结 manifest hash（排除自身）
    body_json = json.dumps({"scene_protocol_v2_sha256": SCENE_PROTOCOL_V2_SHA,
                            "poses": manifest}, sort_keys=True, ensure_ascii=False)
    manifest_sha = hashlib.sha256(body_json.encode()).hexdigest()
    out = {
        "schema_version": "characterization_canonical_pose_manifest_v1",
        "scene_protocol_v2_sha256": SCENE_PROTOCOL_V2_SHA,
        "poses": manifest,
        "manifest_sha256": manifest_sha,
    }
    out_path = os.path.join(out_dir, "characterization_canonical_pose_manifest_v1.json")
    with open(out_path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nmanifest_sha256={manifest_sha}")
    print(f"→ {out_path}")


if __name__ == "__main__":
    main()
