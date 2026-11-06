"""CHARACTERIZATION_SCENE_V1 最小 scene builder。

协议：docs/method/characterization_scene_v1.md
scene_protocol_sha256=145cc8e7635f74645675ba3b79b09ed75636ffce5352f4985eb512238f657f8e

实现：
1. 加载 suite 的 task 场景（作为物理组件宿主）；
2. teleport 被测对象到标准位姿 (0,0) + canonical 姿态；
3. 其它 free-joint 对象 teleport 到工作区外；
4. settling（无动作，确认对象稳定）。

初始化 teleport 合法（trial 开始前），之后动作走 env.step。
"""

from __future__ import annotations

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

SCENE_PROTOCOL_SHA = "145cc8e7635f74645675ba3b79b09ed75636ffce5352f4985eb512238f657f8e"
STD_POS = np.array([0.0, 0.0])  # 标准水平位置（世界系）
STD_YAW = 0.0
OUT_OF_WORKSPACE_Z = 10.0  # 其它对象移出工作区的 z（桌面下方不会撞，直接高处）


def load_env(suite, task_name):
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
        raise ValueError(f"task {task_name} 不在 suite {suite}")
    task = ts.get_task(task_id)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256,
                             camera_widths=256, controller="OSC_POSE")
    env.seed(7)
    env.reset()
    return env


def free_joint_bodies(env):
    """所有 free-joint body 名（可移动对象）。"""
    m = env.sim.model
    out = []
    for j in range(m.njnt):
        if int(m.jnt_type[j]) == 0:
            bid = m.jnt_bodyid[j]
            out.append(m.body_names[bid])
    return out


def set_body_pose(env, body_name, pos, quat_xyzw):
    """teleport 对象 body 到指定 pos + quat（写 free joint qpos）。"""
    m = env.sim.model
    d = env.sim.data._data
    bid = m.body_name2id(body_name)
    for j in range(m.njnt):
        if m.jnt_bodyid[j] == bid:
            qadr = m.jnt_qposadr[j]
            d.qpos[qadr:qadr+3] = np.asarray(pos)
            d.qpos[qadr+3:qadr+7] = np.asarray(quat_xyzw)
            d.qvel[m.jnt_dofadr[j]:m.jnt_dofadr[j]+6] = 0.0
            return
    raise ValueError(f"{body_name} 无 free joint")


def obj_z_min_local(env, body_name):
    """对象 body-local 最低 collision point 的 z（用于高度=桌面-最低点）。"""
    m = env.sim.model
    bid = m.body_name2id(body_name)
    zs = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == bid and m.geom_contype[g] > 0:
            gpos = np.asarray(m.geom_pos[g])
            gsize = np.asarray(m.geom_size[g])
            # box/cylinder: size 是半轴；mesh 用 size 近似
            zs.append(gpos[2] - abs(gsize[2]))
    return min(zs) if zs else 0.0


def table_height(env):
    """支撑面高度（找 table 或 floor body 的 top）。"""
    m = env.sim.model
    d = env.sim.data._data
    for bn in m.body_names:
        if 'table' in bn.lower():
            return float(d.xpos[m.body_name2id(bn)][2])
    return 0.8  # 默认桌面高度（若场景无 table body）


def canonical_quat(shape_family):
    """canonical upright 姿态（yaw=0）。body-local z 对齐世界 z。"""
    # 大多数 SafeLIBERO 对象 body-local z 已是世界 z（upright），
    # 这里用 identity（xyzw = [1,0,0,0]），shape-specific 规则在协议 §3
    return np.array([1.0, 0.0, 0.0, 0.0])


def build_scene(env, target_body, shape_family):
    """teleport 被测对象到标准位姿，其它对象移出工作区，settling。

    关键修正：对象 reset 后已在桌面上（z 自然），只改 x/y 到标准位置 + canonical 姿态，
    不改 z（保持自然桌面高度，避免 table_height 计算错误）。
    """
    m = env.sim.model
    d = env.sim.data._data
    free = free_joint_bodies(env)

    # 记录对象 reset 后的自然 z（桌面高度参考）
    tbid = m.body_name2id(target_body)
    z_natural = float(d.xpos[tbid][2])

    # 1. 其它对象移出工作区（teleport 到高处，避免干扰）
    for bn in free:
        if bn != target_body:
            set_body_pose(env, bn, [0.0, 0.0, OUT_OF_WORKSPACE_Z], [1, 0, 0, 0])

    # 2. 被测对象到标准 x/y + canonical 姿态，保持自然 z
    quat = canonical_quat(shape_family)
    set_body_pose(env, target_body, [STD_POS[0], STD_POS[1], z_natural], quat)

    # 3. forward + settling
    for _ in range(100):
        env.env.sim.forward()
        env.env.sim.step()
        env.env._update_observables()

    # 4. settling 门禁
    vel = np.asarray(d.cvel[tbid])
    if np.linalg.norm(vel) > 0.02:
        return None, False, "CHARACTERIZATION_INITIALIZATION_INVALID: object moving"
    pos = np.asarray(d.xpos[tbid])
    anchor = {
        "target_body": target_body, "target_pos": pos.tolist(),
        "target_quat": np.asarray(d.xquat[tbid]).tolist(),
        "z_natural": z_natural,
        "std_pos": STD_POS.tolist(), "std_yaw": STD_YAW,
        "scene_protocol_sha256": SCENE_PROTOCOL_SHA,
    }
    return anchor, True, None


if __name__ == "__main__":
    # 开发自测：cream_cheese 在 long SCENE1
    env = load_env("safelibero_long",
                   "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket")
    anchor, ok, reason = build_scene(env, "cream_cheese_1_main", "box_like")
    print("cream_cheese 标准场景:", "OK" if ok else reason)
    if ok:
        print("  anchor:", json.dumps(anchor, indent=2))
        # 确认只有 cream_cheese 在工作区，其它在 z=10
        for bn in free_joint_bodies(env):
            p = np.asarray(env.sim.data._data.xpos[env.sim.model.body_name2id(bn)])
            print(f"  {bn}: z={p[2]:.2f}")
    env.close()
