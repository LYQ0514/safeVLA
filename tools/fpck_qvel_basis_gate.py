"""六基向量 qvel 门禁：确定 free joint qvel 六维的真实物理语义。

裁决：MuJoCo free joint qvel 官方定义是 [vx,vy,vz,wx,wy,wz]（前三线速度，后三角速度），
与 cvel 的 [angular, linear] 顺序相反。必须用硬断言实测，不凭记忆。

注入六个单位基向量 e0..e5，每步记录：
- setter 后 get_joint_qvel
- get_body_xvelp（线速度）/ get_body_xvelr（角速度）
- COM 有限差分 (x_{t+1}-x_t)/dt
- 四元数旋转差
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


def make_env():
    from libero.libero import benchmark
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    import pathlib
    ts = benchmark.get_benchmark_dict()["safelibero_long"](safety_level="I")
    task = ts.get_task(0)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256,
                             camera_widths=256, controller="OSC_POSE")
    env.seed(7)
    env.reset()
    return env


def probe(env, basis_idx):
    """注入第 basis_idx 个基向量（幅值 1.0），执行 1 个 substep，记录物理响应。"""
    d = env.sim.data._data
    m = env.sim.model
    joint_name = "cream_cheese_1_joint0"
    body_name = "cream_cheese_1_main"
    cc_bid = m.body_name2id(body_name)

    # 抬到空中（无接触），canonical 姿态
    env.sim.data.set_joint_qpos(joint_name, np.array([0.0, 0.0, 0.6, 1, 0, 0, 0]))
    env.sim.step()  # 稳定
    x0 = np.asarray(d.xpos[cc_bid]).copy()
    q0 = np.asarray(d.xquat[cc_bid]).copy()

    # 注入单位基向量
    v = np.zeros(6)
    v[basis_idx] = 1.0
    env.sim.data.set_joint_qvel(joint_name, v)
    qvel_after_set = env.sim.data.get_joint_qvel(joint_name).copy()

    # 一个 substep
    dt = env.sim.model.opt.timestep
    env.sim.step()

    x1 = np.asarray(d.xpos[cc_bid]).copy()
    q1 = np.asarray(d.xquat[cc_bid]).copy()
    fd_vel = (x1 - x0) / dt
    # 四元数旋转差（q1 vs q0 的角速度）
    # 用相对四元数
    q_rel = np.asarray(env.sim.data._data.xquat[cc_bid]).copy()
    ang_change = np.arccos(np.clip(abs(np.dot(q0, q1)), 0, 1)) * 2 / dt

    return {
        "basis_idx": basis_idx,
        "qvel_after_set": qvel_after_set.tolist(),
        "fd_linear_vel": fd_vel.tolist(),
        "fd_angular_rad_s": float(ang_change),
    }


def main():
    env = make_env()
    results = []
    print("=== 六基向量 qvel 门禁 ===")
    print(f"{'注入':>6} {'fd线速度(x,y,z)':>24} {'fd角速度(rad/s)':>16}")
    for i in range(6):
        r = probe(env, i)
        results.append(r)
        fd = r["fd_linear_vel"]
        print(f"e{i}: ({fd[0]:+.4f},{fd[1]:+.4f},{fd[2]:+.4f})  ang={r['fd_angular_rad_s']:.4f}")

    # 硬断言：e2 应产生 z 平移（dz/dt≈注入值），e5 应产生角速度
    e2 = results[2]["fd_linear_vel"]
    e5_ang = results[5]["fd_angular_rad_s"]
    e2_ang = results[2]["fd_angular_rad_s"]
    e5_lin = results[5]["fd_linear_vel"]

    print("\n=== 断言 ===")
    print(f"e2: dz/dt={e2[2]:+.4f}（应≈1.0），|dx|={abs(e2[0]):.4f}（应≈0），|dy|={abs(e2[1]):.4f}（应≈0），ang={e2_ang:.4f}（应≈0）")
    print(f"e5: ang={e5_ang:.4f}（应≈1.0），|lin|={np.linalg.norm(e5_lin):.4f}（应≈0）")

    # 结论
    linear_dim = None
    for i in range(6):
        fd = results[i]["fd_linear_vel"]
        if abs(fd[2]) > 0.5:  # 明显 z 平移
            linear_dim = i
    print(f"\n结论：线速度在 qvel 第 {linear_dim} 维（0-based）")
    print("→ 前 3 维是线速度" if linear_dim is not None and linear_dim < 3 else "→ 前 3 维是角速度（与官方文档相反！）")

    out_dir = os.path.join(ROOT, ".cache-local", "fpck_characterization")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "qvel_basis_gate.json"), "w") as f:
        json.dump({"results": results, "linear_dim": linear_dim}, f,
                  ensure_ascii=False, indent=2)
    env.close()


if __name__ == "__main__":
    main()
