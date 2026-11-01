"""P0e-1b：Collision-Checked Trajectory Smoothing（S01）。

Gate 0 已完成（见 gate0 脚本）：mug 位移来源 = 机器人运动经桌面/底座间接扰动
（hinge 铰接 mug 的微小摆动），全程 0 个 robot↔mug 接触。

本脚本：
- Gate 1：固定种子的 collision-checked shortcut + 样条平滑，重新稠密碰撞检测；
- Gate 2：平滑轨迹经同一关节 PD 控制器执行，验证 CAR。

对照保留：Raw RRT 与 Smoothed RRT 两组结果。
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time
import warnings

warnings.filterwarnings("ignore")

ROOT = "/data/safeVLA"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "dataset", "SafeLIBERO", "safelibero"))
sys.path.insert(0, "/data/2608.SDL/MojocoProj/src/pi05")
sys.path.insert(0, os.path.join(ROOT, "third_party", "aegis"))

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
import mujoco  # noqa: E402
from scipy.interpolate import CubicSpline  # noqa: E402
from scipy.spatial.transform import Rotation as Rot  # noqa: E402

from src.pnsg.policy_adapter import PnsgPolicy  # noqa: E402
from tools.m1_p1_design_gate_v2 import build_body_geoms  # noqa: E402
from tools.m1_p1_design_gate_v3 import disp_from_p0  # noqa: E402
from tools.m1_p4_p0e_joint_execution import (  # noqa: E402
    reconstruct_nvs, trapezoidal, check_contact_data, KP, KD, V_MAX, A_MAX,
    TORQUE_LO, TORQUE_HI, SETTLE_S, PREGRASP_OFFSETS, IK_RESTARTS, RRT_ITER,
    MUG_BODY, SOUP_BODY, noise_hash,
)
from tools.m1_p4_p0_reachability import (  # noqa: E402
    joint_limits, set_arm_q, get_arm_q, collision_free,
)
from tools.m1_p4_p0_reachability_v2 import ik_multistart, rrt_connect  # noqa: E402

# 平滑参数（预注册，固定）
SHORTCUT_ITER = 300
SHORTCUT_SEED = 42
SPLINE_DENSE_STEP = 0.01  # 样条重采样弧长步长


def shortcut_path(env, rg, og, path, rng):
    """collision-checked shortcut：随机选两路点直连，若直连无碰撞则替换中间段。"""
    path = [np.asarray(q, float).copy() for q in path]

    def seg_free(a, b, step=0.02):
        n = int(np.linalg.norm(b - a) / step) + 1
        for k in range(1, n + 1):
            q = a + (b - a) * (k / n)
            if not collision_free(env, q, rg, og):
                return False
        return True

    for _ in range(SHORTCUT_ITER):
        if len(path) < 3:
            break
        i = rng.integers(0, len(path) - 1)
        j = rng.integers(0, len(path) - 1)
        if abs(i - j) < 2:
            continue
        if i > j:
            i, j = j, i
        if seg_free(path[i], path[j]):
            path = path[:i + 1] + path[j:]
    return path


def smooth_spline(path, dense_step=SPLINE_DENSE_STEP):
    """对 shortcut 后的路径做三次样条平滑 + 弧长重采样。"""
    P = np.array(path)
    n = len(P)
    if n < 4:
        return path
    u = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    L = u[-1]
    if L < 1e-9:
        return path
    cs = CubicSpline(u, P, axis=0)
    u_dense = np.arange(0, L, dense_step)
    u_dense = np.append(u_dense, L)
    return cs(u_dense)


def verify_path_collision(env, rg, og, path):
    """对平滑路径稠密碰撞检测，返回 (all_free, min_gap_info)。"""
    n = len(path)
    for i in range(n - 1):
        a = path[i]
        b = path[i + 1]
        k = int(np.linalg.norm(b - a) / 0.01) + 1
        for j in range(1, k + 1):
            q = a + (b - a) * (j / k)
            if not collision_free(env, q, rg, og):
                return False, i
    # 端点
    if not collision_free(env, path[-1], rg, og):
        return False, n - 1
    return True, -1


def execute_path(env, path, rg, og, p0):
    """关节 PD 执行（computed-torque + 重力补偿），返回执行指标。"""
    m = env.sim.model._model
    d = env.sim.data._data
    dt = env.env.model_timestep
    seg = [float(np.linalg.norm(path[i + 1] - path[i])) for i in range(len(path) - 1)]
    L = sum(seg)
    cumlen = np.concatenate([[0.0], np.cumsum(seg)])
    s_arr, t_arr, T = trapezoidal(L, V_MAX, A_MAX, dt)
    n_settle = int(SETTLE_S / dt)
    q_goal = path[-1].copy()
    M_full = np.ndarray((m.nv, m.nv), dtype=np.float64, order="C")

    def path_at_s_local(s):
        idx = int(np.searchsorted(cumlen, s)) - 1
        idx = max(0, min(idx, len(path) - 2))
        seglen = cumlen[idx + 1] - cumlen[idx]
        frac = (s - cumlen[idx]) / seglen if seglen > 1e-12 else 0.0
        return path[idx] + frac * (path[idx + 1] - path[idx])

    first_contact_t = None
    Dmax = 0.0
    track_err_max = 0.0
    track_err_mean = 0.0
    tau_max_abs = 0.0
    errs = []
    for i in range(len(s_arr) + n_settle):
        q_ref = path_at_s_local(s_arr[i]) if i < len(s_arr) else q_goal
        env.env.sim.forward()
        q = np.asarray(d.qpos[:7])
        qvel = np.asarray(d.qvel[:7])
        mujoco.mj_fullM(m, M_full, d.qM)
        tau = np.clip(M_full[:7, :7] @ (KP * (q_ref - q) - KD * qvel)
                      + np.asarray(d.qfrc_bias[:7]), TORQUE_LO, TORQUE_HI)
        tau_max_abs = max(tau_max_abs, float(np.max(np.abs(tau))))
        d.ctrl[:7] = tau
        d.ctrl[7:] = 0.0
        env.env.sim.step()
        if check_contact_data(env, rg, og) and first_contact_t is None:
            first_contact_t = i * dt
        Dmax = max(Dmax, disp_from_p0(env, p0))
        e = float(np.linalg.norm(q_ref - np.asarray(d.qpos[:7])))
        errs.append(e)
        track_err_max = max(track_err_max, e)
    track_err_mean = float(np.mean(errs)) if errs else 0.0
    return {
        "T": T, "L": L, "first_contact_t": first_contact_t, "Dmax": Dmax,
        "track_err_max": track_err_max, "track_err_mean": track_err_mean,
        "tau_max_abs": tau_max_abs,
        "pos_err": float(np.linalg.norm(np.asarray(d.qpos[:7]) - q_goal)),
        "vel_norm": float(np.linalg.norm(np.asarray(d.qvel[:7]))),
    }


def main():
    from libero.libero import benchmark
    ts = benchmark.get_benchmark_dict()['safelibero_long'](safety_level='I')
    task = ts.get_task(0)
    prompt = task.language
    pol = PnsgPolicy()

    print("=== P0e-1b：轨迹平滑对照 ===", flush=True)
    state_id, stream = 1, 3
    env, layer, scratch, layer_s, rg, og, p0, soup_pos, nvs_step, Dmax0 = \
        reconstruct_nvs(ts, task, prompt, pol, state_id, stream)
    q_start = get_arm_q(env)
    print(f"NVS @ step {nvs_step}，q_start 碰撞自由={collision_free(env, q_start, rg, og)}", flush=True)

    obs = env.env._get_observations()
    iq = np.asarray(obs["robot0_eef_quat"])
    tgt_ori = Rot.from_quat([iq[1], iq[2], iq[3], iq[0]]).as_matrix()
    free_by_off = {}
    for off in PREGRASP_OFFSETS:
        fs, _, _ = ik_multistart(env, soup_pos + off, tgt_ori, rg, og, n=IK_RESTARTS)
        free_by_off[tuple(off.tolist())] = len(fs)
    best_off = max(PREGRASP_OFFSETS, key=lambda o: free_by_off[tuple(o.tolist())])
    free_sols, _, _ = ik_multistart(env, soup_pos + best_off, tgt_ori, rg, og, n=IK_RESTARTS, seed=7)
    q_goal = free_sols[0]

    # 重建 raw RRT（seed=1，与 P0e-1 一致）
    raw_path = None
    for seed in range(8):
        p, f = rrt_connect(env, rg, og, q_start, q_goal, max_iter=RRT_ITER, seed=seed)
        if f:
            raw_path = [np.asarray(q, float) for q in p]
            break
    print(f"[Raw RRT] {len(raw_path)} 路点", flush=True)

    # Gate 1：shortcut + 样条平滑
    rng = np.random.default_rng(SHORTCUT_SEED)
    shortcut = shortcut_path(env, rg, og, raw_path, rng)
    smoothed = smooth_spline(shortcut)
    print(f"[Shortcut] {len(raw_path)} → {len(shortcut)} 路点", flush=True)
    print(f"[Spline] 重采样 → {len(smoothed)} 路点", flush=True)

    # 平滑路径稠密碰撞检测
    free_smoothed, fail_idx = verify_path_collision(env, rg, og, smoothed)
    print(f"[Gate1] 平滑路径碰撞检测: {'通过(全无碰撞)' if free_smoothed else f'失败@路点{fail_idx}'}", flush=True)

    # 恢复执行起点，执行 Raw 与 Smoothed
    results = {}

    def run(path, label):
        set_arm_q(env, q_start)
        mujoco.mj_forward(env.sim.model._model, env.sim.data._data)
        env.sim.data._data.qvel[:] = 0.0
        r = execute_path(env, path, rg, og, p0)
        r["label"] = label
        r["n_waypoints"] = len(path)
        r["L_rad"] = r.pop("L")
        print(f"  [{label}] T={r['T']:.2f}s Dmax={r['Dmax']*1000:.3f}mm "
              f"首接触={r['first_contact_t']} 跟踪max={r['track_err_max']*1000:.0f}mrad "
              f"mean={r['track_err_mean']*1000:.0f}mrad pos_err={r['pos_err']*1000:.1f}mrad", flush=True)
        return r

    results["raw"] = run(raw_path, "Raw RRT")
    results["smoothed"] = run(smoothed, "Smoothed RRT")

    # 验收
    print("\n=== 验收 ===", flush=True)
    for k in ["raw", "smoothed"]:
        r = results[k]
        ok = (r["first_contact_t"] is None) and (r["Dmax"] < 0.001 - 0.0001) \
             and (r["pos_err"] < 0.02) and (r["vel_norm"] < 0.05)
        r["accept"] = bool(ok)
        print(f"  {k}: {'PASS' if ok else 'FAIL'}  (Dmax={r['Dmax']*1000:.3f}mm)", flush=True)

    out = pathlib.Path(ROOT, "data", "audit", "m1_p4_p0e_1b_smoothing.json")
    out.write_text(json.dumps({
        "raw_path_len": len(raw_path), "shortcut_len": len(shortcut),
        "smoothed_len": len(smoothed), "smoothed_collision_free": free_smoothed,
        "results": {k: {kk: (float(vv) if isinstance(vv, (np.floating, np.ndarray)) else vv)
                        for kk, vv in v.items()} for k, v in results.items()},
    }, ensure_ascii=False, indent=2, default=str))
    print(f"\n结果: {out}")
    pol.close()
    env.close()
    scratch.close()


if __name__ == "__main__":
    main()
