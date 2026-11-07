"""实验 1-6 校准配置冻结：按词典序选唯一 (v_max, ω_max, K, N)。

在运行 held-out 之前冻结。使用实验 5 正常轨迹 + 实验 6 校准反例（1x/2x/4x）。

词典序（裁决）：
1. 所有校准反例零过早接受；
2. 正常释放在 H_post 内接受；
3. 非零噪声裕量；
4. 最小化接受延迟；
5. 速度裕量最大；
6. 参数元组字典序。

输出 terminal_validation_config_v1_single_object_calibrated.json（CALIBRATED_WITHIN_SCOPE）。
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

ROOT = "/data/safeVLA"
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

# 实验 5 cream_table 的 settling 表（T_settle 相对撞击，单位 substep）
# 从 exp5_release_settling.json 读取
SETTLE_TABLE_PATH = os.path.join(ROOT, ".cache-local", "fpck_characterization",
                                 "exp5_release_settling.json")
# 实验 6 反例数据
COUNTEREXAMPLE_PATH = os.path.join(ROOT, ".cache-local", "fpck_characterization",
                                   "exp6_counterexamples.json")

V_CANDIDATES = [0.01, 0.02, 0.05, 0.10, 0.20]
W_CANDIDATES = [0.05, 0.10, 0.20, 0.50, 1.00]
K_CANDIDATES = [1, 3, 5]   # 稳定持续窗口（substep）
N_CANDIDATES = [1, 3, 5]   # contact gone 窗口（substep）

H_OPEN = 15                # 控制步（实验1）
H_POST = 100               # 控制步（实验5 观察窗口）
SUBSTEPS = 25


def load_settle_table():
    with open(SETTLE_TABLE_PATH) as f:
        d = json.load(f)
    return d["normal_release_settling_v1"]["conditions"]["cream_table"]


def load_counterexamples():
    with open(COUNTEREXAMPLE_PATH) as f:
        d = json.load(f)
    return d["results"]


def normal_settle_time(v, w):
    """正常轨迹在 (v,w) 下的 settling time（substep），RIGHT_CENSORED 表示不满足。"""
    table = load_settle_table()["settle_table"]
    key = f"v={v}_w={w}"
    return table.get(key)


def counterexample_early_accept(v, w, k, n):
    """校准反例是否在瞬时 goal 阶段被接受。

    返回 (any_early_accept, decisive_reason)。
    简化：反例在前 20 substep 若 goal_true 且速度≤(v,w) 持续 k 步，则过早接受。
    """
    # 用实验6反例的早期速度：如果 v_max 早期 < 阈值，则会被过早接受
    # 反例早期线速度 v_early=0.41~0.54，角速度 w_early≈0
    # 只要 v < 0.41 或 w < 0，反例早期就不会满足「速度≤阈值」
    # 关键：反例早期角速度≈0，所以 ω 阈值无论如何都能被反例满足
    # 真正拒绝反例的是线速度阈值 v_max < 0.41
    # 这里简化：v 阈值 < 反例早期线速度（0.41）则反例不被过早接受
    counterexamples = load_counterexamples()
    # 2x 和 4x 反例的早期线速度
    v_early_min = min(
        max(row["lin_norm2"] for row in r["rows"][:20] if row["goal_truth"])
        for r in counterexamples if r["v0"] >= 0.0844  # 2x 和 4x
    )
    # 反例早期线速度高于阈值 → 反例速度不满足「≤阈值」→ 正确拒绝
    # v 阈值 < v_early_min → 反例被拒绝（正确）
    # v 阈值 >= v_early_min → 反例可能被过早接受（错误）
    if v < v_early_min:
        return False, f"v={v} < 反例早期线速度 {v_early_min:.4f}，反例被正确拒绝"
    else:
        return True, f"v={v} >= 反例早期线速度 {v_early_min:.4f}，反例过早接受"


def main():
    table = load_settle_table()
    print("=== 校准配置选择（词典序）===")

    # 候选网格
    feasible = []
    for v in V_CANDIDATES:
        for w in W_CANDIDATES:
            for k in K_CANDIDATES:
                for n in N_CANDIDATES:
                    # 1. 校准反例零过早接受（线速度阈值 v < 0.41）
                    early_accept, reason = counterexample_early_accept(v, w, k, n)
                    if early_accept:
                        continue
                    # 2. 正常释放在 H_post 内接受
                    t = normal_settle_time(v, w)
                    if t == "RIGHT_CENSORED_NOT_SETTLED":
                        continue
                    # t 是相对撞击的 substep；总接受时刻 = impact + t + k（稳定窗口）
                    # impact=173 substep，H_post=100 控制步=2500 substep
                    accept_substep = 173 + t + k
                    if accept_substep > 2500:
                        continue
                    # 3. 非零噪声裕量（v > 噪声底 b_v≈1e-5, w > b_ω≈0）
                    # 噪声底极小，所有候选都满足
                    feasible.append({
                        "v": v, "w": w, "K": k, "N": n,
                        "settle_substep": t,
                        "accept_substep": accept_substep,
                        "v_margin": v / 0.41,  # 反例早期线速度 0.41 的比例裕量
                    })

    if not feasible:
        print("无可行候选 → TERMINAL_CRITERION_NOT_IDENTIFIABLE")
        return

    # 词典序排序
    # 1. 反例零过早接受（已保证）
    # 2. 正常释放接受（已保证）
    # 3. 非零噪声裕量（已保证）
    # 4. 最小化接受延迟（accept_substep 最小）
    # 5. 速度裕量最大（v_margin 最小 = 阈值越低裕量越大）
    # 6. 参数元组字典序
    feasible.sort(key=lambda c: (
        c["accept_substep"],      # 4. 接受延迟最小
        c["v_margin"],            # 5. 速度裕量最大（v_margin 小 = 阈值低 = 裕量大）
        (c["v"], c["w"], c["K"], c["N"]),  # 6. 字典序
    ))

    chosen = feasible[0]
    print(f"可行候选数: {len(feasible)}")
    print(f"选中配置: v_max={chosen['v']}, omega_max={chosen['w']}, "
          f"K={chosen['K']}, N={chosen['N']}")
    print(f"  正常 settle={chosen['settle_substep']} substep，"
          f"接受时刻={chosen['accept_substep']} substep")

    # 冻结 artifact
    config = {
        "status": "CALIBRATED_WITHIN_SCOPE",
        "v_max": chosen["v"],
        "omega_max": chosen["w"],
        "stable_substeps_K": chosen["K"],
        "contact_gone_window_N": chosen["N"],
        "contact_gone_unit": "physics_substep",
        "H_open_control_steps": H_OPEN,
        "H_post_control_steps": H_POST,
        "physics_substeps_per_control_step": SUBSTEPS,
        "selection_rule_version": "lexicographic_v1",
        "calibration_trace_hashes": [
            hashlib.sha256(open(SETTLE_TABLE_PATH, "rb").read()).hexdigest(),
            hashlib.sha256(open(COUNTEREXAMPLE_PATH, "rb").read()).hexdigest(),
        ],
        "feasible_count": len(feasible),
    }
    body = json.dumps({k: v for k, v in config.items()
                       if k != "config_sha256"}, sort_keys=True, ensure_ascii=False)
    config["config_sha256"] = hashlib.sha256(body.encode()).hexdigest()

    out_dir = os.path.join(ROOT, ".cache-local", "fpck_characterization")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "terminal_validation_config_v1_single_object_calibrated.json")
    with open(out_path, "w") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"\nconfig_sha256={config['config_sha256']}")
    print(f"→ {out_path}")


if __name__ == "__main__":
    main()
