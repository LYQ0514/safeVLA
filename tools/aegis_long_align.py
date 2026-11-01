"""AEGIS long 采集的时间对齐分析（纯读已采集数据，不跑新实验）。

对每个 episode，横轴 = 实际 control step（= 决策点 step，REPLAN_STEPS=5 为间隔），
对齐以下信号：
- 两个原生 goal predicate（soup / cream_cheese 是否 in basket）
- EEF–soup 距离（接近）
- soup 高度（抬升/搬运）
- soup–basket 距离（搬运进度）
- nominal 与 filtered 动作差值（AEGIS 修改量，仅决策点第 1 步的局部观测）
- 每个决策点第 1 步的干预 / QP 状态 / 活跃障碍

明确局限：只保存了决策点第 1 步的 filter info（aegis_first_filter），
执行前缀剩余 4 步的逐步干预未记录；干预统计用 episode 级 intervene_count 做补充，
不补猜剩余步骤。

重点分析窗口：cream_cheese 首次持续满足之后。逐 episode 报告剩余暴露步数、
干预比例、累计修正量，及 soup 的接近/抓取/抬升/搬运是否有变化。
两个 cream_cheese 未完成的 episode 单独呈现。

只报告连续数值与共现，不引入事后停滞阈值，不下因果结论。
"""

from __future__ import annotations

import glob
import json
import os
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")

ROOT = "/data/safeVLA"
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

DESIGN = pathlib.Path(ROOT, "data", "audit", "aegis_long_design")
REPLAN_STEPS = 5


def load_decision(ep, step):
    with np.load(DESIGN / f"ep{ep:02d}" / f"decision_{step:03d}.npz") as f:
        d = {k: f[k] for k in f.files}
    j = json.loads((DESIGN / f"ep{ep:02d}" / f"decision_{step:03d}.json").read_text())
    return d, j


def series(ep, key_fn):
    """沿决策点提取一个标量序列，返回 (steps, values)。"""
    steps, vals = [], []
    for jf in sorted(glob.glob(str(DESIGN / f"ep{ep:02d}" / "decision_*.json"))):
        j = json.loads(open(jf).read())
        d, _ = load_decision(ep, j["step"])
        v = key_fn(d, j)
        if v is not None:
            steps.append(j["step"])
            vals.append(v)
    return np.array(steps), np.array(vals)


def eef_soup_dist(d, j):
    if "robot0_eef_pos" in d and "alphabet_soup_1_pos" in d:
        return float(np.linalg.norm(d["robot0_eef_pos"] - d["alphabet_soup_1_pos"]))
    return None


def soup_height(d, j):
    if "alphabet_soup_1_pos" in d:
        return float(d["alphabet_soup_1_pos"][2])
    return None


def soup_basket_dist(d, j):
    if "alphabet_soup_1_pos" in d and "basket_1_pos" in d:
        return float(np.linalg.norm(d["alphabet_soup_1_pos"][:2] - d["basket_1_pos"][:2]))
    return None


def eef_cream_dist(d, j):
    if "robot0_eef_pos" in d and "cream_cheese_1_pos" in d:
        return float(np.linalg.norm(d["robot0_eef_pos"] - d["cream_cheese_1_pos"]))
    return None


def cream_height(d, j):
    if "cream_cheese_1_pos" in d:
        return float(d["cream_cheese_1_pos"][2])
    return None


def filter_mod(d, j):
    """nominal 第 1 步 与 filtered 第 1 步动作的差值（决策点第 1 步局部观测）。"""
    ff = j.get("aegis_first_filter")
    if ff is None or "action" not in ff:
        return None
    nom0 = d["nominal"][0][:6]
    fil0 = np.array(ff["action"])[:6]
    return float(np.linalg.norm(nom0 - fil0))


def intervened(d, j):
    ff = j.get("aegis_first_filter")
    if ff is None:
        return None
    return 1.0 if ff["info"].get("intervened") else 0.0


def soup_in(d, j):
    for s in j["subgoals"]:
        if s["args"][0] == "alphabet_soup_1":
            return 1.0 if s["value"] else 0.0
    return None


def cream_in(d, j):
    for s in j["subgoals"]:
        if s["args"][0] == "cream_cheese_1":
            return 1.0 if s["value"] else 0.0
    return None


def first_continuous_true(steps, vals):
    """找到 vals 首次为 True 且之后一直 True 的步数；无则 None。"""
    for i in range(len(vals)):
        if vals[i] > 0.5 and all(v > 0.5 for v in vals[i:]):
            return steps[i]
    return None


def analyze_ep(ep):
    epj = json.loads((DESIGN / f"ep{ep:02d}" / "episode.json").read_text())
    n_dec = epj["n_decisions"]
    n_int_total = epj["aegis_stats"]["intervened"]

    # 各信号序列
    s_in, v_in = series(ep, soup_in)
    c_in, v_in2 = series(ep, cream_in)
    st_ees, v_ees = series(ep, eef_soup_dist)
    st_sh, v_sh = series(ep, soup_height)
    st_sb, v_sb = series(ep, soup_basket_dist)
    st_ec, v_ec = series(ep, eef_cream_dist)
    st_ch, v_ch = series(ep, cream_height)
    st_mod, v_mod = series(ep, filter_mod)
    st_int, v_int = series(ep, intervened)

    cream_first = first_continuous_true(c_in, v_in2)

    # cream 首次持续满足之后的分析窗口
    if cream_first is not None:
        # 窗口内（cream_first 之后）的决策点索引
        win_idx = st_ees >= cream_first
        n_win = int(win_idx.sum())
        # 窗口内干预比例（用 episode 级 intervene_count 无法按步切分，
        # 用决策点第 1 步 intervened 序列近似）
        if len(v_int) > 0:
            int_win = v_int[np.searchsorted(st_int, cream_first):]
            win_int_ratio = float(np.mean(int_win)) if len(int_win) > 0 else 0.0
        else:
            win_int_ratio = None
        # 窗口内累计修正量
        mod_win = v_mod[np.searchsorted(st_mod, cream_first):]
        cum_mod = float(np.sum(mod_win)) if len(mod_win) > 0 else 0.0
        # 窗口内 soup 接近/抬升/搬运是否变化（首 vs 末）
        if len(v_ees) > 0:
            ee0 = v_ees[0]; ee_end = v_ees[-1]
            ees_delta = ee_end - ee0
        else:
            ees_delta = None
        if len(v_sh) > 0:
            sh_delta = v_sh[-1] - v_sh[0]
        else:
            sh_delta = None
        if len(v_sb) > 0:
            sb_delta = v_sb[-1] - v_sb[0]
        else:
            sb_delta = None
    else:
        n_win = 0
        win_int_ratio = None
        cum_mod = 0.0
        ees_delta = sh_delta = sb_delta = None

    # soup 全程是否接近过（EEF-soup 最小距离）
    ees_min = float(np.min(v_ees)) if len(v_ees) > 0 else None
    sb_min = float(np.min(v_sb)) if len(v_sb) > 0 else None
    sh_max = float(np.max(v_sh)) if len(v_sh) > 0 else None

    return {
        "ep": ep,
        "success": epj["success"],
        "steps": epj["steps"],
        "n_dec": n_dec,
        "n_int_total": n_int_total,
        "cream_first_cont": cream_first,
        "soup_ever_in": bool(np.any(v_in > 0.5)),
        "cream_ever_in": bool(np.any(v_in2 > 0.5)),
        "ees_min_m": ees_min,
        "sb_min_m": sb_min,
        "sh_max_m": sh_max,
        "cream_window_n": n_win,
        "cream_window_int_ratio": win_int_ratio,
        "cream_window_cum_mod": cum_mod,
        "cream_window_ees_delta": ees_delta,
        "cream_window_sh_delta": sh_delta,
        "cream_window_sb_delta": sb_delta,
        # 全序列存起来供时间线
        "steps_arr": st_ees.tolist(),
        "ees": v_ees.tolist(),
        "sh": v_sh.tolist(),
        "sb": v_sb.tolist(),
        "mod": v_mod.tolist(),
        "intv": v_int.tolist(),
    }


def main():
    results = [analyze_ep(ep) for ep in range(6)]

    print("=" * 78)
    print("AEGIS long 采集时间对齐分析（6 episode）")
    print("横轴 = control step；REPLAN_STEPS=5；干预/修改 = 决策点第 1 步局部观测")
    print("=" * 78)
    print()
    hdr = (f"{'ep':>3} {'soup':>5} {'cream':>5} {'cream首满足':>10} {'EEF-soup_min':>12} "
           f"{'soup-basket_min':>15} {'soup高max':>9} {'干预总数':>8}")
    print(hdr)
    print("-" * 78)
    for r in results:
        cream_first = f"{r['cream_first_cont']}" if r['cream_first_cont'] is not None else "无"
        print(f"{r['ep']:>3} {'✓' if r['soup_ever_in'] else '✗':>5} "
              f"{'✓' if r['cream_ever_in'] else '✗':>5} {cream_first:>10} "
              f"{r['ees_min_m']:.3f}:>12" if False else
              f"{r['ep']:>3} {'✓' if r['soup_ever_in'] else '✗':>5} "
              f"{'✓' if r['cream_ever_in'] else '✗':>5} {cream_first:>10} "
              f"{r['ees_min_m']:.3f}{'':>4} {r['sb_min_m']:.3f}{'':>6} "
              f"{r['sh_max_m']:.3f}{'':>3} {r['n_int_total']:>8}")

    print()
    print("=" * 78)
    print("cream_cheese 首次持续满足之后的分析窗口")
    print("=" * 78)
    print(f"{'ep':>3} {'窗口决策点':>10} {'窗口干预比例':>12} {'窗口累计修正':>12} "
          f"{'EEF-soup Δ':>11} {'soup高 Δ':>9} {'soup-basket Δ':>13}")
    print("-" * 78)
    for r in results:
        if r["cream_first_cont"] is None:
            print(f"{r['ep']:>3} {'(未完成cream)':>10}")
            continue
        def fmt(x, nd=4):
            return f"{x:.{nd}f}" if x is not None else "N/A"
        print(f"{r['ep']:>3} {r['cream_window_n']:>10} "
              f"{fmt(r['cream_window_int_ratio'], 3):>12} {fmt(r['cream_window_cum_mod']):>12} "
              f"{fmt(r['cream_window_ees_delta']):>11} {fmt(r['cream_window_sh_delta']):>9} "
              f"{fmt(r['cream_window_sb_delta']):>13}")

    # 保存
    out = DESIGN / "alignment_analysis.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str))
    print(f"\n已保存: {out}")


if __name__ == "__main__":
    main()
