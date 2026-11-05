"""G0 几何审计（修正版）：用正确的 MuJoCo 真值核验 R1 与 p1。

关键事实（robosuite 源码 single_arm.py:306-312）：
- obs["robot0_eef_pos"]  = sim.data.site_xpos[eef_site_id]   （grip_site）
- obs["robot0_eef_quat"] = convert_quat(body_xquat[eef_name]) （eef_name=robot0_right_hand）

即：eef_pos 来自 grip_site（site），eef_quat 来自 right_hand（body）。这是 AEGIS
上游的固有定义，p1 = eef_pos + R1 @ [0,0,-0.08] 是「grip_site 位置 + right_hand
姿态旋转的局部偏移」。

独立真值（MuJoCo mjData）：
- eef_pos 真值 = site_xpos[grip_site_id]
- R1 真值     = body_xmat[right_hand]（reshape 3x3）
- p1 真值     = eef_pos_truth + R1_truth @ EEF_OFFSET_LOCAL
"""

from __future__ import annotations
import csv
import os
import sys
import warnings

warnings.filterwarnings("ignore")

ROOT = "/data/safeVLA"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "dataset", "SafeLIBERO", "safelibero"))
sys.path.insert(0, os.path.join(ROOT, "third_party", "aegis"))

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

from libero.libero import benchmark  # noqa: E402
from src.safety.aegis_adapted import _quat_wxyz_to_rot, EEF_OFFSET_LOCAL  # noqa: E402
from tools.g1a_soup_trace import make_env  # noqa: E402

SUITE = "safelibero_long"
LEVEL = "I"
TASK_ID = 0
EEF_BODY = "robot0_right_hand"


def main():
    ts = benchmark.get_benchmark_dict()[SUITE](safety_level=LEVEL)
    task = ts.get_task(TASK_ID)

    rows = []
    max_p1_err = 0.0
    max_r1_err = 0.0
    max_eef_err = 0.0
    for state_id in range(50):
        env = make_env(ts, task, state_id)
        obs = env.env._get_observations()
        d = env.sim.data._data

        # AEGIS 路径
        eef_pos_aegis = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
        eef_quat_aegis = np.asarray(obs["robot0_eef_quat"], dtype=np.float64)
        R1_aegis = _quat_wxyz_to_rot(eef_quat_aegis)
        p1_aegis = eef_pos_aegis + R1_aegis @ EEF_OFFSET_LOCAL

        # 独立真值
        eef_site_id = env.env.robots[0].eef_site_id
        eef_pos_truth = np.asarray(d.site_xpos[eef_site_id]).copy()
        bid = env.sim.model.body_name2id(EEF_BODY)
        R1_truth = np.asarray(d.xmat[bid]).reshape(3, 3).copy()
        p1_truth = eef_pos_truth + R1_truth @ EEF_OFFSET_LOCAL

        r1_err = float(np.max(np.abs(R1_aegis - R1_truth)))
        eef_err = float(np.linalg.norm(eef_pos_aegis - eef_pos_truth))
        p1_err = float(np.linalg.norm(p1_aegis - p1_truth))
        max_p1_err = max(max_p1_err, p1_err)
        max_r1_err = max(max_r1_err, r1_err)
        max_eef_err = max(max_eef_err, eef_err)

        rows.append({
            "state_id": state_id,
            "eef_pos_err_mm": round(eef_err * 1000, 6),
            "R1_max_abs_err": round(r1_err, 9),
            "p1_err_mm": round(p1_err * 1000, 6),
        })
        env.close()

    out = os.path.join(ROOT, "experiments", "stg_v1", "audit", "geometry_audit.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"geometry_audit.csv 已写 {len(rows)} 行")
    print(f"max eef_pos_err = {max_eef_err*1000:.6f} mm")
    print(f"max R1_abs_err  = {max_r1_err:.3e}")
    print(f"max p1_err      = {max_p1_err*1000:.6f} mm")
    # 判定：p1 偏差 < 0.1mm（1e-4 m）视为几何一致（16cm 级偏差不可再现）
    verdict = "PASS" if max_p1_err < 1e-4 and max_r1_err < 1e-3 else "FAIL"
    print(f"几何审计 verdict: {verdict}")


if __name__ == "__main__":
    main()
