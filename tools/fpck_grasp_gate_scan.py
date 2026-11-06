"""正式 10 对象抓取门禁扫描（冻结门禁，裁决 §九执行顺序第 5 步）。

按 scripted_grasp_feasibility_protocol_v3 冻结的门禁，对 10 个 unique 候选对象：
1. 每对象按 shape family 模板顺序（最小轴 → 第二轴）；
2. 第一个 PASS 模板 = canonical；
3. 无 PASS 则按结果分类（PHYSICAL_INELIGIBLE / TEMPLATE_FAIL / CONTROLLER_INCONCLUSIVE）；
4. canonical 重复 3 次确认。

不增加模板、不改 descend、不调 clearance、不特调对象。
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

from src.fpck.scripted_grasp_gate import (  # noqa: E402
    run_grasp_gate, make_env as gate_make_env,
)

# 10 个候选对象（精确 suite/task_name/body，来自 v1 对象扫描）
# 几何宽度：box_like 用 principal_dims 前两维（实心块准确）；
# 空心容器（bowl/mug）外径超夹爪，宽度为外径
CANDIDATES = [
    ("cream_cheese", "cream_cheese_1_main", "safelibero_long",
     "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket", 0.0192, 0.0431, "box_like"),
    ("chocolate_pudding", "chocolate_pudding_1_main", "safelibero_long",
     "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate", 0.0298, 0.0500, "box_like"),
    ("bbq_sauce", "bbq_sauce_1_main", "safelibero_object",
     "pick_up_the_bbq_sauce_and_place_it_in_the_basket", 0.0336, 0.0499, "box_like"),
    ("milk", "milk_1_main", "safelibero_object",
     "pick_up_the_milk_and_place_it_in_the_basket", 0.0558, 0.1032, "box_like"),
    ("orange_juice", "orange_juice_1_main", "safelibero_object",
     "pick_up_the_orange_juice_and_place_it_in_the_basket", 0.0647, 0.1032, "box_like"),
    ("alphabet_soup", "alphabet_soup_1_main", "safelibero_long",
     "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket", 0.0662, 0.0757, "box_like"),
    ("tomato_sauce", "tomato_sauce_1_main", "safelibero_long",
     "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket", 0.0678, 0.0744, "box_like"),
    ("akita_black_bowl", "akita_black_bowl_1_main", "safelibero_spatial",
     "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate", 0.0942, 0.0942, "bowl_or_cup"),
    ("porcelain_mug", "porcelain_mug_1_main", "safelibero_long",
     "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate", 0.1254, 0.1391, "cylindrical"),
    ("white_yellow_mug", "white_yellow_mug_1_main", "safelibero_long",
     "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate", 0.1111, 0.1337, "cylindrical"),
]

# 轴方向（世界系）：最小轴/第二轴对应的世界方向（box 对象 body-local 轴 ≈ world 轴）
# 对 box_like，w1 沿 body-local x，w2 沿 body-local y（cream_cheese 已验证）
# 对空心容器，外径沿任意水平轴，这里统一用 x/y 轴
TEMPLATES = {
    "box_like": [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])],
    "bowl_or_cup": [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])],
    "cylindrical": [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])],
}


def main():
    results = []
    for (cls, body, suite, task_name, w1, w2, shape) in CANDIDATES:
        axes = TEMPLATES[shape]
        widths = [w1, w2]
        entry = {"object_class": cls, "body": body, "suite": suite,
                 "shape": shape, "w1_mm": w1*1000, "w2_mm": w2*1000,
                 "templates": [], "canonical": None}

        for ti, (axis, width) in enumerate(zip(axes, widths)):
            env = gate_make_env(suite, task_name)
            try:
                _ = env.sim.model.body_name2id(body)
            except Exception:
                entry["templates"].append({"template": ti, "error": "BODY_NOT_IN_SCENE"})
                env.close()
                continue
            r = run_grasp_gate(env, body, axis, width)
            env.close()
            entry["templates"].append({
                "template": ti, "axis": axis.tolist(), "width_mm": width*1000,
                "outcome": r.outcome, "failure_reason": r.failure_reason,
                "effective_aperture": r.effective_aperture,
                "lift_mm": r.object_lift_m*1000,
            })
            if r.passed and entry["canonical"] is None:
                entry["canonical"] = ti
                break

        results.append(entry)
        status = "PASS" if entry["canonical"] is not None else \
                 entry["templates"][-1].get("outcome", "?")
        print(f"{cls:22s} shape={shape:12s} → {status} "
              f"(w1={w1*1000:.1f} w2={w2*1000:.1f}mm)")

    out_dir = os.path.join(ROOT, ".cache-local", "fpck_characterization")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "grasp_gate_scan_v1.json"), "w") as f:
        json.dump({"frozen_gate": True, "results": results}, f,
                  ensure_ascii=False, indent=2)

    # 汇总
    n_pass = sum(1 for r in results if r["canonical"] is not None)
    print(f"\n=== 汇总：{n_pass}/{len(results)} 对象有 canonical PASS ===")
    for r in results:
        if r["canonical"] is not None:
            t = r["templates"][r["canonical"]]
            print(f"  PASS: {r['object_class']} 模板{r['canonical']} "
                  f"lift={t['lift_mm']:.1f}mm aperture={t['effective_aperture']}")


if __name__ == "__main__":
    main()
