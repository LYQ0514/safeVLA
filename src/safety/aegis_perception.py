"""AEGIS 完整感知链（VLM 识别 + GroundingDINO + 深度点云 + MVEE）。

复现上游 `main_aegis.py` 184-233 行的 obstacle 感知链路，用：
- zhipuai 官方 SDK 替代上游的 `from zai import ZhipuAiClient`（pypi 的 zai 是空包）；
- 本地 GroundingDINO 权重 + 本地 BERT（`src/safety/groundingdino_loader.py`）；
- 上游 `utils.py` 的 `get_point_cloud` / `filtering_points` / `fit_ellipse`。

与 oracle（`obstacle_ellipsoid`）是**两个不同的方法变体**，须在报告中分开：
- oracle：MuJoCo 真值 AABB；
- perception：VLM+GroundingDINO+MVEE（与上游论文一致，但 VLM 结果有不确定性）。
"""

from __future__ import annotations

import base64
import os
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "third_party", "aegis"))

import numpy as np  # noqa: E402


def _load_api_key():
    with open(os.path.join(ROOT, ".env")) as f:
        for line in f:
            if line.startswith("GLM_API_KEY="):
                return line.strip().split("=", 1)[1]
    raise RuntimeError("GLM_API_KEY 未在 .env 中配置")


def obstacle_detection_vlm(agentview_img, instruction, task_suite_name):
    """GLM-4.5V 识别「最可能被碰的 obstacle」，返回颜色+物体类型（如 "red mug"）。

    上游用 `from zai import ZhipuAiClient`（私有封装），改用官方 `zhipuai.ZhipuAI`。
    """
    from zhipuai import ZhipuAI
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    client = ZhipuAI(api_key=_load_api_key())
    # 用进程唯一临时路径，避免多进程并发写同一文件损坏图片（LI/LII 并行时竞态）
    tmp_img = f"/tmp/aegis_obstacle_detection_{os.getpid()}.png"
    plt.imsave(tmp_img, agentview_img)
    try:
        with open(tmp_img, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
    finally:
        try:
            os.remove(tmp_img)
        except OSError:
            pass

    prefer_list = ['yellow rectangular book', 'blue moka pot', 'red mug',
                   'white storage box', 'black wine bottle', 'red milk carton']
    if "long" in task_suite_name:
        prefer_list.append('gray rectangular binder')

    response = client.chat.completions.create(
        model="glm-4.5v",
        messages=[{
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": (
                    f"The robot must follow this instruction: {instruction}. Based on both "
                    f"the instruction and the image, identify exactly one non-robot object "
                    f"that is most likely to obstruct the robot's motion during task execution. "
                    f"You must output a uniquely identifiable obstacle name including both "
                    f"color and object type, preferably from this list when applicable: "
                    f"{prefer_list}. Output only the object name, with no additional words."
                )},
            ],
            "role": "user",
        }],
        temperature=0.1,
        top_p=0.1,
    )
    clean = (response.choices[0].message.content
             .replace("<|begin_of_box|>", "").replace("<|end_of_box|>", ""))
    return clean.strip()


def fit_obstacle_from_perception(env, obs, task_language, task_suite_name,
                                 save_dir=None):
    """完整感知链：VLM 识别 → GroundingDINO → 深度点云 → filtering → MVEE。

    返回 (p2, R2, Q2_diag, obstacle_vlm_name)。
    若任一步失败（无检测/无点），返回 None（= 安全层关闭，退化为 baseline）。
    """
    from src.safety.groundingdino_loader import load_groundingdino
    from utils import get_point_cloud, filtering_points, fit_ellipse

    # 上游需要深度图（env 须 camera_depths=True）
    for key in ("agentview_depth", "backview_depth"):
        if key not in obs:
            raise RuntimeError(
                f"感知模式需要深度图观测 {key}，请用 camera_depths=True 创建 env")

    save_path = pathlib.Path(save_dir) if save_dir else pathlib.Path(
        f"/tmp/aegis_perception_{os.getpid()}")
    save_path.mkdir(parents=True, exist_ok=True)

    # 1. VLM 识别
    agentview_img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    vlm_name = obstacle_detection_vlm(agentview_img, task_language, task_suite_name)

    # 2. GroundingDINO
    model = load_groundingdino(device="cuda")

    # 3. agentview + backview 点云
    agentview_depth = np.ascontiguousarray(obs["agentview_depth"][::-1, ::-1])
    agent_pts = get_point_cloud(agentview_img, agentview_depth, env, "agentview",
                                vlm_name, model, save_path)

    backview_img = np.ascontiguousarray(obs["backview_image"][::-1, ::-1])
    backview_depth = np.ascontiguousarray(obs["backview_depth"][::-1, ::-1])
    back_pts = get_point_cloud(backview_img, backview_depth, env, "backview",
                               vlm_name, model, save_path)

    if agent_pts.shape[1] > 0 and back_pts.shape[1] > 0:
        full = np.vstack([agent_pts, back_pts])
    elif agent_pts.shape[1] > 0:
        full = agent_pts
    elif back_pts.shape[1] > 0:
        full = back_pts
    else:
        return None, vlm_name

    filter_pts = filtering_points(full, task_suite_name)
    if filter_pts.shape[0] == 0:
        return None, vlm_name

    p2, R2, Q2 = fit_ellipse(filter_pts)
    return (np.asarray(p2, dtype=np.float64),
            np.asarray(R2, dtype=np.float64),
            np.asarray(Q2, dtype=np.float64)), vlm_name
