"""GroundingDINO 本地加载器。

为什么不直接用 `groundingdino.util.inference.load_model`：
它的 config 里 `text_encoder_type = "bert-base-uncased"` 是 HF 仓库名，
加载时会联网拉 BERT 权重。容器内 HF 直连不可达，且每次运行都去请求网络不稳定。

这里改为：把 config 里的 text_encoder_type 指向项目内的本地 BERT 目录，
其余完全走上游 `build_model`，**不改模型结构、不改推理逻辑**。
"""

from __future__ import annotations

import os
import types

import groundingdino

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
WEIGHTS_DIR = os.path.join(_ROOT, "third_party", "groundingdino_weights")
CKPT_PATH = os.path.join(WEIGHTS_DIR, "groundingdino_swint_ogc.pth")
BERT_DIR = os.path.join(WEIGHTS_DIR, "bert-base-uncased")


def _load_config_with_local_bert():
    """读上游 config，把 text_encoder_type 换成本地路径，返回 SimpleNamespace。"""
    cfg_path = os.path.join(
        os.path.dirname(groundingdino.__file__), "config", "GroundingDINO_SwinT_OGC.py"
    )
    ns = {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        exec(compile(f.read(), cfg_path, "exec"), ns)  # noqa: S102 上游就是 .py 配置
    ns["text_encoder_type"] = BERT_DIR
    keep = {k: v for k, v in ns.items() if not k.startswith("__")}
    return types.SimpleNamespace(**keep)


def load_groundingdino(device="cuda"):
    """返回可直接用于 `predict()` 的 GroundingDINO 模型。"""
    from groundingdino.models import build_model
    from groundingdino.util.slconfig import SLConfig
    import torch

    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(
            f"缺少 GroundingDINO 权重: {CKPT_PATH}\n"
            "见 third_party/groundingdino_weights/README.md"
        )
    if not os.path.isdir(BERT_DIR):
        raise FileNotFoundError(
            f"缺少本地 BERT: {BERT_DIR}\n"
            "先运行 tools/setup_groundingdino.py 下载"
        )

    args = _load_config_with_local_bert()
    model = build_model(args)
    state = torch.load(CKPT_PATH, map_location="cpu")
    sd = state.get("model", state)
    # 官方 ckpt 是 DataParallel 保存的，key 带 "module." 前缀；不去掉则 strict=False
    # 会静默跳过全部参数（实测交集为 0），模型等于随机初始化 —— 检测输出会变成
    # 900 个框、置信度全 1.0 这种垃圾。这里显式剥前缀并校验。
    if any(k.startswith("module.") for k in sd):
        sd = {k[len("module."):]: v for k, v in sd.items()}

    missing, unexpected = model.load_state_dict(sd, strict=False)
    model_keys = set(model.state_dict().keys())
    loaded = len(model_keys & set(sd.keys()))
    if loaded < 0.95 * len(model_keys):
        raise RuntimeError(
            f"GroundingDINO 权重加载不完整：{loaded}/{len(model_keys)} 参数匹配，"
            f"缺失 {len(missing)} / 多余 {len(unexpected)}"
        )
    model.to(device)
    model.eval()
    return model
