# GroundingDINO 权重（不入 git）

AEGIS 感知链路需要。文件较大，已加入 `.gitignore`。

## 需要两个东西

| 文件/目录 | 大小 | 来源 |
|---|---|---|
| `groundingdino_swint_ogc.pth` | 662 MB | [IDEA-Research/GroundingDINO v0.1.0-alpha](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth) |
| `bert-base-uncased/` | 438 MB | HuggingFace `bert-base-uncased`（本地化，见下）|

## 重建步骤

```bash
cd /data/safeVLA && source tools/activate_env.sh
proxyOn                                  # HF / GitHub 都需要代理

# 1) GroundingDINO 权重
curl -L -o third_party/groundingdino_weights/groundingdino_swint_ogc.pth \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# 2) BERT 本地化（GroundingDINO 的文本编码器）
python -c "
from transformers import BertModel, AutoTokenizer, AutoConfig
out='third_party/groundingdino_weights/bert-base-uncased'
BertModel.from_pretrained('bert-base-uncased').save_pretrained(out)
AutoTokenizer.from_pretrained('bert-base-uncased').save_pretrained(out)
AutoConfig.from_pretrained('bert-base-uncased').save_pretrained(out)
"
proxyOff
```

## ⚠️ 踩过的坑：权重静默不加载

**现象**：检测返回 900 个框、置信度全是 1.000，phrases 里全是 `[SEP]`。

**根因**：官方 ckpt 用 `DataParallel` 保存，key 带 `module.` 前缀。
直接 `load_state_dict(sd, strict=False)` 时 **`strict=False` 会静默跳过全部 940 个参数**
（与模型参数的 key 交集为 0），模型实际是**随机初始化**。
不报错、不警告，只是输出垃圾。

**修复**：剥掉 `module.` 前缀，并**校验匹配率**（`src/safety/groundingdino_loader.py`）：

```python
if any(k.startswith("module.") for k in sd):
    sd = {k[len("module."):]: v for k, v in sd.items()}
missing, unexpected = model.load_state_dict(sd, strict=False)
loaded = len(model_keys & set(sd.keys()))
if loaded < 0.95 * len(model_keys):
    raise RuntimeError(...)
```

**正常后的表现**（spatial task0，1024 分辨率）：

| prompt | 框数 | 最高置信度 |
|---|---|---|
| `blue moka pot.` | 1 | 0.361 |
| `moka pot.` | 2 | 0.373 |
| `black bowl. plate. obstacle.` | 4 | 0.594 |
