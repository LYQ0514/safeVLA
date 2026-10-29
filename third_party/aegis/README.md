# AEGIS（vendored）

来源：https://github.com/THU-RCSCT/vlsa-aegis （MIT License）
论文：VLSA / AEGIS, arXiv 2512.11891, IROS 2026
vendored 日期：2026-10-29
vendored 文件：`main_aegis.py`、`utils.py`、`AEGIS_requirements.txt`（原样拷贝）

## 为什么 vendor 而不是 clone

需要做**最小改动**才能在本机跑，改了什么必须可追溯（见 `PATCHES.md`）。
原样保留上游文件，我们的改动写在 `../../src/safety/aegis_adapted.py`。

## 与上游的差异（务必在报告中注明）

| 项 | 上游 | 我们 | 原因 |
|---|---|---|---|
| 障碍物识别 | 智谱 GLM-4.5v 云端 API | **oracle（仿真真值）** | 上游 api_key 为空，且需付费+外网 |
| `eef_marker` body | 依赖仓库自带场景 XML | 删除（仅可视化） | 公开版 SafeLIBERO 无该 body |
| π0.5 客户端 | websocket server | 复用本项目 Pi05Policy | 我们用文件 IPC |
| 运行框架 | 单脚本 | 接入统一评测框架 | 便于与其他方法同条件对比 |

**这些差异意味着：我们跑出的数字不能直接与论文的 77.85% 对比**，
只能作为「AEGIS 控制层在本机条件下的表现」。见 `../../../docs/explanation/`。
