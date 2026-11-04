# AEGIS 复现协议身份锁定（审计第 1 条）

> 日期：2026-11-04。记录完整复现所需的环境/代码/模型身份，供可追溯复现。

---

## 1. 代码身份

| 组件 | 身份 | 状态 |
|---|---|---|
| 适配代码（safeVLA） | Git SHA `8944837d...` | ⚠️ **已过期**：本轮大量修复（四元数/P0-B/P0-D/P1/感知链）未 commit |
| 上游 AEGIS | THU-RCSCT/vlsa-aegis，论文 arXiv 2512.11891（IROS 2026） | vendored 2026-10-29，未锁定 commit SHA |
| SafeLIBERO fork | 本机 `/data/safeVLA/dataset/SafeLIBERO` | 需记录 init states hash |

**待办**：
1. commit 本轮修复（四元数 bug、几何 MVEE、replan-steps、统计 bug、感知链）；
2. 锁定上游 AEGIS 的确切 commit SHA（当前只记录了论文，未记录代码 SHA）；
3. 计算 SafeLIBERO init states 的 hash。

## 2. 模型身份

| 组件 | 值 |
|---|---|
| π0.5 checkpoint | `/data/2608.SDL/MojocoProj/checkpoints/pi05/pi05_libero`（12GB，ocdbt 格式） |
| checkpoint hash | ⏳ 待计算（12GB 流式 hash，需数分钟） |
| 模型 config | `pi05_libero`（openpi），model.pi05=True |

## 3. 依赖版本

| 包 | 版本 |
|---|---|
| Python | 3.8.18（.venv，模拟器）/ 3.11（openpi worker） |
| numpy | 1.22.4 |
| scipy | 1.10.1 |
| cvxpy | 1.5.2 |
| mujoco | 2.3.7 |
| robosuite | 1.4.1 |
| open3d | 0.19.0 |
| zhipuai | 2.1.5.20250725 |
| GroundingDINO | swint_ogc（权重在 `third_party/groundingdino_weights/`） |

## 4. 运行参数（对齐后）

| 参数 | 值 |
|---|---|
| 渲染分辨率 | 1024 |
| 策略输入分辨率 | 224（π0.5 训练分辨率） |
| num_steps_wait | 20 |
| replan_steps | 5 |
| max_steps | spatial/object/goal=300，long=550 |
| 碰撞判定 | 障碍物位移 > 1mm |
| CBF 参数 | α(h)=10h、动作缩放 0.2、参考速度增益 5、末端椭球 [0.06,0.12,0.11]（长物体 [0.06,0.12,0.20]）、末端偏移 -0.08 |

## 5. 两个方法变体（须分开报告）

| 变体 | 感知 | 表述 |
|---|---|---|
| AEGIS-QP (Oracle Perception) | MuJoCo 真值 MVEE | oracle |
| AEGIS (Perception) | VLM+GroundingDINO+MVEE | 与上游论文一致 |

## 6. 凭据

GLM API key 存于 `.env`（chmod 600，git 忽略），代码通过环境变量读取，不硬编码。

## 7. 关键依赖的确定性风险

- **π0.5 跨 run GPU 非确定**（历史记录：名义动作差约 0.06）——评测需固定 seed 或
  用确定性 PnsgPolicy + CRN；
- **VLM 输出非确定**（temperature=0.1，但仍有随机性）——perception 变体需记录
  VLM 输出到日志；
- **MVEE 的 SCS 求解器数值容差**——已加 1.001 保守缩放保证包围。
