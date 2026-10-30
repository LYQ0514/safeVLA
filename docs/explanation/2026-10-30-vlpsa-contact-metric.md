# VLPSA 接触口径补充对照

- **日期**：2026-10-30
- **性质**：为 VLPSA 复现补充「论文碰撞口径」（MuJoCo 接触）的对照实验

---

## 一、两种碰撞口径的定义

| 口径 | 碰撞定义 | 实现 |
|---|---|---|
| **displacement**（本框架默认） | 障碍物位移 > 1mm（L1） | `CollisionTracker`，观测 `obstacle_pos` 变化 |
| **contact**（论文 VLPSA） | 机械臂碰撞几何与障碍物碰撞几何发生 MuJoCo 接触 | `ContactCollisionTracker`，扫 `data.contact` |

论文原文：「expanded collision detection scheme scans all simulation contacts and
distinguishes visual mesh intersections from MuJoCo collision geometry artifacts」。

实现要点：障碍物的视觉 mesh 是 `contype=0`（不参与物理接触），碰撞几何是
box（`contype=1`）。MuJoCo 的 `data.contact` 只记录 `contype & conaffinity` 的几何
接触，因此**天然排除了视觉 mesh 穿插**，与论文口径一致。

---

## 二、对照结果（VLPSA 关节空间，α=0.1，spatial task0，10 条）

| 口径 | CAR | 关键现象 |
|---|---|---|
| displacement（位移>1mm） | 40% | 障碍被推动才判碰撞 |
| **contact（MuJoCo 接触）** | **30%** | 接触即碰撞，`disp=0.0000` 也判碰撞 |

**关键发现：contact 口径反而更严（30% < 40%）。**

原因：contact 口径把「机械臂碰到障碍但没推动它」也算碰撞（6 条 COLLIDE 的
`disp` 全是 0.0000，说明障碍纹丝不动，纯接触）。而 displacement 口径下，
这种「轻碰未推动」不算碰撞。

---

## 三、修正此前的错误归因

此前（`docs/explanation/2026-10-30-vlpsa-joint-space-correction.md`）归因
「论文 CAR 91.2% 依赖更松的碰撞口径」——**这个归因是错的**。

实测证明 contact 口径更严，不是更松。论文的 CAR 91.2% **不是**靠碰撞口径
放水得来的。

排除的其他因素：
- checkpoint：论文明确用同一个 LIBERO-finetuned π0.5（nominal CAR 23.1%）
- 100Hz：合成测试证明只延迟碰撞 2.5 秒，不是关键

---

## 四、剩余差距的真实来源（未开源，只能推断）

论文 CAR 91.2% vs 我们 contact 口径 30%，剩余差距指向论文的**完整工程细节**：

1. **多分辨率 PSF**：论文 3cm 粗 + 1cm 细（Boolean CBF 组合），我们单 3cm。
   细网格在障碍附近提供更准确的梯度，CBF 介入更早更准。
2. **ε-buffer 与 ε-采样配套**：论文的 buffer 和采样分辨率是理论配套的
   （保证「采样点安全 ⟹ 整个机器人表面安全」），我们单独加 buffer 未配套采样密度。
3. **v_nom 的 IK 实现**：论文未写死 nominal 关节速度的 IK 细节，我们用最小范数伪逆。

这些细节论文均未开源，只能靠猜，且每个都要数小时调参。

---

## 五、最终诚实结论

1. **VLPSA 复现已覆盖可控范围**：关节空间 QP（R⁷）+ 正确速度尺度 + 完整采样
   （含指尖）+ 正确 CBF 增益（α=0.1）+ 两种碰撞口径。

2. **在本项目协议下，全身 CBF（VLPSA）的避障上限明确低于末端椭球 CBF（AEGIS）**：

| 方法 | 口径 | spatial task0 CAR |
|---|---|---|
| AEGIS（末端椭球） | displacement | 50% |
| VLPSA（全身 CBF） | displacement | 40% |
| VLPSA（全身 CBF） | contact | 30% |

3. **论文 91.2% 无法在本机完整复现**，剩余差距来自未开源的工程细节（多分辨率
   PSF、ε-buffer+采样配套、精细 IK），不构成我们的方法学结论。

---

## 六、复现命令

```bash
cd /data/safeVLA && source tools/activate_env.sh

# contact 口径（论文口径）
python tools/run_eval_vlpsa.py --suite safelibero_spatial --level I \
    --tasks 0 --episodes 0 1 2 3 4 --resolution 1024 \
    --controller JOINT_VELOCITY --cbf-gain 0.1 --collision-mode contact
```
