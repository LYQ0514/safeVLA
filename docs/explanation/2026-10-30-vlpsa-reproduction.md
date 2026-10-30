# VLPSA 复现（方案 B：核心思想复现）

- **日期**：2026-10-30
- **上游**：VLPSA (arXiv 2609.22462)，Caltech Ames 组，无代码
- **复现方式**：方案 B —— 复现两个核心创新（PSF + 全身采样），协议对齐到本项目
- **状态**：已跑通，单条/批量验证通过，全量 1600 条进行中

---

## 一、VLPSA 方法原理（读论文理解）

VLPSA 是**全身安全滤波器**，四个组件：

1. **Poisson Safety Function（PSF）**：从 occupancy grid 解 Poisson 方程
   ∇²h = f（f<0），Dirichlet 边界 h=0 在障碍表面。得到全局平滑 SDF：
   自由空间 h>0，障碍表面 h=0。**替代 AEGIS 的椭球近似，可表达任意形状。**

2. **全身采样**：在机械臂多个 link 上采样点（论文 679 点），每个点一条 CBF 约束。
   替代 AEGIS 的单末端椭球。

3. **joint-velocity CBF-QP**：在关节速度空间解 QP，假设 100Hz 内环。

4. **持物保护**：抓取后把物体包围体加入采样集。

论文报告：CAR 91.2%（vs nominal 23.1%）、SR 55.5%（vs 49.1%），
声称「increasing collision avoidance while surpassing task success rate」。

---

## 二、方案 B 的复现范围

按用户决策，复现 PSF + 全身采样两个核心创新，协议对齐本项目：

| 项 | 论文 | 我们（方案 B） | 原因 |
|---|---|---|---|
| 控制 | 100Hz joint-velocity | 20Hz OSC 末端速度 | 与基线/AEGIS 一致 |
| 感知 | 深度传感器 OctoMap | MuJoCo 几何 oracle | 与 AEGIS 的 oracle 一致 |
| 碰撞判定 | 扫描所有接触 | 位移法（1mm） | 统一框架 |
| 采样点 | 679 | 815 | 8 link × mesh AABB 角点+面心 |
| 持物保护 | 有 | **暂未实现** | 先跑通主链路 |

**因此数字不可与论文 91.2% 直接对比**，只与本项目 π0.5/AEGIS 在同一协议下对比。

---

## 三、技术实现

### 3.1 PSF（`src/safety/psf.py`）

- occupancy：向量化体素化 MuJoCo 障碍物 geom（box 精确，mesh 用 size 当半轴近似）
- Poisson：GPU 共轭梯度，拉普拉斯用 3D 卷积（6 邻接）
- 查询：三线性插值 h(y) + 中心差分 ∇h(y)

性能（0.03m 网格，RTX 4090）：

| 步骤 | 耗时 |
|---|---|
| occupancy 生成 | 29ms |
| Poisson 求解（200 iter） | 7ms |
| filter（815 点 QP） | 12.5ms |

### 3.2 全身采样（`src/safety/vlpsa_layer.py`）

- 8 link 的碰撞 mesh，采样 AABB 8 角 + 6 面心 = 815 点
- 雅可比用 `mj_jac`（已验证与 `ctrl.J_pos` 一致，差 0.00000）
- QP：`min ∥v−v_nom∥² s.t. ∇h(y_i)ᵀ(J_i J_ee⁺)v ≥ −γh(y_i)`

---

## 四、关键踩坑：Neumann 边界（决定性 bug）

**现象**：PSF 的 h 场值太小（全场 max 仅 0.017），CBF 约束过度严格，
机械臂被压死，任务全部失败（TSR 0%）。

**根因**：`conv3d` 的 zero-padding 会**隐式把工作空间边界当作 h=0 的
Dirichlet 边界**。机器人的 link 采样点分布在工作空间边缘（z 到 1.5），
被这个伪边界拉低 h。

**修复**：改用 replicate padding（Neumann 边界 ∂h/∂n=0）：

```python
hp = F.pad(h.view(1,1,*shape), (1,1,1,1,1,1), mode="replicate")
lap = F.conv3d(hp, kernel, padding=0)
```

**效果**：h 场 max 从 0.017 → 0.21（12 倍），单条 episode 从 FAIL 翻转为
OK clean（TSR/CAR/SSR 全 100%）。

**教训**：数值求解 Poisson 方程时，边界条件的选择直接决定解的尺度，
进而决定 CBF 的行为。zero-padding ≠ 自由边界，它是 Dirichlet h=0。

---

## 五、初步结果（spatial task0，γ=1）

spatial task0 是「必碰撞」任务（基线 CAR=0%，障碍物 moka pot 紧贴碗）：

| 方法 | TSR | CAR | CAR>5mm | SSR |
|---|---|---|---|---|
| π0.5 基线 | 44% | 0% | — | — |
| VLPSA（10 条） | **50%** | **30%** | **100%** | 10% |

- CAR 从 0% → 30%（安全大幅提升）
- TSR 从 44% → 50%（任务反而略升）
- CAR>5mm 100%（无严重碰撞）

**这正是「兼顾 TSR 和 CAR」的体现**，与论文声称的方向一致。

γ 扫描（5 条/组）：

| γ | TSR | CAR | 判断 |
|---|---|---|---|
| 0.5 | 0% | 40% | 太严 |
| 1 | 50% | 30% | 平衡 |
| 2 | 40% | 40% | 平衡 |
| 5 | 20% | 0% | 太松 |

---

## 六、待办

1. 全量 1600 条（1024，进行中，约 5 小时）
2. 与 π0.5 基线、AEGIS 三方对比
3. 补持物保护（grasp-triggered sample injection）
4. 网格分辨率消融（0.03 vs 0.02m）
