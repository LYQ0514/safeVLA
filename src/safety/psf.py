"""Poisson Safety Function（PSF）—— VLPSA 核心思想的复现。

参考：VLPSA (arXiv 2609.22462) 及 PSF 原始论文（RSS 2025, [14]）。
在方案 B（核心思想复现）下，本模块负责：

1. 从 SafeLIBERO 环境生成 occupancy grid（oracle：直接用 MuJoCo 几何，替代深度传感器）
2. 在 GPU 上解 Poisson 方程 ∇²h = f（f<0），Dirichlet 边界 h=0 在障碍表面
   得到平滑的 SDF 场 h：自由空间 h>0，障碍表面 h=0
3. 提供 h(y) 与 ∇h(y) 的查询（三线性插值），供 CBF-QP 使用

与 AEGIS 的椭球 CBF 的本质区别：
- AEGIS：单个解析椭球，只能表达凸的近似几何
- PSF：从任意 occupancy 场解出全局平滑 SDF，能表达任意形状
"""

from __future__ import annotations

import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")

# 工作空间范围（SafeLIBERO 桌面场景，单位米）
WS_CENTER = np.array([0.0, 0.0, 1.05])
WS_HALF = np.array([0.55, 0.55, 0.30])  # x∈[-0.55,0.55], y∈[-0.55,0.55], z∈[0.75,1.35]


class OccupancyGrid:
    """3D 体素占用栅格。origin=世界坐标最低角，spacing=体素边长（米）。"""

    def __init__(self, origin, spacing, occupied):
        self.origin = np.asarray(origin, dtype=np.float64)
        self.spacing = float(spacing)
        self.occupied = occupied  # bool 数组 [Nx, Ny, Nz]
        self.shape = occupied.shape

    def world_to_grid(self, pts):
        """世界坐标 (...,3) → 体素整数索引。"""
        return np.floor((np.asarray(pts) - self.origin) / self.spacing).astype(int)


def build_occupancy_grid(env, spacing=0.03, buffer=0.0):
    """从 MuJoCo 场景生成 occupancy grid（oracle）。

    只标记**环境障碍物**（obstacle objects），不标记机器人自身、被操作的目标物体
    （bowl/plate 等）和大型家具。理由：与 AEGIS 对比须公平 —— AEGIS 也只约束
    obstacle objects，不约束 table/cabinet。

    buffer：occupancy ε-膨胀（论文 [16] 的核心）。障碍物体素向外膨胀 buffer 米，
    使 PSF 的 h 场梯度更早、更平滑地出现，保证"采样点安全 ⟹ 机器人表面安全"。

    几何处理：
    - box geom（mjGEOM_BOX=6）：精确体素化（向量化）
    - mesh geom（mjGEOM_MESH=7）：用世界 AABB 保守近似
    """
    model = env.sim.model
    data = env.sim.data

    origin = WS_CENTER - WS_HALF
    shape = np.ceil(2 * WS_HALF / spacing).astype(int)
    occupied = np.zeros(shape, dtype=bool)

    obstacle_names = set(
        n.replace("_joint0", "") for n in model.joint_names if "obstacle" in n
    )

    # 所有体素中心坐标（一次生成，向量化）
    xs = origin[0] + (np.arange(shape[0]) + 0.5) * spacing
    ys = origin[1] + (np.arange(shape[1]) + 0.5) * spacing
    zs = origin[2] + (np.arange(shape[2]) + 0.5) * spacing
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    centers = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)  # (Nvox, 3)

    def rasterize_box(center, xmat, half):
        """向量化：标记 box 内所有体素。"""
        R = xmat.T  # world→local
        pl = (centers - center) @ R  # (Nvox,3) 局部坐标
        inside = np.all(np.abs(pl) <= half, axis=1)
        occupied.ravel()[inside] = True

    for g in range(model.ngeom):
        bn = model.body_names[model.geom_bodyid[g]]
        if not any(bn.startswith(o) for o in obstacle_names):
            continue
        center = data.geom_xpos[g].copy()
        xmat = data.geom_xmat[g].copy().reshape(3, 3)
        half = model.geom_size[g].copy()
        # box(6) 精确；mesh(7) 用 size(scale) 当半轴保守近似
        if model.geom_type[g] in (6, 7):
            rasterize_box(center, xmat, half)

    # occupancy ε-膨胀（论文 [16] 的 buffer）
    if buffer > 0:
        occupied = _dilate_occupied(occupied, buffer / spacing)

    return OccupancyGrid(origin, spacing, occupied)


def _dilate_occupied(occupied, radius_voxels):
    """对 bool 占用体素做球形膨胀（近似用 iterated 6-邻接膨胀）。"""
    r = max(1, int(round(radius_voxels)))
    out = occupied.copy()
    for _ in range(r):
        out = out | np.roll(out, 1, axis=0) | np.roll(out, -1, axis=0) \
                 | np.roll(out, 1, axis=1) | np.roll(out, -1, axis=1) \
                 | np.roll(out, 1, axis=2) | np.roll(out, -1, axis=2)
    return out


def solve_psf(occ, f_value=-1.0, max_iter=100, device="cuda"):
    """在 GPU 上解 Poisson 方程 ∇²h = f，返回 h 场（与 occ 同形状的 torch 张量）。

    数值方法：共轭梯度（CG），拉普拉斯算子用 3D 卷积（6 邻接）实现。
    occupied 体素固定 h=0（Dirichlet 边界），free 体素解方程。
    """
    occ_t = torch.as_tensor(occ.occupied, dtype=torch.bool, device=device)
    free = ~occ_t
    n = occ_t.numel()

    # 拉普拉斯核：中心 -6，六邻 +1
    lap_kernel = torch.zeros(1, 1, 3, 3, 3, device=device)
    lap_kernel[0, 0, 1, 1, 1] = -6
    lap_kernel[0, 0, 1, 1, 0] = 1
    lap_kernel[0, 0, 1, 1, 2] = 1
    lap_kernel[0, 0, 1, 0, 1] = 1
    lap_kernel[0, 0, 1, 2, 1] = 1
    lap_kernel[0, 0, 0, 1, 1] = 1
    lap_kernel[0, 0, 2, 1, 1] = 1

    def lap(h):
        # replicate padding 实现 Neumann 边界（∂h/∂n=0），避免工作空间边界被隐式当 h=0
        hp = torch.nn.functional.pad(
            h.view(1, 1, *occ_t.shape), (1, 1, 1, 1, 1, 1), mode="replicate"
        )
        return torch.nn.functional.conv3d(hp, lap_kernel, padding=0).view(*occ_t.shape)

    # 全网格 mask CG：occupied 体素通过乘 free 强制保持 0，
    # CG 在 free 子空间上收敛。A = -free∘lap∘free（正定）。
    def matvec(h):
        hf = h * free.float()
        return -(lap(hf) * free.float())

    b = torch.full(occ_t.shape, -f_value * (occ.spacing ** 2),
                   device=device, dtype=torch.float32)
    b = b * free.float()  # occupied 位置 b=0

    x = torch.zeros(occ_t.shape, device=device, dtype=torch.float32)

    r = b - matvec(x)
    p = r.clone()
    rsold = (r * r).sum()
    for _ in range(max_iter):
        Ap = matvec(p)
        alpha = rsold / (p * Ap).sum()
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = (r * r).sum()
        if rsnew < 1e-14:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew

    h = x  # occupied 位置已恒为 0
    return h


class PSF:
    """查询 PSF 场：h(y) 与 ∇h(y)，三线性插值。"""

    def __init__(self, occ, h_field):
        self.occ = occ
        self.h = h_field  # torch 张量，同 occ 形状，在 GPU
        self.spacing = occ.spacing
        self.origin = torch.as_tensor(occ.origin, dtype=torch.float32,
                                      device=h_field.device)

    def _local(self, pts):
        """世界坐标 → 连续体素坐标。pts: (...,3) torch。"""
        return (pts - self.origin) / self.spacing

    def query(self, pts):
        """返回 h(pts), grad(pts)。pts: (M,3) torch（GPU）。"""
        p = self._local(pts)  # 连续坐标
        shape = torch.tensor(self.h.shape, device=p.device, dtype=torch.float32)
        clamped = torch.clamp(p, torch.zeros_like(shape), shape - 1 - 1e-3)
        i0 = clamped.floor().long()
        i1 = (i0 + 1).clamp(max=torch.tensor(self.h.shape, device=p.device).long() - 1)
        frac = clamped - i0.float()

        def idx(ix, iy, iz):
            return self.h[ix, iy, iz]

        h = (
            idx(i0[:, 0], i0[:, 1], i0[:, 2]) * (1 - frac[:, 0]) * (1 - frac[:, 1]) * (1 - frac[:, 2])
            + idx(i1[:, 0], i0[:, 1], i0[:, 2]) * frac[:, 0] * (1 - frac[:, 1]) * (1 - frac[:, 2])
            + idx(i0[:, 0], i1[:, 1], i0[:, 2]) * (1 - frac[:, 0]) * frac[:, 1] * (1 - frac[:, 2])
            + idx(i1[:, 0], i1[:, 1], i0[:, 2]) * frac[:, 0] * frac[:, 1] * (1 - frac[:, 2])
            + idx(i0[:, 0], i0[:, 1], i1[:, 2]) * (1 - frac[:, 0]) * (1 - frac[:, 1]) * frac[:, 2]
            + idx(i1[:, 0], i0[:, 1], i1[:, 2]) * frac[:, 0] * (1 - frac[:, 1]) * frac[:, 2]
            + idx(i0[:, 0], i1[:, 1], i1[:, 2]) * (1 - frac[:, 0]) * frac[:, 1] * frac[:, 2]
            + idx(i1[:, 0], i1[:, 1], i1[:, 2]) * frac[:, 0] * frac[:, 1] * frac[:, 2]
        )
        # 梯度：中心差分
        eps = 0.5
        gx = (self.query_axis(p, 0, +eps) - self.query_axis(p, 0, -eps)) / (2 * eps) / self.spacing
        gy = (self.query_axis(p, 1, +eps) - self.query_axis(p, 1, -eps)) / (2 * eps) / self.spacing
        gz = (self.query_axis(p, 2, +eps) - self.query_axis(p, 2, -eps)) / (2 * eps) / self.spacing
        grad = torch.stack([gx, gy, gz], dim=-1)
        return h, grad

    def _q(self, p):
        """标量查询（用于差分）。"""
        shape = torch.tensor(self.h.shape, device=p.device, dtype=torch.float32)
        clamped = torch.clamp(p, torch.zeros_like(shape), shape - 1 - 1e-3)
        i0 = clamped.floor().long()
        i1 = (i0 + 1).clamp(max=torch.tensor(self.h.shape, device=p.device).long() - 1)
        frac = clamped - i0.float()
        return (
            self.h[i0[:, 0], i0[:, 1], i0[:, 2]] * (1 - frac[:, 0]) * (1 - frac[:, 1]) * (1 - frac[:, 2])
            + self.h[i1[:, 0], i0[:, 1], i0[:, 2]] * frac[:, 0] * (1 - frac[:, 1]) * (1 - frac[:, 2])
            + self.h[i0[:, 0], i1[:, 1], i0[:, 2]] * (1 - frac[:, 0]) * frac[:, 1] * (1 - frac[:, 2])
            + self.h[i1[:, 0], i1[:, 1], i0[:, 2]] * frac[:, 0] * frac[:, 1] * (1 - frac[:, 2])
            + self.h[i0[:, 0], i0[:, 1], i1[:, 2]] * (1 - frac[:, 0]) * (1 - frac[:, 1]) * frac[:, 2]
            + self.h[i1[:, 0], i0[:, 1], i1[:, 2]] * frac[:, 0] * (1 - frac[:, 1]) * frac[:, 2]
            + self.h[i0[:, 0], i1[:, 1], i1[:, 2]] * (1 - frac[:, 0]) * frac[:, 1] * frac[:, 2]
            + self.h[i1[:, 0], i1[:, 1], i1[:, 2]] * frac[:, 0] * frac[:, 1] * frac[:, 2]
        )

    def query_axis(self, p, axis, delta):
        q = p.clone()
        q[:, axis] += delta
        return self._q(q)
