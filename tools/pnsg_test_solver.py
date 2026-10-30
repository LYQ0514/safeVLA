"""PNSG source_solver 单元测试：合成约束验证（设计文档 17.4 节）。

构造可解析的 1/2/3 约束问题，验证 minimum_norm_halfspaces 与 QP 解一致。
"""

import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pnsg.source_solver import minimum_norm_halfspaces, farkas_certificate  # noqa: E402


def check(name, got, want, tol=1e-6):
    ok = np.allclose(got, want, atol=tol)
    print(f"  [{'OK' if ok else 'FAIL'}] {name}: got={np.round(got,4)} want={np.round(want,4)}")
    return ok


def main():
    print("=== 1 条约束 ===")
    dz, r = minimum_norm_halfspaces(np.array([[1.0, 0.0]]), np.array([1.0]))
    check("A=[[1,0]], b=[1]", dz, np.array([1.0, 0.0]))

    print("=== 2 条正交约束 ===")
    dz, r = minimum_norm_halfspaces(np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([1.0, 1.0]))
    check("A=[[1,0],[0,1]], b=[1,1]", dz, np.array([1.0, 1.0]))

    print("=== 3 条约束（用 QP 对照）===")
    A = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    b = np.array([0.0, 0.0, 1.0])
    dz, r = minimum_norm_halfspaces(A, b)
    # QP 对照
    import cvxpy as cp
    x = cp.Variable(2)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(x)), [A @ x >= b])
    prob.solve(solver=cp.OSQP)
    qp_dz = np.asarray(x.value)
    check("3约束 vs QP", dz, qp_dz, tol=1e-4)
    print(f"      r_crit = {r:.4f} (QP 范数 {np.linalg.norm(qp_dz):.4f})")

    print("=== 已满足约束 → Δz=0 ===")
    dz, r = minimum_norm_halfspaces(np.array([[1.0, 0.0]]), np.array([-1.0]))
    check("b<0 时 Δz=0", dz, np.zeros(2))

    print("=== 不可行 + Farkas witness ===")
    # A = [[1,0],[0,1],[-1,-1]], b = [0,0,1]：x≥0, y≥0, x+y≤-1 不可行
    A_inf = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
    b_inf = np.array([0.0, 0.0, 1.0])
    dz, r = minimum_norm_halfspaces(A_inf, b_inf)
    print(f"  最小范数解: {dz} (应 None)")
    fk = farkas_certificate(A_inf, b_inf)
    print(f"  Farkas: found={fk['found']}, bty={fk['bty']}")

    print("=== 高维形状（60×320 规模）===")
    rng = np.random.default_rng(0)
    A_big = rng.standard_normal((3, 320))
    b_big = np.array([1.0, 0.5, 0.2])
    dz_big, r_big = minimum_norm_halfspaces(A_big, b_big)
    print(f"  320 维 3 约束: r_crit={r_big:.4f}, 满足 A dz ≥ b: {np.all(A_big @ dz_big >= b_big - 1e-9)}")


if __name__ == "__main__":
    main()
