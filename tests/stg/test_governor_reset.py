"""STG governor reset + 风险迟滞单测（G1）。

任务书 §5 / 用户 G1 验收项 4：
- Governor reset 正确，不发生 episode 间状态泄漏；
- 风险离开恢复 B0（λ=0）；
- 风险迟滞（进入/退出阈值）。

注意：STGGuide 是无状态纯函数类，测试其 reset 与跨调用状态隔离。
"""

from __future__ import annotations

import numpy as np
import pytest

from src.safety.stg import STGGuide


def _env_like():
    p_ee = np.array([0.0, 0.0, 0.8])
    a_v = np.array([0.0, 0.0, 1.0])  # 法向 +z（远离障碍物）
    R1 = np.eye(3)
    d = np.array([1.0, 0.0, 0.0])   # 意图 +x
    return p_ee, a_v, R1, d


# ---------------------------------------------------------------------------
# reset 不串 episode
# ---------------------------------------------------------------------------

def test_reset_clears_risk_state():
    g = STGGuide(mode="B3", h_enter=0.01, h_exit=0.02)
    p_ee, a_v, R1, d = _env_like()
    # 进入风险（h < h_enter）
    g.step(p_ee, a_v, R1, d, h=0.005)
    assert g.in_risk()
    assert g._r is not None
    # reset 后应清除
    g.reset()
    assert not g.in_risk()
    assert g._r is None


def test_reset_prevents_cross_episode_leak():
    g = STGGuide(mode="B3", h_enter=0.01, h_exit=0.02)
    p_ee, a_v, R1, d = _env_like()
    # episode 1：进入风险，参考演化到某位置
    g.step(p_ee, a_v, R1, d, h=0.005)
    g.step(p_ee + np.array([0.01, 0, 0]), a_v, R1, d, h=0.005)
    r_ep1 = g._r.copy()
    g.reset()
    # episode 2：重新进入，参考应从当前末端位置重新初始化，非 ep1 残留
    p_ee2 = np.array([0.5, 0.5, 0.8])
    g.step(p_ee2, a_v, R1, d, h=0.005)
    assert not np.allclose(g._r, r_ep1), "reset 后参考不应残留 episode 1 状态"


# ---------------------------------------------------------------------------
# 风险迟滞 + 退出恢复 B0
# ---------------------------------------------------------------------------

def test_no_risk_returns_b0():
    g = STGGuide(mode="B3", h_enter=0.01, h_exit=0.02)
    p_ee, a_v, R1, d = _env_like()
    g_t, lam, info = g.step(p_ee, a_v, R1, d, h=0.1)  # 远离障碍物
    assert g_t is None
    assert lam == 0.0
    assert not g.in_risk()


def test_enter_risk_then_guide():
    g = STGGuide(mode="B3", h_enter=0.01, h_exit=0.02)
    p_ee, a_v, R1, d = _env_like()
    g_t, lam, info = g.step(p_ee, a_v, R1, d, h=0.005)  # 进入风险
    assert g.in_risk()
    assert lam > 0.0
    assert g_t is not None


def test_exit_risk_hysteresis():
    g = STGGuide(mode="B3", h_enter=0.01, h_exit=0.02)
    p_ee, a_v, R1, d = _env_like()
    g.step(p_ee, a_v, R1, d, h=0.005)   # 进入（h < 0.01）
    assert g.in_risk()
    # 仍在退出阈值内（0.015 < 0.02），保持风险
    g_t, lam, _ = g.step(p_ee, a_v, R1, d, h=0.015)
    assert g.in_risk(), "迟滞内不应退出"
    # 超过退出阈值
    g_t, lam, _ = g.step(p_ee, a_v, R1, d, h=0.05)
    assert not g.in_risk()
    assert lam == 0.0


def test_b0_never_guides():
    g = STGGuide(mode="B0")
    p_ee, a_v, R1, d = _env_like()
    g_t, lam, info = g.step(p_ee, a_v, R1, d, h=0.005)
    assert g_t is None
    assert lam == 0.0


# ---------------------------------------------------------------------------
# 各 mode 行为差异（2×2 因子）
# ---------------------------------------------------------------------------

def test_b1_instantaneous_no_memory():
    g = STGGuide(mode="B1", h_enter=0.01, h_exit=0.02, gamma=0.01)
    p_ee, a_v, R1, d = _env_like()
    g1, lam1, _ = g.step(p_ee, a_v, R1, d, h=0.005)
    g2, lam2, _ = g.step(p_ee, a_v, R1, d, h=0.005)
    # B1 无记忆：相同输入相同输出（不依赖上一步）
    assert np.allclose(g1, g2)


def test_b2_governor_has_memory():
    g = STGGuide(mode="B2", h_enter=0.01, h_exit=0.02, alpha=0.5)
    p_ee, a_v, R1, d = _env_like()
    g1, _, _ = g.step(p_ee, a_v, R1, d, h=0.005)
    g2, _, _ = g.step(p_ee + np.array([0.02, 0, 0]), a_v, R1, d, h=0.005)
    # B2 有记忆：r_t 平滑，两次引导不同
    assert not np.allclose(g1, g2)


def test_b3_differs_from_b1_and_b2():
    p_ee, a_v, R1, d = _env_like()
    g_b1 = STGGuide(mode="B1", gamma=0.01).step(p_ee, a_v, R1, d, h=0.005)[0]
    g_b2 = STGGuide(mode="B2", alpha=0.5).step(p_ee, a_v, R1, d, h=0.005)[0]
    g_b3 = STGGuide(mode="B3", alpha=0.5, gamma=0.01).step(p_ee, a_v, R1, d, h=0.005)[0]
    assert not np.allclose(g_b1, g_b3)
    assert not np.allclose(g_b2, g_b3)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
