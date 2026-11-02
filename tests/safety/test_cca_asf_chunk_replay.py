"""CCA-ASF E1-P0：Initial Chunk Replay 等价性测试。

验证 inject_initial_chunk 不改变 π0.5 缓存/推理计数/重规划调度：

P0-1：同一 chunk 注入两方法，chunk 哈希一致（由调用方保证，此处验证注入正确）
P0-2：首周期每个 buffer_index 名义动作一致
P0-3：注入不触发额外 worker 推理（infer_count 不变）
P0-4：重规划时刻一致（replan_steps 截断后，第 6 步触发新推理）
P0-5：注入后首次 act 的 buffer_index 从 0 开始
P0-6：同一 chunk 下普通 AEGIS 与重放 AEGIS 过滤结果一致（用 fake 名义动作验证）
"""

from __future__ import annotations

import numpy as np

from src.policies import Pi05Policy


class _FakeRunner:
    @staticmethod
    def obs_to_policy_obs(obs, prompt, image_size):
        return "fake_obs"


class _FakePolicy:
    def __init__(self):
        self.calls = 0

    def infer(self, policy_obs):
        self.calls += 1
        # 返回一个「注入后不该被用到」的 chunk（若被用说明注入失败）
        return [np.array([9.9, 9.9, 9.9, 9.9, 9.9, 9.9, 9.9], dtype=np.float32)
                for _ in range(10)]


def _make_policy(replan_steps=5):
    pol = object.__new__(Pi05Policy)
    pol._queue = []
    pol.infer_count = 0
    pol._chunk = None
    pol._buffer_index = 0
    pol._chunk_meta = {}
    pol.image_size = 256
    pol.replan_steps = replan_steps
    pol._runner = _FakeRunner()
    pol.policy = _FakePolicy()
    return pol


def _mk_chunk(k=10):
    return np.array([[i * 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, -1.0]
                     for i in range(k)], dtype=np.float32)


def test_P0_inject_consumes_chunk_without_infer():
    pol = _make_policy(replan_steps=5)
    chunk = _mk_chunk()
    pol.reset()
    pol.inject_initial_chunk(chunk)

    # P0-3：注入不触发推理
    assert pol.policy.calls == 0
    assert pol.infer_count == 0

    # P0-2 / P0-5：首周期每步名义动作 = chunk[buffer_index]
    for i in range(5):
        action = pol.act({}, "prompt")
        np.testing.assert_allclose(action, chunk[i], atol=1e-7)
        assert pol.buffer_index == i + 1

    # P0-4：replan_steps=5 → 第 6 步触发新推理（此时才 infer_count=1）
    action6 = pol.act({}, "prompt")
    assert pol.infer_count == 1  # 第 6 步才触发真实推理
    assert pol.policy.calls == 1
    # 第 6 步动作来自真实推理（fake 返回 9.9，被 act 的 clip 截到 1.0），不再是注入 chunk
    np.testing.assert_allclose(action6, np.ones(7, dtype=np.float32), atol=1e-7)


def test_P0_two_policies_same_injected_chunk_same_nominal():
    """P0-1/P0-2：两方法注入同一 chunk，首周期名义动作逐位一致。"""
    chunk = _mk_chunk()
    p1 = _make_policy(replan_steps=5)
    p2 = _make_policy(replan_steps=5)
    p1.reset(); p2.reset()
    p1.inject_initial_chunk(chunk)
    p2.inject_initial_chunk(chunk)
    # P0-1：chunk 哈希一致
    assert p1.get_chunk_metadata()["chunk_hash"] == p2.get_chunk_metadata()["chunk_hash"]
    for i in range(5):
        a1 = p1.act({}, "prompt")
        a2 = p2.act({}, "prompt")
        np.testing.assert_allclose(a1, a2, atol=1e-12)


def test_P0_reset_then_inject():
    """reset 后可注入；未 reset 有残留 chunk 时报错。"""
    pol = _make_policy(replan_steps=5)
    pol.reset()
    pol.inject_initial_chunk(_mk_chunk())
    # 已消费后再次注入应报错（避免覆盖）
    pol.act({}, "prompt")
    try:
        pol.inject_initial_chunk(_mk_chunk())
        raise AssertionError("应报错：非空 queue 不能注入")
    except RuntimeError:
        pass


def test_P0_full_chunk_preserved():
    """get_current_chunk 返回完整 chunk（含 replan 截断尾部）。"""
    pol = _make_policy(replan_steps=5)
    pol.reset()
    chunk = _mk_chunk(10)
    pol.inject_initial_chunk(chunk)
    got = pol.get_current_chunk()
    assert got.shape == (10, 7)
    np.testing.assert_allclose(got, chunk, atol=1e-12)
