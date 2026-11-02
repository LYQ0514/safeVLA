"""CCA-ASF D1 接口测试：Pi05Policy 动作块只读暴露语义（D1-T2）。

用 object.__new__ 绕过 __init__（不 load_policy），验证：
1. act() 返回的动作 == get_current_chunk()[buffer_index]（消费前索引）；
2. 无 replan 时 chunk 完整消费，有 replan 时前 replan_steps 步消费后重新推理；
3. get_current_chunk() 返回完整 chunk（含被 replan 截断的尾部）。
"""

from __future__ import annotations

import numpy as np

from src.policies import Pi05Policy


class _FakeRunner:
    @staticmethod
    def obs_to_policy_obs(obs, prompt, image_size):
        return "fake_obs"


class _FakePolicy:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.calls = 0

    def infer(self, policy_obs):
        # 每次 infer 返回下一个预设 chunk
        idx = self.calls
        self.calls += 1
        return self.chunks[idx]


def _make_policy(chunks, replan_steps=None):
    pol = object.__new__(Pi05Policy)
    pol._queue = []
    pol.infer_count = 0
    pol._chunk = None
    pol._buffer_index = 0
    pol._chunk_meta = {}
    pol.image_size = 256
    pol.replan_steps = replan_steps
    pol._runner = _FakeRunner()
    pol.policy = _FakePolicy(chunks)
    return pol


def _mk_chunk(offset, k):
    # 生成 k 步动作块，第 i 步第一维 = offset*0.1 + i*0.001（都在 [-1,1] 内，避免被 clip）
    return [np.array([offset * 0.1 + i * 0.001, 0.1, 0.2, 0.3, 0.4, 0.5, -1.0], dtype=np.float32)
            for i in range(k)]


def test_D1_T2_chunk_consistency_no_replan():
    pol = _make_policy([_mk_chunk(0, 4)], replan_steps=None)
    for i in range(4):
        action = pol.act({}, "prompt")
        chunk = pol.get_current_chunk()
        assert chunk is not None and len(chunk) == 4
        np.testing.assert_allclose(action, chunk[i], atol=1e-12)
        assert pol.buffer_index == i + 1


def test_D1_T2_chunk_consistency_with_replan():
    # replan_steps=2：chunk 前 2 步消费后重新推理
    pol = _make_policy([_mk_chunk(0, 4), _mk_chunk(10, 4)], replan_steps=2)
    # 前 2 步用第一个 chunk
    for i in range(2):
        action = pol.act({}, "prompt")
        chunk = pol.get_current_chunk()
        np.testing.assert_allclose(action, chunk[i], atol=1e-12)
    # 第 3 步重新推理（第二个 chunk）
    action = pol.act({}, "prompt")
    chunk = pol.get_current_chunk()
    np.testing.assert_allclose(action, chunk[0], atol=1e-12)
    assert abs(chunk[0, 0] - 1.0) < 1e-6  # 第二个 chunk（offset=10）第一维 = 1.0
    assert pol.infer_count == 2


def test_D1_T2_full_chunk_preserved_under_replan():
    # get_current_chunk 应返回完整 chunk（含被 replan 截断的尾部）
    pol = _make_policy([_mk_chunk(0, 4)], replan_steps=2)
    pol.act({}, "prompt")
    chunk = pol.get_current_chunk()
    assert len(chunk) == 4  # 完整 chunk，非截断后的 2
    assert abs(chunk[-1, 0] - 0.003) < 1e-6  # 尾部保留（offset=0, i=3 → 0.003）


def test_D1_T2_reset_clears_chunk():
    pol = _make_policy([_mk_chunk(0, 4)], replan_steps=None)
    pol.act({}, "prompt")
    assert pol.get_current_chunk() is not None
    pol.reset()
    assert pol.get_current_chunk() is None
    assert pol.buffer_index == 0
