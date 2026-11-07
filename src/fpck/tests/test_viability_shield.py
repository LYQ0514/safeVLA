import numpy as np

from src.fpck.frozen_policy import DeterministicFrozenPolicy
from src.fpck.viability_shield import (
    CounterfactualResumeViabilityShield, ViabilityEvaluation,
    goal_pairs_from_manifest,
)


def _evaluation(source="nominal", verified=True, progress=0.0,
                potential=0.0, deviation=0.0, safe_steps=10):
    return ViabilityEvaluation(
        source=source, actions=np.zeros((5, 7), np.float32),
        verified=verified, prefix_safe=verified, resume_safe=verified,
        motion_retained=True, final_mode="q_progress", violation_code=None,
        prefix_steps=5, resume_steps=5, safe_steps=safe_steps,
        progress_delta=progress, goal_potential_delta=potential,
        intent_ratio=1.0, intent_cosine=1.0, min_proxy_clearance=0.01,
        deviation=deviation, resume_chunk_hash="h")


def test_verified_rank_prefers_task_progress_before_deviation():
    progressed = _evaluation("repair", progress=1.0, deviation=2.0)
    nominal = _evaluation("nominal", progress=0.0, deviation=0.0)
    assert CounterfactualResumeViabilityShield._verified_rank(progressed) > \
           CounterfactualResumeViabilityShield._verified_rank(nominal)


def test_fallback_rank_prefers_longer_safe_continuation():
    a = _evaluation("a", verified=False, safe_steps=4)
    b = _evaluation("b", verified=False, safe_steps=8)
    object.__setattr__(a, "prefix_safe", True)
    object.__setattr__(b, "prefix_safe", True)
    assert CounterfactualResumeViabilityShield._fallback_rank(b) > \
           CounterfactualResumeViabilityShield._fallback_rank(a)


def test_goal_pair_resolution_handles_regions():
    manifest = {
        "objects": {"obj": {}, "basket": {}},
        "regions": [{"name": "inside_region", "target": "basket"}],
        "goal_clauses": [("In", ["obj", "basket_inside_region"])],
    }
    assert goal_pairs_from_manifest(manifest) == [("obj", "basket")]


def test_counterfactual_query_does_not_mutate_policy_cursor():
    class Worker:
        H = 2; D = 3
        def generate_with_hashes(self, z, obs):
            return np.zeros((5, 7), np.float32), {"raw": "x"}
    p = DeterministicFrozenPolicy(
        Worker(), suite="s", task_id=1, state_id=2, stream=3,
        replan_steps=5)
    p._policy_obs = lambda obs, prompt: {"obs": 1}
    before = p.snapshot()
    chunk, meta = p.counterfactual_chunk({}, "task", query_idx=7)
    after = p.snapshot()
    assert chunk.shape == (5, 7)
    assert meta["query_idx"] == 7 and meta["counterfactual"] is True
    assert before["query_idx"] == after["query_idx"]
    assert before["buffer_index"] == after["buffer_index"]


def test_deployed_query_is_not_marked_counterfactual():
    class Worker:
        H = 2; D = 3
        def generate_with_hashes(self, z, obs):
            return np.zeros((5, 7), np.float32), {}
    p = DeterministicFrozenPolicy(
        Worker(), suite="s", task_id=1, state_id=2, replan_steps=5)
    p._policy_obs = lambda obs, prompt: {}
    p._query({}, "task")
    assert p.get_chunk_metadata()["counterfactual"] is False
    assert p.query_idx == 1 and p.buffer_index == 0


def test_no_cross_query_padding_contract():
    shield = CounterfactualResumeViabilityShield(horizon=5, execute_steps=2)
    assert shield.horizon == 5 and shield.execute_steps == 2
