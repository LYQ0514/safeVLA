import numpy as np

from src.fpck.predictive_shield import ProductSpacePredictiveShield


def test_seed_sequences_preserve_gripper_intent():
    shield=ProductSpacePredictiveShield(horizon=5,population=8,rollout_budget=4,
                                       iterations=1,elite=2)
    nominal=np.zeros((5,7),np.float32); nominal[:,6]=np.linspace(-1,1,5)
    obs={"robot0_eef_pos":np.array([.2,0,0]),"obs_pos":np.zeros(3)}
    rows=shield._seed_sequences(nominal,obs,"obs")
    assert all(np.array_equal(x[:,6],nominal[:,6]) for x in rows)


def test_analytic_cost_prefers_nominal_when_clearance_equal():
    shield=ProductSpacePredictiveShield(horizon=5,population=8,rollout_budget=4,
                                       iterations=1,elite=2)
    nominal=np.zeros((5,7)); changed=nominal.copy(); changed[:,:3]=.5
    assert shield._analytic_cost(nominal,nominal,1.0) < shield._analytic_cost(changed,nominal,1.0)


def test_progress_ordering_is_monotone():
    class State:
        mode="q_controlled"; lifecycle={"x":"CONTROLLED"}; achieved_mask={"g":False}
    class Auto: state=State()
    class Bridge: auto=Auto()
    a=ProductSpacePredictiveShield._progress(Bridge())
    State.mode="q_progress"; State.lifecycle={"x":"TRANSPORTED"}
    b=ProductSpacePredictiveShield._progress(Bridge())
    assert b>a
