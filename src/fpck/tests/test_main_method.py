import pathlib
import shutil
import tempfile
import json
import numpy as np

from src.fpck.candidate_actions import registered_candidates
from src.fpck.control_state import restore_control_state, snapshot_control_state
from src.fpck.continuation_model import fit_calibrated_ensemble, CalibratedContinuationModel
from src.fpck.direct_governor import DirectContinuationGovernor
from src.fpck.governor import Candidate
from src.fpck.product_features import FEATURE_NAMES, ProductFeatureContext, encode_product_candidate
from src.fpck.task_automaton_acq import GraspAcquisitionAutomaton, AcquisitionEvent
from tools.compare_fpck_methods import load as load_comparison_rows


def test_product_feature_contract_is_fixed_and_finite():
    manifest={"objects":{"obj":{"roles":["MANIPULATED_TARGET"]},
                         "goal":{"roles":["GOAL_RECEPTACLE"]}}}
    auto=GraspAcquisitionAutomaton(["obj"],["in obj goal"])
    obs={"robot0_eef_pos":np.array([.1,.2,.3]),"robot0_gripper_qpos":np.array([.04,-.04]),
         "obj_pos":np.array([.2,.2,.3]),"goal_pos":np.array([.3,.4,.3])}
    ctx=ProductFeatureContext(manifest,auto.state,3,4,9,10)
    phi=encode_product_candidate(obs,np.zeros(7),np.ones(7)*.1,ctx)
    assert phi.shape==(len(FEATURE_NAMES),)
    assert np.all(np.isfinite(phi))


def test_calibrated_ensemble_roundtrip_and_direct_governor():
    rng=np.random.default_rng(1); x=rng.normal(size=(80,6)); y=(x[:,0]+.4*x[:,1]>0).astype(float)
    m=fit_calibrated_ensemble(x[:60],y[:60],x[60:],y[60:],feature_names=[f"x{i}" for i in range(6)],
                              groups=[f"g{i//2}" for i in range(60)],members=4,steps=100,seed=2)
    tmp=pathlib.Path(tempfile.mkdtemp(prefix="fpck_test_",dir=pathlib.Path.cwd()))
    try:
        path=tmp/"model.json"
        m.save(path); loaded=CalibratedContinuationModel.load(path)
        assert np.allclose(m.lower(x[:3]),loaded.lower(x[:3]))
    finally:
        shutil.rmtree(tmp)
    class Fixed:
        def lower(self,z): return np.asarray([.9 if row[0]>0 else .1 for row in z])
    gov=DirectContinuationGovernor(Fixed(),lambda s,n,a,b,h:np.array([a[0]]),threshold=.5)
    d=gov.choose(None,np.zeros(7),[Candidate(np.zeros(7),"nominal"),Candidate(np.r_[1.,np.zeros(6)],"safe")],1,2)
    assert d.status=="empirically_admissible" and d.action[0]==1


def test_candidate_family_preserves_gripper_and_authority():
    nominal=np.array([.2,-.1,.3,0,0,0,.7])
    rows=registered_candidates(nominal,[0,0,0],[.1,0,0],max_delta=.4)
    assert rows[0].source=="nominal"
    assert all(np.isclose(r.action[6],nominal[6]) for r in rows)
    assert all(np.linalg.norm(r.action[:6]-nominal[:6])<=.400001 for r in rows)


def test_v13_acquisition_memory_survives_snapshot_restore():
    auto=GraspAcquisitionAutomaton(["obj"],["in obj goal"])
    auto.process(AcquisitionEvent(step=1,close_intent=True,valid_finger_target_contact=True,
                                  left_finger_contact=True,task_contact=True))
    snap=auto.snapshot(); restored=GraspAcquisitionAutomaton(["obj"],["in obj goal"])
    restored.restore(snap)
    assert restored.snapshot()==snap


def test_single_line_jsonl_is_loaded_as_one_episode():
    tmp=pathlib.Path(tempfile.mkdtemp(prefix="fpck_jsonl_",dir=pathlib.Path.cwd()))
    try:
        row={"suite":"s","level":"I","task_id":0,"episode_idx":0,"stream":0}
        path=tmp/"episodes.jsonl"; path.write_text(json.dumps(row)+"\n",encoding="utf-8")
        assert load_comparison_rows(path)==[row]
    finally:
        shutil.rmtree(tmp)


def test_sticky_gripper_state_is_restored_between_siblings():
    class Obj: pass
    gripper=Obj(); gripper.current_action=np.array([.25,-.25])
    controller=Obj(); controller.goal_pos=np.array([1.,2.,3.]); controller.goal_ori=None
    robot=Obj(); robot.gripper=gripper; robot.controller=controller
    env=Obj(); env.robots=[robot]
    snap=snapshot_control_state(env)
    gripper.current_action[:]=0; controller.goal_pos[:]=9
    restore_control_state(env,snap)
    assert np.allclose(gripper.current_action,[.25,-.25])
    assert np.allclose(controller.goal_pos,[1,2,3])
