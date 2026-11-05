"""safeVLA-compatible policy wrapper. It never changes frozen policy weights."""
import hashlib,json,time
import numpy as np
from .governor import Candidate

class CKGPolicy:
    def __init__(self, base_policy, governor, state_builder, candidate_builder,
                 budget, horizon, log_path=None):
        self.base=base_policy; self.governor=governor; self.state_builder=state_builder
        self.candidate_builder=candidate_builder; self.initial_budget=int(budget); self.initial_horizon=int(horizon)
        self.log_path=log_path; self.reset()
    @property
    def ready(self): return bool(getattr(self.base,"ready",True))
    def reset(self, obs=None, prompt=None):
        self.budget=self.initial_budget; self.horizon=self.initial_horizon; self.step=0
        if hasattr(self.base,"reset"): self.base.reset()
    def act(self, obs, prompt):
        nominal=np.asarray(self.base.act(obs,prompt),dtype=float); state=self.state_builder(obs,prompt,self.base,self.budget,self.horizon)
        candidates=list(self.candidate_builder(state,nominal,obs)); t0=time.perf_counter()
        d=self.governor.choose(state,nominal,candidates,self.budget,self.horizon); runtime=(time.perf_counter()-t0)*1000
        if d.action is None: action=np.zeros_like(nominal); action[-1]=-1.; status=d.status
        else: action=d.action; status=d.status
        dist=float(np.linalg.norm(action-nominal)); intervention=int(dist>self.governor.intervention_eps)
        budget_before=self.budget
        # Never underflow. An emergency action outside the certified set may be
        # executed by the integration layer, but it does not create new budget
        # and carries no continuation guarantee.
        self.budget=max(0,self.budget-intervention); self.horizon=max(0,self.horizon-1); self.step+=1
        if self.log_path:
            meta=getattr(self.base,"get_chunk_metadata",lambda:{})()
            row={"step":self.step,"status":status,"budget_before":budget_before,
                 "budget_remaining":self.budget,"budget_exhausted":bool(self.budget==0),
                 "intervention":bool(intervention),"outside_guarantee":bool(d.action is None),
                 "horizon_remaining":self.horizon,
                 "nominal_action":nominal.tolist(),"selected_action":np.asarray(action).tolist(),"runtime_ms":runtime,
                 "chunk_hash":meta.get("chunk_hash"),"buffer_index":getattr(self.base,"buffer_index",None),"evaluated":d.evaluated}
            with open(self.log_path,"a",encoding="utf-8") as f: f.write(json.dumps(row,ensure_ascii=False)+"\n")
        return np.clip(np.asarray(action,np.float32),-1,1)
    def close(self):
        if hasattr(self.base,"close"): self.base.close()
