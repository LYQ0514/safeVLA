"""Reduced-order smoke factory. Evidence class is always `proxy_smoke_only`."""
import numpy as np
from .governor import Candidate,ContinuationKernelGovernor

def build(args,env,obstacle_name):
    def state_builder(obs,prompt,base,budget,horizon):
        obstacle=np.asarray(obs[obstacle_name+"_pos"],float) if obstacle_name else np.ones(3)*99
        return np.r_[np.asarray(obs["robot0_eef_pos"],float),obstacle]
    def candidates(state,nominal,obs):
        out=[Candidate(np.asarray(nominal),"nominal")]; hold=np.zeros(7); hold[-1]=-1; out.append(Candidate(hold,"hold"))
        away=np.zeros(7); d=state[:3]-state[3:6]; n=np.linalg.norm(d); away[:3]=d/max(n,1e-9); away[-1]=-1; out.append(Candidate(away,"away")); return out
    def transition(s,a,w):
        z=np.asarray(s,float).copy(); z[:3]+=0.05*np.asarray(a[:3])+np.asarray(w); return z
    def disturbances(s): return (np.zeros(3),)
    def value(s,b,h):
        clearance=np.linalg.norm(np.asarray(s[:3])-np.asarray(s[3:6]))-.16
        return min(clearance, float(h)/max(1,args.horizon), float(b+1)/max(1,args.budget+1))
    gov=ContinuationKernelGovernor(transition,disturbances,value,lambda a,n:float(np.linalg.norm(a-n)))
    return gov,state_builder,candidates,"proxy_smoke_only"
