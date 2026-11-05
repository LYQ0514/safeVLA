#!/usr/bin/env python3
"""SafeLIBERO CKG runner using a registered experiment factory.

Factory signature: build(args, env, obstacle_name) ->
    (governor, state_builder, candidate_builder, evidence_class).
The separation prevents this runner from silently treating a proxy as a
certificate. Use from the safeVLA checkout after installing `src/fpck`.
"""
import argparse,datetime as dt,importlib,json,os,pathlib,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"dataset"/"SafeLIBERO"/"safelibero"))

def load_factory(spec):
    mod,name=spec.split(":",1); return getattr(importlib.import_module(mod),name)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--factory",required=True,help="module:function registered factory")
    ap.add_argument("--suite",default="safelibero_spatial"); ap.add_argument("--level",default="I"); ap.add_argument("--tasks",type=int,nargs="+",default=[0]); ap.add_argument("--episodes",type=int,nargs="+",default=[0])
    ap.add_argument("--budget",type=int,default=20); ap.add_argument("--horizon",type=int,default=40); ap.add_argument("--replan-steps",type=int,default=5); ap.add_argument("--max-steps",type=int); ap.add_argument("--out"); ap.add_argument("--no-video",action="store_true"); a=ap.parse_args()
    if not os.environ.get("LIBERO_CONFIG_PATH"): raise SystemExit("source tools/activate_env.sh first")
    os.environ.setdefault("MUJOCO_GL","egl"); os.environ.setdefault("PYOPENGL_PLATFORM","egl")
    from libero.libero import benchmark,get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from src.eval_safelibero import run_episode,save_results,summarize,active_obstacle
    from src.policies import Pi05Policy
    from src.fpck.policy import CKGPolicy
    from tools.run_eval import DEFAULT_MAX_STEPS
    out=pathlib.Path(a.out or ROOT/"data"/"eval"/("fpck_"+dt.datetime.now().strftime("%Y%m%d_%H%M%S"))); out.mkdir(parents=True,exist_ok=True)
    base=Pi05Policy(verbose=True,replan_steps=a.replan_steps,image_size=224); suite=benchmark.get_benchmark_dict()[a.suite](safety_level=a.level); results=[]
    try:
        for task_id in a.tasks:
            task=suite.get_task(task_id); bddl=pathlib.Path(get_libero_path("bddl_files"))/task.problem_folder/task.bddl_file
            env=OffScreenRenderEnv(bddl_file_name=bddl,camera_heights=1024,camera_widths=1024,camera_depths=False); env.seed(7)
            try:
                for ep in a.episodes:
                    env.reset(); obs=env.set_init_state(suite.get_task_init_states(task_id)[ep]); obs_name=active_obstacle(env,obs)
                    gov,state_builder,candidate_builder,evidence_class=load_factory(a.factory)(a,env,obs_name)
                    log=out/f"decisions_t{task_id}_e{ep}.jsonl"; policy=CKGPolicy(base,gov,state_builder,candidate_builder,a.budget,a.horizon,str(log))
                    r=run_episode(env,suite,task_id,ep,policy,max_steps=a.max_steps or DEFAULT_MAX_STEPS[a.suite],num_steps_wait=20,collect_frames=False)
                    r.update({"suite":a.suite,"level":a.level,"evidence_class":evidence_class,"factory":a.factory,"budget":a.budget,"fpck_horizon":a.horizon})
                    results.append(r); save_results(results,{"overall":summarize(results)},out/"summary.json")
            finally: env.close()
    finally: base.close()
    print(json.dumps({"out":str(out),"summary":summarize(results)},indent=2))
if __name__=="__main__": main()
