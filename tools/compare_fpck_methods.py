#!/usr/bin/env python3
"""Paired baseline-vs-CKG SafeLIBERO comparison with bootstrap intervals."""
from __future__ import annotations
import argparse,json,pathlib
import numpy as np

KEYS=("suite","level","task_id","episode_idx","stream")
METRICS={
    "TSR":(lambda r:bool(r["native_success"]), True),
    "STS":(lambda r:bool(r["valid_success"]), True),
    "CAR":(lambda r:bool(r["car_1mm"]), True),
    "CAR_gt5mm":(lambda r:bool(r["car_5mm"]), True),
    "StrictSSR":(lambda r:bool(r["strict_ssr_valid"]), True),
    "UnsafeReturn":(lambda r:r["final_e1_label"]=="UNSAFE_RETURN", False),
    "IntegrityFail":(lambda r:r["final_e1_label"]=="TASK_INTEGRITY_FAIL", False),
    "SafeStall":(lambda r:r["final_e1_label"]=="SAFE_STALL", False),
}

def load(path):
    p=pathlib.Path(path)
    if p.is_dir():
        q=p/"episodes.jsonl"
        if q.exists(): return [json.loads(x) for x in q.read_text(encoding="utf-8").splitlines() if x.strip()]
        p=p/"summary.json"
    text=p.read_text(encoding="utf-8")
    try:
        d=json.loads(text)
        if isinstance(d, dict):
            # A one-line JSONL file is also valid JSON.  Distinguish a single
            # episode record from a summary container before iterating it.
            if "episodes" in d:
                return d["episodes"]
            if all(k in d for k in KEYS):
                return [d]
            raise ValueError("JSON result object has neither episodes nor an episode identity")
        return d
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
def key(r): return tuple(r[k] for k in KEYS)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("baseline"); ap.add_argument("ckg")
    ap.add_argument("--out",default="fpck_comparison.json"); ap.add_argument("--bootstrap",type=int,default=10000); ap.add_argument("--seed",type=int,default=0); a=ap.parse_args()
    b={key(r):r for r in load(a.baseline)}; c={key(r):r for r in load(a.ckg)}
    common=sorted(set(b)&set(c)); missing={"baseline_only":len(set(b)-set(c)),"ckg_only":len(set(c)-set(b))}
    if not common: raise SystemExit("no paired episodes")
    rng=np.random.default_rng(a.seed); out={"paired_n":len(common),"missing":missing,"metrics":{}}
    for name,(fn,higher_is_better) in METRICS.items():
        xb=np.asarray([fn(b[k]) for k in common],float); xc=np.asarray([fn(c[k]) for k in common],float); diff=xc-xb
        idx=rng.integers(0,len(common),size=(a.bootstrap,len(common))); boot=diff[idx].mean(1)
        beneficial=diff if higher_is_better else -diff
        out["metrics"][name]={"baseline":float(xb.mean()),"ckg":float(xc.mean()),"difference":float(diff.mean()),
                              "higher_is_better":higher_is_better,
                              "ci95":[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],
                              "discordant":{"ckg_better":int(np.sum(beneficial>0)),"ckg_worse":int(np.sum(beneficial<0))}}
    pathlib.Path(a.out).write_text(json.dumps(out,indent=2),encoding="utf-8"); print(json.dumps(out,indent=2))
if __name__=="__main__": main()
