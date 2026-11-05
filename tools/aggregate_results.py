#!/usr/bin/env python3
"""Aggregate append-only episode JSONL; never imputes missing outcomes."""
import argparse,json,pathlib,statistics
def rate(rows,key):
    vals=[bool(r[key]) for r in rows if key in r and r[key] is not None]
    return None if not vals else sum(vals)/len(vals)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("inputs",nargs="+"); ap.add_argument("--out",default="aggregate.json"); a=ap.parse_args()
    rows=[]
    for p in a.inputs: rows += [json.loads(x) for x in pathlib.Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]
    runt=[float(r["runtime_ms"]) for r in rows if r.get("runtime_ms") is not None]
    runt.sort(); q=lambda p: None if not runt else runt[min(len(runt)-1,int(p*(len(runt)-1)))]
    out={"n":len(rows),"TSR":rate(rows,"native_success"),"TSR_valid":rate(rows,"valid_success"),
         "CAR":rate(rows,"car_1mm"),"CAR_gt5mm":rate(rows,"car_5mm"),"SSR":rate(rows,"ssr"),
         "StrictSSR_valid":rate(rows,"strict_ssr_valid"),"safe_stall":rate(rows,"safe_stall"),
         "budget_exhausted":rate(rows,"budget_exhausted"),"runtime_ms":{"p50":q(.5),"p95":q(.95),"p99":q(.99)}}
    pathlib.Path(a.out).write_text(json.dumps(out,indent=2),encoding="utf-8"); print(json.dumps(out,indent=2))
if __name__=="__main__": main()
