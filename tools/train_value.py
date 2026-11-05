#!/usr/bin/env python3
"""Fit value baseline from JSONL rows: {split, features, target}."""
import argparse,json,pathlib,sys,numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src" if (ROOT/"src").exists() else ROOT))
from fpck.value_model import fit_ridge
from fpck.governor import one_sided_correction
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("data"); ap.add_argument("--out",default="value.json"); ap.add_argument("--alpha",type=float,default=.05); ap.add_argument("--l2",type=float,default=1e-3); a=ap.parse_args()
    rows=[json.loads(x) for x in pathlib.Path(a.data).read_text(encoding="utf-8").splitlines() if x.strip()]
    tr=[r for r in rows if r["split"]=="train"]; ca=[r for r in rows if r["split"]=="calibration"]
    if not tr or not ca: raise SystemExit("need nonempty train and calibration splits")
    model=fit_ridge([r["features"] for r in tr],[r["target"] for r in tr],a.l2)
    pred=model.predict(np.asarray([r["features"] for r in ca])); truth=np.asarray([r["target"] for r in ca])
    model.correction=one_sided_correction(pred,truth,a.alpha); model.save(a.out)
    print(json.dumps({"train_n":len(tr),"calibration_n":len(ca),"alpha":a.alpha,"correction":model.correction},indent=2))
if __name__=="__main__": main()
