#!/usr/bin/env python3
"""Exact strict-separation experiment; writes JSON without external packages."""
import argparse, json, pathlib, sys
ROOT=pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src" if (ROOT/"src").exists() else ROOT))
from fpck import FiniteProductGame, solve_fpck, solve_physical_viability

def build():
    # A: accepted goal, S: safe stall/unsafe-return state, D: rejected collision.
    states=frozenset({"A","S","D"}); actions={"A":("nom",),"S":("nom","hold","left"),"D":("nom",)}
    table={("A","nom"):("A",),("S","nom"):("D",),("S","hold"):("S",),
           ("S","left"):("A",),("D","nom"):("D",)}
    cost=lambda s,a: int(a!="nom")
    return FiniteProductGame(states,actions,lambda s,a:table[(s,a)],cost,frozenset({"A"}),frozenset({"D"}))
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--horizon",type=int,default=3); ap.add_argument("--budget",type=int,default=0); ap.add_argument("--out",default="e0.json"); a=ap.parse_args()
    g=build(); K,w=solve_fpck(g,a.horizon,a.budget); physical=solve_physical_viability(g)
    result={"horizon":a.horizon,"budget":a.budget,"physical_viability":sorted(physical),
            "fpck":sorted(K[a.horizon][a.budget]),"strict_difference":sorted(physical-K[a.horizon][a.budget]),
            "witnesses":[{"h":h,"b":b,"state":s,"action":act} for (h,b,s),act in sorted(w.items())]}
    pathlib.Path(a.out).write_text(json.dumps(result,indent=2),encoding="utf-8"); print(json.dumps(result,indent=2))
if __name__=="__main__": main()
