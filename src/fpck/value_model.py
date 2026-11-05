"""Auditable NumPy ridge value baseline and serialization.

This is a reproducible baseline, not a claim that a linear model is sufficient
for images. SafeLIBERO collection should write an explicit feature vector made
from observable state, automaton state, policy chunk metadata, budget, and time.
"""
from dataclasses import dataclass
import json
import numpy as np

@dataclass
class RidgeValue:
    mean: np.ndarray
    scale: np.ndarray
    weight: np.ndarray
    correction: float = 0.0

    def predict(self, x):
        x = np.asarray(x, float); z = (x - self.mean) / self.scale
        return z @ self.weight[:-1] + self.weight[-1]

    def lower(self, x): return self.predict(x) - self.correction
    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"mean":self.mean.tolist(),"scale":self.scale.tolist(),
                       "weight":self.weight.tolist(),"correction":self.correction}, f)
    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f: d=json.load(f)
        return cls(*(np.asarray(d[k],float) for k in ("mean","scale","weight")),
                   correction=float(d.get("correction",0.0)))

def fit_ridge(x, y, l2=1e-3):
    x, y = np.asarray(x,float), np.asarray(y,float)
    if x.ndim != 2 or y.shape != (len(x),): raise ValueError("bad shapes")
    mean=x.mean(0); scale=x.std(0); scale[scale<1e-12]=1.0
    z=(x-mean)/scale; design=np.column_stack([z,np.ones(len(z))])
    reg=np.eye(design.shape[1])*float(l2); reg[-1,-1]=0.0
    w=np.linalg.solve(design.T@design+reg, design.T@y)
    return RidgeValue(mean,scale,w)
