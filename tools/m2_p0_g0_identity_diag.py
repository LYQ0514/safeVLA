"""M2-P0 G0 诊断：量化 reconstruct_nvs 的 NVS 步数与 q_start 波动。

多次运行同一 state/stream 的 reconstruct_nvs，记录：
- NVS 步数是否稳定；
- q_start（7 关节）的哈希是否一致；
- soup/mug 位置是否一致。

目的：确定 M2-P0 配对实验是否需要「固定绝对决策步」作为 canonical 起点。
"""

from __future__ import annotations
import hashlib, json, os, sys, warnings, time
warnings.filterwarnings("ignore")
ROOT = "/data/safeVLA"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "dataset", "SafeLIBERO", "safelibero"))
sys.path.insert(0, "/data/2608.SDL/MojocoProj/src/pi05")
sys.path.insert(0, os.path.join(ROOT, "third_party", "aegis"))
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
from libero.libero import benchmark
from src.pnsg.policy_adapter import PnsgPolicy
from tools.g1a_soup_trace import body_state
from tools.m1_p4_p0e_joint_execution import reconstruct_nvs, MUG_BODY, SOUP_BODY
from tools.m1_p4_p0_reachability import get_arm_q

def qhash(q):
    return hashlib.sha256(np.asarray(q).tobytes()).hexdigest()[:12]

def main():
    ts = benchmark.get_benchmark_dict()['safelibero_long'](safety_level='I')
    task = ts.get_task(0); prompt = task.language
    pol = PnsgPolicy()

    print("=== M2-P0 G0：NVS 波动诊断 ===", flush=True)
    for state_id, stream in [(1,3),(4,3),(9,3),(2,0)]:
        nvs_steps = []
        q_hashes = []
        soup_positions = []
        for rep in range(5):
            env, layer, scratch, layer_s, rg, og, p0, soup_pos, nvs_step, Dmax = \
                reconstruct_nvs(ts, task, prompt, pol, state_id, stream)
            nvs_steps.append(nvs_step)
            q_hashes.append(qhash(get_arm_q(env)))
            soup_positions.append(soup_pos.tolist())
            env.close(); scratch.close()
        print(f"\nS{state_id:02d}/st{stream}:", flush=True)
        print(f"  NVS 步数 5 次: {nvs_steps}", flush=True)
        print(f"  q_start 哈希: {q_hashes}", flush=True)
        print(f"  soup 位置一致: {all(s == soup_positions[0] for s in soup_positions)}", flush=True)

    pol.close()

if __name__ == "__main__":
    main()
