"""夹爪动作语义探针（裁决 §十二前置验证）。

目的：物理确认 openpi/LIBERO 动作的 gripper 分量 action[6] 的正负 = 开/闭方向，
以及 aperture 的定义与闭合响应。不能凭经验猜测。

验证点：
1. action[6] 正负方向（close 使 aperture 减小？）
2. aperture 定义（qpos[0] - qpos[1]？）
3. sticky 累积语义（speed=0.01，单次命令变化小）
4. deployed_close_intent 在命令下发时立即成立
5. closing_response 只在 aperture 实际减小时成立
"""
from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = "/data/safeVLA"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "dataset", "SafeLIBERO", "safelibero"))
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

from libero.libero import benchmark  # noqa: E402
from tools.g1a_soup_trace import make_env  # noqa: E402


def aperture(obs):
    gq = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64)
    return gq[0] - gq[1]  # 左指 - 右指，对称闭合时趋近 0


def main():
    ts = benchmark.get_benchmark_dict()["safelibero_spatial"](safety_level="I")
    task = ts.get_task(0)
    env = make_env(ts, task, 0)
    env.env.reset()
    obs = env.env._get_observations()

    print("初始 aperture:", round(aperture(obs), 6))

    # 用零动作（保持位姿，仅改 gripper 分量）测试方向
    # OSC_POSE：action 前 6 维是 delta（0 = 不动），gripper 是第 7 维
    def run_cmd(grip, n=40, label=""):
        env.env.reset()
        a0 = aperture(env.env._get_observations())
        min_a = a0
        for _ in range(n):
            act = np.zeros(7, dtype=np.float64)
            act[6] = grip
            obs, _r, _d, _i = env.step(act)
            a = aperture(obs)
            min_a = min(min_a, a)
        a1 = aperture(obs)
        print(f"{label}: grip={grip:+.2f}  aperture {a0:.6f} -> {a1:.6f}  (min={min_a:.6f})  "
              f"delta={a1 - a0:+.6f}")
        return a0, a1

    run_cmd(-1.0, label="OPEN")
    run_cmd(+1.0, label="CLOSE")

    env.close()


if __name__ == "__main__":
    main()
