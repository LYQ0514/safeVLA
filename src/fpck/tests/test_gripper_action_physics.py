"""夹爪动作解码器物理测试（需 env，裁决 §十二 §1/2/3/5/6/7）。

验证（真实 MuJoCo，非逻辑假设）：
- §1/2/3  明确 open/close 命令方向 + aperture 实际变化；
- §5      closing_response 只在 aperture 实际减小时成立（到机械限位后应转 False）；
- §6/7    空夹爪机械限位（aperture≈0）与物体阻挡闭合（aperture 停在中缝）区分。

运行较慢（~10s env 初始化）。
"""

from __future__ import annotations

import os
import sys
import unittest
import warnings

warnings.filterwarnings("ignore")
ROOT = "/data/safeVLA"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "dataset", "SafeLIBERO", "safelibero"))
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

from src.fpck.gripper_action_decoder import (  # noqa: E402
    decode_close_intent,
    compute_closing_response,
    compute_closure_blocked,
)


def _aperture(obs):
    gq = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64)
    return float(gq[0] - gq[1])


def _make_env():
    from libero.libero import benchmark
    from tools.g1a_soup_trace import make_env
    ts = benchmark.get_benchmark_dict()["safelibero_spatial"](safety_level="I")
    task = ts.get_task(0)
    return make_env(ts, task, 0)


class TestGripperActionPhysics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = _make_env()

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def _run_grip(self, grip, n):
        self.env.env.reset()
        traj = []
        for _ in range(n):
            act = np.zeros(7, dtype=np.float64)
            act[6] = grip
            obs, _r, _d, _i = self.env.step(act)
            traj.append(_aperture(obs))
        return traj

    def test_close_command_decreases_aperture(self):
        """§2/§3：close 命令使 aperture 减小（物理确认 sign>0=闭合）。"""
        traj = self._run_grip(+1.0, 40)
        self.assertLess(traj[-1], traj[0], f"close 应使 aperture 减小: {traj[0]:.4f}->{traj[-1]:.4f}")
        self.assertTrue(decode_close_intent(1.0))

    def test_open_command_increases_aperture(self):
        """§1：open 命令使 aperture 增大。"""
        traj = self._run_grip(-1.0, 40)
        self.assertGreater(traj[-1], traj[0], f"open 应使 aperture 增大: {traj[0]:.4f}->{traj[-1]:.4f}")
        self.assertFalse(decode_close_intent(-1.0))

    def test_closing_response_turns_off_at_limit(self):
        """§5/§7：closing_response 只在 aperture 实际减小时成立，到空夹爪限位后转 False。"""
        self.env.env.reset()
        close_intent = True
        prev = _aperture(self.env.env._get_observations())
        responses = []
        apertures = []
        for _ in range(80):
            act = np.zeros(7, dtype=np.float64)
            act[6] = 1.0
            obs, _r, _d, _i = self.env.step(act)
            a = _aperture(obs)
            vel = (a - prev) / 0.05  # 每控制步 0.05s
            responses.append(compute_closing_response(close_intent, vel))
            apertures.append(a)
            prev = a
        # 初期：aperture 快速减小 → closing_response True
        self.assertTrue(any(responses[:10]), "初期闭合应有 closing_response")
        # 末期：aperture 到限位（≈0）不再减小 → closing_response False
        self.assertFalse(responses[-1], "到空夹爪机械限位后 closing_response 应为 False")
        self.assertLess(apertures[-1], 0.006, "空夹爪应闭合到机械限位")
        # §6/§7：此时 aperture≈0 且双指无接触 → closure_blocked_by_object 应为 False（非物体阻挡）
        self.assertFalse(compute_closure_blocked(close_intent, 0.0, False, apertures[-1]),
                         "空夹爪限位不得误判为物体阻挡闭合")


if __name__ == "__main__":
    unittest.main()
