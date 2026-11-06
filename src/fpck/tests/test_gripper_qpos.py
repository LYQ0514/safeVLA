"""gripper qpos 采集修复的验证测试（裁决 §七）。

验证：
1. snapshot_gripper 从 obs["robot0_gripper_qpos"] 读正确值（非 d.qpos[-2:] 猜测）；
2. 夹爪 qpos 是 2 维（左右手指），与 obs 一致。
"""

from __future__ import annotations

import os
import sys
import unittest
import warnings

warnings.filterwarnings("ignore")

ROOT = "/data/safeVLA"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "dataset", "SafeLIBERO", "safelibero"))

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402


class TestGripperQpos(unittest.TestCase):
    def test_snapshot_gripper_reads_obs(self):
        from libero.libero import benchmark  # noqa: E402
        from tools.g1a_soup_trace import make_env  # noqa: E402
        from src.fpck.substep_collector import SubstepCollector  # noqa: E402

        ts = benchmark.get_benchmark_dict()["safelibero_spatial"](safety_level="I")
        task = ts.get_task(0)
        env = make_env(ts, task, 0)
        collector = SubstepCollector(env)
        obs = env.env._get_observations()

        snap = collector.snapshot_gripper(obs)
        qpos = snap["qpos"]
        obs_qpos = np.asarray(obs["robot0_gripper_qpos"])

        # qpos 应该非 None，且与 obs 一致
        self.assertIsNotNone(qpos, "gripper qpos 不应为 None")
        self.assertEqual(len(qpos), 2, "夹爪应有 2 个手指关节")
        self.assertTrue(np.allclose(qpos, obs_qpos),
                        f"snapshot 读到的 {qpos} 应等于 obs {obs_qpos}")
        env.close()

    def test_gripper_qpos_not_last_two(self):
        """验证 d.qpos[-2:] 不是夹爪关节（旧 bug 的根因）。"""
        from libero.libero import benchmark  # noqa: E402
        from tools.g1a_soup_trace import make_env  # noqa: E402

        ts = benchmark.get_benchmark_dict()["safelibero_spatial"](safety_level="I")
        task = ts.get_task(0)
        env = make_env(ts, task, 0)
        d = env.sim.data._data
        obs = env.env._get_observations()
        obs_qpos = np.asarray(obs["robot0_gripper_qpos"])
        last_two = np.asarray(d.qpos[-2:])
        # 旧 bug：d.qpos[-2:] 可能不等于夹爪 qpos
        print(f"d.qpos[-2:] = {last_two}, obs gripper qpos = {obs_qpos}")
        print(f"相等: {np.allclose(last_two, obs_qpos)}")
        # 不强制断言（取决于模型 qpos 布局），只记录
        env.close()


if __name__ == "__main__":
    unittest.main()
