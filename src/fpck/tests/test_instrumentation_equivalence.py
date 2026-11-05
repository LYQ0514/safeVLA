"""执行等价测试：历史循环 vs instrumented 循环（相同状态 + 相同路径）。

裁决测试 8：对相同保存状态和路径，历史循环与 instrumented 循环的
qpos/qvel、动作和终止结果必须一致。

这里用真实 env + 固定路径（简单直线路径，避免 RRT 非确定性），对比
历史 execute_joint_path vs execute_joint_path_instrumented 的 Dmax + 最终 qpos。
"""

from __future__ import annotations
import json
import os
import sys
import tempfile
import unittest
import warnings

warnings.filterwarnings("ignore")

ROOT = "/data/safeVLA"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "dataset", "SafeLIBERO", "safelibero"))
sys.path.insert(0, "/data/2608.SDL/MojocoProj/src/pi05")
sys.path.insert(0, os.path.join(ROOT, "third_party", "aegis"))

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

from libero.libero import benchmark  # noqa: E402
from tools.m1_p4_p1_reentry import execute_joint_path  # noqa: E402
from tools.m1_p4_p0e_joint_execution import SOUP_BODY, MUG_BODY  # noqa: E402
from tools.m1_p4_p0_reachability import get_arm_q  # noqa: E402
from src.fpck.instrumented_loops import execute_joint_path_instrumented  # noqa: E402
from src.fpck.task_event_bridge import TaskEventBridge  # noqa: E402
from src.fpck.task_automaton import TaskAutomaton  # noqa: E402
from tools.g1a_soup_trace import body_state  # noqa: E402

SUITE = "safelibero_long"
LEVEL = "I"
TASK_ID = 0


class TestExecutionEquivalence(unittest.TestCase):
    def test_joint_path_equivalence(self):
        ts = benchmark.get_benchmark_dict()[SUITE](safety_level=LEVEL)
        task = ts.get_task(TASK_ID)

        # 构造 env（settled）
        import pathlib
        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file

        def make_env():
            env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256,
                                     camera_widths=256, controller="OSC_POSE")
            env.seed(7); env.reset()
            env.set_init_state(ts.get_task_init_states(TASK_ID)[0])
            dummy = np.zeros(7, dtype=np.float32); dummy[-1] = -1.0
            for _ in range(20):
                env.step(dummy)
            return env

        env1 = make_env()
        env2 = make_env()

        q_start = get_arm_q(env1)
        # 固定简单路径：q_start → q_start + 小扰动（2 个 waypoint，短路径）
        delta = np.zeros(7)
        delta[0] = 0.01  # 微小关节扰动
        path = np.array([q_start.copy(), q_start.copy() + delta])

        p0_1 = body_state(env1, MUG_BODY)["pos"].copy()
        p0_2 = body_state(env2, MUG_BODY)["pos"].copy()

        # 历史版本
        Dmax_hist = execute_joint_path(env1, path, p0_1)
        qpos_hist = np.asarray(env1.sim.data._data.qpos[:7]).copy()

        # instrumented 版本（bridge 空日志）
        with tempfile.TemporaryDirectory() as d:
            auto = TaskAutomaton(["alphabet_soup_1", "cream_cheese_1"],
                                 ["in alphabet_soup_1 basket_1_contain_region",
                                  "in cream_cheese_1 basket_1_contain_region"])
            bridge = TaskEventBridge(auto, os.path.join(d, "raw.jsonl"),
                                     ["alphabet_soup_1", "cream_cheese_1"],
                                     ["in alphabet_soup_1 basket_1_contain_region",
                                      "in cream_cheese_1 basket_1_contain_region"])
            settled = {"alphabet_soup_1_main": body_state(env2, "alphabet_soup_1_main")["pos"].copy(),
                       "cream_cheese_1_main": body_state(env2, "cream_cheese_1_main")["pos"].copy()}
            bridge.reset(env2, settled, ["alphabet_soup_1_main", "cream_cheese_1_main"],
                         MUG_BODY)
            Dmax_inst = execute_joint_path_instrumented(
                env2, path, p0_2, bridge,
                ["alphabet_soup_1_main", "cream_cheese_1_main"], settled, MUG_BODY)
            qpos_inst = np.asarray(env2.sim.data._data.qpos[:7]).copy()

        # 对比
        dmax_eq = abs(Dmax_hist - Dmax_inst) <= 1e-9
        qpos_eq = np.allclose(qpos_hist, qpos_inst, atol=1e-9)
        print(f"Dmax_hist={Dmax_hist:.6e} Dmax_inst={Dmax_inst:.6e} → {dmax_eq}")
        print(f"qpos max-err={np.max(np.abs(qpos_hist - qpos_inst)):.2e} → {qpos_eq}")

        env1.close(); env2.close()
        self.assertTrue(dmax_eq, "Dmax 必须一致")
        self.assertTrue(qpos_eq, "最终 qpos 必须一致")


if __name__ == "__main__":
    unittest.main()
