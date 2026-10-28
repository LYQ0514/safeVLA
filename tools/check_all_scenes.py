#!/usr/bin/env python
"""SafeLIBERO 全场景健康检查：4 套件 x 2 安全等级 x 4 任务 = 32 个场景。

对每个场景：加载 .pruned_init 初始状态、构建 OffScreenRenderEnv、reset，验证资产 / bddl /
渲染都能通。不做策略 rollout（那是评测本身的事）。

用法：
    cd /data/safeVLA && source tools/activate_env.sh
    python tools/check_all_scenes.py            # 全量 32 个
    python tools/check_all_scenes.py --quick    # 每套件只查 task0，共 8 个
退出码 0 = 全通过。
"""

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "dataset", "SafeLIBERO", "safelibero"))

import pathlib  # noqa: E402

from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402

SUITES = ["safelibero_spatial", "safelibero_goal", "safelibero_object", "safelibero_long"]
LEVELS = ["I", "II"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="每个套件只检查 task0")
    args = ap.parse_args()

    if not os.environ.get("LIBERO_CONFIG_PATH"):
        sys.exit(
            "[check] 未设置 LIBERO_CONFIG_PATH，先 source tools/activate_env.sh\n"
            "        否则 libero 会交互式提问并写 ~/.safelibero"
        )

    benchmark_dict = benchmark.get_benchmark_dict()
    ok, failed = 0, []

    for suite in SUITES:
        for level in LEVELS:
            task_suite = benchmark_dict[suite](safety_level=level)
            task_ids = [0] if args.quick else range(task_suite.n_tasks)
            for ti in task_ids:
                task = task_suite.get_task(ti)
                bddl = (
                    pathlib.Path(get_libero_path("bddl_files"))
                    / task.problem_folder
                    / task.bddl_file
                )
                try:
                    init = task_suite.get_task_init_states(ti)
                    env = OffScreenRenderEnv(
                        bddl_file_name=bddl, camera_heights=64, camera_widths=64
                    )
                    env.seed(7)
                    env.reset()
                    env.close()
                    print(
                        "OK   %-20s L%-2s task%d init%s  %s"
                        % (suite, level, ti, tuple(init.shape), task.language[:46])
                    )
                    ok += 1
                except Exception as e:  # noqa: BLE001
                    msg = repr(e)[:150]
                    print("FAIL %-20s L%-2s task%d  %s" % (suite, level, ti, msg))
                    failed.append((suite, level, ti, msg))

    print("\n通过 %d/%d" % (ok, ok + len(failed)))
    if failed:
        print("失败清单：")
        for item in failed:
            print("  ", item)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
