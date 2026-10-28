#!/usr/bin/env python
"""π0.5 策略加载 + 单次推理烟囱测试（不建环境，纯验证模型链路）。

用途：确认 12G 权重的 Orbax 加载、JAX 编译、跨 Python 版本的 IPC
（父进程 py3.8/.venv  ↔  子进程 py3.11//opt/venv/openpi）都通。

用法：
    cd /data/safeVLA && source tools/activate_env.sh
    python tools/smoke_policy.py
"""

import os
import sys
import time

sys.path.insert(0, "/data/2608.SDL/MojocoProj/src/pi05")

import numpy as np  # noqa: E402

import pi05_libero_runner as R  # noqa: E402


def main():
    print("[1/3] 检查权重")
    print("  ckpt:", R.DEFAULT_CKPT)
    print("  ready:", R.ckpt_ready(R.DEFAULT_CKPT))
    print("  openpi py:", R.OPENPI_PY)

    print("[2/3] 加载策略（首次含 JAX 编译，可能数分钟）")
    t0 = time.time()
    # 注意：runner 的 OPENPI_DATA_HOME 是模块级常量（默认 /opt/assets/.cache/openpi，
    # root 只读，openpi 启动时会 chmod 它 → PermissionError），必须显式传可写目录。
    policy = R.load_policy(
        "pi05_libero", R.DEFAULT_CKPT, num_infer_steps=10, verbose=True,
        openpi_data_home=os.environ["OPENPI_DATA_HOME"],
    )
    print("  加载耗时 %.1fs  ready=%s" % (time.time() - t0, policy.ready))

    print("[3/3] 单次推理（假观测）")
    dummy = {
        "observation/image": np.zeros((256, 256, 3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((256, 256, 3), dtype=np.uint8),
        "observation/state": np.zeros(8, dtype=np.float32),
        "prompt": "pick up the black bowl and place it on the plate",
    }
    t1 = time.time()
    chunk = policy.infer(dummy)
    print("  推理耗时 %.2fs  输出 shape=%s dtype=%s" % (time.time() - t1, chunk.shape, chunk.dtype))
    print("  动作范围 [%.3f, %.3f]" % (float(chunk.min()), float(chunk.max())))

    policy.close()
    print("\n烟囱测试通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
