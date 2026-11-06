"""确定性实验启动器（Commit 1，裁决 §一/§二）。

在 JAX/openpi import 之前设置 XLA_FLAGS，启动后执行硬门禁。

设计原则（裁决 §一/§二）：
- 不静默修改 PnsgPolicy 的通用默认语义（`xla_flags=None` 仍是默认非确定配置）；
- 本模块提供显式的确定性 factory，任何 FPCK/CKG 正式实验必须经过它；
- XLA_FLAGS 通过 subprocess env 在 worker 进程 import jax 之前生效
  （worker 独立进程，import jax 前读 os.environ["XLA_FLAGS"]）；
- 硬门禁：worker 启动后验证 xla_flags_at_import 是否包含确定性配置，
  缺失 → 抛 DETERMINISM_CONFIGURATION_MISSING（不能只 warning 后继续）。

确定性配置（裁决 §一，经 fpck_determinism_matrix 实测）：
- --xla_gpu_autotune_level=0       消除编译期 kernel 选择非确定（已确认的根因）
- --xla_gpu_exclude_nondeterministic_ops=true  防止未来模型图引入运行期非确定算子
"""

from __future__ import annotations

from typing import Optional, Tuple

from src.pnsg.policy_adapter import PnsgPolicy

# 确定性配置（正式评测强制，裁决 §一）
DETERMINISTIC_XLA_FLAGS = (
    "--xla_gpu_autotune_level=0 "
    "--xla_gpu_exclude_nondeterministic_ops=true"
)

# 门禁版本（冻结，裁决 §一 expected_gate_version）
GATE_VERSION = "fpck_xla_det_v1"

# 硬门禁失败错误码（裁决 §二）
DETERMINISM_CONFIGURATION_MISSING = "DETERMINISM_CONFIGURATION_MISSING"


class DeterminismGateError(RuntimeError):
    """确定性门禁失败：正式实验不得在缺失确定性配置下启动。"""


def _check_xla_flags(xla_at_import: Optional[str]) -> None:
    """硬门禁：xla_flags_at_import 必须包含确定性配置。"""
    if not xla_at_import:
        raise DeterminismGateError(
            f"{DETERMINISM_CONFIGURATION_MISSING}: worker 未设置 XLA_FLAGS"
            f"（xla_flags_at_import=None），正式实验拒绝启动")
    missing = []
    for frag in ("--xla_gpu_autotune_level=0",
                 "--xla_gpu_exclude_nondeterministic_ops"):
        if frag not in xla_at_import:
            missing.append(frag)
    if missing:
        raise DeterminismGateError(
            f"{DETERMINISM_CONFIGURATION_MISSING}: 缺少确定性 flag {missing}，"
            f"实际 xla_flags_at_import={xla_at_import!r}")


def create_deterministic_policy(
        ckpt_dir=None, num_steps: int = 10,
        openpi_py=None) -> Tuple[PnsgPolicy, dict]:
    """创建确定性 policy + 执行硬门禁。

    返回 (policy, gate)，其中 gate 是运行时确定性门禁记录（写入 run manifest）。

    参数透传给 PnsgPolicy，但 xla_flags 固定为 DETERMINISTIC_XLA_FLAGS。
    """
    kwargs = {"xla_flags": DETERMINISTIC_XLA_FLAGS, "num_steps": num_steps}
    if ckpt_dir is not None:
        kwargs["ckpt_dir"] = ckpt_dir
    if openpi_py is not None:
        kwargs["openpi_py"] = openpi_py

    policy = PnsgPolicy(**kwargs)

    # 硬门禁：验证 worker 运行时身份
    try:
        info = policy.info
        xla_at_import = info.get("xla_flags_at_import")
        _check_xla_flags(xla_at_import)
    except DeterminismGateError:
        policy.close()
        raise
    except Exception as e:
        policy.close()
        raise DeterminismGateError(
            f"{DETERMINISM_CONFIGURATION_MISSING}: 读取 worker 身份失败: {e}")

    gate = {
        "gate_version": GATE_VERSION,
        "passed": True,
        "xla_flags": DETERMINISTIC_XLA_FLAGS,
        "xla_flags_at_import": xla_at_import,
        "xla_flags_env": info.get("xla_flags_env"),
        "jax_version": info.get("jax_version"),
        "jaxlib_version": info.get("jaxlib_version"),
        "xla_client_version": info.get("xla_client_version"),
        "default_backend": info.get("default_backend"),
        "devices": info.get("devices"),
    }
    return policy, gate


def determinism_gate_record(gate: dict) -> dict:
    """写入 run manifest 的确定性门禁记录（裁决 §二）。"""
    return {
        "determinism_gate_version": gate.get("gate_version"),
        "determinism_gate_passed": gate.get("passed"),
        "xla_flags": gate.get("xla_flags"),
        "xla_flags_at_import": gate.get("xla_flags_at_import"),
        "jax_version": gate.get("jax_version"),
        "jaxlib_version": gate.get("jaxlib_version"),
        "default_backend": gate.get("default_backend"),
        "devices": gate.get("devices"),
    }
