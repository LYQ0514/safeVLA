"""DUAL_ENDPOINT_HARNESS_V1 — 双终止 harness 结构（Commit 2）。

裁决 §四~§八：native success 只记录不终止，integrity 终止路由，horizon 计算接口，
显式必填配置，生产环境未验证配置硬失败。

本模块是**纯逻辑状态机**（不依赖 env/policy），单元测试覆盖全部转移。
科学参数（H_post、稳定窗口、速度阈值等）全部必填，无默认值；由 Commit 4 物理表征
回填并标 verified=true。生产环境未验证配置 → UNVERIFIED_TERMINAL_VALIDATION_CONFIG。

状态机（裁决 §四）：
    RUNNING_PRE_NATIVE → RUNNING_POST_NATIVE → TERMINATED_*
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# harness 状态
# ---------------------------------------------------------------------------
H_RUNNING_PRE_NATIVE = "RUNNING_PRE_NATIVE"
H_RUNNING_POST_NATIVE = "RUNNING_POST_NATIVE"
H_TERMINATED_INTEGRITY = "TERMINATED_INTEGRITY"
H_TERMINATED_TIMEOUT = "TERMINATED_TIMEOUT"
H_TERMINATED_RUNTIME_FAILURE = "TERMINATED_RUNTIME_FAILURE"

# integrity 终止路由（裁决 §八）
TERM_SAFE_REENTRY = "SAFE_REENTRY"
TERM_UNSAFE_RETURN = "UNSAFE_RETURN"
TERM_TASK_INTEGRITY_FAIL = "TASK_INTEGRITY_FAIL"
TERM_PREHANDOFF_SAFETY_FAIL = "PREHANDOFF_SAFETY_FAIL"
TERM_UNCERTAIN = "UNCERTAIN"
TERM_NATIVE_SUCCESS_WITHOUT_TERMINAL_WITNESS = "NATIVE_SUCCESS_WITHOUT_TERMINAL_WITNESS"
TERM_UNCERTAIN_TIMEOUT_PROGRESSING = "UNCERTAIN_TIMEOUT_PROGRESSING"
TERM_SAFE_STALL = "SAFE_STALL"
TERM_TASK_NOT_COMPLETED = "TASK_NOT_COMPLETED"
TERM_RUNTIME_FAILURE = "RUNTIME_FAILURE"
TERM_UNKNOWN_IDENTITY = "UNKNOWN_IDENTITY"

# 配置门禁错误码
UNVERIFIED_TERMINAL_VALIDATION_CONFIG = "UNVERIFIED_TERMINAL_VALIDATION_CONFIG"
MISSING_TERMINAL_VALIDATION_CONFIG = "MISSING_TERMINAL_VALIDATION_CONFIG"


class TerminalConfigError(RuntimeError):
    """未验证/缺失的终端验证配置：正式实验不得启动。"""


# ---------------------------------------------------------------------------
# 配置：所有科学参数必填，无默认值（裁决 §二）
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TerminalValidationConfig:
    """终端验证配置（Commit 4 表征回填后 verified=True）。"""

    # 时间域（裁决 §五）
    task_horizon: int                  # 原 SafeLIBERO episode horizon（控制步）
    post_native_seconds: float         # native success 后最短后验证窗口（秒）
    min_full_policy_chunks: int        # 要求观察的完整 policy chunk 数（≥1）
    # 物理阈值（Commit 4 表征）
    release_contact_gone_substeps: int  # finger contact 连续消失确认窗口
    stable_control_steps: int           # 目标稳定确认窗口（控制步）
    linear_velocity_max_mps: float      # 稳定线速度上界
    angular_velocity_max_radps: float   # 稳定角速度上界
    # 门禁元数据
    control_hz: float                  # 控制频率（物理表征记录，用于秒↔步换算）
    chunk_len: int                     # 单 chunk 控制步数（REPLAN）
    provenance: str                    # 表征出处（如 characterization artifact path/描述）
    characterization_sha256: str       # 表征 artifact hash（非空）
    verified: bool                     # 必须 True 才能进生产

    def __post_init__(self):
        if not self.characterization_sha256:
            raise TerminalConfigError(
                f"{MISSING_TERMINAL_VALIDATION_CONFIG}: characterization_sha256 为空")
        if self.min_full_policy_chunks < 1:
            raise TerminalConfigError(
                f"{MISSING_TERMINAL_VALIDATION_CONFIG}: min_full_policy_chunks 必须 ≥1")
        if self.post_native_seconds < 0:
            raise TerminalConfigError(
                f"{MISSING_TERMINAL_VALIDATION_CONFIG}: post_native_seconds 必须 ≥0")
        if self.chunk_len < 1:
            raise TerminalConfigError(
                f"{MISSING_TERMINAL_VALIDATION_CONFIG}: chunk_len 必须 ≥1")
        if self.control_hz <= 0:
            raise TerminalConfigError(
                f"{MISSING_TERMINAL_VALIDATION_CONFIG}: control_hz 必须 >0")

    def require_verified(self) -> None:
        """生产入口硬门禁：未验证配置拒绝启动（裁决 §二/§九.21）。"""
        if not self.verified:
            raise TerminalConfigError(
                f"{UNVERIFIED_TERMINAL_VALIDATION_CONFIG}: "
                f"provenance={self.provenance!r} 未通过物理表征验证")

    # -- horizon 计算（裁决 §五/§六）--
    def hard_cap(self) -> int:
        """绝对最大扩展时域 H_hard = H_task + H_post（控制步）。"""
        return self.task_horizon + self.post_steps()

    def post_steps(self) -> int:
        """H_post 的物理部分（秒 → 控制步）。"""
        return max(1, int(round(self.post_native_seconds * self.control_hz)))

    def post_chunk_steps(self, remaining_in_chunk: int) -> int:
        """H_chunk = r_n + K*L（裁决 §六：完成当前 chunk 剩余 + K 个完整 chunk）。

        remaining_in_chunk：native success 时当前 chunk 剩余控制步数。
        """
        return remaining_in_chunk + self.min_full_policy_chunks * self.chunk_len

    def post_window(self, remaining_in_chunk: int) -> int:
        """H_post = max(H_physical, H_chunk)。"""
        return max(self.post_steps(), self.post_chunk_steps(remaining_in_chunk))

    def integrity_deadline(self, t_native: int, remaining_in_chunk: int) -> int:
        """H_integrity = min(H_hard, max(H_task, t_n + H_post))。"""
        return min(self.hard_cap(),
                   max(self.task_horizon, t_native + self.post_window(remaining_in_chunk)))

    def snapshot(self) -> dict:
        return {
            "task_horizon": self.task_horizon,
            "post_native_seconds": self.post_native_seconds,
            "min_full_policy_chunks": self.min_full_policy_chunks,
            "release_contact_gone_substeps": self.release_contact_gone_substeps,
            "stable_control_steps": self.stable_control_steps,
            "linear_velocity_max_mps": self.linear_velocity_max_mps,
            "angular_velocity_max_radps": self.angular_velocity_max_radps,
            "control_hz": self.control_hz,
            "chunk_len": self.chunk_len,
            "provenance": self.provenance,
            "characterization_sha256": self.characterization_sha256,
            "verified": self.verified,
        }


# ---------------------------------------------------------------------------
# Native endpoint 不可变快照（裁决 §四）
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NativeEndpointRecord:
    """首次 native success 的不可变快照，后续不覆盖。"""
    step: int
    event_index: int
    simulator_time: float
    goal_clause_truth: Dict[str, Optional[bool]]
    automaton_mode: str
    target_lifecycle: Dict[str, str]
    target_linear_velocity: Dict[str, float]
    target_angular_velocity: Dict[str, float]
    contact_summary: Dict[str, bool]
    policy_chunk_sha256: str
    query_index: int
    buffer_index: int
    full_state_sha256: str

    def as_dict(self) -> dict:
        return {
            "step": self.step, "event_index": self.event_index,
            "simulator_time": self.simulator_time,
            "goal_clause_truth": dict(self.goal_clause_truth),
            "automaton_mode": self.automaton_mode,
            "target_lifecycle": dict(self.target_lifecycle),
            "target_linear_velocity": dict(self.target_linear_velocity),
            "target_angular_velocity": dict(self.target_angular_velocity),
            "contact_summary": dict(self.contact_summary),
            "policy_chunk_sha256": self.policy_chunk_sha256,
            "query_index": self.query_index, "buffer_index": self.buffer_index,
            "full_state_sha256": self.full_state_sha256,
        }


# ---------------------------------------------------------------------------
# 双 endpoint 状态机（纯逻辑）
# ---------------------------------------------------------------------------
class DualEndpointHarness:
    """双终止 harness（不依赖 env/policy，供采集循环驱动）。"""

    def __init__(self, config: TerminalValidationConfig):
        # 生产入口：未验证配置硬失败（裁决 §二）
        config.require_verified()
        self.config = config
        self.state = H_RUNNING_PRE_NATIVE
        self.native_record: Optional[NativeEndpointRecord] = None
        self._integrity_deadline: Optional[int] = None
        self.terminal_label: Optional[str] = None
        self.terminal_reason: Optional[str] = None

    # -- 每控制步驱动 --
    def observe(self, *, step: int, native_success: bool,
                remaining_in_chunk: int, **native_fields) -> str:
        """每步调用：处理 native endpoint 状态转移，返回当前 harness 状态。

        首次 native_success=True 时创建不可变快照并切到 RUNNING_POST_NATIVE，
        计算 integrity deadline；后续 native_success 不覆盖首次记录。
        native_fields 为 NativeEndpointRecord 所需字段。
        """
        if self.state == H_RUNNING_PRE_NATIVE and native_success:
            self.native_record = NativeEndpointRecord(
                step=step,
                event_index=native_fields.get("event_index", -1),
                simulator_time=native_fields.get("simulator_time", 0.0),
                goal_clause_truth=native_fields.get("goal_clause_truth", {}),
                automaton_mode=native_fields.get("automaton_mode", ""),
                target_lifecycle=native_fields.get("target_lifecycle", {}),
                target_linear_velocity=native_fields.get("target_linear_velocity", {}),
                target_angular_velocity=native_fields.get("target_angular_velocity", {}),
                contact_summary=native_fields.get("contact_summary", {}),
                policy_chunk_sha256=native_fields.get("policy_chunk_sha256", ""),
                query_index=native_fields.get("query_index", -1),
                buffer_index=native_fields.get("buffer_index", -1),
                full_state_sha256=native_fields.get("full_state_sha256", ""),
            )
            self._integrity_deadline = self.config.integrity_deadline(
                t_native=step, remaining_in_chunk=remaining_in_chunk)
            self.state = H_RUNNING_POST_NATIVE
        return self.state

    def deadline(self) -> Optional[int]:
        return self._integrity_deadline

    # -- integrity 终止（裁决 §七 优先级）--
    def terminate_integrity(self, *, identity_failure: bool = False,
                            automaton_mode: str = "",
                            violation_code: Optional[str] = None,
                            accepted: bool = False) -> str:
        """按固定优先级路由 integrity 终止，返回 terminal label。"""
        if self.state == H_TERMINATED_INTEGRITY:
            return self.terminal_label
        if identity_failure:
            self._set_term(H_TERMINATED_INTEGRITY, TERM_UNKNOWN_IDENTITY)
        elif automaton_mode == "q_bad":
            self._set_term(H_TERMINATED_INTEGRITY,
                           self._route_bad(violation_code))
        elif automaton_mode == "q_unknown":
            self._set_term(H_TERMINATED_INTEGRITY, TERM_UNCERTAIN)
        elif accepted or automaton_mode == "q_acc":
            self._set_term(H_TERMINATED_INTEGRITY, TERM_SAFE_REENTRY)
        else:
            # 未到终止条件：保持运行（不推进状态）
            return self.state
        return self.state

    def _route_bad(self, violation_code: Optional[str]) -> str:
        # 物理安全违规（handoff 前）= PREHANDOFF_SAFETY_FAIL；其余归 TASK_INTEGRITY_FAIL
        safety = {"FORBIDDEN_CONTACT", "STATIC_OBSTACLE_MOVED",
                  "PAYLOAD_OBSTACLE_CONTACT", "NON_FINGER_ROBOT_TARGET_CONTACT"}
        if violation_code in safety:
            return TERM_PREHANDOFF_SAFETY_FAIL
        return TERM_TASK_INTEGRITY_FAIL

    def terminate_timeout(self, *, native_seen: bool = False,
                          progressing: bool = False, stalled: bool = False,
                          task_completed: bool = False) -> str:
        """timeout 路由（裁决 §八）。"""
        if self.state == H_TERMINATED_TIMEOUT:
            return self.terminal_label
        if native_seen and not self.native_record is None:
            self._set_term(H_TERMINATED_TIMEOUT,
                           TERM_NATIVE_SUCCESS_WITHOUT_TERMINAL_WITNESS)
        elif progressing:
            self._set_term(H_TERMINATED_TIMEOUT, TERM_UNCERTAIN_TIMEOUT_PROGRESSING)
        elif stalled:
            self._set_term(H_TERMINATED_TIMEOUT, TERM_SAFE_STALL)
        else:
            self._set_term(H_TERMINATED_TIMEOUT, TERM_TASK_NOT_COMPLETED)
        return self.state

    def terminate_runtime_failure(self, reason: str = "") -> str:
        if self.state == H_TERMINATED_RUNTIME_FAILURE:
            return self.terminal_label
        self._set_term(H_TERMINATED_RUNTIME_FAILURE, TERM_RUNTIME_FAILURE, reason)
        return self.state

    def _set_term(self, state, label, reason=None):
        self.state = state
        self.terminal_label = label
        self.terminal_reason = reason

    def summary(self) -> dict:
        return {
            "harness_state": self.state,
            "terminal_label": self.terminal_label,
            "terminal_reason": self.terminal_reason,
            "native_success_seen": self.native_record is not None,
            "native_endpoint": self.native_record.as_dict() if self.native_record else None,
            "integrity_deadline": self._integrity_deadline,
        }
