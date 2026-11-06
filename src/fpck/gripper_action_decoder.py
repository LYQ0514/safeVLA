"""夹爪动作解码器（deployed close intent 形式化定义，裁决 §十二前置）。

语义（PandaGripper，物理探针 `tools/fpck_gripper_action_probe.py` 确认）：
- 动作夹爪分量 action[6] ∈ [-1,1]，sign 决定方向：>0 = 闭合，<0 = 打开；
- robosuite `PandaGripper.format_action` 用 `np.sign(action)` 做 sticky 累积
  （speed=0.01，`current_action += [-1,1]*speed*sign(action)`）；
- aperture = gripper_qpos[0] - gripper_qpos[1]（对称手指，闭合→0，初始 0.0417）。

裁决核心原则：
- close_intent 只能来自「最终 deployed command 的夹爪分量」，**不能从 gripper_qpos 反推**；
- nominal（冻结 π0.5 意图）与 deployed（governor/安全层处理后实际执行）必须分开记录；
- actual closure（aperture / aperture_velocity / 受阻响应）是独立物理观测，与 intent 分开；
- 进入 GRASP_ACQUISITION 用 deployed_close_intent；nominal 仅用于审计策略意图与实际执行是否一致。

本模块是纯逻辑（不依赖 MuJoCo / env），可独立单元测试。
物理量（aperture、closing_response、controller_gripper_target）由 bridge 从 env 读入后
构造 GripperActionDecode，本模块只负责「命令 → 意图」的解码与迟滞。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

DECODER_VERSION = "panda-gripper-sign-v1"

# 阈值来自动作接口定义（[-1,1] 连续、sign 决定方向），非 state1 反向拟合。
# sign 边界天然在 0；为防单步噪声用死区 + 连续步迟滞（裁决 §五）。
THETA_ENTER = 0.0     # 命令 > 0 => 闭合方向
THETA_EXIT = -0.05    # 命令 < -0.05 => 明确打开（死区）
N_EXIT_STEPS = 2      # 连续 N_EXIT 步明确打开才清除 close 锁定（迟滞 θ_enter > θ_exit）

# 空夹爪机械限位：aperture 下界（闭合到物体阻挡前的最窄缝，来自 init_qpos 0.020833*2 ≈ 0.0417；
# 完全闭合≈0。此阈值仅用于区分「空夹爪到机械限位」与「物体阻挡闭合」，见裁决 §七）
APERTURE_FULLY_CLOSED = 0.005   # aperture < 此值 = 空夹爪已到机械限位（无物体阻挡）


@dataclass
class GripperActionDecode:
    """单控制步的夹爪解码结果（裁决 §十一 + §三流水线字段）。"""

    decoder_version: str = DECODER_VERSION

    # -- 命令管线（裁决 §三：policy → denorm → adapter → sticky → controller）--
    policy_gripper_raw: Optional[float] = None        # 冻结 π0.5 原始 external 动作夹爪分量
    deployed_gripper_command: Optional[float] = None  # governor/安全层处理后送入 env 的夹爪分量
    controller_gripper_target: Optional[Tuple] = None  # sticky 累积后的夹爪控制目标（gripper.current_action）

    # -- 意图（裁决 §一 四层：intent 与 physical response 分开）--
    nominal_close_intent: bool = False     # sign(policy_gripper_raw) > 0
    deployed_close_intent: bool = False    # sign(deployed_gripper_command) > θ_enter
    close_intent_latched: bool = False     # 迟滞锁定（enter/exit 阈值 + 连续步）
    close_intent_start_step: Optional[int] = None  # 首次锁定 close 的控制步

    # -- 物理响应（独立观测，bridge 从 env 读入）--
    aperture: Optional[float] = None              # qpos[0] - qpos[1]
    aperture_velocity: Optional[float] = None     # da/dt（每控制步）
    closing_response: bool = False                # close intent ∧ da/dt < -ε
    closure_blocked_by_object: bool = False       # close intent ∧ |da/dt|≈0 ∧ 双指接触 ∧ 未到空夹爪限位

    def snapshot(self) -> dict:
        return {
            "decoder_version": self.decoder_version,
            "policy_gripper_raw": self.policy_gripper_raw,
            "deployed_gripper_command": self.deployed_gripper_command,
            "controller_gripper_target": (list(self.controller_gripper_target)
                                          if self.controller_gripper_target is not None else None),
            "nominal_close_intent": self.nominal_close_intent,
            "deployed_close_intent": self.deployed_close_intent,
            "close_intent_latched": self.close_intent_latched,
            "close_intent_start_step": self.close_intent_start_step,
            "aperture": self.aperture,
            "aperture_velocity": self.aperture_velocity,
            "closing_response": self.closing_response,
            "closure_blocked_by_object": self.closure_blocked_by_object,
        }

    @classmethod
    def restore(cls, d: dict) -> "GripperActionDecode":
        out = cls()
        out.decoder_version = d.get("decoder_version", DECODER_VERSION)
        out.policy_gripper_raw = d.get("policy_gripper_raw")
        out.deployed_gripper_command = d.get("deployed_gripper_command")
        ct = d.get("controller_gripper_target")
        out.controller_gripper_target = tuple(ct) if ct is not None else None
        out.nominal_close_intent = bool(d.get("nominal_close_intent"))
        out.deployed_close_intent = bool(d.get("deployed_close_intent"))
        out.close_intent_latched = bool(d.get("close_intent_latched"))
        out.close_intent_start_step = d.get("close_intent_start_step")
        out.aperture = d.get("aperture")
        out.aperture_velocity = d.get("aperture_velocity")
        out.closing_response = bool(d.get("closing_response"))
        out.closure_blocked_by_object = bool(d.get("closure_blocked_by_object"))
        return out


def decode_close_intent(gripper_command: Optional[float]) -> bool:
    """sign 解码：>0 = 闭合方向（裁决 §四 sign 型）。

    None（命令缺失）→ False，不臆造意图。
    """
    if gripper_command is None:
        return False
    return float(gripper_command) > THETA_ENTER


class CloseIntentTracker:
    """close intent 迟滞跟踪器（裁决 §五：θ_enter > θ_exit + 连续步防噪）。

    命令 > THETA_ENTER → 立即锁定 close（deployed_close_intent 在命令下发时立即成立）；
    命令 < THETA_EXIT 连续 N_EXIT_STEPS 步才清除锁定（迟滞，防止单步噪声抖动）。
    """

    def __init__(self, theta_enter: float = THETA_ENTER,
                 theta_exit: float = THETA_EXIT,
                 n_exit_steps: int = N_EXIT_STEPS):
        self.theta_enter = theta_enter
        self.theta_exit = theta_exit
        self.n_exit_steps = n_exit_steps
        self.latched = False
        self.start_step: Optional[int] = None
        self._open_run = 0

    def update(self, gripper_command: Optional[float], step: int) -> bool:
        """更新迟滞状态，返回当前 close_intent_latched。"""
        if gripper_command is None:
            return self.latched
        cmd = float(gripper_command)
        if cmd > self.theta_enter:
            if not self.latched:
                self.latched = True
                self.start_step = step
            self._open_run = 0
        elif cmd < self.theta_exit:
            self._open_run += 1
            if self._open_run >= self.n_exit_steps:
                self.latched = False
                self.start_step = None
        else:
            # 死区：保持当前（不累积 open，也不清除）
            pass
        return self.latched

    def snapshot(self) -> dict:
        return {"latched": self.latched, "start_step": self.start_step,
                "open_run": self._open_run,
                "theta_enter": self.theta_enter, "theta_exit": self.theta_exit,
                "n_exit_steps": self.n_exit_steps}

    def restore(self, d: dict):
        self.latched = bool(d.get("latched"))
        self.start_step = d.get("start_step")
        self._open_run = int(d.get("open_run", 0))
        self.theta_enter = d.get("theta_enter", THETA_ENTER)
        self.theta_exit = d.get("theta_exit", THETA_EXIT)
        self.n_exit_steps = int(d.get("n_exit_steps", N_EXIT_STEPS))
        return self


def compute_closing_response(close_intent: bool, aperture_velocity: Optional[float],
                             eps: float = -1e-4) -> bool:
    """closing_response：close intent ∧ 实际 aperture 减小（裁决 §六 R_t^close）。

    eps 为每控制步 aperture 变化速率下界（负值 = 闭合方向），
    来自控制频率（0.05s/step）与 speed=0.01 的数量级，非 state1 拟合。
    """
    if not close_intent or aperture_velocity is None:
        return False
    return float(aperture_velocity) < eps


def compute_closure_blocked(close_intent: bool, aperture_velocity: Optional[float],
                            two_finger_contact: bool, aperture: Optional[float],
                            stall_eps: float = 2e-4) -> bool:
    """closure_blocked_by_object：close intent ∧ 双指接触 ∧ aperture 不再缩小 ∧ 未到空夹爪限位。

    裁决 §七：不能把「空夹爪到机械限位」误判为「物体阻挡闭合」。
    """
    if not close_intent or aperture_velocity is None or aperture is None:
        return False
    if not two_finger_contact:
        return False
    if float(aperture) <= APERTURE_FULLY_CLOSED:
        return False  # 空夹爪机械限位，非物体阻挡
    return abs(float(aperture_velocity)) <= stall_eps
