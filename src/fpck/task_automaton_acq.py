"""FPCK 任务自动机 v1.3 — GRASP_ACQUISITION 状态（task_automaton_spec_v1.3）。

替代 v1.2 的 5mm 累计位移规则（v1.2 已冻结为
`AUTOMATON_V1_2_LEGACY_INVALID_FOR_GRASP_ACQUISITION`）。

核心变更（裁决）：
1. 新增 `q_acquisition` mode 与 `GRASP_ACQUISITION` lifecycle；
2. acquisition memory ξ_acq 进入状态（马尔可夫性）；
3. 5mm 累计位移在 acquisition 期间停止作为立即失败条件；
4. 任务对象运动改用组合条件（d_c>d_max ∧ ¬bilateral ∧ ¬unloading ∧ lateral_sweep）；
5. 进入 CONTROLLED 要求持续双指约束 + 物理控制证据（不要求夹爪完全闭合）；
6. acquisition 超时三分类（Resting / q_bad / q_unknown）。

本模块子类化 `TaskAutomaton`，复用 process 骨架中不变的部分（_detect_unknown、
_check_accept、_update_mode），仅覆盖 violation/lifecycle/controlled 相关方法。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from src.fpck.task_automaton import (
    TaskAutomaton, TaskEvent, AutomatonState,
    MODE_INIT, MODE_ENGAGED, MODE_CONTROLLED, MODE_PROGRESS, MODE_RELEASED,
    MODE_ACC, MODE_BAD, MODE_UNKNOWN,
    LIFT_THRESHOLD_MM, RELATIVE_DRIFT_MM, CONTROL_WITNESS_STEPS,
    TARGET_MOTION_MM, TRANSPORTED_MM,
    RELEASE_CONTACT_GONE_STEPS, RELEASE_VEL_STEPS, RELEASE_VEL_MPS,
)

SPEC_VERSION = "fpck-task-automaton-v1.3-grasp-acquisition"

MODE_ACQUISITION = "q_acquisition"

# 新违规码（裁决 §五.5：机器人非手指部位推动任务对象 = 非法推动）
VIOLATION_NON_FINGER_PUSH = "NON_FINGER_ROBOT_TARGET_CONTACT"

# ---------------------------------------------------------------------------
# 参数（独立依据，见 spec §7；非 state1 反向拟合）
# ---------------------------------------------------------------------------
ACQ_TIMEOUT_STEPS = 60        # acquisition 超时：夹爪全行程≈42 步 + 接触顺应裕量
ACQ_BILATERAL_STEPS = 3       # 双指约束持续步数（跨控制步持续，高于接触抖动尺度）
ACQ_DISP_MAX_MM = 15.0        # acquisition 包络：任务对象尺度~5-8cm，数 cm 级顺应合理
ACQ_LATERAL_SWEEP_MM = 10.0   # EEF 横向扫动阈值：抓取以竖向/旋转为主
ACQ_ENTRY_WINDOW = 5          # 接触先/命令先的 pending 窗口（控制步）

# 新 lifecycle 顺序（插入 GRASP_ACQUISITION）
LIFECYCLE_V13 = ("RESTING", "GRASP_ACQUISITION", "CONTROLLED",
                 "TRANSPORTED", "RELEASED")


@dataclass
class AcquisitionEvent(TaskEvent):
    """v1.3 扩展事件：acquisition 见证字段（裁决 §五）。"""

    # 接触身份（§五.2：有效接触仅限注册 finger pad）
    left_finger_contact: bool = False
    right_finger_contact: bool = False
    valid_finger_target_contact: bool = False      # finger pad 接触目标
    non_finger_robot_target_contact: bool = False  # link/wrist/housing 接触目标 = 非法推动
    # 夹爪意图/响应（§一/§十二：意图来自命令，响应来自 aperture）
    close_intent: bool = False
    closing_response: bool = False
    closure_blocked: bool = False
    aperture: Optional[float] = None
    # 支撑/抬升/横向/耦合
    support_contact: bool = False
    lift_mm: float = 0.0
    lateral_sweep_mm: float = 0.0
    relative_pose_error: Optional[float] = None


@dataclass
class AcquisitionMemory:
    """ξ_acq（spec §1.2）：必须进入状态（序列化），非仅日志。"""

    active: bool = False
    target: Optional[str] = None
    t_c: Optional[int] = None      # 首次合法 finger–target 接触步
    t_g: Optional[int] = None      # 首次 deployed close intent 步
    d_c0: float = 0.0              # 首次接触时的 baseline 位移（L1）
    d_c: float = 0.0               # 首次接触后累计位移（mm）
    C_L: bool = False
    C_R: bool = False
    S: bool = False                # 支撑面接触
    G: bool = False                # 物理闭合响应（closing_response ∨ closure_blocked）
    tau: int = 0                   # acquisition 计时（控制步）
    bilateral_steps: int = 0       # 双指约束持续步数
    lateral_sweep_mm: float = 0.0  # 末端横向累计位移
    pending_intent: bool = False   # 接触先：等 close intent
    pending_contact: bool = False  # 命令先：等 finger contact
    pending_tau: int = 0

    def snapshot(self) -> dict:
        return {
            "active": self.active, "target": self.target,
            "t_c": self.t_c, "t_g": self.t_g,
            "d_c0": self.d_c0, "d_c": self.d_c,
            "C_L": self.C_L, "C_R": self.C_R, "S": self.S, "G": self.G,
            "tau": self.tau, "bilateral_steps": self.bilateral_steps,
            "lateral_sweep_mm": self.lateral_sweep_mm,
            "pending_intent": self.pending_intent,
            "pending_contact": self.pending_contact,
            "pending_tau": self.pending_tau,
        }

    @classmethod
    def restore(cls, d: dict) -> "AcquisitionMemory":
        m = cls()
        m.active = bool(d.get("active")); m.target = d.get("target")
        m.t_c = d.get("t_c"); m.t_g = d.get("t_g")
        m.d_c0 = float(d.get("d_c0", 0.0)); m.d_c = float(d.get("d_c", 0.0))
        m.C_L = bool(d.get("C_L")); m.C_R = bool(d.get("C_R"))
        m.S = bool(d.get("S")); m.G = bool(d.get("G"))
        m.tau = int(d.get("tau", 0)); m.bilateral_steps = int(d.get("bilateral_steps", 0))
        m.lateral_sweep_mm = float(d.get("lateral_sweep_mm", 0.0))
        m.pending_intent = bool(d.get("pending_intent"))
        m.pending_contact = bool(d.get("pending_contact"))
        m.pending_tau = int(d.get("pending_tau", 0))
        return m


class GraspAcquisitionAutomaton(TaskAutomaton):
    """v1.3 自动机：显式 GRASP_ACQUISITION 状态。"""

    def __init__(self, manipulated_targets, goal_clause_keys):
        super().__init__(manipulated_targets, goal_clause_keys)
        self.state.spec_version = SPEC_VERSION
        self.acq_mem: Dict[str, AcquisitionMemory] = {
            t: AcquisitionMemory() for t in self.manipulated_targets
        }
        self._acq_active_target: Optional[str] = None  # 当前正在 acquisition 的 target

    # ------------------------------------------------------------------
    # 状态扩展
    # ------------------------------------------------------------------
    def snapshot(self) -> dict:
        snap = super().snapshot() if hasattr(super(), "snapshot") else self.state.snapshot()
        snap["acq_mem"] = {t: self.acq_mem[t].snapshot() for t in self.acq_mem}
        snap["acq_active_target"] = self._acq_active_target
        return snap

    def restore(self, snap: dict):
        # 先恢复自动机状态（父类有 snapshot/restore 语义则复用）
        st = snap.get("automaton", snap)
        self.state = AutomatonState(
            mode=st.get("mode", MODE_INIT),
            lifecycle=dict(st.get("lifecycle", {})),
            achieved_mask=dict(st.get("achieved_mask", {})),
            violation_code=st.get("violation_code"),
            spec_version=SPEC_VERSION,
        )
        for t in self.manipulated_targets:
            self.state.lifecycle.setdefault(t, "RESTING")
        acq = snap.get("acq_mem", {})
        for t in self.acq_mem:
            if t in acq:
                self.acq_mem[t] = AcquisitionMemory.restore(acq[t])
        self._acq_active_target = snap.get("acq_active_target")

    # ------------------------------------------------------------------
    # 主处理：覆盖 process 以控制 acquisition 顺序
    # ------------------------------------------------------------------
    def process(self, ev: AcquisitionEvent) -> AutomatonState:
        s = self.state
        if s.mode in (MODE_BAD, MODE_UNKNOWN, MODE_ACC):
            return s

        # 1. 硬违规（与 acquisition 无关，立即 q_bad）
        hard = self._detect_hard_violation(ev)
        if hard is not None:
            s.mode = MODE_BAD
            s.violation_code = hard
            return s

        # 2. 更新 acquisition 状态（进入/计时/累计位移/解除）——先于违规判定，
        #    使 acquisition 感知的违规判定能看到「本步是否已进入 acquisition」。
        self._update_acquisition(ev)

        # 3. acquisition-aware 目标运动违规（组合条件 + 超时三分类）
        motion_violation = self._detect_motion_violation(ev)
        if motion_violation is not None:
            s.mode = MODE_BAD
            s.violation_code = motion_violation
            return s

        # 4. 超时歧义 → q_unknown（传感器/机制不明）
        if self._acquisition_ambiguous(ev):
            s.mode = MODE_UNKNOWN
            return s

        # 5. 不确定检查（父类）
        if self._detect_unknown(ev):
            s.mode = MODE_UNKNOWN
            return s

        # 6. 更新 achieved_mask
        for clause, truth in ev.goal_clause_truth.items():
            if clause in s.achieved_mask and truth is not None:
                s.achieved_mask[clause] = truth

        # 7. 生命周期见证（v1.3）
        self._update_lifecycle(ev)

        # 8. NATIVE_FALSE_POSITIVE
        if ev.native_success and not self._check_accept(ev):
            s.mode = MODE_BAD
            s.violation_code = "NATIVE_FALSE_POSITIVE"
            return s

        # 9. 接受检查
        if self._check_accept(ev):
            s.mode = MODE_ACC
            return s

        # 10. 阶段推进
        self._update_mode(ev)
        return s

    # ------------------------------------------------------------------
    # 硬违规（§五.5：与 acquisition 无关，仍立即 q_bad）
    # ------------------------------------------------------------------
    def _detect_hard_violation(self, ev) -> Optional[str]:
        if ev.forbidden_contact:
            return "FORBIDDEN_CONTACT"
        if ev.obstacle_displacement_l1_mm > 1.0:
            return "STATIC_OBSTACLE_MOVED"
        if ev.payload_obstacle_contact:
            return "PAYLOAD_OBSTACLE_CONTACT"
        if ev.non_finger_robot_target_contact:
            return VIOLATION_NON_FINGER_PUSH
        return None

    # ------------------------------------------------------------------
    # acquisition 状态更新
    # ------------------------------------------------------------------
    def _active_mem(self) -> Optional[AcquisitionMemory]:
        if self._acq_active_target is not None:
            return self.acq_mem.get(self._acq_active_target)
        return None

    def _update_acquisition(self, ev: AcquisitionEvent):
        mem = self._active_mem()

        # 进入 acquisition（§3：close_intent ∧ valid_finger_target_contact）
        if mem is None:
            if ev.close_intent and ev.valid_finger_target_contact:
                t = self._which_target(ev)
                m = self.acq_mem[t]
                m.active = True
                m.target = t
                m.t_c = ev.step
                m.t_g = ev.step
                m.d_c0 = ev.target_displacement_l1_mm
                m.d_c = 0.0
                m.C_L = ev.left_finger_contact
                m.C_R = ev.right_finger_contact
                m.S = ev.support_contact
                m.G = ev.closing_response or ev.closure_blocked
                m.tau = 0
                m.bilateral_steps = 0
                m.lateral_sweep_mm = ev.lateral_sweep_mm
                self._acq_active_target = t
                self._advance_lifecycle(t, "GRASP_ACQUISITION")
            return

        # 已在 acquisition：计时 + 累计位移 + 双指持续 + 物理响应
        mem.tau += 1
        mem.d_c = max(0.0, ev.target_displacement_l1_mm - mem.d_c0)
        mem.C_L = ev.left_finger_contact
        mem.C_R = ev.right_finger_contact
        mem.S = ev.support_contact
        mem.G = ev.closing_response or ev.closure_blocked
        mem.lateral_sweep_mm = max(mem.lateral_sweep_mm, ev.lateral_sweep_mm)
        if ev.left_finger_contact and ev.right_finger_contact:
            mem.bilateral_steps += 1
        else:
            mem.bilateral_steps = 0

    def _which_target(self, ev) -> str:
        # 单目标任务直接取唯一 target；多目标时由 bridge 通过 valid 接触指定（占位取第一个）
        return self.manipulated_targets[0]

    def _release_acquisition(self):
        if self._acq_active_target is not None:
            self.acq_mem[self._acq_active_target].active = False
            self._acq_active_target = None

    # ------------------------------------------------------------------
    # acquisition-aware 目标运动违规（§4/§6）
    # ------------------------------------------------------------------
    def _detect_motion_violation(self, ev) -> Optional[str]:
        mem = self._active_mem()
        if mem is not None and mem.active:
            # acquisition 期间（含超时）：5mm 规则停止，只用组合条件
            if self._illegal_push(ev, mem):
                return "UNAUTHORIZED_TARGET_MOTION"
            return None

        # 非 acquisition：保留 v1.2 的 5mm 规则（无 close intent 保护时）
        if (ev.target_displacement_l1_mm > TARGET_MOTION_MM
                and not self._any_controlled()):
            return "UNAUTHORIZED_TARGET_MOTION"
        return None

    def _illegal_push(self, ev, mem) -> bool:
        bilateral = mem.bilateral_steps >= ACQ_BILATERAL_STEPS
        support_unloading = ev.lift_mm > 0.0
        lateral_sweep = mem.lateral_sweep_mm > ACQ_LATERAL_SWEEP_MM
        return (mem.d_c > ACQ_DISP_MAX_MM
                and not bilateral
                and not support_unloading
                and lateral_sweep)

    def _acquisition_ambiguous(self, ev) -> bool:
        """超时三分类之三：证据不足/机制不明 → q_unknown。"""
        mem = self._active_mem()
        if mem is None or not mem.active:
            return False
        if mem.tau <= ACQ_TIMEOUT_STEPS:
            return False
        # 超时但已满足 Controlled 见证：交给 lifecycle 推进，不判歧义
        if self._controlled_witness(ev):
            return False
        # 超时：目标明显移动但未满足 illegal_push（非持续推动、又非 Resting）→ 歧义
        return mem.d_c > TARGET_MOTION_MM

    # ------------------------------------------------------------------
    # 生命周期（v1.3）
    # ------------------------------------------------------------------
    def _advance_lifecycle(self, target, new_stage):
        if target not in self.state.lifecycle:
            self.state.lifecycle[target] = "RESTING"
        cur = self.state.lifecycle[target]
        order = {s: i for i, s in enumerate(LIFECYCLE_V13)}
        if new_stage == cur:
            return
        if new_stage == "RESTING" and cur in ("TASK_CONTACT", "GRASP_ACQUISITION"):
            self.state.lifecycle[target] = "RESTING"
            return
        if order.get(new_stage, -1) > order.get(cur, -1):
            self.state.lifecycle[target] = new_stage

    def _update_lifecycle(self, ev):
        mem = self._active_mem()
        for t in self.manipulated_targets:
            cur = self.state.lifecycle.get(t, "RESTING")
            if cur == "RESTING" and ev.task_contact:
                self._advance_lifecycle(t, "TASK_CONTACT")
            elif cur == "TASK_CONTACT" and not ev.task_contact:
                self._advance_lifecycle(t, "RESTING")
            elif cur in ("TASK_CONTACT", "GRASP_ACQUISITION") and self._controlled_witness(ev):
                self._advance_lifecycle(t, "CONTROLLED")
                if mem is not None and mem.target == t:
                    self._release_acquisition()
            elif cur == "GRASP_ACQUISITION" and self._acquisition_timed_out_resting():
                # 超时三分类之一：目标几乎未移动 → 返回 Resting（不 q_bad）
                self._advance_lifecycle(t, "RESTING")
                self._release_acquisition()
            elif cur == "CONTROLLED" and ev.target_displacement_l1_mm > TRANSPORTED_MM:
                self._advance_lifecycle(t, "TRANSPORTED")
            elif cur in ("CONTROLLED", "TRANSPORTED") and self._released_witness(ev):
                self._advance_lifecycle(t, "RELEASED")

    def _acquisition_timed_out_resting(self) -> bool:
        mem = self._active_mem()
        if mem is None or not mem.active:
            return False
        if mem.tau <= ACQ_TIMEOUT_STEPS:
            return False
        # 目标几乎未移动（d_c 未超包络）且无危险接触 → Resting
        return mem.d_c <= TARGET_MOTION_MM

    def _controlled_witness(self, ev) -> bool:
        """v1.3 CONTROLLED（§5）：持续双指约束 + 物理控制证据，不要求夹爪完全闭合。"""
        mem = self._active_mem()
        bilateral = mem is not None and mem.active and mem.bilateral_steps >= ACQ_BILATERAL_STEPS
        if not bilateral:
            return False
        close_sustained = ev.close_intent
        if not close_sustained:
            return False
        # 物理控制证据：受阻闭合 / 支撑卸载 / 抬升 / 耦合
        physical = (ev.closure_blocked
                    or ev.lift_mm >= LIFT_THRESHOLD_MM
                    or (ev.relative_pose_error is not None
                        and ev.relative_pose_error <= RELATIVE_DRIFT_MM))
        return bool(physical)

    # ------------------------------------------------------------------
    # mode 推进（含 q_acquisition）
    # ------------------------------------------------------------------
    def _update_mode(self, ev):
        s = self.state
        if self._any_transported_or_released():
            s.mode = MODE_RELEASED
        elif any(v is True for v in s.achieved_mask.values()):
            s.mode = MODE_PROGRESS
        elif self._any_controlled():
            s.mode = MODE_CONTROLLED
        elif any(v == "GRASP_ACQUISITION" for v in s.lifecycle.values()):
            s.mode = MODE_ACQUISITION
        elif any(v == "TASK_CONTACT" for v in s.lifecycle.values()):
            s.mode = MODE_ENGAGED
        else:
            # 无任何进展：acquisition 失败/超时回退后回到 q_init
            s.mode = MODE_INIT
