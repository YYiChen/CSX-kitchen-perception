"""活动状态机：把每帧观测变成稳定的活动状态与事件。

设计原则：
- 状态由“候选状态 + 持续时间 + 冷却”确认，不把每帧原始检测直接当状态；
- ACTIVE_A/B 只由切菜节奏触发（用户确认的口径）；普通移动是 MOVING；
- 手检测不到时先给宽限，超时进入 OCCLUDED，绝不直接判 PAUSED；
- 只有确认的状态变化才写事件，事件本身带 evidence 供研究分析。
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Sequence

from .config import Config
from .geometry import clamp
from .motion import MotionFeatures

STATE_VALUES = (
    "UNKNOWN",
    "ACTIVE_A",
    "ACTIVE_B",
    "MOVING",
    "PAUSED",
    "TRANSITION_A_TO_B",
    "TRANSITION_B_TO_A",
    "OCCLUDED",
    "AWAY",
)

DEFAULT_STATE_ZH = {
    "UNKNOWN": "等待观察",
    "ACTIVE_A": "A 区切菜中",
    "ACTIVE_B": "B 区切菜中",
    "MOVING": "手部移动（未见切菜节奏）",
    "PAUSED": "暂停（未见切菜节奏）",
    "TRANSITION_A_TO_B": "A→B 区域转移",
    "TRANSITION_B_TO_A": "B→A 区域转移",
    "OCCLUDED": "手部被遮挡/观察不足",
    "AWAY": "暂未观察到用户",
}


@dataclass
class ActivityObservation:
    """一帧的输入；所有字段都来自已计算好的视觉/运动结果。"""

    ts_ms: float
    frame_index: int = 0
    person_visible: bool = False
    hands_valid: bool = False
    hands_valid_count: int = 0
    body_zone: str = "unknown"
    chopping_zone: str = "unknown"
    chopping: Optional[MotionFeatures] = None
    movement_level: str = "unknown"
    detection_gap_ms: float = 0.0
    stale: bool = False
    camera_status: str = "CONNECTED"


@dataclass
class ActivityEvent:
    event: str
    ts_ms: float
    frame: int
    state: str
    previous_state: str
    zone: str = "unknown"
    duration_ms: float = 0.0
    quality: str = "UNKNOWN"
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "ts_ms": round(float(self.ts_ms), 1),
            "frame": int(self.frame),
            "event": self.event,
            "state": self.state,
            "previous_state": self.previous_state,
            "zone": self.zone,
            "duration_ms": round(float(self.duration_ms), 1),
            "quality": self.quality,
            "event_source": "AUTOMATIC",
            "evidence": dict(self.evidence),
        }


class ActivityStateMachine:
    def __init__(self, config: Config) -> None:
        self.config = config
        activity = config.get("activity", {}) or {}
        chopping = config.get("chopping", {}) or {}
        self.enter_ms = float(activity.get("enter_ms", 500.0))
        self.pause_after_ms = float(activity.get("pause_after_ms", 2000.0))
        self.exit_ms = float(activity.get("exit_ms", 1500.0))
        self.occluded_after_ms = float(activity.get("occluded_after_ms", 800.0))
        self.away_after_ms = float(activity.get("away_after_ms", 5000.0))
        self.transition_window_ms = float(activity.get("transition_window_ms", 8000.0))
        self.transition_hold_ms = float(activity.get("transition_hold_ms", 1200.0))
        self.min_valid_ratio = float(activity.get("min_valid_ratio", 0.5))
        self.event_cooldown_ms = float(activity.get("event_cooldown_ms", 3000.0))
        self.min_duration_ms = float(activity.get("min_duration_ms", 500.0))
        self.hand_score_min = float(activity.get("hand_valid_score_min", 0.5))
        self.start_confirm_ms = float(chopping.get("start_confirm_ms", 600.0))
        self.stop_after_ms = float(chopping.get("stop_after_ms", 1500.0))
        labels = activity.get("labels_zh", {}) or {}
        self.state_zh = {**DEFAULT_STATE_ZH, **{str(k): str(v) for k, v in labels.items()}}
        self.reset()

    def reset(self, ts_ms: float = 0.0) -> None:
        self.state = "UNKNOWN"
        self.state_since = float(ts_ms)
        self.previous_state = "UNKNOWN"
        self.previous_stable_state = "UNKNOWN"
        self._candidate: Optional[str] = None
        self._candidate_since: Optional[float] = None
        self._candidate_mono: Optional[float] = None
        self._transition_target: Optional[str] = None
        self._transition_since: Optional[float] = None
        self._last_person_ms: Optional[float] = None
        self._last_hand_ms: Optional[float] = None
        self._last_activity_ms: Optional[float] = None
        self._last_seen_any_ms: Optional[float] = None
        self._start_ms: Optional[float] = None
        self._current_quality = "UNKNOWN"
        self._chopping_reported = False
        self._last_chop_ms: Optional[float] = None
        self._last_active_zone: Optional[str] = None
        self._last_active_end_ms: Optional[float] = None
        self._valid_samples: deque[tuple[float, bool]] = deque(maxlen=90)
        self._event_last: dict[str, float] = {}
        self._last_chopping = False
        self._last_state_event: str = ""
        self.events: list[ActivityEvent] = []

    # ------------------------------------------------------------------ 工具
    @staticmethod
    def _is_active(state: str) -> bool:
        return state in ("ACTIVE_A", "ACTIVE_B")

    @staticmethod
    def _zone_of(state: str) -> str:
        if state.endswith("_A"):
            return "A"
        if state.endswith("_B"):
            return "B"
        return "unknown"

    def _valid_ratio(self, now_ms: float) -> float:
        self._valid_samples.append((now_ms, bool(self._last_input_valid)))
        window_start = now_ms - 1000.0
        while self._valid_samples and self._valid_samples[0][0] < window_start:
            self._valid_samples.popleft()
        if not self._valid_samples:
            return 0.0
        return sum(1 for _, value in self._valid_samples if value) / len(self._valid_samples)

    def _cooldown_ok(self, event: str, now_ms: float) -> bool:
        last = self._event_last.get(event)
        if last is not None and now_ms - last < self.event_cooldown_ms:
            return False
        self._event_last[event] = now_ms
        return True

    def _emit(self, event: str, obs: ActivityObservation, previous: str, state: str, evidence: dict, force: bool = False) -> None:
        if not force and not self._cooldown_ok(event, obs.ts_ms):
            return
        if force:
            self._event_last[event] = obs.ts_ms
        duration = max(0.0, obs.ts_ms - self.state_since)
        self.events.append(
            ActivityEvent(
                event=event,
                ts_ms=obs.ts_ms,
                frame=obs.frame_index,
                state=state,
                previous_state=previous,
                zone=self._zone_of(state) if state.startswith(("ACTIVE", "TRANSITION")) else obs.body_zone,
                duration_ms=duration,
                quality=self._current_quality,
                evidence=evidence,
            )
        )

    def _set_state(self, new_state: str, obs: ActivityObservation, evidence: Optional[dict] = None) -> None:
        if new_state == self.state:
            return
        old = self.state
        self.previous_state = old
        if self._is_active(old) and old != new_state:
            zone = self._zone_of(old)
            self.previous_stable_state = old
            self._last_active_zone = zone
            self._last_active_end_ms = obs.ts_ms
            # A→B 转移时切菜没有真正停止，不在这里发 CHOPPING_STOPPED
            if self._chopping_reported and not new_state.startswith("TRANSITION"):
                self._emit(
                    "CHOPPING_STOPPED",
                    obs,
                    old,
                    new_state,
                    {"zone": zone, "state_duration_ms": round(max(0.0, obs.ts_ms - self.state_since), 1)},
                )
                self._chopping_reported = False
        if self._is_active(new_state):
            self.previous_stable_state = new_state
            self._last_active_zone = self._zone_of(new_state)
            self._last_active_end_ms = None
            if not self._chopping_reported:
                self._emit("CHOPPING_STARTED", obs, old, new_state, evidence or {})
                self._chopping_reported = True
        self.state = new_state
        self.state_since = obs.ts_ms
        self._candidate = None
        self._candidate_since = None
        self._candidate_mono = None
        self._last_state_event = new_state

    def _state_delay(self, candidate: str) -> float:
        if candidate in ("ACTIVE_A", "ACTIVE_B"):
            return self.start_confirm_ms
        if candidate == "MOVING":
            return self.enter_ms
        if candidate == "PAUSED":
            return self.pause_after_ms
        if candidate == "OCCLUDED":
            # 调用方已经按 last_hand 等满 occluded_after_ms，这里不再叠加一次延迟
            return 0.0
        if candidate == "AWAY":
            return self.away_after_ms
        if candidate.startswith("TRANSITION"):
            return self.min_duration_ms
        return 0.0

    def _offer_candidate(self, candidate: str, obs: ActivityObservation) -> bool:
        """候选状态持续够久才提交；返回本帧是否切换。"""
        now = float(obs.ts_ms)
        if candidate == self.state:
            self._candidate = None
            self._candidate_since = None
            return False
        if candidate == "AWAY" and self.state != "AWAY":
            # 离开是安全优先状态，满足条件后直接进入，避免被“最短持续时间”拖住
            old = self.state
            self._set_state(candidate, obs, {"last_person_ms": self._last_person_ms})
            self._emit("USER_AWAY", obs, old, candidate, {"last_person_ms": self._last_person_ms})
            return True
        if self._candidate != candidate:
            self._candidate = candidate
            self._candidate_since = now
            self._candidate_mono = now
            return False
        elapsed = now - (self._candidate_since if self._candidate_since is not None else now)
        if elapsed >= self._state_delay(candidate):
            # 最短持续时间：刚进入的状态不因抖动立刻离开（AWAY/OCCLUDED 例外）
            if now - self.state_since < self.min_duration_ms and candidate not in ("OCCLUDED", "AWAY"):
                return False
            old = self.state
            self._set_state(candidate, obs, {"candidate_ms": round(elapsed, 1)})
            self._emit_state_specific_events(old, candidate, obs)
            return True
        return False

    def _emit_state_specific_events(self, old: str, new: str, obs: ActivityObservation) -> None:
        evidence = {"movement_level": obs.movement_level, "hands_valid_count": obs.hands_valid_count, "body_zone": obs.body_zone}
        if old == "AWAY" and new != "AWAY":
            self._emit("USER_RETURNED", obs, old, new, evidence)
        if new in ("ACTIVE_A", "ACTIVE_B"):
            event = "ACTIVITY_A_STARTED" if new == "ACTIVE_A" else "ACTIVITY_B_STARTED"
            self._emit(event, obs, old, new, evidence, force=True)
        elif new == "PAUSED":
            self._emit("ACTIVITY_PAUSED", obs, old, new, evidence, force=True)
        elif new == "MOVING":
            self._emit("HAND_MOVING", obs, old, new, evidence)
        elif new == "OCCLUDED":
            self._emit("HAND_OCCLUDED", obs, old, new, evidence, force=True)
        elif new in ("TRANSITION_A_TO_B", "TRANSITION_B_TO_A"):
            event = "ZONE_TRANSITION_A_TO_B" if new.endswith("A_TO_B") else "ZONE_TRANSITION_B_TO_A"
            self._emit(event, obs, old, new, evidence, force=True)
        elif new == "AWAY":
            self._emit("USER_AWAY", obs, old, new, evidence, force=True)

    # ------------------------------------------------------------------ 主逻辑
    def update(self, obs: ActivityObservation) -> dict:
        """输入一帧观测，输出 activity 字典；事件挂在 self.events（调用方取走后清空）。"""
        self.events = []
        now = float(obs.ts_ms)
        if self._start_ms is None:
            self._start_ms = now
        self._last_input_valid = bool(obs.person_visible and obs.hands_valid)
        quality = self._observation_quality(obs)
        self._current_quality = quality
        self._last_seen_any_ms = now
        if obs.person_visible:
            self._last_person_ms = now
        if obs.hands_valid:
            self._last_hand_ms = now
            self._last_activity_ms = now
        if obs.chopping is not None and obs.chopping.detected:
            self._last_activity_ms = now
            self._last_chop_ms = obs.chopping.last_chop_ms or now

        person_visible = bool(obs.person_visible)
        hands_valid = bool(obs.hands_valid)
        hand_ratio = self._valid_ratio(now)
        chopping = obs.chopping
        chopping_detected = bool(chopping is not None and chopping.detected)
        zone = obs.body_zone if obs.body_zone in ("A", "B") else (obs.chopping_zone if obs.chopping_zone in ("A", "B") else "unknown")
        # 切菜已经停止：ACTIVE 状态里先发一次 CHOPPING_STOPPED，再等状态退出延迟
        if (
            self._is_active(self.state)
            and not chopping_detected
            and self._chopping_reported
            and self._last_chop_ms is not None
            and now - self._last_chop_ms >= self.stop_after_ms
        ):
            self._emit(
                "CHOPPING_STOPPED",
                obs,
                self.state,
                self.state,
                {"silent_ms": round(now - self._last_chop_ms, 1), "zone": self._zone_of(self.state)},
            )
            self._chopping_reported = False

        # 1) 无人/长时间无有效观察 → AWAY
        last_person = self._last_person_ms if self._last_person_ms is not None else self._start_ms
        last_activity = self._last_activity_ms if self._last_activity_ms is not None else self._start_ms
        away_reference = max(last_person, last_activity)
        if not person_visible and now - away_reference >= self.away_after_ms:
            if self.state != "AWAY":
                self._set_state("AWAY", obs, {"no_person_ms": round(now - away_reference, 1)})
                self._emit("USER_AWAY", obs, self.previous_state, "AWAY", {"no_person_ms": round(now - away_reference, 1)}, force=True)
            self._last_chopping = False
            return self._result(obs, quality, hand_ratio, zone, chopping)

        # 2) 人还在/刚离开，手不可靠 → 宽限后 OCCLUDED；不要判 PAUSED
        if not hands_valid:
            if not person_visible:
                # 画面里没有人：先回 UNKNOWN，不把“无人”误报成“手部遮挡”；
                # 等 away_after_ms 满足后再进入 AWAY。
                if self.state != "UNKNOWN":
                    self._offer_candidate("UNKNOWN", obs)
                self._last_chopping = False
                return self._result(obs, quality, hand_ratio, zone, chopping)
            last_hand = self._last_hand_ms if self._last_hand_ms is not None else self._start_ms
            if now - last_hand >= self.occluded_after_ms:
                self._offer_candidate("OCCLUDED", obs)
            else:
                self._candidate = None
                self._candidate_since = None
            self._last_chopping = False
            return self._result(obs, quality, hand_ratio, zone, chopping)

        # 3) 手可靠：切菜 → ACTIVE/TRANSITION；否则 MOVING/PAUSED
        if chopping_detected and zone in ("A", "B"):
            target_state = "ACTIVE_" + zone
            if self._is_active(self.state):
                prior_zone = self._zone_of(self.state)
                active_gap_ms = 0.0
            else:
                prior_zone = self._last_active_zone or ""
                active_gap_ms = (
                    max(0.0, now - self._last_active_end_ms)
                    if self._last_active_end_ms is not None
                    else float("inf")
                )
            within_transition = prior_zone in ("A", "B") and prior_zone != zone and active_gap_ms <= self.transition_window_ms
            if self.state.startswith("TRANSITION"):
                # 转移保持期结束后进入目标 ACTIVE
                if now - self.state_since >= self.transition_hold_ms:
                    self._set_state(target_state, obs, {"transition_done_ms": round(now - self.state_since, 1)})
                    self._emit(
                        "ACTIVITY_A_STARTED" if target_state == "ACTIVE_A" else "ACTIVITY_B_STARTED",
                        obs,
                        "TRANSITION",
                        target_state,
                        {"zone": zone},
                        force=True,
                    )
                    self.previous_stable_state = target_state
            elif within_transition and not self.state.startswith("TRANSITION"):
                transition = f"TRANSITION_{prior_zone}_TO_{zone}"
                self._set_state(transition, obs, {"from": prior_zone, "to": zone})
                self._emit(
                    "ZONE_TRANSITION_A_TO_B" if zone == "B" else "ZONE_TRANSITION_B_TO_A",
                    obs,
                    self.previous_state,
                    transition,
                    {"from": prior_zone, "to": zone, "gap_ms": round(active_gap_ms, 1)},
                    force=True,
                )
                self._transition_target = target_state
            else:
                self._offer_candidate(target_state, obs)
            self._last_chopping = chopping_detected
        elif chopping_detected and zone not in ("A", "B"):
            self._last_chopping = False
            self._offer_candidate("MOVING", obs)
        else:
            self._last_chopping = False
            moving = obs.movement_level in ("MEDIUM", "HIGH")
            if moving:
                self._offer_candidate("MOVING", obs)
            else:
                self._offer_candidate("PAUSED", obs)
        return self._result(obs, quality, hand_ratio, zone, chopping)

    def _observation_quality(self, obs: ActivityObservation) -> str:
        if not obs.person_visible:
            return "NO_PERSON"
        if not obs.hands_valid:
            return "OCCLUDED"
        if obs.stale or obs.detection_gap_ms > 200.0:
            return "LOW"
        return "VALID"

    def _result(
        self,
        obs: ActivityObservation,
        quality: str,
        hand_ratio: float,
        zone: str,
        chopping: Optional[MotionFeatures],
    ) -> dict:
        duration = max(0.0, obs.ts_ms - self.state_since)
        chopping_dict = {}
        if chopping is not None:
            chopping_dict = chopping.to_dict()
            chopping_dict["zone"] = zone if zone in ("A", "B") else "unknown"
        state_zone = self._zone_of(self.state) if self.state.startswith(("ACTIVE", "TRANSITION")) else zone
        last_event = self.events[-1].event if self.events else ""
        return {
            "state": self.state,
            "state_zh": self.state_zh.get(self.state, self.state),
            "name": self.state,
            "duration_ms": round(duration, 1),
            "previous_state": self.previous_state,
            "previous_stable_state": self.previous_stable_state,
            "zone": state_zone,
            "movement_level": obs.movement_level,
            "observation_quality": quality,
            "hands_valid_ratio": round(clamp(hand_ratio, 0.0, 1.0), 3),
            "event": last_event,
            "event_source": "AUTOMATIC",
            "state_since_ms": round(self.state_since, 1),
            "last_chop_ms": None if self._last_chop_ms is None else round(float(self._last_chop_ms), 1),
            "chopping": chopping_dict,
            "zone_source": "body" if obs.body_zone in ("A", "B") else ("hand" if obs.chopping_zone in ("A", "B") else "none"),
        }

    def take_events(self) -> list[ActivityEvent]:
        events = list(self.events)
        self.events = []
        return events
