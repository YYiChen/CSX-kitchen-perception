"""感知流水线：视频源 → 姿态/手部推理 → 躯干/手指分析 → 快照。

这一层是 UI、CLI、benchmark 共用的核心，不依赖 Tkinter。
- 延迟预算（LatencyBudget）按实测 p95 自动在 L0–L3 降级；
- 降级时复用上一次推理结果，并在 frame 里标记 stale_pose / stale_hands；
- 直立基准标定结果保存在 outputs/calibration.json（可选）。
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .activity import ActivityObservation, ActivityStateMachine
from .budget import FpsMeter, LatencyBudget
from .config import Config, load_config
from .engines import HandEngine, ModelError, PoseEngine, resolve_model_paths, verify_models
from .events import EventLogger
from .fingers import HandAnalyzer
from .motion import HandMotionTracker, MotionObservation
from .overlay import OverlayRenderer
from .sources import FrameSource, create_source
from .torso import TorsoAnalyzer, hands_in_workspace
from .types import HandState, PerceptionSnapshot, PoseFrame, TorsoState
from .zones import ZoneManager, classify_x, view_x

LOGGER = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, config: Config, source: Optional[FrameSource] = None, enable_overlay: bool = False) -> None:
        self.config = config
        errors = verify_models(config)
        if errors:
            raise ModelError("模型校验失败：\n" + "\n".join(errors))
        paths = resolve_model_paths(config)
        self.pose_engine = PoseEngine(paths["pose"], config)
        self.hand_engine = HandEngine(paths["hand"], config)
        self.torso = TorsoAnalyzer(config, offsets=self._load_calibration())
        self.hand_analyzer = HandAnalyzer(config)
        self.budget = LatencyBudget(config)
        self.fps = FpsMeter()
        self.source = source
        self.renderer = OverlayRenderer(config) if enable_overlay else None
        self.frame_index = 0
        self.last_overlay: Optional[np.ndarray] = None
        self.last_snapshot: Optional[dict] = None
        self._last_pose: Optional[PoseFrame] = None
        self._last_pose_ts = 0.0
        self._last_hands: list = []
        self._last_hand_ts = 0.0
        self._hand_tracks: dict[str, tuple[float, float, float]] = {}
        self._prev_posture: Optional[str] = None
        self._prev_hands_state: Optional[str] = None
        self._calibration_path = config.resolve_path(config.get("torso.calibration_file", "outputs/calibration.json"))
        self._source_kind = source.kind if source else "none"
        self._source_desc = getattr(source, "description", "") if source else ""
        self.schema_version = str(config.get("project.schema_version", "1.1"))
        self.zones = ZoneManager(config)
        self.motion = HandMotionTracker(config)
        self.activity = ActivityStateMachine(config)
        self.event_logger = EventLogger(config)
        self.camera_status = "CONNECTED"
        self._session_logged = False
        self._last_ts_ms = 0.0

    # ------------------------------------------------------------------ 配置
    def _load_calibration(self) -> Optional[dict]:
        path = self.config.resolve_path(self.config.get("torso.calibration_file", "outputs/calibration.json"))
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as fp:
                    data = json.load(fp)
                LOGGER.info("加载直立基准: %s", data)
                return data
            except (OSError, json.JSONDecodeError) as exc:
                LOGGER.warning("直立基准读取失败: %s", exc)
        return None

    def calibrate(self) -> dict:
        offsets = self.torso.calibrate()
        try:
            self._calibration_path.parent.mkdir(parents=True, exist_ok=True)
            with self._calibration_path.open("w", encoding="utf-8") as fp:
                json.dump(offsets, fp, ensure_ascii=False, indent=2)
        except OSError as exc:
            LOGGER.warning("直立基准保存失败: %s", exc)
        return offsets

    def clear_calibration(self) -> None:
        self.torso.clear_calibration()
        try:
            if self._calibration_path.exists():
                self._calibration_path.unlink()
        except OSError:
            pass

    # ------------------------------------------------------------------ 分区配置
    def apply_zone_settings(self, data: dict, persist: bool = False) -> dict:
        """由 UI 调用：应用 A/B 分区配置；persist=True 时同时写 config/zones.json。"""
        settings = self.zones.apply_settings(data, persist=persist)
        if persist:
            try:
                self.event_logger.write(
                    {
                        "ts_ms": round(float(self._last_ts_ms), 1),
                        "frame": int(self.frame_index),
                        "event": "ZONE_CONFIG_CHANGED",
                        "state": self.activity.state,
                        "previous_state": self.activity.previous_state,
                        "zone": self.activity._zone_of(self.activity.state),
                        "duration_ms": 0.0,
                        "quality": "UNKNOWN",
                        "event_source": "AUTOMATIC",
                        "evidence": settings.to_dict(),
                    }
                )
            except Exception:
                LOGGER.exception("记录分区配置变更失败")
        return settings.to_dict()

    # ------------------------------------------------------------------ 主流程
    def _hands_busy(self, states: list[HandState], hand_frames: list, ts_ms: float) -> None:
        busy_speed = float(self.config.get("motion.busy_speed_norm_per_s", 0.35))
        for index, state in enumerate(states):
            if state.wrist_norm is None:
                continue
            key = f"{state.label}#{index}"
            x, y = state.wrist_norm
            prev = self._hand_tracks.get(key)
            speed = 0.0
            if prev is not None:
                dt_s = max((ts_ms - prev[2]) / 1000.0, 1e-3)
                dist = math.hypot(x - prev[0], y - prev[1])
                speed = dist / dt_s
            if speed > busy_speed:
                state.busy = True
            self._hand_tracks[key] = (x, y, ts_ms)

    def _torso_scale(self, torso: TorsoState, height: int) -> float:
        """把竖直位移换算成躯干尺度，降低人离镜头远近对阈值的影响。"""
        if height <= 0:
            return 0.25
        length = float(torso.features.torso_length_px) if torso.features.torso_length_px else 0.0
        if math.isfinite(length) and length > 0:
            return max(0.05, min(0.5, length / float(height)))
        width = float(torso.features.shoulder_width_px) if torso.features.shoulder_width_px else 0.0
        if math.isfinite(width) and width > 0:
            return max(0.03, min(0.4, width / float(height)))
        return 0.25

    def _motion_observations(
        self,
        pose: PoseFrame,
        hand_states: list[HandState],
        hand_frames: list,
    ) -> list[MotionObservation]:
        """掌心优先；某侧没有可靠手时用姿态手腕兜底，保证降级隔帧也能算节奏。"""
        min_score = float(self.config.get("chopping.min_hand_score", 0.5))
        observations: list[MotionObservation] = []
        for index, state in enumerate(hand_states):
            frame = hand_frames[index] if index < len(hand_frames) else None
            valid = bool(state.palm_norm is not None and frame is not None and float(frame.score) >= min_score)
            if valid and state.palm_norm is not None:
                observations.append(
                    MotionObservation(
                        key=f"{state.label}#{index}",
                        label=state.label,
                        x=float(state.palm_norm[0]),
                        y=float(state.palm_norm[1]),
                        valid=True,
                        source="palm",
                        score=float(frame.score),
                    )
                )
        if not bool(self.config.get("chopping.use_pose_wrist_fallback", True)):
            return observations
        if not pose.present:
            return observations
        min_visibility = float(self.config.get("zones.body_visibility_min", 0.3))
        covered = {obs.label for obs in observations if obs.valid}
        for label, index in (("left", 15), ("right", 16)):
            if label in covered or index >= len(pose.landmarks):
                continue
            visibility = pose.visibility(index)
            if visibility < min_visibility:
                continue
            landmark = pose.landmarks[index]
            observations.append(
                MotionObservation(
                    key=f"pose:{label}",
                    label=label,
                    x=float(landmark.x),
                    y=float(landmark.y),
                    valid=True,
                    source="wrist",
                    score=visibility,
                )
            )
        return observations

    def _motion_zone(self, feature, hand_states: list[HandState], observations: list[MotionObservation]) -> str:
        """把选中的运动轨迹映射回 A/B；手部用 ZoneManager 写好的 zone，手腕兜底现算。"""
        for index, state in enumerate(hand_states):
            if feature is not None and feature.key == f"{state.label}#{index}":
                return state.zone if state.zone in ("A", "B") else "unknown"
        if feature is not None:
            for obs in observations:
                if obs.key == feature.key and obs.x is not None:
                    return classify_x(
                        view_x(obs.x, self.zones.mirror),
                        self.zones.settings,
                        "unknown",
                    )
        return "unknown"

    def process(self, frame: np.ndarray, ts_ms: float) -> dict:
        t_start = time.perf_counter()
        self._last_ts_ms = float(ts_ms)
        height, width = frame.shape[:2]

        self.frame_index += 1
        run_pose = self._last_pose is None or (self.frame_index - 1) % self.budget.pose_every_n == 0
        run_hands = self._last_hands is None or (self.frame_index - 1) % self.budget.hand_every_n == 0

        t0 = time.perf_counter()
        if run_pose:
            pose = self.pose_engine.detect(frame, ts_ms)
            self._last_pose = pose
            self._last_pose_ts = ts_ms
        else:
            pose = self._last_pose
        pose_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        if run_hands:
            hand_frames = self.hand_engine.detect(frame, ts_ms)
            self._last_hands = hand_frames
            self._last_hand_ts = ts_ms
        else:
            hand_frames = self._last_hands
        hand_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        torso_state = self.torso.analyze(pose, width, height, ts_ms)

        aspect = float(width) / float(height) if height else 1.0
        hand_states = [self.hand_analyzer.analyze(hf, aspect=aspect) for hf in hand_frames]
        for hand_state in hand_states:
            # 隔帧复用的手势结果要标 stale，避免下游把旧手部结果当成本帧新观测。
            hand_state.stale = not run_hands
        margin = float(self.config.get("motion.in_workspace_margin", 0.1))
        hands_in_ws = hands_in_workspace([hs.bbox_norm for hs in hand_states], torso_state.features.bbox_norm, margin)
        torso_state.hands_in_workspace = hands_in_ws
        for hs in hand_states:
            hs.in_workspace = hands_in_ws
        self._hands_busy(hand_states, hand_frames, ts_ms)
        analyze_ms = (time.perf_counter() - t0) * 1000.0

        if not hand_states:
            hands_state = "no_hands"
        elif any(h.busy for h in hand_states):
            hands_state = "hands_busy"
        elif any(h.in_workspace for h in hand_states):
            hands_state = "in_workspace"
        else:
            hands_state = "hands_visible"

        # ---------------------------------------------------------- 情境感知
        activity_start = time.perf_counter()
        zones_info = self.zones.update(pose, torso_state, hand_states, hand_frames, ts_ms)
        observations = self._motion_observations(pose, hand_states, hand_frames)
        motion_features = self.motion.update(observations, ts_ms, self._torso_scale(torso_state, height))
        best = HandMotionTracker.best(motion_features)
        chopping_zone = self._motion_zone(best, hand_states, observations)
        hands_valid_count = sum(1 for hs in hand_states if hs.palm_norm is not None)
        hands_valid = hands_valid_count > 0
        person_visible = bool(pose.present and torso_state.posture not in ("no_person", "not_visible"))
        detection_gap_ms = max((float(f.sampling_gap_ms) for f in motion_features), default=0.0)
        activity_obs = ActivityObservation(
            ts_ms=float(ts_ms),
            frame_index=self.frame_index,
            person_visible=person_visible,
            hands_valid=hands_valid,
            hands_valid_count=hands_valid_count,
            body_zone=str(zones_info.get("body_zone", "unknown")),
            chopping_zone=chopping_zone,
            chopping=best,
            movement_level=(best.movement_level if best is not None else "unknown"),
            detection_gap_ms=detection_gap_ms,
            stale=not run_hands,
            camera_status=self.camera_status,
        )
        activity_info = self.activity.update(activity_obs)
        activity_events = self.activity.take_events()
        activity_ms = (time.perf_counter() - activity_start) * 1000.0

        if not self._session_logged:
            self._session_logged = True
            self.event_logger.write(
                {
                    "ts_ms": round(float(ts_ms), 1),
                    "frame": int(self.frame_index),
                    "event": "SESSION_STARTED",
                    "state": self.activity.state,
                    "previous_state": "UNKNOWN",
                    "zone": zones_info.get("body_zone", "unknown"),
                    "duration_ms": 0.0,
                    "quality": activity_info.get("observation_quality", "UNKNOWN"),
                    "event_source": "AUTOMATIC",
                    "evidence": {
                        "schema_version": self.schema_version,
                        "source": self._source_kind,
                        "source_description": self._source_desc,
                        "split_x": zones_info.get("split_x"),
                        "view_mirrored": zones_info.get("view_mirrored"),
                        "zone_warnings": zones_info.get("warnings", []),
                    },
                }
            )

        events: list[dict] = []
        for event in activity_events:
            payload = event.to_dict()
            events.append(payload)
            self.event_logger.write(payload)
        if self._prev_posture is not None and torso_state.posture != self._prev_posture:
            event = {"type": "posture_change", "from": self._prev_posture, "to": torso_state.posture, "ts_ms": ts_ms}
            events.append(event)
            self.event_logger.write(
                {
                    "ts_ms": round(float(ts_ms), 1),
                    "frame": int(self.frame_index),
                    "event": "POSTURE_CHANGED",
                    "state": torso_state.posture,
                    "previous_state": self._prev_posture,
                    "zone": zones_info.get("body_zone", "unknown"),
                    "duration_ms": 0.0,
                    "quality": activity_info.get("observation_quality", "UNKNOWN"),
                    "event_source": "AUTOMATIC",
                    "evidence": {"from": self._prev_posture, "to": torso_state.posture},
                }
            )
        self._prev_posture = torso_state.posture
        if self._prev_hands_state is not None and hands_state != self._prev_hands_state:
            event = {"type": "hands_state_change", "from": self._prev_hands_state, "to": hands_state, "ts_ms": ts_ms}
            events.append(event)
            self.event_logger.write(
                {
                    "ts_ms": round(float(ts_ms), 1),
                    "frame": int(self.frame_index),
                    "event": "HANDS_STATE_CHANGED",
                    "state": hands_state,
                    "previous_state": self._prev_hands_state,
                    "zone": zones_info.get("body_zone", "unknown"),
                    "duration_ms": 0.0,
                    "quality": activity_info.get("observation_quality", "UNKNOWN"),
                    "event_source": "AUTOMATIC",
                    "evidence": {"from": self._prev_hands_state, "to": hands_state},
                }
            )
        self._prev_hands_state = hands_state

        total_ms = (time.perf_counter() - t_start) * 1000.0
        self.budget.observe(total_ms)
        fps = self.fps.tick(ts_ms)

        budget_stats = self.budget.stats()
        perf = {
            "total_ms": round(total_ms, 2),
            "pose_ms": round(pose_ms, 2),
            "hand_ms": round(hand_ms, 2),
            "analyze_ms": round(analyze_ms, 2),
            "activity_ms": round(activity_ms, 2),
            "p50_ms": budget_stats.get("p50_ms"),
            "p95_ms": budget_stats.get("p95_ms"),
            "max_ms": budget_stats.get("max_ms"),
            "level": budget_stats.get("level"),
            "level_index": budget_stats.get("level_index"),
            "pose_every_n": budget_stats.get("pose_every_n"),
            "hand_every_n": budget_stats.get("hand_every_n"),
            "fps": round(fps, 1),
            "stale_pose": not run_pose,
            "stale_hands": not run_hands,
        }
        frame_info = {
            "index": self.frame_index,
            "ts_ms": round(float(ts_ms), 1),
            "width": int(width),
            "height": int(height),
            "source": self._source_kind,
            "source_description": self._source_desc,
            "calibration": dict(self.torso.offsets),
        }
        snapshot = PerceptionSnapshot(
            schema_version=self.schema_version,
            frame=frame_info,
            perf=perf,
            torso=torso_state,
            hands=hand_states,
            hands_state=hands_state,
            activity=activity_info,
            zones=zones_info,
            camera_status=self.camera_status,
            events=events,
        ).to_dict()
        self.last_snapshot = snapshot

        if self.renderer is not None:
            try:
                self.last_overlay = self.renderer.render(
                    frame,
                    pose,
                    hand_frames,
                    hand_states,
                    torso_state,
                    perf,
                    fps=fps,
                    zones=zones_info,
                    activity=activity_info,
                    extra_lines=[
                        f"帧 {self.frame_index}  手 {len(hand_states)} 只",
                        f"活动 {activity_info.get('state_zh', activity_info.get('state', 'UNKNOWN'))}"
                        f"  区 {activity_info.get('zone', 'unknown')}"
                        f"  切菜 {float(getattr(best, 'score', 0.0) or 0.0):.2f}"
                        f"  {float(getattr(best, 'rhythm_hz', 0.0) or 0.0):.1f}Hz"
                        f"  {'稳定' if bool(getattr(best, 'stable', False)) else '未稳定'}",
                    ],
                )
            except Exception as exc:  # 渲染失败不能中断检测
                LOGGER.warning("叠加层渲染失败: %s", exc)
                self.last_overlay = frame.copy()
        return snapshot

    # ------------------------------------------------------------------ 无头运行
    def run(self, max_frames: Optional[int] = None, on_snapshot=None, sleep_live: bool = True) -> None:
        if self.source is None:
            raise RuntimeError("Pipeline.run 需要 source")
        failures = 0
        while True:
            ok, frame, ts_ms = self.source.read()
            if not ok:
                if self.source.is_live():
                    failures += 1
                    if failures > 30:
                        self.notify_camera_error(self.source.last_error or "摄像头读取失败")
                        break
                    time.sleep(0.05)
                    continue
                break
            failures = 0
            snapshot = self.process(frame, ts_ms)
            if on_snapshot is not None:
                on_snapshot(snapshot, frame)
            if max_frames is not None and self.frame_index >= max_frames:
                break

    def notify_camera_error(self, message: str) -> None:
        """采集最终失败时由 worker/run 调用：记录断连事件并更新状态。"""
        if self.camera_status == "DISCONNECTED":
            return
        self.camera_status = "DISCONNECTED"
        try:
            self.event_logger.write(
                {
                    "ts_ms": round(float(self._last_ts_ms), 1),
                    "frame": int(self.frame_index),
                    "event": "CAMERA_DISCONNECTED",
                    "state": self.activity.state,
                    "previous_state": self.activity.previous_state,
                    "zone": self.activity._zone_of(self.activity.state),
                    "duration_ms": 0.0,
                    "quality": "UNKNOWN",
                    "event_source": "AUTOMATIC",
                    "evidence": {"error": message},
                }
            )
        except Exception:
            LOGGER.exception("记录摄像头断连事件失败")

    def close(self) -> None:
        try:
            self.event_logger.close()
        except Exception:
            pass
        for engine in (self.pose_engine, self.hand_engine):
            try:
                engine.close()
            except Exception:
                pass
        if self.source is not None:
            try:
                self.source.close()
            except Exception:
                pass