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

from .budget import FpsMeter, LatencyBudget
from .config import Config, load_config
from .engines import HandEngine, ModelError, PoseEngine, resolve_model_paths, verify_models
from .fingers import HandAnalyzer
from .overlay import OverlayRenderer
from .sources import FrameSource, create_source
from .torso import TorsoAnalyzer, hands_in_workspace
from .types import HandState, PerceptionSnapshot, PoseFrame, TorsoState

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
        self.schema_version = str(config.get("project.schema_version", "1.0"))

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

    def process(self, frame: np.ndarray, ts_ms: float) -> dict:
        t_start = time.perf_counter()
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

        events: list[dict] = []
        if self._prev_posture is not None and torso_state.posture != self._prev_posture:
            events.append({"type": "posture_change", "from": self._prev_posture, "to": torso_state.posture, "ts_ms": ts_ms})
        self._prev_posture = torso_state.posture
        if self._prev_hands_state is not None and hands_state != self._prev_hands_state:
            events.append({"type": "hands_state_change", "from": self._prev_hands_state, "to": hands_state, "ts_ms": ts_ms})
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
            activity={"name": "unknown", "confidence": 0.0, "note": "阶段一未实现活动识别"},
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
                    extra_lines=[f"帧 {self.frame_index}  手 {len(hand_states)} 只"],
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

    def close(self) -> None:
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