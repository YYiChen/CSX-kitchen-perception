"""A/B 操作区、身体重心与手部分区。

坐标约定（关键）：
- MediaPipe 原始归一化 x：画面左边缘为 0，向右为 1；
- 界面默认镜像预览（ui.mirror_preview=true）；
- 分区配置按“用户看到的显示画面”定义，因此统一先换算成 view_x：
  view_x = 1 - raw_x（镜像）或 raw_x（不镜像）。
这样 split_x=1/3 在镜像画面里就是用户看到的左侧三分之一，不会左右颠倒。
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from .config import Config
from .geometry import clamp
from .types import HandFrame, HandState, PoseFrame, TorsoState

LOGGER = logging.getLogger(__name__)

ZONE_VALUES = ("A", "B", "unknown")
HAND_LANDMARK_COUNT = 21
PALM_INDICES = (0, 5, 9, 13, 17)  # 腕 + 四个掌指关节，比单个指尖稳定
HIP_POINTS = (23, 24)
SHOULDER_POINTS = (11, 12)


def view_x(raw_x: float, mirror: bool) -> float:
    """原始归一化 x → 显示画面 x。"""
    x = float(raw_x)
    return clamp(1.0 - x if mirror else x, 0.0, 1.0)


def view_point(point: Sequence[float], mirror: bool) -> tuple[float, float]:
    """原始归一化点 (x, y) → 显示画面点；y 不受镜像影响。"""
    return (view_x(point[0], mirror), clamp(float(point[1]), 0.0, 1.0))


def palm_center(hand: HandFrame) -> Optional[tuple[float, float]]:
    """掌心参考点 = 腕 + 四掌指关节均值；点数不足返回 None。"""
    if hand is None or len(hand.landmarks) < HAND_LANDMARK_COUNT:
        return None
    try:
        xs = [float(hand.landmarks[i].x) for i in PALM_INDICES]
        ys = [float(hand.landmarks[i].y) for i in PALM_INDICES]
    except (IndexError, TypeError, ValueError):
        return None
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def hand_zone_code(label: str, zone: str) -> str:
    """给下游/日志用的手部区域代码。"""
    if zone not in ("A", "B"):
        return "HAND_NOT_VISIBLE"
    prefix = {"left": "LEFT_HAND", "right": "RIGHT_HAND"}.get(str(label).lower(), "HAND")
    return f"{prefix}_IN_{zone}"


@dataclass
class BodyCenter:
    """身体重心估计结果（已同时保留原始坐标与显示坐标）。"""

    raw_x: float
    raw_y: float
    view_x: float
    view_y: float
    source: str = "none"
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "raw_norm": [round(self.raw_x, 5), round(self.raw_y, 5)],
            "view_norm": [round(self.view_x, 5), round(self.view_y, 5)],
            "source": self.source,
            "confidence": round(clamp(self.confidence, 0.0, 1.0), 3),
        }


def _visible(pose: PoseFrame, index: int, min_visibility: float) -> bool:
    if not pose.present or index >= len(pose.landmarks):
        return False
    return pose.visibility(index) >= min_visibility


def _midpoint(pose: PoseFrame, a: int, b: int) -> tuple[float, float]:
    la, lb = pose.landmarks[a], pose.landmarks[b]
    return ((float(la.x) + float(lb.x)) / 2.0, (float(la.y) + float(lb.y)) / 2.0)


def estimate_body_center(
    pose: PoseFrame,
    torso: Optional[TorsoState],
    mirror: bool,
    min_visibility: float = 0.3,
) -> Optional[BodyCenter]:
    """髋中点 → 肩中点 → 躯干框中心 的降级估计。

    髋部常在笔记本电脑摄像头画面外，但仍优先使用髋部；不可用时才降级，
    并把 source 写进快照，方便判断分区可信度。
    """
    if pose is not None and pose.present:
        if all(_visible(pose, i, min_visibility) for i in HIP_POINTS):
            raw_x, raw_y = _midpoint(pose, *HIP_POINTS)
            conf = min(pose.visibility(i) for i in HIP_POINTS)
            return BodyCenter(raw_x, raw_y, view_x(raw_x, mirror), clamp(raw_y, 0, 1), "hips", conf)
        if all(_visible(pose, i, min_visibility) for i in SHOULDER_POINTS):
            raw_x, raw_y = _midpoint(pose, *SHOULDER_POINTS)
            conf = min(pose.visibility(i) for i in SHOULDER_POINTS)
            return BodyCenter(raw_x, raw_y, view_x(raw_x, mirror), clamp(raw_y, 0, 1), "shoulders", conf)

    bbox = None
    if torso is not None:
        bbox = torso.features.bbox_norm
    if bbox is not None and len(bbox) == 4:
        raw_x = (float(bbox[0]) + float(bbox[2])) / 2.0
        raw_y = (float(bbox[1]) + float(bbox[3])) / 2.0
        conf = clamp(float(getattr(torso, "confidence", 0.0)) * 0.6, 0.0, 1.0)
        return BodyCenter(raw_x, raw_y, view_x(raw_x, mirror), clamp(raw_y, 0, 1), "bbox", conf)
    return None


@dataclass
class ZoneSettings:
    """A/B 分区配置；split_x / deadband 都是显示画面归一化坐标。"""

    split_x: float = 0.5
    deadband: float = 0.02
    ema_alpha: float = 0.35
    enter_ms: float = 300.0
    labels: dict[str, str] = field(default_factory=lambda: {"A": "备菜区", "B": "处理区"})
    enabled: bool = True

    def normalized(self) -> "ZoneSettings":
        split = clamp(float(self.split_x), 0.05, 0.95)
        deadband = clamp(float(self.deadband), 0.0, 0.2)
        alpha = clamp(float(self.ema_alpha), 0.01, 1.0)
        enter_ms = max(0.0, float(self.enter_ms))
        labels = {
            "A": str(self.labels.get("A", "备菜区")),
            "B": str(self.labels.get("B", "处理区")),
        }
        return ZoneSettings(split, deadband, alpha, enter_ms, labels, bool(self.enabled))

    def to_dict(self) -> dict:
        return {
            "enabled": bool(self.enabled),
            "split_x": round(float(self.split_x), 4),
            "deadband": round(float(self.deadband), 4),
            "ema_alpha": round(float(self.ema_alpha), 3),
            "enter_ms": round(float(self.enter_ms), 1),
            "labels": dict(self.labels),
        }

    @classmethod
    def from_dict(cls, data: dict, base: Optional["ZoneSettings"] = None) -> "ZoneSettings":
        base = base or cls()
        labels = dict(base.labels)
        raw_labels = data.get("labels")
        if isinstance(raw_labels, dict):
            for key in ("A", "B"):
                value = raw_labels.get(key)
                if isinstance(value, str) and value.strip():
                    labels[key] = value.strip()
                elif isinstance(value, dict) and value.get("zh"):
                    labels[key] = str(value["zh"])
        return cls(
            split_x=float(data.get("split_x", base.split_x)),
            deadband=float(data.get("deadband", base.deadband)),
            ema_alpha=float(data.get("ema_alpha", base.ema_alpha)),
            enter_ms=float(data.get("enter_ms", base.enter_ms)),
            labels=labels,
            enabled=bool(data.get("enabled", base.enabled)),
        ).normalized()

    def label(self, zone: str) -> str:
        return self.labels.get(zone, zone)


def _settings_from_config(config: Config) -> ZoneSettings:
    section = config.get("zones", {}) or {}
    if not isinstance(section, dict):
        section = {}
    return ZoneSettings.from_dict(section)


def zones_file(config: Config) -> Path:
    return config.resolve_path(config.get("zones.file", "config/zones.json"))


def load_settings(config: Config) -> tuple[ZoneSettings, list[str], Path]:
    """默认配置 + config/zones.json 覆盖；损坏时回退默认并返回 warning。"""
    defaults = _settings_from_config(config)
    path = zones_file(config)
    warnings: list[str] = []
    if not path.exists():
        return defaults, warnings, path
    try:
        with path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
        if not isinstance(data, dict):
            raise ValueError("根节点不是对象")
        settings = ZoneSettings.from_dict(data, defaults)
        return settings, warnings, path
    except (OSError, ValueError, TypeError) as exc:
        warnings.append(f"分区配置读取失败，已回退默认：{exc}")
        LOGGER.warning("分区配置读取失败: %s", exc)
        return defaults, warnings, path


def save_settings(path: Path, settings: ZoneSettings) -> None:
    """原子写入：临时文件 + os.replace，避免中途失败留下半个 JSON。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = settings.normalized().to_dict()
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
        fp.write("\n")
    os.replace(tmp, path)


def classify_x(view_x_value: float, settings: ZoneSettings, previous: str = "unknown") -> str:
    """按 split_x ± deadband 判 A/B；死区内保持 previous。"""
    x = clamp(float(view_x_value), 0.0, 1.0)
    split = settings.split_x
    deadband = settings.deadband
    if x < split - deadband:
        return "A"
    if x > split + deadband:
        return "B"
    return previous if previous in ("A", "B") else ("A" if x < split else "B")


class ZoneManager:
    """有状态的分区管理器：重心平滑 + 换区确认 + 手部分区 + 配置热加载。"""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.mirror = bool(config.get("ui.mirror_preview", True))
        self.path = zones_file(config)
        self.settings, self.warnings, _ = load_settings(config)
        self.min_hand_score = float(config.get("hands.min_hand_score", 0.5))
        self.min_visibility = float(config.get("zones.body_visibility_min", 0.3))
        self._center_ema: Optional[tuple[float, float]] = None
        self._center_missing_since: Optional[float] = None
        self._body_zone = "unknown"
        self._candidate_zone: Optional[str] = None
        self._candidate_since: Optional[float] = None
        self._settings_mtime: Optional[float] = None
        self._last_ts_ms = 0.0
        try:
            self._settings_mtime = self.path.stat().st_mtime
        except OSError:
            self._settings_mtime = None

    # ------------------------------------------------------------------ 配置
    def reload_if_changed(self) -> bool:
        """配置文件被外部修改时热加载；返回是否发生变化。"""
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            mtime = None
        if mtime == self._settings_mtime:
            return False
        self._settings_mtime = mtime
        settings, warnings, _ = load_settings(self.config)
        self.settings = settings
        self.warnings = warnings
        self._candidate_zone = None
        self._candidate_since = None
        LOGGER.info("分区配置已重新加载: %s", self.settings.to_dict())
        return True

    def apply_settings(self, data: dict, persist: bool = False) -> ZoneSettings:
        """应用界面传来的配置；persist=True 时写入 config/zones.json。"""
        self.settings = ZoneSettings.from_dict(data, self.settings).normalized()
        self._candidate_zone = None
        self._candidate_since = None
        if persist:
            save_settings(self.path, self.settings)
            try:
                self._settings_mtime = self.path.stat().st_mtime
            except OSError:
                self._settings_mtime = None
        return self.settings

    def save(self) -> None:
        save_settings(self.path, self.settings)
        try:
            self._settings_mtime = self.path.stat().st_mtime
        except OSError:
            self._settings_mtime = None

    # ------------------------------------------------------------------ 主逻辑
    def _smooth_center(self, center: Optional[BodyCenter], ts_ms: float) -> tuple[Optional[tuple[float, float]], str, float]:
        if center is None:
            if self._center_missing_since is None:
                self._center_missing_since = ts_ms
            elif ts_ms - self._center_missing_since > 1000.0:
                self._center_ema = None
            return None, "none", 0.0
        self._center_missing_since = None
        point = (center.view_x, center.view_y)
        alpha = self.settings.ema_alpha
        if self._center_ema is None:
            self._center_ema = point
        else:
            ex, ey = self._center_ema
            self._center_ema = (alpha * point[0] + (1 - alpha) * ex, alpha * point[1] + (1 - alpha) * ey)
        return self._center_ema, center.source, center.confidence

    def _confirmed_zone(self, desired: str, ts_ms: float) -> str:
        if desired not in ("A", "B"):
            return self._body_zone
        if self._body_zone == "unknown":
            self._body_zone = desired
            return desired
        if desired == self._body_zone:
            self._candidate_zone = None
            self._candidate_since = None
            return self._body_zone
        if self._candidate_zone != desired:
            self._candidate_zone = desired
            self._candidate_since = ts_ms
            return self._body_zone
        if self._candidate_since is None or ts_ms - self._candidate_since >= self.settings.enter_ms:
            self._body_zone = desired
            self._candidate_zone = None
            self._candidate_since = None
        return self._body_zone

    def update(
        self,
        pose: PoseFrame,
        torso: Optional[TorsoState],
        hand_states: Sequence[HandState],
        hand_frames: Sequence[HandFrame],
        ts_ms: float,
    ) -> dict:
        """每帧调用：更新身体重心/手部分区，返回快照里的 zones 节点。"""
        self._last_ts_ms = float(ts_ms)
        self.reload_if_changed()
        center = estimate_body_center(pose, torso, self.mirror, self.min_visibility)
        smoothed, source, confidence = self._smooth_center(center, float(ts_ms))
        desired = "unknown"
        if smoothed is not None:
            desired = classify_x(smoothed[0], self.settings, self._body_zone)
        body_zone = self._confirmed_zone(desired, float(ts_ms))
        if smoothed is None:
            self._body_zone = "unknown"
            body_zone = "unknown"

        hand_info: list[dict] = []
        for index, state in enumerate(hand_states):
            frame = hand_frames[index] if index < len(hand_frames) else None
            palm = palm_center(frame) if frame is not None else None
            valid = bool(palm is not None and frame is not None and float(frame.score) >= self.min_hand_score)
            if valid and palm is not None:
                px, py = view_point(palm, self.mirror)
                previous = state.zone if state.zone in ("A", "B") else "unknown"
                zone = classify_x(px, self.settings, previous)
                state.palm_norm = [round(float(palm[0]), 6), round(float(palm[1]), 6)]
                state.zone = zone
                state.zone_code = hand_zone_code(state.label, zone)
                hand_info.append(
                    {
                        "label": state.label,
                        "palm_norm": list(state.palm_norm),
                        "view_norm": [round(px, 6), round(py, 6)],
                        "zone": zone,
                        "zone_code": state.zone_code,
                        "valid": True,
                        "score": round(float(frame.score), 3),
                    }
                )
            else:
                state.palm_norm = None
                state.zone = "unknown"
                state.zone_code = hand_zone_code(state.label, "unknown")
                hand_info.append(
                    {
                        "label": state.label,
                        "palm_norm": None,
                        "view_norm": None,
                        "zone": "unknown",
                        "zone_code": state.zone_code,
                        "valid": False,
                        "score": round(float(frame.score), 3) if frame is not None else 0.0,
                    }
                )

        hand_zones = {item["zone"] for item in hand_info if item["zone"] in ("A", "B")}
        mismatch = bool(body_zone in ("A", "B") and hand_zones and body_zone not in hand_zones)
        return {
            "enabled": bool(self.settings.enabled),
            "split_x": round(self.settings.split_x, 4),
            "deadband": round(self.settings.deadband, 4),
            "view_mirrored": bool(self.mirror),
            "labels": dict(self.settings.labels),
            "body_center_norm": [round(smoothed[0], 5), round(smoothed[1], 5)] if smoothed else None,
            "body_zone": body_zone,
            "zone_source": source if smoothed is not None else "none",
            "zone_confidence": round(clamp(confidence, 0.0, 1.0), 3) if smoothed is not None else 0.0,
            "hand_zone_mismatch": mismatch,
            "hands": hand_info,
            "warnings": list(self.warnings),
        }

    @property
    def body_zone(self) -> str:
        return self._body_zone
