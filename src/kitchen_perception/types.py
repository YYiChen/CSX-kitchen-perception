"""数据契约：内部结构 + 对外 JSON 快照。

所有对外字段必须可 JSON 序列化；NaN/Inf 统一转 null，避免下游解析失败。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Optional


def _clean(value: Any) -> Any:
    """递归把 NaN/Inf 转 None，dataclass 转 dict。"""
    if isinstance(value, float):
        return None if (math.isnan(value) or math.isinf(value)) else value
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def to_jsonable(obj: Any) -> Any:
    return _clean(asdict(obj))


@dataclass
class Landmark:
    """归一化图像坐标（x,y,z）+ 可见度/存在度。"""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    visibility: float = 1.0
    presence: float = 1.0

    @classmethod
    def from_mp(cls, lm: Any) -> "Landmark":
        return cls(
            x=float(getattr(lm, "x", 0.0)),
            y=float(getattr(lm, "y", 0.0)),
            z=float(getattr(lm, "z", 0.0)),
            visibility=float(getattr(lm, "visibility", 1.0) or 0.0),
            presence=float(getattr(lm, "presence", 1.0) or 0.0),
        )

    def px(self, width: int, height: int) -> tuple[float, float]:
        return (self.x * width, self.y * height)

    def score(self) -> float:
        return min(max(self.visibility, 0.0), 1.0) * min(max(self.presence, 0.0), 1.0)


@dataclass
class PoseFrame:
    landmarks: list[Landmark] = field(default_factory=list)
    world: list[Landmark] = field(default_factory=list)
    present: bool = False
    mean_visibility: float = 0.0

    def point(self, index: int, width: int, height: int) -> tuple[float, float] | None:
        if index >= len(self.landmarks):
            return None
        return self.landmarks[index].px(width, height)

    def world_point(self, index: int) -> tuple[float, float, float] | None:
        if index >= len(self.world):
            return None
        lm = self.world[index]
        return (lm.x, lm.y, lm.z)

    def visibility(self, index: int) -> float:
        if index >= len(self.landmarks):
            return 0.0
        lm = self.landmarks[index]
        return min(max(lm.visibility, 0.0), 1.0) * min(max(lm.presence, 0.0), 1.0)


@dataclass
class HandFrame:
    landmarks: list[Landmark] = field(default_factory=list)
    world: list[Landmark] = field(default_factory=list)
    handedness: str = "unknown"
    score: float = 0.0


@dataclass
class FingerMetrics:
    finger: str
    pip_deg: float = float("nan")
    dip_deg: float = float("nan")
    curl_deg: float = float("nan")
    dist_ratio: float = float("nan")
    valid: bool = False
    note: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return _clean(asdict(self))


@dataclass
class FingerState:
    finger: str
    state: str = "unknown"
    metric: FingerMetrics | None = None


@dataclass
class HandState:
    label: str = "unknown"
    label_source: str = "model"
    score: float = 0.0
    fingers: dict[str, str] = field(default_factory=dict)
    finger_metrics: dict[str, FingerMetrics] = field(default_factory=dict)
    grip: str = "unknown"
    grip_zh: str = ""
    in_workspace: bool = False
    busy: bool = False
    stale: bool = False
    wrist_norm: list[float] | None = None
    bbox_norm: list[float] | None = None
    # V2：掌心参考点（原始归一化坐标）与 A/B 分区
    palm_norm: list[float] | None = None
    zone: str = "unknown"
    zone_code: str = "HAND_NOT_VISIBLE"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = _clean(asdict(self))
        data["fingers"] = {k: v for k, v in self.fingers.items()}
        data["finger_metrics"] = {k: v.to_dict() for k, v in self.finger_metrics.items()}
        return data


@dataclass
class TorsoFeatures:
    pitch_deg: float = float("nan")
    roll_deg: float = float("nan")
    yaw_deg: float = float("nan")
    facing: str = "unknown"
    visibility: float = 0.0
    shoulder_width_px: float = float("nan")
    torso_length_px: float = float("nan")
    bbox_norm: list[float] | None = None
    source: str = "world"
    inclination_2d_deg: float = float("nan")
    foreshorten_ratio: float = float("nan")
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return _clean(asdict(self))


@dataclass
class TorsoState:
    present: bool = False
    posture: str = "no_person"
    posture_zh: str = ""
    confidence: float = 0.0
    stable_frames: int = 0
    features: TorsoFeatures = field(default_factory=TorsoFeatures)
    hands_in_workspace: bool = False
    note: str = ""

    def to_dict(self) -> dict:
        data = _clean(asdict(self))
        data["features"] = self.features.to_dict()
        # 为下游兼容，把核心角度同时提到 torso 顶层
        data["pitch_deg"] = _clean(self.features.pitch_deg)
        data["roll_deg"] = _clean(self.features.roll_deg)
        data["yaw_deg"] = _clean(self.features.yaw_deg)
        data["facing"] = self.features.facing
        data["visibility"] = _clean(self.features.visibility)
        data["source"] = self.features.source
        data["bbox_norm"] = _clean(self.features.bbox_norm)
        return data


@dataclass
class PerceptionSnapshot:
    schema_version: str = "1.0"
    camera_status: str = "UNKNOWN"
    frame: dict = field(default_factory=dict)
    perf: dict = field(default_factory=dict)
    zones: dict = field(default_factory=dict)
    torso: TorsoState = field(default_factory=TorsoState)
    hands: list[HandState] = field(default_factory=list)
    hands_state: str = "unknown"
    activity: dict = field(default_factory=dict)
    demo: dict = field(default_factory=dict)
    planned_step: Any = None
    observed_step: Any = None
    events: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "camera_status": self.camera_status,
            "frame": _clean(self.frame),
            "perf": _clean(self.perf),
            "zones": _clean(self.zones),
            "torso": self.torso.to_dict(),
            "hands": [h.to_dict() for h in self.hands],
            "hands_state": self.hands_state,
            "activity": _clean(self.activity),
            "demo": _clean(self.demo),
            "planned_step": _clean(self.planned_step),
            "observed_step": _clean(self.observed_step),
            "events": _clean(self.events),
        }