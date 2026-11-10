"""手部运动与切菜节奏检测。

只做“可解释的几何/时间特征”，不训练模型：
- 位置：掌心点优先，手部丢失时用姿态手腕兜底；
- 时间：真实毫秒时间戳，变帧率/降级隔帧时速度不虚高；
- 竖直方向：EMA 平滑后统计方向反转、行程、半周期；
- 切菜：反转次数、频率、周期稳定性、幅度稳定性综合评分。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

from .config import Config
from .geometry import clamp

MOTION_LEVELS = ("LOW", "MEDIUM", "HIGH", "unknown")


@dataclass
class MotionObservation:
    """一帧里某只手的观测；key 用来跨帧关联同一只手。"""

    key: str
    label: str = "unknown"
    x: Optional[float] = None
    y: Optional[float] = None
    valid: bool = False
    source: str = "none"  # palm / wrist / none
    score: float = 0.0


@dataclass
class MotionFeatures:
    key: str = ""
    label: str = "unknown"
    valid: bool = False
    source: str = "none"
    view_y: Optional[float] = None
    speed_norm_per_s: float = 0.0
    movement_level: str = "unknown"
    reversals: int = 0
    rhythm_hz: float = 0.0
    amplitude_ratio: float = 0.0
    period_cv: float = 1.0
    amplitude_cv: float = 1.0
    score: float = 0.0
    detected: bool = False
    stable: bool = False
    last_chop_ms: Optional[float] = None
    sampling_gap_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "valid": self.valid,
            "source": self.source,
            "speed_norm_per_s": round(self.speed_norm_per_s, 4),
            "movement_level": self.movement_level,
            "reversals": int(self.reversals),
            "rhythm_hz": round(self.rhythm_hz, 3),
            "amplitude_ratio": round(self.amplitude_ratio, 4),
            "period_cv": round(self.period_cv, 3),
            "amplitude_cv": round(self.amplitude_cv, 3),
            "score": round(self.score, 3),
            "detected": bool(self.detected),
            "stable": bool(self.stable),
            "last_chop_ms": None if self.last_chop_ms is None else round(float(self.last_chop_ms), 1),
            "sampling_gap_ms": round(self.sampling_gap_ms, 1),
        }


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def _cv(values: Sequence[float], minimum: float = 1e-6) -> float:
    if len(values) < 2:
        return 1.0
    mean = abs(_mean(values))
    if mean < minimum:
        return 1.0
    return _std(values) / mean


class _Track:
    """单只手的运动轨迹（每帧只保留少量状态，内存有界）。"""

    def __init__(self) -> None:
        self.reset(0.0)

    def reset(self, ts_ms: float) -> None:
        self.last_ts = float(ts_ms)
        self.last_x: Optional[float] = None
        self.last_y: Optional[float] = None
        self.smooth_y: Optional[float] = None
        self.direction = 0
        self.excursion = 0.0
        self.reversal_times: list[float] = []
        self.reversal_amplitudes: list[float] = []
        self.speed_ema = 0.0
        self.source = "none"

    def update(
        self,
        ts_ms: float,
        x: float,
        y: float,
        source: str,
        scale: float,
        alpha: float,
        reversal_deadband: float,
        reset_gap_ms: float,
    ) -> float:
        """返回本帧采样间隔 ms；折线/换源时重置轨迹。"""
        ts = float(ts_ms)
        if source != self.source or self.last_x is None or ts - self.last_ts > reset_gap_ms:
            gap = 0.0 if self.last_ts <= 0 else max(0.0, ts - self.last_ts)
            self.reset(ts)
            self.source = source
            self.last_x = float(x)
            self.last_y = float(y)
            self.smooth_y = float(y)
            return gap
        gap = max(0.0, ts - self.last_ts)
        dt_s = max(gap / 1000.0, 1e-4)
        dx = float(x) - self.last_x
        dy = float(y) - self.last_y
        # 速度：归一化坐标/秒，再做 EMA；幅值用躯干尺度归一化，不直接当阈值
        speed = math.hypot(dx, dy) / dt_s
        self.speed_ema = alpha * speed + (1.0 - alpha) * self.speed_ema
        smooth = alpha * float(y) + (1.0 - alpha) * (self.smooth_y if self.smooth_y is not None else float(y))
        # 方向判定用平滑后的增量，阈值随采样间隔缩放，避免 5 FPS 时噪声被当反转
        dy_smooth = smooth - (self.smooth_y if self.smooth_y is not None else smooth)
        threshold = max(1e-5, reversal_deadband * max(1.0, dt_s / 0.033))
        new_direction = self.direction
        if dy_smooth > threshold:
            new_direction = 1
        elif dy_smooth < -threshold:
            new_direction = -1
        if new_direction != 0 and new_direction != self.direction:
            if self.direction != 0:
                amplitude = abs(self.excursion) / max(scale, 1e-6)
                if amplitude > 0.0:
                    self.reversal_times.append(ts)
                    self.reversal_amplitudes.append(amplitude)
            self.excursion = abs(self.smooth_y - float(y)) if self.smooth_y is not None else abs(dy_smooth)
            self.direction = new_direction
        else:
            self.excursion += abs(dy_smooth)
        self.smooth_y = smooth
        self.last_x = float(x)
        self.last_y = float(y)
        self.last_ts = ts
        self.source = source
        return gap

    def prune(self, now_ms: float, window_ms: float) -> None:
        cutoff = now_ms - window_ms
        while self.reversal_times and self.reversal_times[0] < cutoff:
            self.reversal_times.pop(0)
            self.reversal_amplitudes.pop(0)


class HandMotionTracker:
    """管理多只手的轨迹，输出每只手的 MotionFeatures。"""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.window_ms = float(config.get("chopping.window_s", 2.5)) * 1000.0
        self.smooth_alpha = float(config.get("chopping.smooth_alpha", 0.4))
        self.reversal_deadband = float(config.get("chopping.reversal_deadband_norm", 0.0015))
        self.min_amplitude_ratio = float(config.get("chopping.min_amplitude_ratio", 0.04))
        self.min_reversals = int(config.get("chopping.min_reversals", 4))
        self.min_rhythm = float(config.get("chopping.min_rhythm_hz", 0.8))
        self.max_rhythm = float(config.get("chopping.max_rhythm_hz", 4.0))
        self.period_cv_max = float(config.get("chopping.period_cv_max", 0.35))
        self.amplitude_cv_max = float(config.get("chopping.amplitude_cv_max", 0.45))
        self.reset_gap_ms = float(config.get("chopping.reset_gap_ms", 800.0))
        self.low_speed = float(config.get("motion.low_speed_norm_per_s", 0.08))
        self.high_speed = float(config.get("motion.high_speed_norm_per_s", 0.35))
        self._tracks: dict[str, _Track] = {}

    def reset(self) -> None:
        self._tracks.clear()

    def update(self, observations: Sequence[MotionObservation], ts_ms: float, scale: float) -> list[MotionFeatures]:
        """每帧调用；observations 里没有的手会被标记为 invalid。"""
        now = float(ts_ms)
        results: list[MotionFeatures] = []
        seen: set[str] = set()
        for obs in observations:
            key = obs.key
            track = self._tracks.get(key)
            if track is None:
                track = _Track()
                track.reset(now)
                self._tracks[key] = track
            feature = MotionFeatures(key=key, label=obs.label, source=obs.source)
            if not obs.valid or obs.x is None or obs.y is None:
                feature.valid = False
                feature.movement_level = "unknown"
                feature.sampling_gap_ms = max(0.0, now - track.last_ts) if track.last_ts else 0.0
                results.append(feature)
                seen.add(key)
                continue
            seen.add(key)
            gap = track.update(
                now,
                float(obs.x),
                float(obs.y),
                obs.source,
                float(scale),
                self.smooth_alpha,
                self.reversal_deadband,
                self.reset_gap_ms,
            )
            track.prune(now, self.window_ms)
            feature.valid = True
            feature.sampling_gap_ms = gap
            feature.view_y = round(float(obs.y), 5)
            feature.speed_norm_per_s = track.speed_ema
            if track.speed_ema >= self.high_speed:
                feature.movement_level = "HIGH"
            elif track.speed_ema >= self.low_speed:
                feature.movement_level = "MEDIUM"
            else:
                feature.movement_level = "LOW"
            results.append(feature)
            self._fill_chopping(feature, track, now)
        # 超时没再出现的手：清轨迹，避免下次出现时沿用旧数据
        for key in list(self._tracks.keys()):
            if key not in seen and now - self._tracks[key].last_ts > self.reset_gap_ms:
                del self._tracks[key]
        return results

    def _fill_chopping(self, feature: MotionFeatures, track: _Track, now_ms: float) -> None:
        times = track.reversal_times
        amps = track.reversal_amplitudes
        feature.reversals = len(times)
        if len(times) >= 2:
            half_periods = [times[i] - times[i - 1] for i in range(1, len(times))]
            mean_half = _mean(half_periods)
            if mean_half > 1e-6:
                feature.rhythm_hz = 1.0 / (2.0 * mean_half / 1000.0)
            feature.period_cv = _cv(half_periods)
        if amps:
            feature.amplitude_ratio = _mean(amps)
            feature.amplitude_cv = _cv(amps)
        feature.last_chop_ms = times[-1] if times else None
        amp_score = clamp(feature.amplitude_ratio / max(self.min_amplitude_ratio * 1.5, 1e-6), 0.0, 1.0)
        if feature.rhythm_hz <= 0:
            rate_score = 0.0
        elif self.min_rhythm <= feature.rhythm_hz <= self.max_rhythm:
            rate_score = 1.0
        else:
            distance = self.min_rhythm - feature.rhythm_hz if feature.rhythm_hz < self.min_rhythm else feature.rhythm_hz - self.max_rhythm
            rate_score = clamp(1.0 - distance / max(self.max_rhythm - self.min_rhythm, 0.1), 0.0, 1.0)
        period_score = clamp(1.0 - feature.period_cv, 0.0, 1.0)
        amp_cv_score = clamp(1.0 - feature.amplitude_cv, 0.0, 1.0)
        feature.score = clamp(
            0.45 * amp_score + 0.25 * rate_score + 0.20 * period_score + 0.10 * amp_cv_score,
            0.0,
            1.0,
        )
        feature.detected = bool(
            feature.reversals >= self.min_reversals
            and self.min_rhythm <= feature.rhythm_hz <= self.max_rhythm
            and feature.amplitude_ratio >= self.min_amplitude_ratio
            and feature.score >= 0.5
        )
        feature.stable = bool(
            feature.detected
            and feature.reversals >= self.min_reversals
            and feature.period_cv <= self.period_cv_max
            and feature.amplitude_cv <= self.amplitude_cv_max
        )

    @staticmethod
    def best(features: Sequence[MotionFeatures]) -> Optional[MotionFeatures]:
        """选切菜评分最高的手；分数相同优先右手（更常见拿刀手）。"""
        valid = [f for f in features if f.valid]
        if not valid:
            return None
        return max(valid, key=lambda f: (f.score, f.stable, f.label == "right"))
