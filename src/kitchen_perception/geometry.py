"""几何与平滑工具。

约定：图像坐标 x 向右、y 向下；角度单位为度。
所有输入都是普通 tuple/float，便于单元测试，不依赖 MediaPipe。
"""

from __future__ import annotations

import math
from collections import deque
from typing import Iterable, Sequence

Point = tuple[float, float]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def midpoint(a: Sequence[float], b: Sequence[float]) -> Point:
    return ((float(a[0]) + float(b[0])) / 2.0, (float(a[1]) + float(b[1])) / 2.0)


def angle_deg(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float:
    """角 ABC（顶点 B），三点共线时为 0/180，退化时返回 nan。"""
    v1 = (float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))
    v2 = (float(c[0]) - float(b[0]), float(c[1]) - float(b[1]))
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return float("nan")
    cos_value = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    return math.degrees(math.acos(clamp(cos_value, -1.0, 1.0)))


def atan2_deg(dy: float, dx: float) -> float:
    return math.degrees(math.atan2(float(dy), float(dx)))


def make_ema(alpha: float) -> "EMA":
    return EMA(alpha)


class EMA:
    """一阶指数滑动平均；value 为 None 时透传。"""

    def __init__(self, alpha: float) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha 必须在 (0, 1] 内")
        self.alpha = alpha
        self.value: float | None = None

    def update(self, value: float | None) -> float | None:
        if value is None or math.isnan(value):
            return self.value
        if self.value is None:
            self.value = float(value)
        else:
            self.value = self.alpha * float(value) + (1.0 - self.alpha) * self.value
        return self.value

    def reset(self) -> None:
        self.value = None


class Hysteresis:
    """状态滞回：连续 confirm_frames 帧同一候选才切换，避免抖动。"""

    def __init__(self, confirm_frames: int) -> None:
        self.confirm_frames = max(1, int(confirm_frames))
        self.state: str | None = None
        self.candidate: str | None = None
        self.candidate_count = 0

    def update(self, candidate: str | None) -> tuple[str | None, int]:
        """返回 (当前稳定状态, 已稳定帧数)。candidate=None 表示本帧无观测。"""
        if candidate is None:
            return self.state, 0
        if candidate == self.state:
            self.candidate = candidate
            self.candidate_count = 1
            return self.state, self.candidate_count
        if candidate == self.candidate:
            self.candidate_count += 1
        else:
            self.candidate = candidate
            self.candidate_count = 1
        if self.candidate_count >= self.confirm_frames or self.state is None:
            self.state = candidate
        return self.state, self.candidate_count

    def reset(self) -> None:
        self.state = None
        self.candidate = None
        self.candidate_count = 0


class MotionSpeedTracker:
    """归一化速度：位移 / 时间，再除以参考长度（躯干长）。"""

    def __init__(self, window_s: float = 0.6) -> None:
        self.window_ms = int(window_s * 1000)
        self.samples: deque[tuple[float, float, float]] = deque()

    def update(self, x: float, y: float, ts_ms: float, ref_len_px: float) -> float:
        self.samples.append((float(ts_ms), float(x), float(y)))
        cutoff = ts_ms - self.window_ms
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()
        if len(self.samples) < 2 or ref_len_px <= 1e-6:
            return 0.0
        t0, x0, y0 = self.samples[0]
        t1, x1, y1 = self.samples[-1]
        dt_s = max((t1 - t0) / 1000.0, 1e-3)
        dist = math.hypot(x1 - x0, y1 - y0)
        return (dist / ref_len_px) / dt_s

    def reset(self) -> None:
        self.samples.clear()


def mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return sum(vals) / len(vals) if vals else float("nan")


def percentile(values: Sequence[float], q: float) -> float:
    """简单百分位（线性插值），q 取 0-100。"""
    vals = sorted(float(v) for v in values if v is not None and not math.isnan(float(v)))
    if not vals:
        return float("nan")
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * clamp(q, 0.0, 100.0) / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)
