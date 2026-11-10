"""延迟预算：按帧耗时在 L0–L3 降级级别间自动切换。

意义：演示机性能可能和开发机不同，与其硬编码隔帧推理，不如根据实测 p95
自动调节。规则写在 config.latency.levels 里：
- pose_every_n / hand_every_n：每 N 帧做一次推理，中间帧复用上次结果；
- analysis_scale：分析分辨率缩放（给未来在更弱机器上降级留口子）。

只统计“单帧处理耗时”（read 之后到分析结束，不含摄像头等帧），避免把
摄像头阻塞时间算进延迟。
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Deque, Optional

from .config import Config
from .geometry import percentile


class LatencyBudget:
    def __init__(self, config: Config) -> None:
        section = config.section("latency")
        self.target_p95_ms = float(section.get("target_p95_ms", 250.0))
        self.hard_cap_ms = float(section.get("hard_cap_ms", 500.0))
        self.eval_every = max(1, int(section.get("eval_every_frames", 30)))
        self.eval_count = 0
        self.frame_count = 0
        levels = section.get("levels") or [{"name": "L0", "pose_every_n": 1, "hand_every_n": 1, "analysis_scale": 1.0}]
        self.levels = levels
        self.level_index = 0
        self.window: Deque[float] = deque(maxlen=max(10, int(section.get("window_frames", 120))))
        self.infer_window: Deque[float] = deque(maxlen=max(10, int(section.get("window_frames", 120))))
        self.max_observed_ms = 0.0
        self.total_frames = 0

    @property
    def level(self) -> dict:
        return self.levels[self.level_index]

    @property
    def level_name(self) -> str:
        return str(self.level.get("name", f"L{self.level_index}"))

    @property
    def pose_every_n(self) -> int:
        return max(1, int(self.level.get("pose_every_n", 1)))

    @property
    def hand_every_n(self) -> int:
        return max(1, int(self.level.get("hand_every_n", 1)))

    @property
    def analysis_scale(self) -> float:
        return float(self.level.get("analysis_scale", 1.0))

    def observe(self, frame_ms: float) -> None:
        """每帧处理完调用一次；frame_ms 为该帧处理耗时。"""
        self.window.append(float(frame_ms))
        self.total_frames += 1
        self.max_observed_ms = max(self.max_observed_ms, float(frame_ms))
        if self.total_frames % self.eval_every == 0:
            self._adjust()

    def _adjust(self) -> None:
        if len(self.window) < 10:
            return
        p95 = percentile(self.window, 95.0)
        if math.isnan(p95):
            return
        if p95 > self.hard_cap_ms:
            new_index = min(len(self.levels) - 1, self.level_index + 2)
        elif p95 > self.target_p95_ms:
            new_index = min(len(self.levels) - 1, self.level_index + 1)
        elif p95 < self.target_p95_ms * 0.55 and self.level_index > 0:
            new_index = self.level_index - 1
        else:
            new_index = self.level_index
        self.level_index = new_index

    def stats(self) -> dict:
        return {
            "level": self.level_name,
            "level_index": self.level_index,
            "target_p95_ms": self.target_p95_ms,
            "hard_cap_ms": self.hard_cap_ms,
            "frames": self.total_frames,
            "p50_ms": _safe(percentile(self.window, 50.0)),
            "p95_ms": _safe(percentile(self.window, 95.0)),
            "max_ms": _safe(self.max_observed_ms),
            "pose_every_n": self.pose_every_n,
            "hand_every_n": self.hand_every_n,
            "analysis_scale": self.analysis_scale,
        }

    def reset(self) -> None:
        self.window.clear()
        self.infer_window.clear()
        self.level_index = 0
        self.total_frames = 0
        self.max_observed_ms = 0.0


def _safe(value: float) -> Optional[float]:
    return None if value is None or math.isnan(value) else round(value, 3)


class FpsMeter:
    """最近 N 帧的 FPS 估计。"""

    def __init__(self, window: int = 30) -> None:
        self.times: Deque[float] = deque(maxlen=max(2, window))

    def tick(self, ts_ms: Optional[float] = None) -> float:
        now = ts_ms if ts_ms is not None else time.perf_counter() * 1000.0
        self.times.append(float(now))
        if len(self.times) < 2:
            return 0.0
        dt = (self.times[-1] - self.times[0]) / 1000.0
        if dt <= 1e-6:
            return 0.0
        return (len(self.times) - 1) / dt

    def reset(self) -> None:
        self.times.clear()
