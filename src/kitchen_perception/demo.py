"""两状态切菜演示：正在切菜 / 切菜完成。

这一层不改变阶段二的活动状态机（ACTIVE_A/B、MOVING、PAUSED 等），只为录像
演示提供一个稳定的二值状态。判定直接复用 HandMotionTracker 已算好的几何/
时间特征，不训练模型；阈值在 config/default.json 的 demo 节。

状态：
- waiting：还没有确认过切菜节奏；界面显示“等待切菜”。
- chopping：检测到连续的抬放节奏；界面显示“正在切菜”。
- done：切菜节奏停止超过 done_after_ms；界面显示“切菜完成”。
"""

from __future__ import annotations

from typing import Optional

from .config import Config
from .motion import MotionFeatures

FALLBACK_LABELS = {
    "waiting": "等待切菜",
    "chopping": "正在切菜",
    "done": "切菜完成",
}


class DemoStateTracker:
    """把逐帧切菜特征压成两个可显示状态（外加首次检测前的等待）。"""

    def __init__(self, config: Config) -> None:
        labels = config.get("demo.labels_zh", {}) or {}
        self.labels = dict(FALLBACK_LABELS)
        if isinstance(labels, dict):
            self.labels.update({str(k): str(v) for k, v in labels.items()})
        self.min_reversals = int(config.get("demo.min_reversals", 2))
        self.min_rhythm = float(config.get("demo.min_rhythm_hz", 0.5))
        self.max_rhythm = float(config.get("demo.max_rhythm_hz", 4.0))
        self.min_amplitude = float(config.get("demo.min_amplitude_ratio", 0.02))
        self.min_score = float(config.get("demo.min_score", 0.5))
        self.start_confirm_ms = float(config.get("demo.start_confirm_ms", 300.0))
        self.done_after_ms = float(config.get("demo.done_after_ms", 900.0))
        self.reset()

    def reset(self) -> None:
        self.state = "waiting"
        self._since_ms = 0.0
        self._last_detected_ms: Optional[float] = None
        self._raw_since_ms: Optional[float] = None

    def _detected(self, best: Optional[MotionFeatures]) -> bool:
        if best is None or not best.valid:
            return False
        return bool(
            best.reversals >= self.min_reversals
            and self.min_rhythm <= best.rhythm_hz <= self.max_rhythm
            and best.amplitude_ratio >= self.min_amplitude
            and best.score >= self.min_score
        )

    def _set_state(self, state: str, now_ms: float) -> None:
        if state != self.state:
            self.state = state
            self._since_ms = float(now_ms)

    def update(self, ts_ms: float, best: Optional[MotionFeatures]) -> dict:
        """每帧调用；返回可直接写进 snapshot 的 demo 字典。"""
        now = float(ts_ms)
        raw = self._detected(best)
        if raw:
            self._last_detected_ms = now
            if self._raw_since_ms is None:
                self._raw_since_ms = now
        else:
            self._raw_since_ms = None

        if self.state == "chopping":
            if self._last_detected_ms is None or now - self._last_detected_ms >= self.done_after_ms:
                self._set_state("done", now)
        elif raw and self._raw_since_ms is not None and now - self._raw_since_ms >= self.start_confirm_ms:
            # waiting 或 done 中重新出现连续节奏，都切回“正在切菜”。
            self._set_state("chopping", now)

        age = None if self._last_detected_ms is None else max(0.0, now - self._last_detected_ms)
        feature = best if best is not None else MotionFeatures()
        return {
            "state": self.state,
            "state_zh": self.labels.get(self.state, self.state),
            "since_ms": round(self._since_ms, 1),
            "duration_ms": round(max(0.0, now - self._since_ms), 1),
            "last_detected_ms": None if self._last_detected_ms is None else round(self._last_detected_ms, 1),
            "last_detected_age_ms": None if age is None else round(age, 1),
            "raw_detected": bool(raw),
            "rhythm_hz": round(float(feature.rhythm_hz), 3),
            "reversals": int(feature.reversals),
            "amplitude_ratio": round(float(feature.amplitude_ratio), 4),
            "score": round(float(feature.score), 3),
            "hand": feature.key,
        }
