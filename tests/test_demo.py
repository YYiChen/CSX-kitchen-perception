import unittest

import helpers  # noqa: F401

from kitchen_perception.config import load_config
from kitchen_perception.demo import DemoStateTracker
from kitchen_perception.motion import MotionFeatures


def _feature(reversals: int = 3, rhythm: float = 1.2, amplitude: float = 0.06, score: float = 0.8) -> MotionFeatures:
    return MotionFeatures(
        key="right#0",
        label="right",
        valid=True,
        reversals=reversals,
        rhythm_hz=rhythm,
        amplitude_ratio=amplitude,
        score=score,
    )


class TestDemoState(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(
            overrides={
                "demo": {
                    "start_confirm_ms": 100.0,
                    "done_after_ms": 300.0,
                }
            }
        )

    def _tracker(self) -> DemoStateTracker:
        return DemoStateTracker(self.config)

    def test_starts_waiting(self) -> None:
        tracker = self._tracker()
        info = tracker.update(0.0, None)
        self.assertEqual(info["state"], "waiting")
        self.assertEqual(info["state_zh"], "等待切菜")

    def test_chopping_then_done(self) -> None:
        tracker = self._tracker()
        self.assertEqual(tracker.update(0.0, _feature())["state"], "waiting")
        info = tracker.update(100.0, _feature())
        self.assertEqual(info["state"], "chopping")
        self.assertEqual(info["state_zh"], "正在切菜")
        self.assertEqual(tracker.update(200.0, None)["state"], "chopping")
        info = tracker.update(400.0, None)
        self.assertEqual(info["state"], "done")
        self.assertEqual(info["state_zh"], "切菜完成")
        self.assertFalse(info["raw_detected"])
        self.assertEqual(info["last_detected_ms"], 100.0)

    def test_short_gap_keeps_chopping(self) -> None:
        tracker = self._tracker()
        tracker.update(0.0, _feature())
        tracker.update(100.0, _feature())
        tracker.update(200.0, None)
        info = tracker.update(300.0, _feature())
        self.assertEqual(info["state"], "chopping")

    def test_resumes_after_done(self) -> None:
        tracker = self._tracker()
        tracker.update(0.0, _feature())
        tracker.update(100.0, _feature())
        tracker.update(500.0, None)
        self.assertEqual(tracker.state, "done")
        self.assertEqual(tracker.update(600.0, _feature())["state"], "done")
        self.assertEqual(tracker.update(700.0, _feature())["state"], "chopping")

    def test_non_chopping_motion_not_detected(self) -> None:
        tracker = self._tracker()
        info = tracker.update(0.0, _feature(reversals=1, rhythm=0.2, amplitude=0.005, score=0.2))
        self.assertEqual(info["state"], "waiting")
        self.assertFalse(info["raw_detected"])

    def test_to_dict_fields(self) -> None:
        tracker = self._tracker()
        tracker.update(0.0, _feature())
        info = tracker.update(100.0, _feature())
        for key in ("state", "state_zh", "since_ms", "duration_ms", "last_detected_ms",
                    "last_detected_age_ms", "raw_detected", "rhythm_hz", "reversals",
                    "amplitude_ratio", "score", "hand"):
            self.assertIn(key, info)
        self.assertEqual(info["hand"], "right#0")
        self.assertEqual(info["rhythm_hz"], 1.2)


if __name__ == "__main__":
    unittest.main()
