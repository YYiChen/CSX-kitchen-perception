import math
import unittest

import helpers

from kitchen_perception.config import load_config
from kitchen_perception.motion import HandMotionTracker, MotionFeatures, MotionObservation


def _obs(x, y, key="right#0", label="right", source="palm", valid=True, score=0.95):
    return MotionObservation(key=key, label=label, x=x, y=y, valid=valid, source=source, score=score)


def _feed_chop(tracker, fps=30.0, seconds=4.0, freq=2.0, amp=0.03, scale=0.25, source="palm"):
    dt = 1000.0 / fps
    feature = None
    for i in range(int(seconds * fps) + 1):
        ts = i * dt
        y = 0.60 + amp * math.sin(2 * math.pi * freq * ts / 1000.0)
        features = tracker.update([_obs(0.5, y, source=source)], ts, scale)
        feature = features[0]
    return feature


class TestMotion(unittest.TestCase):
    def setUp(self):
        self.config = load_config()

    def test_regular_chop_detected_and_stable(self):
        tracker = HandMotionTracker(self.config)
        feature = _feed_chop(tracker)
        self.assertTrue(feature.detected)
        self.assertTrue(feature.stable)
        self.assertGreaterEqual(feature.reversals, 4)
        self.assertAlmostEqual(feature.rhythm_hz, 2.0, delta=0.4)
        self.assertGreaterEqual(feature.amplitude_ratio, 0.08)
        self.assertGreater(feature.score, 0.6)

    def test_variable_frame_rate_uses_real_time(self):
        tracker = HandMotionTracker(self.config)
        feature = _feed_chop(tracker, fps=15.0, seconds=4.0)
        self.assertTrue(feature.detected)
        self.assertAlmostEqual(feature.rhythm_hz, 2.0, delta=0.5)

    def test_still_hand_not_detected(self):
        tracker = HandMotionTracker(self.config)
        for i in range(120):
            features = tracker.update([_obs(0.5, 0.60)], i * 33.3, 0.25)
        feature = features[0]
        self.assertFalse(feature.detected)
        self.assertLess(feature.score, 0.5)
        self.assertEqual(feature.movement_level, "LOW")

    def test_irregular_amplitude_unstable(self):
        tracker = HandMotionTracker(self.config)
        # 频率稳定但每个周期幅度交替大小，制造 amplitude_cv 过大
        feature = None
        for i in range(150):
            ts = i * 33.3
            cycle = int(ts / 500.0)
            amp = 0.015 if cycle % 2 == 0 else 0.08
            y = 0.60 + amp * math.sin(2 * math.pi * 2.0 * ts / 1000.0)
            features = tracker.update([_obs(0.5, y)], ts, 0.25)
            feature = features[0]
        self.assertFalse(feature.stable)
        self.assertGreaterEqual(feature.amplitude_cv, 0.4)

    def test_source_switch_resets_track(self):
        tracker = HandMotionTracker(self.config)
        _feed_chop(tracker, seconds=2.0, source="palm")
        features = tracker.update([_obs(0.5, 0.60, source="wrist", key="pose:right")], 5000.0, 0.25)
        feature = features[0]
        self.assertTrue(feature.valid)
        self.assertEqual(feature.reversals, 0)
        self.assertEqual(feature.sampling_gap_ms, 0.0)

    def test_hand_switch_resets(self):
        tracker = HandMotionTracker(self.config)
        _feed_chop(tracker, seconds=2.0)
        features = tracker.update([_obs(0.5, 0.60, key="left#0", label="left")], 2100.0, 0.25)
        self.assertEqual(features[0].reversals, 0)

    def test_best_prefers_right_on_tie(self):
        left = MotionFeatures(key="left#0", label="left", valid=True, score=0.8)
        right = MotionFeatures(key="right#0", label="right", valid=True, score=0.8)
        self.assertIs(HandMotionTracker.best([left, right]), right)
        self.assertIsNone(HandMotionTracker.best([MotionFeatures(valid=False)]))


if __name__ == "__main__":
    unittest.main()
