import math
import unittest

import helpers  # noqa: F401  (负责 sys.path 引导)

from kitchen_perception.geometry import EMA, Hysteresis, MotionSpeedTracker, angle_deg, percentile


class TestGeometry(unittest.TestCase):
    def test_angle_deg_right_angle(self):
        self.assertAlmostEqual(angle_deg((0, 0), (0, 1), (1, 1)), 90.0, places=3)

    def test_angle_deg_straight(self):
        # 手指伸直时 PIP/DIP 内角应为 180°（三点共线、方向相反）
        self.assertAlmostEqual(angle_deg((0, 0), (0, 1), (0, 2)), 180.0, places=3)

    def test_angle_deg_degenerate_nan(self):
        self.assertTrue(math.isnan(angle_deg((0, 0), (0, 0), (1, 1))))

    def test_ema_smoothing(self):
        ema = EMA(0.5)
        self.assertEqual(ema.update(10.0), 10.0)
        self.assertAlmostEqual(ema.update(20.0), 15.0)
        self.assertAlmostEqual(ema.update(None), 15.0)

    def test_hysteresis_requires_confirm_frames(self):
        hyst = Hysteresis(confirm_frames=3)
        state, _ = hyst.update("upright")
        self.assertEqual(state, "upright")
        state, _ = hyst.update("bent_over")
        self.assertEqual(state, "upright")
        state, _ = hyst.update("bent_over")
        self.assertEqual(state, "upright")
        state, _ = hyst.update("bent_over")
        self.assertEqual(state, "bent_over")

    def test_percentile_simple(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.assertAlmostEqual(percentile(values, 50), 3.0)
        self.assertAlmostEqual(percentile(values, 0), 1.0)
        self.assertAlmostEqual(percentile(values, 100), 5.0)

    def test_motion_speed(self):
        tracker = MotionSpeedTracker(window_s=1.0)
        self.assertEqual(tracker.update(0.1, 0.1, 0.0, 100.0), 0.0)
        speed = tracker.update(0.2, 0.1, 500.0, 100.0)
        self.assertGreater(speed, 0.0)


if __name__ == "__main__":
    unittest.main()
