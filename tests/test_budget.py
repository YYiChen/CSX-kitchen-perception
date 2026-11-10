import time
import unittest

import helpers  # noqa: F401

from kitchen_perception.budget import FpsMeter, LatencyBudget
from kitchen_perception.config import load_config


class TestBudget(unittest.TestCase):
    def test_budget_degrades_on_slow_frames(self):
        config = load_config(overrides={"latency": {"eval_every_frames": 5, "window_frames": 20}})
        budget = LatencyBudget(config)
        for _ in range(40):
            budget.observe(600.0)
        self.assertGreater(budget.level_index, 0)

    def test_budget_recovers_on_fast_frames(self):
        config = load_config(overrides={"latency": {"eval_every_frames": 5, "window_frames": 20}})
        budget = LatencyBudget(config)
        for _ in range(40):
            budget.observe(600.0)
        level_after_slow = budget.level_index
        for _ in range(80):
            budget.observe(20.0)
        self.assertLess(budget.level_index, level_after_slow)

    def test_fps_meter(self):
        meter = FpsMeter(window=10)
        meter.tick(0.0)
        meter.tick(100.0)
        meter.tick(200.0)
        self.assertAlmostEqual(meter.tick(300.0), 10.0, delta=0.5)


if __name__ == "__main__":
    unittest.main()
