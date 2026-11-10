import unittest

import helpers

from kitchen_perception.config import load_config
from kitchen_perception.fingers import HandAnalyzer, analyze_fingers, classify_grip
from kitchen_perception.types import FingerMetrics


class TestFingerAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()

    def test_open_hand_all_extended(self):
        states, metrics, info = analyze_fingers(helpers.make_hand("open"), self.config, aspect=1.0)
        self.assertEqual(states, {"thumb": "extended", "index": "extended", "middle": "extended", "ring": "extended", "pinky": "extended"})
        self.assertTrue(all(m.valid for m in metrics.values()))
        self.assertEqual(info["valid_count"], 5)

    def test_fist_all_flexed(self):
        states, metrics, _ = analyze_fingers(helpers.make_hand("fist"), self.config, aspect=1.0)
        for finger in ("index", "middle", "ring", "pinky"):
            self.assertEqual(states[finger], "flexed")
        self.assertEqual(states["thumb"], "flexed")

    def test_short_landmarks_unknown(self):
        from kitchen_perception.types import HandFrame

        states, metrics, info = analyze_fingers(HandFrame(landmarks=[]), self.config)
        self.assertEqual(set(states.values()), {"unknown"})
        self.assertFalse(info["valid"])

    def test_grip_open_and_fist(self):
        open_states, open_metrics, _ = analyze_fingers(helpers.make_hand("open"), self.config)
        self.assertEqual(classify_grip(open_states, open_metrics, self.config)[0], "open")
        fist_states, fist_metrics, _ = analyze_fingers(helpers.make_hand("fist"), self.config)
        self.assertEqual(classify_grip(fist_states, fist_metrics, self.config)[0], "fist")

    def test_grip_point(self):
        states = {"thumb": "flexed", "index": "extended", "middle": "flexed", "ring": "flexed", "pinky": "flexed"}
        grip, confidence, _ = classify_grip(states, {}, self.config)
        self.assertEqual(grip, "point")
        self.assertGreater(confidence, 0.5)

    def test_grip_unknown_too_few_fingers(self):
        states = {"thumb": "unknown", "index": "unknown", "middle": "extended", "ring": "unknown", "pinky": "unknown"}
        self.assertEqual(classify_grip(states, {}, self.config)[0], "unknown")

    def test_hand_analyzer_state_shape(self):
        state = HandAnalyzer(self.config).analyze(helpers.make_hand("open"), aspect=1.0)
        self.assertEqual(state.label, "right")
        self.assertEqual(state.grip, "open")
        self.assertIn("index", state.fingers)
        self.assertIsNotNone(state.bbox_norm)


if __name__ == "__main__":
    unittest.main()
