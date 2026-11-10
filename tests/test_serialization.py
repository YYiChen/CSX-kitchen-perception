import json
import math
import unittest

import helpers  # noqa: F401

from kitchen_perception.types import FingerMetrics, HandState, PerceptionSnapshot, TorsoFeatures, TorsoState


class TestSerialization(unittest.TestCase):
    def test_nan_becomes_null(self):
        torso = TorsoState(features=TorsoFeatures(pitch_deg=float("nan")))
        snapshot = PerceptionSnapshot(torso=torso).to_dict()
        text = json.dumps(snapshot, ensure_ascii=False)
        self.assertNotIn("NaN", text)
        self.assertIsNone(snapshot["torso"]["pitch_deg"])

    def test_hand_state_to_dict(self):
        state = HandState(label="left", grip="open", fingers={"index": "extended"})
        state.finger_metrics["index"] = FingerMetrics(finger="index", pip_deg=170.0, valid=True)
        data = state.to_dict()
        self.assertEqual(data["label"], "left")
        self.assertEqual(data["finger_metrics"]["index"]["pip_deg"], 170.0)


if __name__ == "__main__":
    unittest.main()
