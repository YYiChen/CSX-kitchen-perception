import math
import unittest

import helpers

from kitchen_perception.config import load_config
from kitchen_perception.torso import (
    TorsoAnalyzer,
    classify,
    compute_features,
    hands_in_workspace,
)
from kitchen_perception.types import TorsoFeatures


def features(pitch=0.0, roll=0.0, yaw=0.0, vis=1.0, hips=True, source="world"):
    f = TorsoFeatures(pitch_deg=pitch, roll_deg=roll, yaw_deg=yaw, visibility=vis, source=source)
    f.extra["shoulder_visible"] = True
    f.extra["hips_image_visible"] = hips
    return f


class TestTorsoClassify(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()

    def test_no_person(self):
        f = TorsoFeatures()
        f.extra["shoulder_visible"] = False
        self.assertEqual(classify(f, self.config)[0], "no_person")

    def test_not_visible(self):
        f = features(vis=0.2)
        self.assertEqual(classify(f, self.config)[0], "not_visible")

    def test_upright(self):
        self.assertEqual(classify(features(pitch=2.0, roll=1.0), self.config)[0], "upright")

    def test_leaning_forward(self):
        self.assertEqual(classify(features(pitch=30.0), self.config)[0], "leaning_forward")

    def test_bent_over(self):
        self.assertEqual(classify(features(pitch=55.0), self.config)[0], "bent_over")

    def test_leaning_backward(self):
        self.assertEqual(classify(features(pitch=-30.0), self.config)[0], "leaning_backward")

    def test_leaning_side_roll(self):
        self.assertEqual(classify(features(pitch=2.0, roll=25.0), self.config)[0], "leaning_side")

    def test_hip_penalty_note(self):
        state = TorsoAnalyzer(self.config).analyze(helpers.make_pose(), 640, 480)
        # 合成姿态里髋部可见，不应带“估算”提示
        self.assertNotIn("估算", state.note)

    def test_confidence_drops_without_hips(self):
        with_hips = classify(features(pitch=2.0, hips=True), self.config)
        without = classify(features(pitch=2.0, hips=False), self.config)
        self.assertGreater(with_hips[1], without[1])


class TestTorsoFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()

    def test_upright_pose_features(self):
        pose = helpers.make_pose()
        f = compute_features(pose, 640, 480)
        self.assertEqual(f.source, "world")
        self.assertAlmostEqual(f.pitch_deg, 0.0, delta=1.0)
        self.assertAlmostEqual(f.roll_deg, 0.0, delta=1.0)
        self.assertEqual(f.facing, "toward_camera")
        self.assertGreater(f.visibility, 0.9)

    def test_forward_pitch_sign(self):
        pose = helpers.make_pose(up=(0.0, -0.42, -0.2), nose_dz=-0.1)
        f = compute_features(pose, 640, 480)
        self.assertGreater(f.pitch_deg, 20.0)

    def test_roll_sign(self):
        pose = helpers.make_pose(up=(0.2, -0.37, 0.0))
        f = compute_features(pose, 640, 480)
        self.assertGreater(f.roll_deg, 20.0)

    def test_2d_fallback_without_world(self):
        pose = helpers.make_pose(with_world=False)
        f = compute_features(pose, 640, 480)
        self.assertEqual(f.source, "2d")
        self.assertGreaterEqual(f.visibility, 0.9)

    def test_hips_invisible_but_world_available(self):
        pose = helpers.make_pose(image_points={
            23: (0.5, 1.4, 0.0, 0.0),
            24: (0.5, 1.4, 0.0, 0.0),
        })
        f = compute_features(pose, 640, 480)
        self.assertEqual(f.source, "world")
        self.assertFalse(f.extra["hips_image_visible"])
        state = TorsoAnalyzer(self.config).analyze(pose, 640, 480)
        self.assertIn("估算", state.note)

    def test_analyzer_calibration(self):
        analyzer = TorsoAnalyzer(self.config)
        pose = helpers.make_pose(up=(0.0, -0.42, -0.2), nose_dz=-0.1)
        state = analyzer.analyze(pose, 640, 480)
        self.assertGreater(state.features.pitch_deg, 20.0)
        analyzer.calibrate()
        state = analyzer.analyze(pose, 640, 480)
        self.assertLess(abs(state.features.pitch_deg), 2.0)

    def test_hands_in_workspace(self):
        torso_bbox = [0.3, 0.3, 0.7, 0.9]
        self.assertTrue(hands_in_workspace([[0.4, 0.4, 0.5, 0.5]], torso_bbox))
        self.assertFalse(hands_in_workspace([[0.0, 0.0, 0.05, 0.05]], torso_bbox))


if __name__ == "__main__":
    unittest.main()
