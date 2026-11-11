import json
import tempfile
import unittest
from pathlib import Path

import helpers

from kitchen_perception.config import load_config
from kitchen_perception.fingers import HandAnalyzer
from kitchen_perception.types import HandFrame, TorsoState
from kitchen_perception.zones import (
    BodyCenter,
    ZoneManager,
    ZoneSettings,
    classify_x,
    estimate_body_center,
    hand_zone_code,
    load_settings,
    palm_center,
    save_settings,
    view_x,
)


class TestCoordinates(unittest.TestCase):
    def test_view_x_mirror(self):
        self.assertAlmostEqual(view_x(0.25, True), 0.75)
        self.assertAlmostEqual(view_x(0.25, False), 0.25)
        self.assertAlmostEqual(view_x(-1.0, False), 0.0)
        self.assertAlmostEqual(view_x(2.0, False), 1.0)

    def test_palm_center_and_codes(self):
        hand = helpers.make_hand("open")
        palm = palm_center(hand)
        self.assertIsNotNone(palm)
        self.assertAlmostEqual(palm[0], sum(hand.landmarks[i].x for i in (0, 5, 9, 13, 17)) / 5)
        self.assertEqual(hand_zone_code("right", "A"), "RIGHT_HAND_IN_A")
        self.assertEqual(hand_zone_code("left", "B"), "LEFT_HAND_IN_B")
        self.assertEqual(hand_zone_code("right", "unknown"), "HAND_NOT_VISIBLE")
        self.assertIsNone(palm_center(HandFrame(landmarks=[])))


class TestBodyCenter(unittest.TestCase):
    def test_prefers_hips(self):
        pose = helpers.make_pose(image_points={23: (0.30, 0.60, 0.0, 1.0), 24: (0.10, 0.60, 0.0, 1.0)})
        center = estimate_body_center(pose, None, mirror=True)
        self.assertIsNotNone(center)
        self.assertEqual(center.source, "hips")
        self.assertAlmostEqual(center.raw_x, 0.20)
        self.assertAlmostEqual(center.view_x, 0.80)

    def test_shoulders_fallback(self):
        pose = helpers.make_pose(
            image_points={
                23: (0.5, 0.6, 0.0, 0.0),
                24: (0.5, 0.6, 0.0, 0.0),
                11: (0.40, 0.30, 0.0, 1.0),
                12: (0.20, 0.30, 0.0, 1.0),
            }
        )
        center = estimate_body_center(pose, None, mirror=False)
        self.assertEqual(center.source, "shoulders")
        self.assertAlmostEqual(center.raw_x, 0.30)

    def test_bbox_fallback(self):
        pose = helpers.make_pose()
        pose.present = False
        torso = TorsoState(present=True, posture="upright", confidence=0.8)
        torso.features.bbox_norm = [0.2, 0.2, 0.6, 0.8]
        center = estimate_body_center(pose, torso, mirror=False)
        self.assertEqual(center.source, "bbox")
        self.assertAlmostEqual(center.raw_x, 0.4)
        self.assertAlmostEqual(center.view_x, 0.4)

    def test_no_center(self):
        pose = helpers.make_pose()
        pose.present = False
        self.assertIsNone(estimate_body_center(pose, None, mirror=False))


class TestSettingsAndClassify(unittest.TestCase):
    def test_deadband_hysteresis(self):
        settings = ZoneSettings(split_x=0.5, deadband=0.02)
        self.assertEqual(classify_x(0.49, settings, "unknown"), "A")
        self.assertEqual(classify_x(0.51, settings, "unknown"), "B")
        self.assertEqual(classify_x(0.49, settings, "B"), "B")  # 死区内保持上一侧
        self.assertEqual(classify_x(0.51, settings, "A"), "A")
        self.assertEqual(classify_x(0.47, settings, "B"), "A")  # 越过死区才切换
        self.assertEqual(classify_x(0.53, settings, "A"), "B")

    def test_normalized_and_labels(self):
        settings = ZoneSettings(split_x=2.0, deadband=-1.0, ema_alpha=0.0, enter_ms=-5).normalized()
        self.assertLessEqual(settings.split_x, 0.95)
        self.assertGreaterEqual(settings.deadband, 0.0)
        self.assertGreaterEqual(settings.ema_alpha, 0.01)
        self.assertEqual(settings.enter_ms, 0.0)
        self.assertEqual(settings.label("A"), "备菜区")
        self.assertTrue(settings.enabled)

    def test_enabled_roundtrip(self):
        settings = ZoneSettings(split_x=0.4, enabled=False)
        self.assertFalse(settings.to_dict()["enabled"])
        restored = ZoneSettings.from_dict(settings.to_dict())
        self.assertFalse(restored.enabled)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "zones.json"
            save_settings(path, settings)
            loaded, warnings, _ = load_settings(load_config(overrides={"zones": {"file": str(path)}}))
            self.assertFalse(warnings)
            self.assertFalse(loaded.enabled)

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "zones.json"
            save_settings(path, ZoneSettings(split_x=0.3333, deadband=0.01, labels={"A": "左区", "B": "右区"}))
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertAlmostEqual(data["split_x"], 0.3333)
            self.assertEqual(data["labels"]["A"], "左区")
            self.assertFalse((Path(tmp) / "zones.json.tmp").exists())

    def test_bad_file_falls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "zones.json"
            path.write_text("{bad json", encoding="utf-8")
            config = load_config(overrides={"zones": {"file": str(path)}})
            settings, warnings, resolved = load_settings(config)
            self.assertTrue(warnings)
            self.assertEqual(settings.split_x, 0.5)
            self.assertEqual(resolved, path)


class TestZoneManager(unittest.TestCase):
    def _manager(self, tmp: str, **overrides) -> ZoneManager:
        zones = {"file": str(Path(tmp) / "zones.json"), "enter_ms": 120}
        zones.update(overrides)
        return ZoneManager(load_config(overrides={"zones": zones}))

    def test_zone_switch_with_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = self._manager(tmp, ema_alpha=1.0)
            pose = helpers.make_pose(image_points={23: (0.30, 0.60, 0.0, 1.0), 24: (0.10, 0.60, 0.0, 1.0)})
            torso = TorsoState(present=True, posture="upright")
            info = manager.update(pose, torso, [], [], 0.0)
            self.assertEqual(info["body_zone"], "B")  # 镜像后 view_x=0.8
            pose2 = helpers.make_pose(image_points={23: (0.80, 0.60, 0.0, 1.0), 24: (0.60, 0.60, 0.0, 1.0)})
            info = manager.update(pose2, torso, [], [], 40.0)
            self.assertEqual(info["body_zone"], "B")
            info = manager.update(pose2, torso, [], [], 170.0)
            self.assertEqual(info["body_zone"], "A")

    def test_hand_zone_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = self._manager(tmp, ema_alpha=1.0)
            pose = helpers.make_pose(image_points={23: (0.30, 0.60, 0.0, 1.0), 24: (0.10, 0.60, 0.0, 1.0)})
            torso = TorsoState(present=True, posture="upright")
            hand = helpers.make_hand("open")  # x 约 0.28-0.58，镜像后为 0.42-0.72
            state = manager.update(pose, torso, [HandAnalyzer(load_config()).analyze(hand, aspect=1.0)], [hand], 0.0)
            self.assertEqual(len(state["hands"]), 1)
            self.assertIn(state["hands"][0]["zone"], ("A", "B"))

    def test_apply_and_persist(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = self._manager(tmp)
            manager.apply_settings({"split_x": 0.33, "deadband": 0.01, "labels": {"A": "左", "B": "右"}, "enabled": False}, persist=True)
            self.assertAlmostEqual(manager.settings.split_x, 0.33)
            self.assertFalse(manager.settings.enabled)
            self.assertTrue(Path(tmp, "zones.json").exists())
            manager2 = self._manager(tmp)
            self.assertAlmostEqual(manager2.settings.split_x, 0.33)
            self.assertEqual(manager2.settings.labels["A"], "左")
            self.assertFalse(manager2.settings.enabled)

    def test_reload_if_changed(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = self._manager(tmp)
            path = Path(tmp) / "zones.json"
            save_settings(path, ZoneSettings(split_x=0.4))
            manager.path = path
            self.assertTrue(manager.reload_if_changed())
            self.assertAlmostEqual(manager.settings.split_x, 0.4)


if __name__ == "__main__":
    unittest.main()
