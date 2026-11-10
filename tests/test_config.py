import json
import unittest

import helpers  # noqa: F401

from kitchen_perception.config import DEFAULT_CONFIG_PATH, Config, load_config


class TestConfig(unittest.TestCase):
    def test_default_config_sections(self):
        config = load_config()
        for section in ("project", "models", "camera", "inference", "torso", "hands", "motion", "zones", "chopping", "activity", "latency", "ui", "output", "logging"):
            self.assertIsInstance(config.section(section), dict)

    def test_dotted_get_and_default(self):
        config = load_config()
        self.assertEqual(config.get("models.pose"), "lite")
        self.assertEqual(config.get("nope.nope", 42), 42)

    def test_overrides_deep_merge(self):
        config = load_config(overrides={"torso": {"thresholds": {"upright_pitch_deg": 12.0}}})
        self.assertEqual(config.get("torso.thresholds.upright_pitch_deg"), 12.0)
        # 未被覆盖的兄弟键仍在
        self.assertEqual(config.get("torso.thresholds.bent_pitch_deg"), 45.0)

    def test_default_config_file_exists(self):
        self.assertTrue(DEFAULT_CONFIG_PATH.exists())
        data = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertEqual(data["project"]["schema_version"], "1.1")

    def test_custom_rule_module_resolvable(self):
        config = load_config()
        module_name = config.get("torso.custom_rule_module")
        self.assertTrue(module_name)
        config.resolve_path("models/hand_landmarker.task")


if __name__ == "__main__":
    unittest.main()
