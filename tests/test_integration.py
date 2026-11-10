import json
import os
import shutil
import tempfile
import unittest

import helpers

from kitchen_perception.config import load_config
from kitchen_perception.media_io import imread_u
from kitchen_perception.pipeline import Pipeline


def _has_official_image() -> bool:
    return (helpers.PROJECT_ROOT / "assets" / "test" / "official" / "pose.jpg").exists()


@unittest.skipUnless(_has_official_image(), "缺少官方测试图片 assets/test/official/pose.jpg")
class TestPipelineIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()
        cls.image_path = helpers.PROJECT_ROOT / "assets" / "test" / "official" / "pose.jpg"
        cls.frame = imread_u(cls.image_path)
        cls.tmp = tempfile.mkdtemp(prefix="kp_test_pipeline_")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _new_pipeline(self, overlay: bool = False) -> Pipeline:
        return Pipeline(self.config, source=None, enable_overlay=overlay)

    def test_snapshot_schema(self):
        pipeline = self._new_pipeline()
        try:
            snapshot = pipeline.process(self.frame, 0.0)
        finally:
            pipeline.close()
        for key in ("schema_version", "frame", "perf", "torso", "hands", "hands_state", "activity", "events"):
            self.assertIn(key, snapshot)
        self.assertIn(snapshot["torso"]["posture"], {"upright", "leaning_forward", "leaning_backward", "leaning_side", "bent_over", "not_visible", "uncertain"})
        json.dumps(snapshot, ensure_ascii=False)

    def test_overlay_renders(self):
        pipeline = self._new_pipeline(overlay=True)
        try:
            pipeline.process(self.frame, 0.0)
            overlay = pipeline.last_overlay
        finally:
            pipeline.close()
        self.assertIsNotNone(overlay)
        self.assertEqual(overlay.shape, self.frame.shape)


if __name__ == "__main__":
    unittest.main()
