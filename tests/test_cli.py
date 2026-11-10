import contextlib
import io
import json
import os
import unittest

import helpers

from kitchen_perception import cli


class TestCli(unittest.TestCase):
    def test_verify_models(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = cli.main(["verify-models"])
        self.assertEqual(code, 0)
        self.assertIn("通过", buffer.getvalue())

    def test_snapshot_command(self):
        image = helpers.PROJECT_ROOT / "assets" / "test" / "official" / "pose.jpg"
        if not image.exists():
            self.skipTest("缺少官方测试图片")
        out_json = helpers.PROJECT_ROOT / "outputs" / "test_cli_snapshot.json"
        out_png = helpers.PROJECT_ROOT / "outputs" / "test_cli_snapshot.jpg"
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = cli.main([
                "snapshot",
                "--image", str(image),
                "--json", str(out_json.relative_to(helpers.PROJECT_ROOT)),
                "--overlay", str(out_png.relative_to(helpers.PROJECT_ROOT)),
            ])
        self.assertEqual(code, 0)
        self.assertTrue(out_json.exists())
        self.assertTrue(out_png.exists())
        data = json.loads(out_json.read_text(encoding="utf-8"))
        self.assertIn("torso", data)
        for path in (out_json, out_png):
            try:
                os.remove(path)
            except OSError:
                pass


if __name__ == "__main__":
    unittest.main()
