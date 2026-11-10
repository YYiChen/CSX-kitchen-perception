"""合成观测序列 CLI 的端到端测试（不需要摄像头/MediaPipe 推理）。

运行：
  py -3.12 -X utf8 tests/run_tests.py -p test_simulate.py
"""

import json
import tempfile
import unittest
from pathlib import Path

import helpers  # noqa: F401  （把 src/ 加入 sys.path）

from kitchen_perception import cli


class TestSimulate(unittest.TestCase):
    def _run(self, scenario: str) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / f"simulate_{scenario}.json"
            code = cli.main(["simulate", "--scenario", scenario, "--json", str(out_path)])
            self.assertEqual(code, 0)
            self.assertTrue(out_path.exists())
            return json.loads(out_path.read_text(encoding="utf-8"))

    def test_a_to_b_reports_transition_and_active_b(self):
        report = self._run("a_to_b")
        self.assertEqual(report["summary"]["final_state"], "ACTIVE_B")
        names = [item["event"] for item in report["events"]]
        self.assertEqual(names.count("ACTIVITY_A_STARTED"), 1)
        self.assertEqual(names.count("ZONE_TRANSITION_A_TO_B"), 1)
        self.assertEqual(names.count("ACTIVITY_B_STARTED"), 1)
        states = [item["state"] for item in report["timeline"]]
        self.assertIn("ACTIVE_A", states)
        self.assertIn("TRANSITION_A_TO_B", states)

    def test_occluded_after_hands_disappear(self):
        report = self._run("occluded")
        self.assertEqual(report["summary"]["final_state"], "OCCLUDED")
        names = [item["event"] for item in report["events"]]
        self.assertIn("HAND_OCCLUDED", names)
        self.assertNotIn("ACTIVITY_PAUSED", names)

    def test_no_person_becomes_away_not_occluded(self):
        report = self._run("no_person")
        self.assertEqual(report["summary"]["final_state"], "AWAY")
        names = [item["event"] for item in report["events"]]
        self.assertIn("USER_AWAY", names)
        self.assertNotIn("HAND_OCCLUDED", names)

    def test_away_return_resumes_paused(self):
        report = self._run("away_return")
        self.assertEqual(report["summary"]["final_state"], "PAUSED")
        names = [item["event"] for item in report["events"]]
        self.assertIn("USER_AWAY", names)
        self.assertIn("USER_RETURNED", names)


if __name__ == "__main__":
    unittest.main()
