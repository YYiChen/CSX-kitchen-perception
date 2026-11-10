import json
import tempfile
import unittest
from pathlib import Path

import helpers

from kitchen_perception.config import load_config
from kitchen_perception.events import EventLogger


class TestEventLogger(unittest.TestCase):
    def test_write_jsonl_and_recent(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(overrides={"logging": {"events_dir": tmp, "recent_events_in_memory": 3}})
            logger = EventLogger(config)
            try:
                self.assertTrue(logger.write({"event": "A", "ts_ms": 1.0, "frame": 1}))
                self.assertTrue(logger.write({"event": "B", "ts_ms": 2.0, "frame": 2}))
                self.assertIsNotNone(logger.path)
                self.assertTrue(logger.path.exists())
                lines = logger.path.read_text(encoding="utf-8").strip().splitlines()
                self.assertEqual(len(lines), 2)
                first = json.loads(lines[0])
                self.assertEqual(first["event"], "A")
                self.assertEqual(first["event_source"], "AUTOMATIC")
                self.assertIn("wall_time", first)
                recent = logger.recent()
                self.assertEqual([item["event"] for item in recent], ["A", "B"])
                logger.write({"event": "C", "ts_ms": 3.0})
                logger.write({"event": "D", "ts_ms": 4.0})
                self.assertEqual([item["event"] for item in logger.recent()], ["B", "C", "D"])
                self.assertEqual(logger.recent(1)[0]["event"], "D")
            finally:
                logger.close()

    def test_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(overrides={"logging": {"events_enabled": False, "events_dir": tmp}})
            logger = EventLogger(config)
            try:
                self.assertFalse(logger.write({"event": "A"}))
                self.assertEqual(len(logger.recent()), 1)
                self.assertIsNone(logger.path)
            finally:
                logger.close()

    def test_unwritable_degrades(self):
        with tempfile.TemporaryDirectory() as tmp:
            blocker = Path(tmp) / "not_a_dir"
            blocker.write_text("x", encoding="utf-8")
            config = load_config(overrides={"logging": {"events_dir": str(blocker / "events")}})
            logger = EventLogger(config)
            try:
                self.assertFalse(logger.write({"event": "A"}))
                self.assertIsNotNone(logger.error)
                self.assertEqual(len(logger.recent()), 1)
            finally:
                logger.close()

    def test_close_idempotent(self):
        logger = EventLogger(load_config(overrides={"logging": {"events_enabled": False}}))
        logger.close()
        logger.close()


if __name__ == "__main__":
    unittest.main()
