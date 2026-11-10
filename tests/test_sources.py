import os
import shutil
import tempfile
import unittest

import cv2
import numpy as np

import helpers  # noqa: F401

from kitchen_perception.sources import VideoFileSource


def write_video(path: str, frames: int = 10, fps: float = 10.0) -> None:
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    assert writer.isOpened()
    for i in range(frames):
        frame = np.full((48, 64, 3), i * 10 % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


class TestVideoSource(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="kp_test_src_")
        self.video = os.path.join(self.tmp, "clip.mp4")
        write_video(self.video, frames=10)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_reads_all_frames_with_monotonic_timestamps(self):
        source = VideoFileSource(self.video)
        timestamps = []
        count = 0
        while True:
            ok, frame, ts = source.read()
            if not ok:
                break
            count += 1
            timestamps.append(ts)
            self.assertEqual(frame.shape[0], 48)
        source.close()
        self.assertEqual(count, 10)
        self.assertEqual(timestamps, sorted(timestamps))
        self.assertGreater(len(set(timestamps)), 8)

    def test_loop_wraps(self):
        source = VideoFileSource(self.video, loop=True, max_frames=15)
        count = 0
        while True:
            ok, _, _ = source.read()
            if not ok:
                break
            count += 1
        source.close()
        self.assertEqual(count, 15)

    def test_chinese_filename(self):
        chinese = os.path.join(self.tmp, "中文 视频.mp4")
        shutil.copyfile(self.video, chinese)
        source = VideoFileSource(chinese)
        ok, frame, _ = source.read()
        source.close()
        self.assertTrue(ok)
        self.assertIsNotNone(frame)

    def test_missing_file(self):
        source = VideoFileSource(os.path.join(self.tmp, "nope.mp4"))
        ok, frame, _ = source.read()
        self.assertFalse(ok)
        self.assertIsNone(frame)
        self.assertTrue(source.last_error)


if __name__ == "__main__":
    unittest.main()
