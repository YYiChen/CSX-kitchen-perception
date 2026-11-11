"""视频导出任务测试（用合成短视频，不依赖真实素材）。"""

import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

import helpers  # noqa: F401

from kitchen_perception.config import load_config
from kitchen_perception.exporter import VideoExportJob
from kitchen_perception.media_io import compute_frame_stride, resize_to_max_width


def _write_synthetic_video(path: Path, frames: int = 24, width: int = 320, height: int = 240, fps: float = 30.0) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("无法创建测试视频")
    for index in range(frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (30, 30, 30)
        x = int(20 + (index * 7) % max(1, width - 80))
        cv2.rectangle(frame, (x, 80), (x + 50, 160), (40, 180, 240), -1)
        cv2.putText(frame, f"F{index}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        writer.write(frame)
    writer.release()


class TestMediaHelpers(unittest.TestCase):
    def test_stride_by_target_fps(self):
        self.assertEqual(compute_frame_stride(59.82, 30.0), 2)
        self.assertEqual(compute_frame_stride(30.0, 30.0), 1)
        self.assertEqual(compute_frame_stride(0.0, 30.0), 1)
        self.assertEqual(compute_frame_stride(25.0, 60.0), 1)

    def test_resize_to_max_width_even_dimensions(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        out = resize_to_max_width(frame, 1280)
        self.assertEqual(out.shape[1], 1280)
        self.assertEqual(out.shape[0] % 2, 0)
        self.assertEqual(out.shape[0], 720)
        odd = np.zeros((101, 201, 3), dtype=np.uint8)
        result = resize_to_max_width(odd, 1000)
        self.assertEqual(result.shape[0] % 2, 0)
        self.assertEqual(result.shape[1] % 2, 0)


class TestVideoExportJob(unittest.TestCase):
    def test_export_writes_mp4_and_reports_done(self):
        with tempfile.TemporaryDirectory(prefix="kp_export_test_") as tmp:
            video = Path(tmp) / "in.mp4"
            output = Path(tmp) / "out.mp4"
            _write_synthetic_video(video)
            job = VideoExportJob(
                load_config(),
                str(video),
                str(output),
                max_width=320,
                max_fps=30.0,
                max_frames=6,
            )
            job.start()
            self.assertTrue(job.wait(timeout=60.0), "导出线程未在 60s 内结束")
            status = job.status()
            self.assertEqual(status["state"], "done", status)
            self.assertEqual(status["frames"], 6)
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)
            cap = cv2.VideoCapture(str(output))
            try:
                self.assertTrue(cap.isOpened())
                ok, frame = cap.read()
                self.assertTrue(ok)
                self.assertEqual(frame.shape[1], 320)
                self.assertEqual(frame.shape[0], 240)
            finally:
                cap.release()

    def test_export_missing_video_fails(self):
        with tempfile.TemporaryDirectory(prefix="kp_export_fail_") as tmp:
            output = Path(tmp) / "out.mp4"
            job = VideoExportJob(load_config(), str(Path(tmp) / "missing.mp4"), str(output))
            job.start()
            self.assertTrue(job.wait(timeout=30.0), "失败任务未在 30s 内结束")
            status = job.status()
            self.assertEqual(status["state"], "failed")
            self.assertTrue(status["error"])
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
