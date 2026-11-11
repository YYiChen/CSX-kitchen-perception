"""把视频逐帧标注后导出 MP4。

设计目标：
- 与 UI 实时预览共用 Pipeline 和 OverlayRenderer，导出的画面/状态与界面一致；
- 视频先按 video.max_width 等比缩放、按 video.max_fps 抽帧，避免 4K 素材把
  演示机拖慢；输出仍是 MP4，帧率 = 源帧率 / 抽帧步长；
- 导出在后台线程执行，UI 只轮询进度；输出路径支持中文（先写 ASCII 临时文件）。
- 导出视频不包含音轨（避免依赖 ffmpeg）；如果需要音频可用原视频工具再合成。
"""

from __future__ import annotations

import logging
import math
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

import cv2

from .config import Config
from .media_io import compute_frame_stride, move_file_with_retry, resize_to_max_width
from .pipeline import Pipeline
from .sources import create_source

LOGGER = logging.getLogger(__name__)


class VideoExportJob:
    """后台视频导出任务；start() 后由 UI/CLI 轮询 status()。"""

    def __init__(
        self,
        config: Config,
        video_path: str,
        output_path: str,
        max_width: Optional[int] = None,
        max_fps: Optional[float] = None,
        zone_settings: Optional[dict] = None,
        max_frames: Optional[int] = None,
    ) -> None:
        self.config = config
        self.video_path = str(video_path)
        self.output_path = Path(output_path)
        self.max_width = int(
            max_width if max_width is not None else config.get("video.export_max_width", 1280)
        )
        self.max_fps = float(
            max_fps if max_fps is not None else config.get("video.export_max_fps", 30.0)
        )
        self.zone_settings = dict(zone_settings) if isinstance(zone_settings, dict) else None
        self.max_frames = int(max_frames) if max_frames else None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._state = "idle"
        self._error = ""
        self._frames = 0
        self._total = 0
        self._fps = 0.0
        self._stride = 1
        self._started_at: Optional[float] = None
        self._finished_at: Optional[float] = None

    # ------------------------------------------------------------------ 生命周期
    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._state = "running"
            self._error = ""
            self._frames = 0
            self._total = 0
            self._started_at = time.time()
            self._finished_at = None
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="VideoExportJob", daemon=True)
            self._thread.start()

    def cancel(self) -> None:
        self._stop.set()

    def is_running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def wait(self, timeout: Optional[float] = None) -> bool:
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
            return not thread.is_alive()
        return True

    def status(self) -> dict:
        with self._lock:
            elapsed = 0.0
            if self._started_at is not None:
                end = self._finished_at if self._finished_at is not None else time.time()
                elapsed = max(0.0, end - self._started_at)
            progress = round(self._frames / self._total, 4) if self._total > 0 else None
            return {
                "state": self._state,
                "frames": int(self._frames),
                "total_frames": int(self._total),
                "progress": progress,
                "fps": round(self._fps, 3),
                "stride": int(self._stride),
                "elapsed_s": round(elapsed, 2),
                "output_path": str(self.output_path),
                "error": self._error,
            }

    # ------------------------------------------------------------------ 后台执行
    def _set_progress(self, frames: int, total: Optional[int] = None) -> None:
        with self._lock:
            self._frames = int(frames)
            if total is not None:
                self._total = int(total)

    def _finish(self, state: str, error: str = "") -> None:
        with self._lock:
            self._state = state
            self._error = error
            self._finished_at = time.time()

    def _run(self) -> None:
        source = None
        pipeline = None
        writer = None
        tmp_dir: Optional[Path] = None
        try:
            source = create_source("video", path=self.video_path)
            if getattr(source, "cap", None) is None:
                raise RuntimeError(getattr(source, "last_error", "") or "无法打开视频文件")

            source_fps = float(getattr(source, "fps", 25.0) or 25.0)
            stride = compute_frame_stride(source_fps, self.max_fps)
            output_fps = max(1.0, source_fps / stride)
            estimate = int(getattr(source, "frame_count_estimate", 0) or 0)
            total = max(1, int(math.ceil(estimate / stride))) if estimate > 0 else 0
            with self._lock:
                self._fps = output_fps
                self._stride = stride
            self._set_progress(0, total)

            pipeline = Pipeline(self.config, source=source, enable_overlay=True)
            if self.zone_settings is not None:
                pipeline.apply_zone_settings(self.zone_settings, persist=False)
            tmp_dir = Path(tempfile.mkdtemp(prefix="kp_export_"))
            tmp_path = tmp_dir / "out.mp4"

            processed = 0
            raw_index = 0
            while not self._stop.is_set():
                ok, frame, ts_ms = source.read()
                if not ok:
                    break
                if stride > 1 and (raw_index % stride) != 0:
                    raw_index += 1
                    continue
                raw_index += 1
                frame = resize_to_max_width(frame, self.max_width)
                pipeline.process(frame, ts_ms)
                overlay = pipeline.last_overlay
                if overlay is None:
                    overlay = frame
                height, width = overlay.shape[:2]
                even_w = max(2, width - width % 2)
                even_h = max(2, height - height % 2)
                if (even_w, even_h) != (width, height):
                    overlay = cv2.resize(overlay, (even_w, even_h), interpolation=cv2.INTER_AREA)
                if writer is None:
                    self.output_path.parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(
                        str(tmp_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        output_fps,
                        (even_w, even_h),
                    )
                    if not writer.isOpened():
                        raise RuntimeError("无法创建 MP4 写入器（mp4v）")
                writer.write(overlay)
                processed += 1
                self._set_progress(processed)
                if self.max_frames is not None and processed >= self.max_frames:
                    break

            if self._stop.is_set():
                self._finish("cancelled")
                return
            if processed == 0:
                raise RuntimeError("没有可导出的画面")

            writer.release()
            writer = None
            try:
                if self.output_path.exists():
                    self.output_path.unlink()
            except OSError:
                LOGGER.warning("覆盖已有导出文件失败：%s", self.output_path)
            move_file_with_retry(tmp_path, self.output_path)
            self._finish("done")
        except Exception as exc:  # 后台线程异常必须转成状态给 UI/CLI
            LOGGER.exception("视频导出失败")
            self._finish("failed", f"{type(exc).__name__}: {exc}")
        finally:
            if writer is not None:
                try:
                    writer.release()
                except Exception:
                    pass
            if pipeline is not None:
                try:
                    pipeline.close()
                except Exception:
                    LOGGER.warning("关闭导出流水线失败", exc_info=True)
            elif source is not None:
                try:
                    source.close()
                except Exception:
                    pass
            if tmp_dir is not None:
                shutil.rmtree(tmp_dir, ignore_errors=True)

