"""后台采集/推理线程，供 Tkinter 界面轮询最新结果。

UI 主线程只做显示，所有 MediaPipe 调用都在 worker 线程里，避免界面卡顿。
MediaPipe Tasks 的对象不是线程安全的，因此 Pipeline 只在一个线程里创建和使用。
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import numpy as np

from .config import Config
from .pipeline import Pipeline
from .sources import create_source

LOGGER = logging.getLogger(__name__)


class PerceptionWorker:
    def __init__(
        self,
        config: Config,
        source_kind: str = "camera",
        index: int = 0,
        video_path: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        loop_video: bool = False,
    ) -> None:
        self.config = config
        self.source_kind = source_kind
        self.index = int(index)
        self.video_path = video_path
        self.width = int(width or config.get("camera.width", 640))
        self.height = int(height or config.get("camera.height", 480))
        self.loop_video = loop_video
        self.pipeline: Optional[Pipeline] = None
        self.error: Optional[str] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: Optional[tuple[dict, Optional[np.ndarray]]] = None
        self._latest_monotonic = 0.0
        self.frame_count = 0

    # ------------------------------------------------------------------ 生命周期
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self.error = None
        self._thread = threading.Thread(target=self._run, name="PerceptionWorker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        self._thread = None

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------ 主循环
    def _run(self) -> None:
        source = None
        try:
            mirror = bool(self.config.get("ui.mirror_preview", True))
            source = create_source(
                self.source_kind,
                index=self.index,
                path=self.video_path,
                width=self.width,
                height=self.height,
                fps=int(self.config.get("camera.fps", 30)),
                loop=self.loop_video,
            )
            self.pipeline = Pipeline(self.config, source=source, enable_overlay=mirror or True)
            LOGGER.info("感知线程启动: %s", getattr(source, "description", ""))
            failures = 0
            while not self._stop.is_set():
                ok, frame, ts_ms = source.read()
                if not ok:
                    if source.is_live():
                        failures += 1
                        if failures > 30:
                            self.error = source.last_error or "摄像头读取失败"
                            break
                        time.sleep(0.05)
                        continue
                    break
                failures = 0
                snapshot = self.pipeline.process(frame, ts_ms)
                overlay = self.pipeline.last_overlay
                with self._lock:
                    self._latest = (snapshot, overlay)
                    self._latest_monotonic = time.monotonic()
                    self.frame_count += 1
                if self.source_kind == "video":
                    time.sleep(0.001)
        except Exception as exc:  # 线程内异常必须记录下来给 UI 显示
            LOGGER.exception("感知线程异常")
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            if self.pipeline is not None:
                self.pipeline.close()
            elif source is not None:
                source.close()

    # ------------------------------------------------------------------ 对外接口
    def get_latest(self) -> Optional[tuple[dict, Optional[np.ndarray]]]:
        with self._lock:
            return self._latest

    def latest_age_ms(self) -> float:
        with self._lock:
            if not self._latest:
                return float("inf")
            return (time.monotonic() - self._latest_monotonic) * 1000.0

    def calibrate(self) -> Optional[dict]:
        if self.pipeline is None:
            return None
        return self.pipeline.calibrate()

    def clear_calibration(self) -> None:
        if self.pipeline is not None:
            self.pipeline.clear_calibration()

    def set_paused(self, paused: bool) -> None:
        if paused:
            self._stop.set()
        else:
            self.start()
