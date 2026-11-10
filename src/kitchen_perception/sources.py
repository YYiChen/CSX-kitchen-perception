"""视频源：默认摄像头 / 视频文件。

摄像头：
- 自动尝试 Windows 后端（DSHOW → MSMF → ANY），因为不同机器对 index 的支持不同。
- CAP_PROP_BUFFERSIZE=1，避免驱动缓冲造成额外延迟。
- read 失败时自动重连（最多 3 次），并把错误写入 last_error。

视频文件：
- 支持中文路径（OpenCV VideoCapture 在 Windows 上对中文路径不稳定，
  先复制到 ASCII 临时路径再打开）。
- 时间戳 = 帧序号 / fps * 1000，保证单调递增；支持循环播放。
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

BACKENDS = {
    "dshow": getattr(cv2, "CAP_DSHOW", 700),
    "msmf": getattr(cv2, "CAP_MSMF", 1400),
    "any": getattr(cv2, "CAP_ANY", 0),
}


class FrameSource:
    """帧源抽象。read() 返回 (ok, frame_bgr, ts_ms)。"""

    kind = "unknown"
    description = ""

    def read(self) -> tuple[bool, Optional[np.ndarray], float]:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def is_live(self) -> bool:
        return False


def _open_camera(index: int, backends: list[str], width: int, height: int, fps: int) -> tuple[Optional[cv2.VideoCapture], str]:
    for name in backends:
        code = BACKENDS.get(name, cv2.CAP_ANY)
        cap = cv2.VideoCapture(index, code)
        if cap is not None and cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            ok, frame = cap.read()
            if ok and frame is not None:
                return cap, name
            cap.release()
        else:
            if cap is not None:
                cap.release()
    return None, ""


def probe_cameras(max_index: int = 5, backends: Optional[list[str]] = None) -> list[int]:
    """探测可用摄像头 index（会短暂占用设备，UI 刷新时串行调用即可）。"""
    found: list[int] = []
    for index in range(max_index):
        cap, _ = _open_camera(index, backends or ["dshow", "msmf", "any"], 640, 480, 30)
        if cap is not None:
            found.append(index)
            cap.release()
    return found


class CameraSource(FrameSource):
    kind = "camera"

    def __init__(
        self,
        index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        backends: Optional[list[str]] = None,
        reconnect_attempts: int = 3,
    ) -> None:
        self.index = int(index)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.backends = backends or ["dshow", "msmf", "any"]
        self.reconnect_attempts = int(reconnect_attempts)
        self.cap: Optional[cv2.VideoCapture] = None
        self.backend_name = ""
        self.last_error = ""
        self._start = time.perf_counter()
        self._open()

    def _open(self) -> bool:
        cap, name = _open_camera(self.index, self.backends, self.width, self.height, self.fps)
        if cap is None:
            self.last_error = f"无法打开摄像头 index={self.index}（已尝试 {self.backends}）"
            return False
        self.cap = cap
        self.backend_name = name
        self.last_error = ""
        return True

    def read(self) -> tuple[bool, Optional[np.ndarray], float]:
        if self.cap is None:
            if not self._open():
                return False, None, self._now_ms()
        for attempt in range(self.reconnect_attempts + 1):
            ok, frame = self.cap.read()
            if ok and frame is not None:
                return True, frame, self._now_ms()
            # 掉线重连
            if self.cap is not None:
                self.cap.release()
            self.cap = None
            if attempt < self.reconnect_attempts:
                self.last_error = "摄像头读取失败，正在重连..."
                time.sleep(0.15)
                if not self._open():
                    time.sleep(0.3)
        self.last_error = "摄像头读取失败（已停止重连）"
        return False, None, self._now_ms()

    def _now_ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000.0

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def is_live(self) -> bool:
        return True

    @property
    def description(self) -> str:
        return f"camera:{self.index} ({self.backend_name or 'unknown'})"


class VideoFileSource(FrameSource):
    kind = "video"

    def __init__(self, path: str | os.PathLike, loop: bool = False, max_frames: Optional[int] = None) -> None:
        self.path = Path(path)
        self.loop = bool(loop)
        self.max_frames = max_frames
        self._tmp_dir: Optional[str] = None
        self._local_path = str(self.path)
        self.cap: Optional[cv2.VideoCapture] = None
        self.last_error = ""
        self.frame_index = 0
        self.total_frames = 0
        self.fps = 25.0
        self._open()

    def _open(self) -> bool:
        if not self.path.exists():
            self.last_error = f"视频文件不存在: {self.path}"
            return False
        # OpenCV 在 Windows 中文路径上可能打不开，复制到 ASCII 临时目录
        if any(ord(ch) > 127 for ch in str(self.path)):
            self._tmp_dir = tempfile.mkdtemp(prefix="kp_video_src_")
            tmp = os.path.join(self._tmp_dir, "source.mp4")
            try:
                shutil.copyfile(str(self.path), tmp)
                self._local_path = tmp
            except OSError as exc:
                self.last_error = f"视频文件复制失败: {exc}"
                return False
        cap = cv2.VideoCapture(self._local_path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(self._local_path)
        if not cap.isOpened():
            self.last_error = f"无法打开视频文件: {self.path}"
            return False
        self.cap = cap
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.fps = fps if 1.0 <= fps <= 240.0 else 25.0
        self.frame_index = 0
        self.total_frames = 0
        return True

    def read(self) -> tuple[bool, Optional[np.ndarray], float]:
        if self.cap is None:
            self.last_error = "视频未打开"
            return False, None, 0.0
        if self.max_frames is not None and self.total_frames >= self.max_frames:
            self.last_error = "到达 max_frames 上限"
            return False, None, self._ts(self.total_frames)
        ok, frame = self.cap.read()
        if not ok or frame is None:
            if self.loop:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.frame_index = 0
                ok, frame = self.cap.read()
            if not ok or frame is None:
                self.last_error = "视频播放结束"
                return False, None, self._ts(self.total_frames)
        # 用累计帧数生成时间戳，保证循环播放时也单调递增
        ts = self._ts(self.total_frames)
        self.total_frames += 1
        self.frame_index += 1
        return True, frame, ts

    def _ts(self, index: int) -> float:
        return index * 1000.0 / max(self.fps, 1.0)

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self._tmp_dir:
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
            self._tmp_dir = None

    @property
    def description(self) -> str:
        return f"video:{self.path.name}"


def create_source(
    kind: str = "camera",
    index: int = 0,
    path: str | os.PathLike | None = None,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    loop: bool = False,
    backends: Optional[list[str]] = None,
) -> FrameSource:
    if kind == "camera":
        return CameraSource(index=index, width=width, height=height, fps=fps, backends=backends)
    if kind == "video":
        if path is None:
            raise ValueError("video 源需要 path")
        return VideoFileSource(path=path, loop=loop)
    raise ValueError(f"未知视频源类型: {kind}")