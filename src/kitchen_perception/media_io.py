"""图像/字体 IO（Windows 中文路径安全）。

要点：
- cv2.imread / cv2.imwrite 在含中文的 Windows 路径上会失败，统一改用
  np.fromfile + cv2.imdecode 与 cv2.imencode + tofile。
- 视频写入同理：先写 ASCII 临时文件，再移动到目标路径。
- 中文文字用 Pillow 绘制，避免 cv2.putText 不支持 CJK。
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
import math
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DEFAULT_CJK_FONTS = (
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
)


def imread_u(path: str | os.PathLike, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """读取图像，支持中文/空格路径。"""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_u(path: str | os.PathLike, image: np.ndarray) -> bool:
    """写图像，支持中文/空格路径。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    suffix = target.suffix or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        return False
    encoded.tofile(str(target))
    return True


def bgr_to_pil(frame_bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def resize_keep_ratio(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """等比缩放到不超过 width x height。"""
    h, w = frame.shape[:2]
    scale = min(width / max(w, 1), height / max(h, 1))
    if abs(scale - 1.0) < 1e-6:
        return frame
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def compute_frame_stride(fps: float, max_fps: float) -> int:
    """按目标最大 FPS 计算抽帧步长；<=1 表示逐帧处理。"""
    try:
        fps_value = float(fps)
        max_value = float(max_fps)
    except (TypeError, ValueError):
        return 1
    if fps_value <= 0.0 or max_value <= 0.0:
        return 1
    return max(1, int(math.ceil(fps_value / max_value - 1e-9)))


def resize_to_max_width(frame: np.ndarray, max_width: int) -> np.ndarray:
    """等比缩放到最大宽度，并把宽高取偶数（视频编码器要求）。"""
    h, w = frame.shape[:2]
    try:
        limit = int(max_width)
    except (TypeError, ValueError):
        limit = 0
    if limit <= 0:
        return frame
    scale = min(1.0, float(limit) / max(float(w), 1.0))
    new_w = max(2, int(round(w * scale)))
    new_h = max(2, int(round(h * scale)))
    new_w -= new_w % 2
    new_h -= new_h % 2
    if (new_w, new_h) == (w, h):
        return frame
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def fit_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """保持比例缩放并用黑色填充到固定画布，返回 width x height。"""
    resized = resize_keep_ratio(frame, width, height)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    h, w = resized.shape[:2]
    x = (width - w) // 2
    y = (height - h) // 2
    canvas[y : y + h, x : x + w] = resized
    return canvas


def load_cjk_font(size: int, candidates: Iterable[str] = DEFAULT_CJK_FONTS) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def draw_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int] = (255, 255, 255),
    stroke_width: int = 0,
    stroke_fill: tuple[int, int, int] = (0, 0, 0),
) -> None:
    draw.text(xy, text, font=font, fill=fill, stroke_width=stroke_width, stroke_fill=stroke_fill)


def write_video_u(path: str | os.PathLike, frames: Iterable[np.ndarray], fps: float) -> bool:
    """把 BGR 帧写成 mp4，支持中文路径（先写临时 ASCII 路径）。

    注意：必须先 writer.release() 再移动文件。OpenCV 5 / Windows 上 writer 仍持有
    文件句柄时 shutil.move 会抛 PermissionError [WinError 32]（实现在任务里踩到过）。
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="kp_video_"))
    tmp_path = tmp_dir / "out.mp4"
    writer = None
    try:
        for frame in frames:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                if not writer.isOpened():
                    return False
            writer.write(frame)
        if writer is None:
            return False
        writer.release()
        writer = None
        move_file_with_retry(tmp_path, target)
        return True
    finally:
        if writer is not None:
            writer.release()
        shutil.rmtree(tmp_dir, ignore_errors=True)


def move_file_with_retry(src: Path, dst: Path, attempts: int = 6, delay_s: float = 0.2) -> None:
    """Windows 上杀毒/索引服务可能短暂占用刚写完的文件，移动失败时退避重试。"""
    last_error: OSError | None = None
    for attempt in range(attempts):
        try:
            shutil.move(str(src), str(dst))
            return
        except OSError as exc:
            last_error = exc
            time.sleep(delay_s * (attempt + 1))
    if last_error is not None:
        raise last_error


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
