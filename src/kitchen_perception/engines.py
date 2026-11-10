"""MediaPipe Tasks 推理封装：PoseLandmarker / HandLandmarker。

关键实现细节（都踩过坑，别改回去）：
- 模型用 model_asset_buffer 加载，而不是 model_asset_path：Windows 中文路径会让
  MediaPipe 的 C++ 文件读取失败（实测 FileNotFoundError）。
- VIDEO 模式要求时间戳严格递增，重复/回退会抛
  ValueError: Input timestamp must be monotonically increasing。
- 每次 detect 都新建 mp.Image，避免跨线程复用同一份像素缓冲。
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision
from mediapipe.tasks.python.vision import RunningMode

from .config import Config
from .types import HandFrame, Landmark, PoseFrame

LOGGER = logging.getLogger(__name__)


class ModelError(RuntimeError):
    """模型缺失/损坏/加载失败。"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(config: Config) -> dict:
    manifest_path = config.resolve_path(config.get("models.manifest", "models/MANIFEST.json"))
    if not manifest_path.exists():
        raise ModelError(f"模型清单不存在: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def verify_models(config: Config, check_hash: Optional[bool] = None) -> list[str]:
    """校验模型文件；返回错误列表（空表示通过）。"""
    manifest = load_manifest(config)
    do_hash = config.get("models.verify_sha256", True) if check_hash is None else check_hash
    errors: list[str] = []
    for item in manifest.get("models", []):
        path = config.resolve_path(item["path"])
        if not path.exists():
            errors.append(f"缺少模型: {path}（运行 scripts/download_models.py）")
            continue
        size = path.stat().st_size
        if item.get("bytes") and size != int(item["bytes"]):
            errors.append(f"模型大小不符: {path} 期望 {item['bytes']} 实际 {size}")
            continue
        if do_hash:
            actual = sha256_file(path)
            if actual != item.get("sha256"):
                errors.append(f"模型 sha256 不符: {path} 期望 {item['sha256']} 实际 {actual}")
    return errors


def resolve_model_paths(config: Config) -> dict[str, Path]:
    variant = str(config.get("models.pose", "lite"))
    pose_rel = config.get(f"models.pose_files.{variant}")
    if not pose_rel:
        raise ModelError(f"未知姿态模型变体: {variant}")
    return {
        "pose": config.resolve_path(pose_rel),
        "hand": config.resolve_path(config.get("models.hand", "models/hand_landmarker.task")),
    }


def _to_mp_image(frame_bgr: np.ndarray) -> mp.Image:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))


class _TimestampGuard:
    def __init__(self) -> None:
        self.last = -1

    def next(self, ts_ms: float) -> int:
        ts = int(round(float(ts_ms)))
        if ts <= self.last:
            ts = self.last + 1
        self.last = ts
        return ts


# MediaPipe 的 Clearcut 遥测在离线/受限网络下会让 close() 同步等待约 42s（本机实测）。
# 关闭动作放守护线程、最多等 1.5s，超时就放弃等待并把引用留在进程级列表避免提前析构；
# 进程退出由入口 os._exit() 直接终止（见 __main__.py 与 tests/run_tests.py）。
_CLOSE_WAIT_S = 1.5
_ABANDONED_LANDMARKERS: list[object] = []


def _release_landmarker(landmarker: object) -> None:
    """尽力关闭 landmarker；离线环境下不阻塞调用方。"""
    _ABANDONED_LANDMARKERS.append(landmarker)

    def _close() -> None:
        try:
            landmarker.close()  # type: ignore[attr-defined]
        except Exception:
            pass

    thread = threading.Thread(target=_close, name="mp-landmarker-close", daemon=True)
    thread.start()
    thread.join(timeout=_CLOSE_WAIT_S)


class PoseEngine:
    def __init__(self, model_path: Path, config: Config) -> None:
        if not model_path.exists():
            raise ModelError(f"姿态模型不存在: {model_path}")
        self.model_path = model_path
        buf = model_path.read_bytes()
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=buf),
            running_mode=RunningMode.VIDEO,
            num_poses=int(config.get("inference.num_poses", 1)),
            min_pose_detection_confidence=float(config.get("inference.min_pose_detection_confidence", 0.5)),
            min_pose_presence_confidence=float(config.get("inference.min_pose_presence_confidence", 0.5)),
            min_tracking_confidence=float(config.get("inference.min_tracking_confidence", 0.5)),
            output_segmentation_masks=False,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._ts = _TimestampGuard()
        LOGGER.info("PoseLandmarker 已加载: %s", model_path.name)

    def detect(self, frame_bgr: np.ndarray, ts_ms: float) -> PoseFrame:
        image = _to_mp_image(frame_bgr)
        result = self._landmarker.detect_for_video(image, self._ts.next(ts_ms))
        if not result.pose_landmarks:
            return PoseFrame(present=False)
        landmarks = [Landmark.from_mp(lm) for lm in result.pose_landmarks[0]]
        world = [Landmark.from_mp(lm) for lm in result.pose_world_landmarks[0]] if result.pose_world_landmarks else []
        key_ids = (11, 12, 23, 24)
        vis = [min(1.0, max(0.0, landmarks[i].visibility)) for i in key_ids if i < len(landmarks)]
        return PoseFrame(
            landmarks=landmarks,
            world=world,
            present=True,
            mean_visibility=float(np.mean(vis)) if vis else 0.0,
        )

    def close(self) -> None:
        _release_landmarker(self._landmarker)


class HandEngine:
    def __init__(self, model_path: Path, config: Config) -> None:
        if not model_path.exists():
            raise ModelError(f"手部模型不存在: {model_path}")
        self.model_path = model_path
        buf = model_path.read_bytes()
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=buf),
            running_mode=RunningMode.VIDEO,
            num_hands=int(config.get("inference.num_hands", 2)),
            min_hand_detection_confidence=float(config.get("inference.min_hand_detection_confidence", 0.5)),
            min_hand_presence_confidence=float(config.get("inference.min_hand_presence_confidence", 0.5)),
            min_tracking_confidence=float(config.get("inference.min_tracking_confidence", 0.5)),
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._ts = _TimestampGuard()
        LOGGER.info("HandLandmarker 已加载: %s", model_path.name)

    def detect(self, frame_bgr: np.ndarray, ts_ms: float) -> list[HandFrame]:
        image = _to_mp_image(frame_bgr)
        result = self._landmarker.detect_for_video(image, self._ts.next(ts_ms))
        frames: list[HandFrame] = []
        for idx, hand in enumerate(result.hand_landmarks):
            landmarks = [Landmark.from_mp(lm) for lm in hand]
            world = []
            if result.hand_world_landmarks and idx < len(result.hand_world_landmarks):
                world = [Landmark.from_mp(lm) for lm in result.hand_world_landmarks[idx]]
            handedness = "unknown"
            score = 0.0
            if result.handedness and idx < len(result.handedness) and result.handedness[idx]:
                category = result.handedness[idx][0]
                handedness = str(category.category_name or "unknown").lower()
                score = float(category.score or 0.0)
            frames.append(HandFrame(landmarks=landmarks, world=world, handedness=handedness, score=score))
        return frames

    def close(self) -> None:
        _release_landmarker(self._landmarker)
