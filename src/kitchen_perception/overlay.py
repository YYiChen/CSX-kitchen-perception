"""叠加层渲染：OpenCV 画骨架 + PIL 画中文 HUD。

为什么用 PIL 画字：OpenCV 的 putText 不支持中文，而界面需要中文状态标签。
为了不每帧重复创建字体，OverlayRenderer 在构造时缓存字体。
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .config import Config
from .fingers import FINGER_ORDER, FINGER_ZH
from .media_io import bgr_to_pil, load_cjk_font, pil_to_bgr
from .types import HandFrame, HandState, PoseFrame, TorsoState

COLOR_POSE = (0, 230, 120)
COLOR_HAND = (255, 170, 40)
COLOR_OK = (0, 220, 0)
COLOR_PARTIAL = (0, 200, 255)
COLOR_BAD = (60, 60, 255)
COLOR_UNKNOWN = (170, 170, 170)
COLOR_WARN = (0, 165, 255)

POSE_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
)

HAND_PALM = ((0, 5), (5, 9), (9, 13), (13, 17), (0, 17))
HAND_FINGER_CHAINS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}

FACING_ZH = {
    "toward_camera": "面向镜头",
    "away_from_camera": "背对镜头",
    "sideways": "侧身",
    "unknown": "朝向未知",
}

STATE_COLORS = {
    "extended": COLOR_OK,
    "partial": COLOR_PARTIAL,
    "flexed": COLOR_BAD,
    "unknown": COLOR_UNKNOWN,
}


class OverlayRenderer:
    def __init__(self, config: Config) -> None:
        self.config = config
        family = str(config.get("ui.font_family", "Microsoft YaHei UI"))
        self.font_hud = load_cjk_font(17)
        self.font_small = load_cjk_font(15)
        self.font_title = load_cjk_font(19)

    # ------------------------------------------------------------------ 骨架
    @staticmethod
    def _xy(lm, width: int, height: int, mirror: bool) -> tuple[int, int]:
        x = (1.0 - float(lm.x)) if mirror else float(lm.x)
        return int(round(x * width)), int(round(lm.y * height))

    def _draw_pose(self, canvas: np.ndarray, pose: PoseFrame, mirror: bool) -> None:
        h, w = canvas.shape[:2]
        landmarks = pose.landmarks
        for a, b in POSE_CONNECTIONS:
            if a >= len(landmarks) or b >= len(landmarks):
                continue
            vis = min(float(landmarks[a].visibility), float(landmarks[b].visibility))
            color = COLOR_POSE if vis >= 0.5 else COLOR_UNKNOWN
            cv2.line(canvas, self._xy(landmarks[a], w, h, mirror), self._xy(landmarks[b], w, h, mirror), color, 2, cv2.LINE_AA)
        for i, lm in enumerate(landmarks):
            if i not in (0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26):
                continue
            color = COLOR_POSE if float(lm.visibility) >= 0.5 else COLOR_UNKNOWN
            cv2.circle(canvas, self._xy(lm, w, h, mirror), 4, color, -1, cv2.LINE_AA)

    def _draw_hands(self, canvas: np.ndarray, hand_frames: Sequence[HandFrame], hand_states: Sequence[HandState], mirror: bool) -> None:
        h, w = canvas.shape[:2]
        for index, frame in enumerate(hand_frames):
            if len(frame.landmarks) < 21:
                continue
            state = hand_states[index] if index < len(hand_states) else None
            fingers = state.fingers if state else {}
            pts = [self._xy(lm, w, h, mirror) for lm in frame.landmarks]
            for a, b in HAND_PALM:
                cv2.line(canvas, pts[a], pts[b], COLOR_HAND, 2, cv2.LINE_AA)
            for finger in FINGER_ORDER:
                color = STATE_COLORS.get(fingers.get(finger, "unknown"), COLOR_UNKNOWN)
                chain = HAND_FINGER_CHAINS[finger]
                for a, b in zip(chain[:-1], chain[1:]):
                    cv2.line(canvas, pts[a], pts[b], color, 2, cv2.LINE_AA)
                cv2.circle(canvas, pts[chain[-1]], 4, color, -1, cv2.LINE_AA)
            for point in pts:
                cv2.circle(canvas, point, 2, COLOR_HAND, -1, cv2.LINE_AA)

    # ------------------------------------------------------------------ HUD
    def _hud_lines(self, torso: TorsoState, hand_states: Sequence[HandState], perf: dict, fps: float, extra: Iterable[str]) -> list[str]:
        lines: list[str] = []
        posture = torso.posture_zh or torso.posture
        lines.append(f"躯干：{posture}  置信度 {torso.confidence:.2f}")
        f = torso.features
        pitch = "--" if f.pitch_deg != f.pitch_deg else f"{f.pitch_deg:+.1f}°"
        roll = "--" if f.roll_deg != f.roll_deg else f"{f.roll_deg:+.1f}°"
        yaw = "--" if f.yaw_deg != f.yaw_deg else f"{f.yaw_deg:+.1f}°"
        lines.append(f"前倾 {pitch}  侧倾 {roll}  转向 {yaw}  {FACING_ZH.get(f.facing, f.facing)}")
        if hand_states:
            for hand in hand_states:
                label = {"left": "左手", "right": "右手"}.get(hand.label, "手部")
                fingers = " ".join(f"{FINGER_ZH[k]}:{'伸直' if v == 'extended' else '半屈' if v == 'partial' else '弯曲' if v == 'flexed' else '未知'}" for k, v in hand.fingers.items())
                lines.append(f"{label}：{hand.grip_zh or hand.grip}  {fingers}")
        else:
            lines.append("手部：未检测到手（做动作时把手放进画面）")
        level = perf.get("level", "?")
        p95 = perf.get("p95_ms")
        p95_text = "--" if p95 is None else f"{p95:.0f}ms"
        lines.append(f"FPS {fps:4.1f}   延迟 p95 {p95_text}   降级 {level}")
        lines.extend(str(line) for line in extra if line)
        return lines

    def _draw_hud(self, canvas: np.ndarray, lines: Sequence[str], warning: Optional[str]) -> np.ndarray:
        if not lines and not warning:
            return canvas
        pil = bgr_to_pil(canvas)
        overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        pad = 10
        line_h = 23
        text_w = max((draw.textlength(line, font=self.font_hud) for line in lines), default=0.0)
        box_w = int(text_w) + pad * 2
        box_h = line_h * len(lines) + pad * 2
        draw.rounded_rectangle([(0, 0), (box_w, box_h)], radius=10, fill=(0, 0, 0, 150))
        y = pad
        for line in lines:
            draw.text((pad, y), line, font=self.font_hud, fill=(240, 240, 240, 255))
            y += line_h
        if warning:
            w, h = pil.size
            text_w = draw.textlength(warning, font=self.font_title)
            x = max(10, int((w - text_w) / 2))
            y = h - 54
            draw.rounded_rectangle([(x - 12, y - 8), (x + text_w + 12, y + 30)], radius=10, fill=(0, 0, 0, 160))
            draw.text((x, y), warning, font=self.font_title, fill=COLOR_WARN + (255,))
        composite = Image.alpha_composite(pil.convert("RGBA"), overlay).convert("RGB")
        return pil_to_bgr(composite)

    def render(
        self,
        frame: np.ndarray,
        pose: PoseFrame,
        hand_frames: Sequence[HandFrame],
        hand_states: Sequence[HandState],
        torso: TorsoState,
        perf: dict,
        fps: float = 0.0,
        mirror: Optional[bool] = None,
        extra_lines: Optional[Iterable[str]] = None,
    ) -> np.ndarray:
        if mirror is None:
            mirror = bool(self.config.get("ui.mirror_preview", True))
        canvas = cv2.flip(frame, 1) if mirror else frame.copy()
        if pose.present:
            self._draw_pose(canvas, pose, mirror)
        if hand_frames:
            self._draw_hands(canvas, hand_frames, hand_states, mirror)

        warning = None
        if torso.posture == "no_person":
            warning = "未检测到人物：请坐到摄像头前"
        elif torso.posture == "not_visible":
            warning = "躯干关键点不足：请让上半身进入画面（必要时后退）"
        elif not hand_states:
            warning = "把手放进画面即可显示五指状态"

        lines = self._hud_lines(torso, hand_states, perf, fps, extra_lines or [])
        return self._draw_hud(canvas, lines, warning)
