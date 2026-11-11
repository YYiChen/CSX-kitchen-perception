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
COLOR_ZONE_A = (220, 180, 40)   # BGR：偏蓝
COLOR_ZONE_B = (40, 170, 230)   # BGR：偏橙
COLOR_BODY = (255, 80, 220)
DEMO_COLORS_RGB = {
    "waiting": (220, 150, 20),
    "chopping": (0, 190, 90),
    "done": (40, 110, 230),
}

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
        self.font_demo = load_cjk_font(40)

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
    def _draw_zones(self, canvas: np.ndarray, zones: dict) -> None:
        """画 A/B 半透明分区、分割线、活动区高亮、身体重心。"""
        if not bool(zones.get("enabled", True)):
            return
        h, w = canvas.shape[:2]
        try:
            split = float(zones.get("split_x", 0.5))
        except (TypeError, ValueError):
            split = 0.5
        split = min(max(split, 0.05), 0.95)
        x = int(round(split * w))
        layer = canvas.copy()
        cv2.rectangle(layer, (0, 0), (max(0, x - 1), h), COLOR_ZONE_A, -1)
        cv2.rectangle(layer, (min(w - 1, x), 0), (w - 1, h), COLOR_ZONE_B, -1)
        cv2.addWeighted(layer, 0.10, canvas, 0.90, 0, canvas)
        active_zone = str(zones.get("body_zone", "unknown"))
        if active_zone in ("A", "B"):
            color = COLOR_ZONE_A if active_zone == "A" else COLOR_ZONE_B
            if active_zone == "A":
                cv2.rectangle(canvas, (2, 2), (max(2, x - 3), h - 3), color, 2)
            else:
                cv2.rectangle(canvas, (min(w - 3, x + 1), 2), (w - 3, h - 3), color, 2)
        cv2.line(canvas, (x, 0), (x, h), (255, 255, 255), 2, cv2.LINE_AA)
        # 画在底部两角：左上角会被 HUD 盖住，底部中央留给警示条。
        label_y = max(30, h - 12)
        cv2.putText(canvas, "A", (12, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_ZONE_A, 2, cv2.LINE_AA)
        cv2.putText(canvas, "B", (max(12, w - 30), label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_ZONE_B, 2, cv2.LINE_AA)
        center = zones.get("body_center_norm")
        if isinstance(center, (list, tuple)) and len(center) == 2:
            cx = int(round(float(center[0]) * w))
            cy = int(round(float(center[1]) * h))
            cv2.drawMarker(canvas, (cx, cy), COLOR_BODY, markerType=cv2.MARKER_CROSS, markerSize=16, thickness=2)
            cv2.circle(canvas, (cx, cy), 5, COLOR_BODY, -1, cv2.LINE_AA)

    def _draw_hand_zones(self, canvas: np.ndarray, hand_states: Sequence[HandState], mirror: bool) -> None:
        h, w = canvas.shape[:2]
        for state in hand_states:
            if not state.palm_norm or len(state.palm_norm) != 2:
                continue
            x = (1.0 - float(state.palm_norm[0])) if mirror else float(state.palm_norm[0])
            px = int(round(x * w))
            py = int(round(float(state.palm_norm[1]) * h))
            color = COLOR_ZONE_A if state.zone == "A" else COLOR_ZONE_B if state.zone == "B" else COLOR_UNKNOWN
            cv2.circle(canvas, (px, py), 6, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, (px, py), 9, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, str(state.zone), (px + 10, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    def _hud_lines(self, torso: TorsoState, hand_states: Sequence[HandState], perf: dict, fps: float, extra: Iterable[str]) -> list[str]:
        lines: list[str] = []
        posture = torso.posture_zh or torso.posture
        if hand_states and torso.posture in ("no_person", "not_visible"):
            posture = "躯干未入镜（手部可见）"
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

    def _draw_hud(
        self,
        canvas: np.ndarray,
        lines: Sequence[str],
        warning: Optional[str],
        demo: Optional[dict] = None,
    ) -> np.ndarray:
        if not lines and not warning:
            return canvas
        pil = bgr_to_pil(canvas)
        overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        w, h = pil.size
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
            text_w = draw.textlength(warning, font=self.font_title)
            x = max(10, int((w - text_w) / 2))
            y = h - 54
            draw.rounded_rectangle([(x - 12, y - 8), (x + text_w + 12, y + 30)], radius=10, fill=(0, 0, 0, 160))
            draw.text((x, y), warning, font=self.font_title, fill=COLOR_WARN + (255,))
        if demo:
            state = str(demo.get("state") or "waiting").lower()
            text = str(demo.get("state_zh") or demo.get("state") or "")
            if text:
                font = self.font_demo if w >= 720 else self.font_title
                text_w = draw.textlength(text, font=font)
                x = max(10, int((w - text_w) / 2))
                y = max(10, h - 128)
                draw.rounded_rectangle(
                    [(x - 14, y - 10), (x + text_w + 14, y + 34)],
                    radius=12,
                    fill=(0, 0, 0, 185),
                )
                draw.text((x, y), text, font=font, fill=DEMO_COLORS_RGB.get(state, (230, 230, 230)) + (255,))
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
        zones: Optional[dict] = None,
        activity: Optional[dict] = None,
        demo: Optional[dict] = None,
    ) -> np.ndarray:
        if mirror is None:
            mirror = bool(self.config.get("ui.mirror_preview", True))
        canvas = cv2.flip(frame, 1) if mirror else frame.copy()
        zones_enabled = bool((zones or {}).get("enabled", True))
        if zones and zones_enabled:
            self._draw_zones(canvas, zones)
        if pose.present:
            self._draw_pose(canvas, pose, mirror)
        if hand_frames:
            self._draw_hands(canvas, hand_frames, hand_states, mirror)
        if hand_states and zones_enabled:
            self._draw_hand_zones(canvas, hand_states, mirror)

        warning = None
        if torso.posture == "no_person" and not hand_states:
            warning = "未检测到人物：请坐到摄像头前"
        elif torso.posture == "not_visible" and not hand_states:
            warning = "躯干关键点不足：请让上半身进入画面（必要时后退）"
        elif not hand_states:
            warning = "把手放进画面即可显示五指状态"
        # 演示版只关心“等待/正在切菜/切菜完成”。切菜中或完成后手离开画面是正常现象，
        # 不再弹旧提示，避免和底部状态标签互相矛盾。
        if demo and str(demo.get("state") or "") != "waiting":
            warning = None

        lines = self._hud_lines(torso, hand_states, perf, fps, extra_lines or [])
        return self._draw_hud(canvas, lines, warning, demo=demo)
