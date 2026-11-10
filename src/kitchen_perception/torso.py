"""躯干状态分析。

设计原则：状态定义“不过死”，任何环节想要不同口径都能改配置或插件覆盖：
- 阈值、中文标签全部来自 config/default.json；
- 分类规则是纯函数，方便单元测试与按环节替换；
- custom_rules.torso_state(features, config) 可以覆盖默认结果（返回 None 表示不覆盖）。

几何口径：
- 图像坐标 x 向右、y 向下、归一化到 [0,1]。
- world landmarks 单位约为米、原点在髋中心；y 向下。
- facing：通过鼻尖相对肩中点的 z 偏移判断朝向摄像头/背向/侧身。
- pitch > 0 表示上身向“脸朝向的方向”前倾；roll > 0 表示向图像右侧倾斜；
  yaw 表示肩线相对正面的旋转。

可见性口径（实测踩坑）：笔记本摄像头经常只拍到上半身，髋部在画面外但
MediaPipe 仍会给出合理的 world 髋位置。因此：
- 只要求双肩可见即可做世界坐标系姿态分析；
- 髋部不在画面内时仍可判 posture，但置信度乘惩罚系数并标注“估算”；
- 连 world 坐标都缺失、且髋部不可见时才判 not_visible。
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

from .config import Config
from .geometry import EMA, Hysteresis, clamp
from .types import PoseFrame, TorsoFeatures, TorsoState

NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26

KEY_POINTS = (NOSE, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_KNEE, R_KNEE)
TORSO_POINTS = (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)
SHOULDER_POINTS = (L_SHOULDER, R_SHOULDER)
HIP_POINTS = (L_HIP, R_HIP)

POSTURE_VALUES = (
    "no_person",
    "not_visible",
    "upright",
    "leaning_forward",
    "leaning_backward",
    "leaning_side",
    "bent_over",
    "uncertain",
)


def _lm(points: Sequence, index: int):
    return points[index] if index < len(points) else None


def _p3(points: Sequence, index: int) -> tuple[float, float, float] | None:
    lm = _lm(points, index)
    if lm is None:
        return None
    return (float(lm.x), float(lm.y), float(lm.z))


def _mid3(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0, (a[2] + b[2]) / 2.0)


def _sub(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _norm(v: Sequence[float]) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def angle3_deg(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float:
    """三维角 ABC（顶点 B）；退化返回 nan。"""
    v1 = _sub(a, b)
    v2 = _sub(c, b)
    n1, n2 = _norm(v1), _norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return float("nan")
    cos_value = (v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]) / (n1 * n2)
    return math.degrees(math.acos(clamp(cos_value, -1.0, 1.0)))


def _vis(lm) -> float:
    return clamp(min(float(lm.visibility), float(lm.presence)), 0.0, 1.0)


def bbox_from_landmarks(landmarks: Sequence, indices: Iterable[int] = KEY_POINTS, min_visibility: float = 0.3) -> list[float] | None:
    xs: list[float] = []
    ys: list[float] = []
    for idx in indices:
        lm = _lm(landmarks, idx)
        if lm is None or _vis(lm) < min_visibility:
            continue
        xs.append(lm.x)
        ys.append(lm.y)
    if len(xs) < 2:
        return None
    return [clamp(min(xs), 0.0, 1.0), clamp(min(ys), 0.0, 1.0), clamp(max(xs), 0.0, 1.0), clamp(max(ys), 0.0, 1.0)]


def hands_in_workspace(hand_bboxes: Iterable[Sequence[float] | None], torso_bbox: Sequence[float] | None, margin: float = 0.1) -> bool:
    """手的包围盒中点是否落在躯干包围盒（外扩 margin）内。"""
    if torso_bbox is None:
        return False
    x0, y0, x1, y1 = torso_bbox
    for bbox in hand_bboxes:
        if bbox is None:
            continue
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        if x0 - margin <= cx <= x1 + margin and y0 - margin <= cy <= y1 + margin:
            return True
    return False


def compute_features(
    pose: PoseFrame,
    width: int,
    height: int,
    aspect: Optional[float] = None,
    turn_yaw_deg: float = 30.0,
    front_z_threshold: float = 0.02,
    hip_vis_min: float = 0.3,
) -> TorsoFeatures:
    """从一帧姿态结果计算躯干几何特征（不做平滑/滞回）。"""
    features = TorsoFeatures()
    features.extra["shoulder_visible"] = False
    features.extra["hips_image_visible"] = False
    if not pose.present or len(pose.landmarks) <= R_SHOULDER:
        features.source = "none"
        return features

    landmarks = pose.landmarks
    aspect = float(aspect) if aspect else (float(width) / float(height) if height else 1.0)

    shoulder_vis = min(_vis(landmarks[i]) if _lm(landmarks, i) is not None else 0.0 for i in SHOULDER_POINTS)
    hip_vis = min(_vis(landmarks[i]) if _lm(landmarks, i) is not None else 0.0 for i in HIP_POINTS)
    features.visibility = shoulder_vis
    features.extra["shoulder_visible"] = shoulder_vis > 0.0
    features.extra["shoulder_visibility"] = shoulder_vis
    features.extra["hip_visibility"] = hip_vis
    features.extra["hips_image_visible"] = hip_vis >= hip_vis_min

    bbox = bbox_from_landmarks(landmarks)
    features.bbox_norm = bbox

    # 2D 肩宽/躯干长（像素），用于诊断与兜底
    lsh, rsh = landmarks[L_SHOULDER], landmarks[R_SHOULDER]
    shoulder_px = math.hypot((lsh.x - rsh.x) * width, (lsh.y - rsh.y) * height)
    features.shoulder_width_px = shoulder_px if shoulder_px > 1e-6 else float("nan")

    if hip_vis >= hip_vis_min and _lm(landmarks, R_HIP) is not None:
        lhip, rhip = landmarks[L_HIP], landmarks[R_HIP]
        sh_mid_2d = ((lsh.x + rsh.x) / 2.0, (lsh.y + rsh.y) / 2.0)
        hip_mid_2d = ((lhip.x + rhip.x) / 2.0, (lhip.y + rhip.y) / 2.0)
        torso_len_px = math.hypot((sh_mid_2d[0] - hip_mid_2d[0]) * width, (sh_mid_2d[1] - hip_mid_2d[1]) * height)
        features.torso_length_px = torso_len_px if torso_len_px > 1e-6 else float("nan")
        dx = (sh_mid_2d[0] - hip_mid_2d[0]) * width
        dy = (sh_mid_2d[1] - hip_mid_2d[1]) * height
        features.inclination_2d_deg = math.degrees(math.atan2(dx, -dy)) if abs(dx) + abs(dy) > 1e-6 else float("nan")

    world = pose.world
    ls, rs = _p3(world, L_SHOULDER), _p3(world, R_SHOULDER)
    lh, rh = _p3(world, L_HIP), _p3(world, R_HIP)
    nose = _p3(world, NOSE)

    if not all((ls, rs, lh, rh)):
        # world 缺失：只能用 2D 侧倾兜底，且要求髋部可见
        if hip_vis >= hip_vis_min and not math.isnan(features.inclination_2d_deg):
            features.source = "2d"
            features.roll_deg = features.inclination_2d_deg
            features.facing = "unknown"
        else:
            features.source = "none"
        return features

    shoulder_mid = _mid3(ls, rs)
    hip_mid = _mid3(lh, rh)
    up = _sub(shoulder_mid, hip_mid)  # 髋 → 肩，人体“向上”方向
    if _norm(up) < 1e-6:
        features.source = "none"
        return features

    nose_dz = (nose[2] - shoulder_mid[2]) if nose is not None else 0.0
    front = 1.0 if nose_dz > 0 else -1.0

    features.pitch_deg = math.degrees(math.atan2(up[2] * front, -up[1]))
    features.roll_deg = math.degrees(math.atan2(up[0], -up[1]))
    shoulder_vec = _sub(rs, ls)
    features.yaw_deg = math.degrees(
        math.atan2(shoulder_vec[2], abs(shoulder_vec[0]) if abs(shoulder_vec[0]) > 1e-6 else 1e-6)
    )
    if abs(features.yaw_deg) > float(turn_yaw_deg):
        features.facing = "sideways"
    elif nose_dz < -float(front_z_threshold):
        features.facing = "toward_camera"
    elif nose_dz > float(front_z_threshold):
        features.facing = "away_from_camera"
    else:
        features.facing = "unknown"

    shoulder_3d = _norm(shoulder_vec)
    torso_3d = _norm(up)
    if shoulder_3d > 1e-6 and torso_3d > 1e-6 and shoulder_px > 1e-6 and features.torso_length_px == features.torso_length_px:
        ratio_3d = torso_3d / shoulder_3d
        ratio_2d = features.torso_length_px / shoulder_px
        features.foreshorten_ratio = ratio_2d / ratio_3d if ratio_3d > 1e-6 else float("nan")

    features.source = "world"
    features.extra["nose_dz"] = nose_dz
    features.extra["front"] = front
    return features


def _margin_to_thresholds(pitch: float, roll: float, th: dict) -> float:
    boundaries: list[float] = []
    if not math.isnan(pitch):
        boundaries.extend(
            [abs(pitch - th["upright_pitch_deg"]), abs(pitch - th["backward_pitch_deg"]), abs(pitch - th["bent_pitch_deg"])]
        )
    if not math.isnan(roll):
        boundaries.append(abs(abs(roll) - th["side_roll_deg"]))
    return min(boundaries) if boundaries else 0.0


def classify(features: TorsoFeatures, config: Config) -> tuple[str, float, str]:
    """纯规则分类，返回 (posture, confidence, note)。"""
    th = config.section("torso")["thresholds"]
    if not features.extra.get("shoulder_visible"):
        return "no_person", 0.0, ""

    vis_min = float(config.get("torso.visibility_min", 0.5))
    base_vis = float(features.visibility)
    if base_vis < vis_min:
        return "not_visible", clamp(base_vis, 0.0, 1.0), "双肩关键点可见度不足"

    hips_penalty = 1.0 if features.extra.get("hips_image_visible") else float(config.get("torso.hip_visibility_penalty", 0.85))
    note = "" if features.extra.get("hips_image_visible") else "髋部不在画面内，姿态为估算"

    pitch = features.pitch_deg
    roll = features.roll_deg
    upright_pitch = float(th["upright_pitch_deg"])
    backward_pitch = float(th["backward_pitch_deg"])
    bent_pitch = float(th["bent_pitch_deg"])
    side_roll = float(th["side_roll_deg"])

    if features.source == "2d":
        if not math.isnan(roll) and abs(roll) > side_roll:
            posture = "leaning_side"
            note = "仅 2D 特征"
        else:
            posture = "upright"
            note = "仅 2D 特征，pitch 不可用"
        margin = _margin_to_thresholds(float("nan"), roll, th)
        conf = clamp(base_vis * (0.45 + 0.25 * clamp(margin / 20.0, 0.0, 1.0)) * hips_penalty, 0.0, 1.0)
        return posture, conf, note

    if not math.isnan(pitch) and pitch >= bent_pitch:
        posture = "bent_over"
    elif not math.isnan(pitch) and pitch >= upright_pitch:
        posture = "leaning_forward"
    elif not math.isnan(pitch) and pitch <= backward_pitch:
        posture = "leaning_backward"
    elif not math.isnan(roll) and abs(roll) > side_roll:
        posture = "leaning_side"
    else:
        posture = "upright"

    if math.isnan(pitch) or math.isnan(roll):
        return "uncertain", clamp(base_vis * 0.4, 0.0, 1.0), "角度不可用"

    margin = _margin_to_thresholds(pitch, roll, th)
    conf = clamp(base_vis * (0.6 + 0.75 * clamp(margin / 20.0, 0.0, 1.0)) * hips_penalty, 0.0, 1.0)
    return posture, conf, note


class TorsoAnalyzer:
    """带平滑与滞回的躯干状态分析器（有状态，按视频流顺序调用）。

    支持“直立基准”标定：摄像头角度导致整体 pitch 偏移时，点一次“标定”把
    当前姿态记为 0°，后续角度都相对该基准计算。
    """

    def __init__(self, config: Config, offsets: Optional[dict] = None) -> None:
        self.config = config
        self.vis_min = float(config.get("torso.visibility_min", 0.5))
        alpha = float(config.get("torso.ema_alpha", 0.3))
        self.ema_pitch = EMA(alpha) if alpha < 1.0 else None
        self.ema_roll = EMA(alpha) if alpha < 1.0 else None
        self.ema_yaw = EMA(alpha) if alpha < 1.0 else None
        self.hyst = Hysteresis(int(config.get("torso.stable_frames", 5)))
        self.last: TorsoState | None = None
        self.frame_count = 0
        self.offsets = {"pitch": 0.0, "roll": 0.0, "yaw": 0.0}
        if offsets:
            for key in self.offsets:
                try:
                    self.offsets[key] = float(offsets.get(key, 0.0))
                except (TypeError, ValueError):
                    pass

    def reset(self) -> None:
        for ema in (self.ema_pitch, self.ema_roll, self.ema_yaw):
            if ema is not None:
                ema.reset()
        self.hyst.reset()
        self.last = None
        self.frame_count = 0

    def calibrate(self) -> dict:
        """把当前平滑后的角度记为 0° 基准，返回偏移量。"""
        if self.last is not None:
            f = self.last.features
            if not math.isnan(f.pitch_deg):
                self.offsets["pitch"] = f.pitch_deg
            if not math.isnan(f.roll_deg):
                self.offsets["roll"] = f.roll_deg
            if not math.isnan(f.yaw_deg):
                self.offsets["yaw"] = f.yaw_deg
        return dict(self.offsets)

    def clear_calibration(self) -> None:
        self.offsets = {"pitch": 0.0, "roll": 0.0, "yaw": 0.0}

    def analyze(self, pose: PoseFrame, width: int, height: int, ts_ms: float = 0.0) -> TorsoState:
        self.frame_count += 1
        labels = self.config.section("torso").get("labels_zh", {})
        th = self.config.section("torso")["thresholds"]

        if not pose.present or len(pose.landmarks) <= R_SHOULDER:
            state = TorsoState(present=False, posture="no_person", posture_zh=labels.get("no_person", "未检测到人物"), confidence=1.0)
            self.last = state
            return state

        features = compute_features(
            pose,
            width,
            height,
            turn_yaw_deg=float(th.get("turn_yaw_deg", 30.0)),
            front_z_threshold=float(self.config.get("torso.front_z_threshold", 0.02)),
        )

        if features.source == "none":
            state = TorsoState(
                present=True,
                posture="not_visible",
                posture_zh=labels.get("not_visible", "躯干不可见"),
                confidence=clamp(features.visibility, 0.0, 1.0),
                features=features,
                note="未检测到可用的躯干关键点（请让上半身进入画面）",
            )
            self.last = state
            self.hyst.reset()
            return state

        # EMA 平滑
        if self.ema_pitch is not None:
            value = self.ema_pitch.update(features.pitch_deg)
            if value is not None:
                features.pitch_deg = value
        if self.ema_roll is not None:
            value = self.ema_roll.update(features.roll_deg)
            if value is not None:
                features.roll_deg = value
        if self.ema_yaw is not None:
            value = self.ema_yaw.update(features.yaw_deg)
            if value is not None:
                features.yaw_deg = value

        # 直立基准
        if self.offsets["pitch"] or self.offsets["roll"] or self.offsets["yaw"]:
            if not math.isnan(features.pitch_deg):
                features.pitch_deg -= self.offsets["pitch"]
            if not math.isnan(features.roll_deg):
                features.roll_deg -= self.offsets["roll"]
            if not math.isnan(features.yaw_deg):
                features.yaw_deg -= self.offsets["yaw"]

        posture, confidence, note = classify(features, self.config)

        # 扩展点：允许外部模块按炒菜环节覆盖结论
        try:
            from . import custom_rules

            override = custom_rules.apply_torso_rule(features.to_dict(), self.config)
        except Exception:  # 插件错误不能影响主流程
            override = None
        if isinstance(override, dict):
            posture = str(override.get("posture", posture))
            if posture not in POSTURE_VALUES:
                posture = "uncertain"
            confidence = clamp(float(override.get("confidence", confidence)), 0.0, 1.0)
            note = str(override.get("note", note))

        stable, stable_frames = self.hyst.update(posture)
        posture = stable or posture
        state = TorsoState(
            present=True,
            posture=posture,
            posture_zh=labels.get(posture, posture),
            confidence=confidence,
            stable_frames=int(stable_frames),
            features=features,
            note=note,
        )
        self.last = state
        return state
