"""手部五指状态分析（规则化，不训练）。

MediaPipe Hand 21 点：0 腕，1-4 拇指（CMC/MCP/IP/TIP），5-8 食指，9-12 中指，
13-16 无名指，17-20 小指；每指依次 MCP/PIP/DIP/TIP。

指标口径：
- pip_deg：PIP 关节内角（180 = 完全伸直），拇指用 IP 内角；
- dip_deg：DIP 关节内角，拇指用 MCP 内角；
- curl_deg = (180 - pip_deg) + (180 - dip_deg)，0 = 伸直；
- dist_ratio：拇指尖到食指尖/掌长，用于区分“拇指外展/内收”。
所有阈值都在 config.hands.finger_thresholds 中，方便按人/按环节标定。
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

from .config import Config
from .geometry import angle_deg, clamp, distance
from .types import FingerMetrics, HandFrame, HandState, Landmark

FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")
FINGER_ZH = {"thumb": "拇指", "index": "食指", "middle": "中指", "ring": "无名指", "pinky": "小指"}

# 每根手指：(mcp, pip_or_ip, dip_or_tip) —— 拇指 2/3/4，其余 MCP/PIP/DIP
FINGER_JOINTS = {
    "thumb": (2, 3, 4),
    "index": (5, 6, 7),
    "middle": (9, 10, 11),
    "ring": (13, 14, 15),
    "pinky": (17, 18, 19),
}
FINGERTIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
WRIST = 0
MIDDLE_MCP = 9
INDEX_MCP = 5
THUMB_TIP = 4


def _pt(lm: Landmark, aspect: float) -> tuple[float, float]:
    return (lm.x * aspect, lm.y)


def _dist_ratio(a: Landmark, b: Landmark, palm_len: float, aspect: float) -> float:
    if palm_len <= 1e-9:
        return float("nan")
    return distance(_pt(a, aspect), _pt(b, aspect)) / palm_len


def bbox_from_hand(hand: HandFrame) -> list[float] | None:
    if not hand.landmarks:
        return None
    xs = [lm.x for lm in hand.landmarks]
    ys = [lm.y for lm in hand.landmarks]
    return [clamp(min(xs), 0.0, 1.0), clamp(min(ys), 0.0, 1.0), clamp(max(xs), 0.0, 1.0), clamp(max(ys), 0.0, 1.0)]


def analyze_fingers(hand: HandFrame, config: Config, aspect: float = 1.0) -> tuple[dict[str, str], dict[str, FingerMetrics], dict]:
    """返回 (每指标签, 每指指标, 附加信息)。"""
    thresholds = config.section("hands")["finger_thresholds"]
    min_vis = 0.4
    states: dict[str, str] = {}
    metrics: dict[str, FingerMetrics] = {}
    if len(hand.landmarks) < 21:
        return {finger: "unknown" for finger in FINGER_ORDER}, {}, {"valid": False, "reason": "landmarks<21"}

    marks = hand.landmarks
    palm_len = distance(_pt(marks[WRIST], aspect), _pt(marks[MIDDLE_MCP], aspect))
    if palm_len <= 1e-6:
        palm_len = 1e-6

    for finger in FINGER_ORDER:
        mcp_i, pip_i, dip_i = FINGER_JOINTS[finger]
        tip_i = FINGERTIPS[finger]
        needed = [mcp_i, pip_i, dip_i]
        # 注意：MediaPipe Hand 的 landmark visibility/presence 常常恒为 0（不提供），
        # 只有明确给出 >0 的值时才当作可见度不足处理。
        vis_ok = True
        for i in needed:
            v = min(marks[i].visibility, marks[i].presence)
            if v > 1e-6 and v < min_vis:
                vis_ok = False
                break
        metric = FingerMetrics(finger=finger)
        if not vis_ok:
            metric.note = "关键点可见度不足"
            metrics[finger] = metric
            states[finger] = "unknown"
            continue

        pts = [_pt(marks[i], aspect) for i in needed]
        joint_pip = angle_deg(pts[0], pts[1], pts[2])
        # 指尖相对 DIP 的弯曲：没有 TIP 的指节（拇指）使用 4 点，这里用 TIP 补一个 DIP 弯曲
        tip_pt = _pt(marks[tip_i], aspect) if tip_i < len(marks) else None
        dip_pt = pts[2]
        joint_dip = angle_deg(pts[1], dip_pt, tip_pt) if tip_pt is not None else float("nan")
        if math.isnan(joint_dip):
            joint_dip = angle_deg(pts[0], pts[1], pts[2])
        metric.pip_deg = joint_pip
        metric.dip_deg = joint_dip
        metric.curl_deg = (180.0 - joint_pip) + (180.0 - joint_dip)
        metric.valid = not (math.isnan(metric.curl_deg) or math.isnan(metric.pip_deg))

        if finger == "thumb":
            tip = marks[THUMB_TIP]
            metric.dist_ratio = _dist_ratio(tip, marks[INDEX_MCP], palm_len, aspect)
            metric.extra["pinch_ratio"] = _dist_ratio(tip, marks[FINGERTIPS["index"]], palm_len, aspect)
            ext = (
                metric.pip_deg >= float(thresholds["thumb_extended_ip_min_deg"])
                and metric.dist_ratio >= float(thresholds["thumb_extended_dist_ratio"])
            )
            flex = (
                metric.pip_deg <= float(thresholds["thumb_flexed_ip_max_deg"])
                or metric.dist_ratio <= float(thresholds["thumb_flexed_dist_ratio"])
            )
            if ext:
                states[finger] = "extended"
            elif flex:
                states[finger] = "flexed"
            else:
                states[finger] = "partial"
        else:
            if metric.curl_deg <= float(thresholds["extended_curl_max_deg"]) and metric.pip_deg >= float(thresholds["extended_pip_min_deg"]):
                states[finger] = "extended"
            elif metric.curl_deg >= float(thresholds["flexed_curl_min_deg"]) or metric.pip_deg <= float(thresholds["flexed_pip_max_deg"]):
                states[finger] = "flexed"
            else:
                states[finger] = "partial"
        metrics[finger] = metric

    info = {
        "palm_len_norm": palm_len,
        "valid_count": sum(1 for s in states.values() if s != "unknown"),
    }
    return states, metrics, info


def classify_grip(states: dict[str, str], metrics: dict[str, FingerMetrics], config: Config) -> tuple[str, float, str]:
    """规则化手势分类，返回 (grip, confidence, note)。"""
    thresholds = config.section("hands")["finger_thresholds"]
    known = [s for s in states.values() if s != "unknown"]
    if len(known) < 3:
        return "unknown", 0.0, "有效手指不足"

    ext = [f for f in ("index", "middle", "ring", "pinky") if states.get(f) == "extended"]
    flex = [f for f in ("index", "middle", "ring", "pinky") if states.get(f) == "flexed"]
    thumb = states.get("thumb", "unknown")

    # 捏取：拇指尖与食指尖靠得很近（用掌长归一化，捏合时通常 < 0.4）
    thumb_metric = metrics.get("thumb")
    if thumb_metric is not None and thumb_metric.valid:
        pinch_ratio = thumb_metric.extra.get("pinch_ratio")
        index_not_flexed = states.get("index") in ("extended", "partial")
        if pinch_ratio is not None and pinch_ratio <= float(thresholds["pinch_dist_ratio"]) and index_not_flexed:
            return "pinch", 0.8, "拇指-食指靠近"

    if len(ext) == 4 and thumb == "extended":
        return "open", 0.9, ""
    if len(flex) == 4 and thumb in ("flexed", "unknown"):
        return "fist", 0.85, ""
    if states.get("index") == "extended" and len(flex) >= 3:
        return "point", 0.8, "仅食指伸直"
    if len(ext) >= 2:
        return "partial", 0.6, "多指半伸"
    if len(flex) >= 2:
        return "partial", 0.6, "多指半屈"
    return "unknown", 0.3, ""


class HandAnalyzer:
    """把 HandFrame 序列转成 HandState；带手别纠正与简单速度跟踪。"""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.flip_handedness = bool(config.get("hands.handedness_flip", False))
        self.min_score = float(config.get("hands.min_hand_score", 0.5))

    def analyze(self, hand: HandFrame, aspect: float = 1.0) -> HandState:
        labels_zh = self.config.section("hands").get("labels_zh", {})
        grip_zh = self.config.section("hands").get("grip_labels_zh", {})
        state = HandState()
        state.score = float(hand.score)
        label = hand.handedness.lower() if hand.handedness else "unknown"
        if self.flip_handedness and label in ("left", "right"):
            label = "right" if label == "left" else "left"
        state.label = label
        state.label_source = "model"
        state.bbox_norm = bbox_from_hand(hand)
        if not hand.landmarks:
            state.fingers = {finger: "unknown" for finger in FINGER_ORDER}
            state.grip = "unknown"
            state.grip_zh = grip_zh.get("unknown", "未知")
            state.notes.append("无手部关键点")
            return state

        states, metrics, info = analyze_fingers(hand, self.config, aspect=aspect)
        grip, confidence, note = classify_grip(states, metrics, self.config)

        try:
            from . import custom_rules

            override = custom_rules.apply_grip_rule(states, metrics, self.config)
        except Exception:
            override = None
        if isinstance(override, str) and override:
            grip = override
            confidence = max(confidence, 0.5)
        elif isinstance(override, dict):
            grip = str(override.get("grip", grip))
            confidence = clamp(float(override.get("confidence", confidence)), 0.0, 1.0)
            note = str(override.get("note", note))

        state.fingers = states
        state.finger_metrics = metrics
        state.grip = grip
        state.grip_zh = grip_zh.get(grip, grip)
        state.score = max(state.score, clamp(confidence, 0.0, 1.0))
        if note:
            state.notes.append(note)
        if hand.landmarks:
            state.wrist_norm = [float(hand.landmarks[WRIST].x), float(hand.landmarks[WRIST].y)]
        if info.get("valid_count", 0) < 5:
            state.notes.append(f"有效手指 {info.get('valid_count', 0)}/5")
        return state