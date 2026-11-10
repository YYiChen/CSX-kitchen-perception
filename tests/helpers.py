"""测试公共工具：把 src/ 加入 sys.path，并提供合成姿态数据。"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _fill_landmarks(count: int, defaults: dict[int, tuple[float, float, float, float]]):
    from kitchen_perception.types import Landmark

    points = []
    for index in range(count):
        if index in defaults:
            x, y, z, vis = defaults[index]
            points.append(Landmark(x=x, y=y, z=z, visibility=vis, presence=vis))
        else:
            points.append(Landmark(x=0.5, y=0.5, z=0.0, visibility=0.0, presence=0.0))
    return points


def make_pose(
    up=(0.0, -0.42, 0.0),
    nose_dz: float = -0.10,
    shoulder_half_width: float = 0.15,
    hip_half_width: float = 0.10,
    visibility: float = 1.0,
    with_world: bool = True,
    image_points: dict | None = None,
):
    """合成一个 33 点姿态。

    up 是“髋中点 → 肩中点”的世界向量；nose_dz 是鼻尖相对肩中点的 z 偏移
    （负值表示朝向摄像头）。返回 PoseFrame。
    """
    from kitchen_perception.types import PoseFrame

    image_defaults = {
        0: (0.5, 0.20, 0.0, visibility),
        11: (0.5 + 0.15, 0.30, 0.0, visibility),
        12: (0.5 - 0.15, 0.30, 0.0, visibility),
        13: (0.5 + 0.25, 0.42, 0.0, visibility),
        14: (0.5 - 0.25, 0.42, 0.0, visibility),
        15: (0.5 + 0.30, 0.50, 0.0, visibility),
        16: (0.5 - 0.30, 0.50, 0.0, visibility),
        23: (0.5 + 0.10, 0.60, 0.0, visibility),
        24: (0.5 - 0.10, 0.60, 0.0, visibility),
        25: (0.5 + 0.10, 0.78, 0.0, visibility),
        26: (0.5 - 0.10, 0.78, 0.0, visibility),
        27: (0.5 + 0.10, 0.95, 0.0, visibility),
        28: (0.5 - 0.10, 0.95, 0.0, visibility),
    }
    if image_points:
        image_defaults.update(image_points)
    landmarks = _fill_landmarks(33, image_defaults)
    world = []
    if with_world:
        sx, sy, sz = up
        nose_z = sz + nose_dz
        # up 是"髋中点 → 肩中点"的完整世界向量，因此肩线整体沿 x 平移 sx，
        # 这样单侧倾斜（roll）和前后倾（pitch）都能由 up 独立控制。
        world_defaults = {
            0: (sx, sy - 0.15, nose_z, 1.0),
            11: (sx + shoulder_half_width, sy, sz, 1.0),
            12: (sx - shoulder_half_width, sy, sz, 1.0),
            23: (hip_half_width, 0.0, 0.0, 1.0),
            24: (-hip_half_width, 0.0, 0.0, 1.0),
        }
        world = _fill_landmarks(33, world_defaults)
    pose = PoseFrame(landmarks=landmarks, world=world, present=True, mean_visibility=visibility)
    return pose


def make_hand(states: str = "open"):
    """合成 21 点手部关键点。

    states: "open"（五指伸直）或 "fist"（四指弯曲、拇指内收）。
    坐标系是归一化图像坐标，y 向下。
    """
    from kitchen_perception.types import HandFrame, Landmark

    pts = {}
    # 腕与掌骨
    pts[0] = (0.50, 0.90)
    mcp_x = {"index": 0.46, "middle": 0.50, "ring": 0.54, "pinky": 0.58}
    pts[1] = (0.44, 0.84)  # thumb CMC
    pts[2] = (0.42, 0.80)  # thumb MCP
    for name, index in (("index", 5), ("middle", 9), ("ring", 13), ("pinky", 17)):
        pts[index] = (mcp_x[name], 0.72)

    if states == "open":
        chain = {"index": (5, 6, 7, 8), "middle": (9, 10, 11, 12), "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}
        for name, (mcp, pip, dip, tip) in chain.items():
            x = pts[mcp][0]
            pts[pip] = (x, 0.62)
            pts[dip] = (x, 0.55)
            pts[tip] = (x, 0.48)
        pts[3] = (0.36, 0.74)  # thumb IP
        pts[4] = (0.28, 0.70)  # thumb TIP（外展，离食指 MCP 远）
        # 让拇指方向沿掌侧向上一点，避免完全水平
        pts[3] = (0.35, 0.74)
        pts[4] = (0.26, 0.70)
    else:
        # 四指弯曲：PIP/DIP/TIP 折回掌心，PIP 内角很小
        chain = {"index": (5, 6, 7, 8), "middle": (9, 10, 11, 12), "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}
        for name, (mcp, pip, dip, tip) in chain.items():
            x = pts[mcp][0]
            pts[pip] = (x, 0.78)   # 向掌侧折
            pts[dip] = (x, 0.70)   # 折回
            pts[tip] = (x, 0.76)   # 指尖靠近掌心
        pts[3] = (0.44, 0.78)
        pts[4] = (0.47, 0.73)  # 拇指内收，靠近食指 MCP

    landmarks = []
    for index in range(21):
        x, y = pts.get(index, (0.5, 0.8))
        landmarks.append(Landmark(x=x, y=y, z=0.0, visibility=0.0, presence=0.0))
    return HandFrame(landmarks=landmarks, world=[], handedness="right", score=0.95)
