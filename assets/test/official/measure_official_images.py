
"""Measure official MediaPipe test images with MediaPipe Tasks 0.10.35.

Read-only w.r.t. src/models. Writes only into assets/test/official/.
Run: py -3.12 assets/test/official/measure_official_images.py
"""
from __future__ import annotations

import json
import math
import os
import statistics
import time

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
MODELS = os.path.join(PROJ, "models")
IMAGES = [
    "pose.jpg",
    "male_full_height_hands.jpg",
    "right_hands.jpg",
    "left_hands.jpg",
    "fist.png",
    "woman_hands.jpg",
]

POSE_MODEL_LITE = os.path.join(MODELS, "pose_landmarker_lite.task")
POSE_MODEL_FULL = os.path.join(MODELS, "pose_landmarker_full.task")
HAND_MODEL = os.path.join(MODELS, "hand_landmarker.task")


def read_model_buffer(path):
    with open(path, "rb") as f:
        return f.read()


def read_image_bgr(path):
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("imdecode failed: " + path)
    return img


def to_mp_image(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))


def create_pose(model_path, running_mode):
    opts = vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_buffer=read_model_buffer(model_path)),
        running_mode=running_mode,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    return vision.PoseLandmarker.create_from_options(opts)


def create_hand(running_mode):
    opts = vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_buffer=read_model_buffer(HAND_MODEL)),
        running_mode=running_mode,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(opts)


def interior_angle(a, b, c, dims=3):
    """Interior angle at b (degrees), vectors a-b and c-b."""
    if dims == 3:
        v1 = np.array([a.x - b.x, a.y - b.y, a.z - b.z], dtype=np.float64)
        v2 = np.array([c.x - b.x, c.y - b.y, c.z - b.z], dtype=np.float64)
    else:
        v1 = np.array([a.x - b.x, a.y - b.y], dtype=np.float64)
        v2 = np.array([c.x - b.x, c.y - b.y], dtype=np.float64)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return float("nan")
    cos = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return math.degrees(math.acos(cos))


# Hand landmark index chains: thumb 1-2-3-4, index 5-6-7-8, middle 9-10-11-12,
# ring 13-14-15-16, pinky 17-18-19-20.
FINGERS = {
    "thumb": {"flex": [("mcp", 1, 2, 3), ("ip", 2, 3, 4)]},
    "index": {"flex": [("pip", 5, 6, 7), ("dip", 6, 7, 8)]},
    "middle": {"flex": [("pip", 9, 10, 11), ("dip", 10, 11, 12)]},
    "ring": {"flex": [("pip", 13, 14, 15), ("dip", 14, 15, 16)]},
    "pinky": {"flex": [("pip", 17, 18, 19), ("dip", 18, 19, 20)]},
}


def hand_curls(lms, dims=None):
    """Per-finger total flexion (deg) = sum(180 - interior joint angle).

    Uses hand world landmarks (3D, meters) by default; 2D fallback on request.
    Straight extended finger ~ 0-20 deg total; closed fist ~ 150-270 deg.
    """
    out = {}
    for name, spec in FINGERS.items():
        joints = {}
        total = 0.0
        for jname, a, b, c in spec["flex"]:
            ang = interior_angle(lms[a], lms[b], lms[c], dims=3 if dims is None else dims)
            joints[jname + "_interior"] = round(ang, 1) if not math.isnan(ang) else None
            if not math.isnan(ang):
                total += 180.0 - ang
        out[name] = {
            "total_flexion_deg": round(total, 1),
            "joints": joints,
        }
    return out


def classify_finger(total):
    if total is None:
        return "unknown"
    if total < 30.0:
        return "extended"
    if total <= 110.0:
        return "partial"
    return "curled"


def summarize_hand(res, hand_idx):
    lm_norm = res.hand_landmarks[hand_idx]
    lm_world = res.hand_world_landmarks[hand_idx] if res.hand_world_landmarks else lm_norm
    hd = res.handedness[hand_idx][0] if res.handedness else None
    curls_world = hand_curls(lm_world, dims=3)
    curls_norm = hand_curls(lm_norm, dims=2)
    fingers = {}
    for name, c in curls_world.items():
        fingers[name] = {
            "total_flexion_deg": c["total_flexion_deg"],
            "prj2d_total_flexion_deg": curls_norm[name]["total_flexion_deg"],
            "joints_world": c["joints"],
            "state_rule": classify_finger(c["total_flexion_deg"]),
        }
    return {
        "handedness_label": hd.category_name if hd else None,
        "handedness_score": round(hd.score, 4) if hd else None,
        "landmark_count": len(lm_norm),
        "world_origin_note": "world landmarks: meters, origin near hand geometric center",
        "sample_world_points": {
            "wrist_0": [round(lm_world[0].x, 4), round(lm_world[0].y, 4), round(lm_world[0].z, 4)],
            "middle_mcp_9": [round(lm_world[9].x, 4), round(lm_world[9].y, 4), round(lm_world[9].z, 4)],
            "middle_tip_12": [round(lm_world[12].x, 4), round(lm_world[12].y, 4), round(lm_world[12].z, 4)],
        },
        "fingers": fingers,
        "open_finger_count_rule": sum(
            1 for f in fingers.values() if f["state_rule"] == "extended"
        ),
    }


def summarize_pose(res, model_name):
    if not res.pose_landmarks:
        return {"num_poses": 0, "pose_model": os.path.basename(model_name)}
    lm = res.pose_landmarks[0]
    world = res.pose_world_landmarks[0] if res.pose_world_landmarks else []
    sample_norm = {i: [round(lm[i].x, 4), round(lm[i].y, 4), round(lm[i].z, 4)] for i in (0, 11, 12, 23, 24)}
    sample_world = {}
    if world:
        sample_world = {
            str(i): [round(world[i].x, 4), round(world[i].y, 4), round(world[i].z, 4)]
            for i in (0, 11, 12, 23, 24)
        }
    nose_z_minus_shoulder_z = None
    if world:
        shoulder_z = 0.5 * (world[11].z + world[12].z)
        nose_z_minus_shoulder_z = round(world[0].z - shoulder_z, 4)
    return {
        "pose_model": os.path.basename(model_name),
        "num_poses": len(res.pose_landmarks),
        "landmark_count": len(lm),
        "normalized_xy_sample": sample_norm,
        "world_xyz_sample_m": sample_world,
        "nose_z_minus_shoulder_mid_z_m": nose_z_minus_shoulder_z,
        "world_y_shoulder_minus_hip_m": (
            round(0.5 * (world[11].y + world[12].y) - 0.5 * (world[23].y + world[24].y), 4)
            if world else None
        ),
    }


def bench(fn, n=20, warmup=3):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts_sorted = sorted(ts)
    return {
        "n": n,
        "mean_ms": round(statistics.mean(ts), 2),
        "median_ms": round(statistics.median(ts), 2),
        "mean_plus_std_ms": round(statistics.mean(ts) + statistics.stdev(ts), 2) if n > 1 else None,
        "std_ms": round(statistics.stdev(ts), 2) if n > 1 else None,
        "p95_ms": round(ts_sorted[min(n - 1, int(round(0.95 * (n - 1))))], 2),
        "min_ms": round(min(ts), 2),
        "max_ms": round(max(ts), 2),
    }


def main():
    report = {
        "environment": {
            "python": os.sys.version.split()[0],
            "mediapipe": mp.__version__,
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "cpu_logical_count": os.cpu_count(),
            "platform": os.sys.platform,
        },
        "models": {},
        "legacy_solutions": None,
        "images": {},
        "video_mode_timestamp_test": None,
        "benchmarks": {},
        "gpu_delegate_test": None,
    }
    for p in (POSE_MODEL_LITE, POSE_MODEL_FULL, HAND_MODEL):
        report["models"][os.path.basename(p)] = {
            "bytes": os.path.getsize(p),
            "sha256_short": None,
        }
    import hashlib
    for p in (POSE_MODEL_LITE, POSE_MODEL_FULL, HAND_MODEL):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        report["models"][os.path.basename(p)]["sha256"] = h.hexdigest()

    # Legacy API availability (mp.solutions in 0.10.35?)
    try:
        _ = mp.solutions
        report["legacy_solutions"] = {"available": True}
    except Exception as e:
        report["legacy_solutions"] = {
            "available": False,
            "error": f"{type(e).__name__}: {e}",
            "has_attr": hasattr(mp, "solutions"),
        }

    pose_lite = create_pose(POSE_MODEL_LITE, vision.RunningMode.IMAGE)
    pose_full = create_pose(POSE_MODEL_FULL, vision.RunningMode.IMAGE)
    hand = create_hand(vision.RunningMode.IMAGE)

    for name in IMAGES:
        path = os.path.join(HERE, name)
        bgr = read_image_bgr(path)
        h, w = bgr.shape[:2]
        mp_img = to_mp_image(bgr)
        t0 = time.perf_counter()
        pose_lite_res = pose_lite.detect(mp_img)
        t_pose_lite = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        pose_full_res = pose_full.detect(mp_img)
        t_pose_full = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        hand_res = hand.detect(mp_img)
        t_hand = (time.perf_counter() - t0) * 1000
        entry = {
            "file": name,
            "bytes": os.path.getsize(path),
            "width": w,
            "height": h,
            "channels": bgr.shape[2] if bgr.ndim == 3 else 1,
            "pose_lite": summarize_pose(pose_lite_res, POSE_MODEL_LITE),
            "pose_full": summarize_pose(pose_full_res, POSE_MODEL_FULL),
            "first_run_ms": {
                "pose_lite": round(t_pose_lite, 1),
                "pose_full": round(t_pose_full, 1),
                "hand": round(t_hand, 1),
            },
            "hands": [
                summarize_hand(hand_res, i) for i in range(len(hand_res.hand_landmarks))
            ],
        }
        entry["pose_lite"]["detected"] = entry["pose_lite"]["num_poses"] > 0
        entry["pose_full"]["detected"] = entry["pose_full"]["num_poses"] > 0
        entry["num_hands"] = len(hand_res.hand_landmarks)
        report["images"][name] = entry
        print(f"[image] {name} {w}x{h} pose_lite={entry['pose_lite']['num_poses']} "
              f"pose_full={entry['pose_full']['num_poses']} hands={entry['num_hands']}")

    # VIDEO mode timestamp constraint.
    video_pose = create_pose(POSE_MODEL_LITE, vision.RunningMode.VIDEO)
    bgr = read_image_bgr(os.path.join(HERE, "pose.jpg"))
    mp_img = to_mp_image(bgr)
    vt = {"frames_ok": [], "violations": []}

    def try_frame(ts):
        try:
            res = video_pose.detect_for_video(mp_img, ts)
            vt["frames_ok"].append({"timestamp_ms": ts, "num_poses": len(res.pose_landmarks)})
            return True
        except Exception as e:  # noqa: BLE001 - recording exact failure text
            vt["violations"].append(
                {"timestamp_ms": ts, "error": f"{type(e).__name__}: {e}"}
            )
            return False

    try_frame(0)
    try_frame(33)
    try_frame(33)   # duplicate timestamp
    try_frame(10)   # decreasing timestamp
    try_frame(66)
    video_pose.close()
    report["video_mode_timestamp_test"] = vt
    print("[video] timestamps:", json.dumps(vt, ensure_ascii=False)[:600])

    # Benchmarks: 640x480 and 1280x720, IMAGE (detector every frame) and VIDEO (tracking).
    bench_report = {}
    for label, size in (("640x480", (640, 480)), ("1280x720", (1280, 720))):
        bgr = cv2.resize(read_image_bgr(os.path.join(HERE, "male_full_height_hands.jpg")), size)
        mp_img = to_mp_image(bgr)
        entry = {}
        entry["pose_lite_image_mode"] = bench(lambda: pose_lite.detect(mp_img))
        entry["pose_full_image_mode"] = bench(lambda: pose_full.detect(mp_img))
        entry["hand_image_mode"] = bench(lambda: hand.detect(mp_img))

        vp = create_pose(POSE_MODEL_LITE, vision.RunningMode.VIDEO)
        vh = create_hand(vision.RunningMode.VIDEO)
        state = {"ts": 0}

        def pose_video_call():
            state["ts"] += 33
            return vp.detect_for_video(mp_img, state["ts"])

        def hand_video_call():
            state_hand = state.get("hand_ts", 0) + 33
            state["hand_ts"] = state_hand
            return vh.detect_for_video(mp_img, state_hand)

        entry["pose_lite_video_tracking"] = bench(pose_video_call)
        entry["hand_video_tracking"] = bench(hand_video_call)

        # Fresh pair for the combined pipeline: each VIDEO landmarker keeps its own
        # last timestamp, and reusing vp/vh from the benches above would violate it.
        vp2 = create_pose(POSE_MODEL_LITE, vision.RunningMode.VIDEO)
        vh2 = create_hand(vision.RunningMode.VIDEO)
        state_combined = {"ts": 0, "hand_ts": 0}

        def combined():
            state_combined["ts"] += 33
            state_combined["hand_ts"] += 33
            r1 = vp2.detect_for_video(mp_img, state_combined["ts"])
            r2 = vh2.detect_for_video(mp_img, state_combined["hand_ts"])
            return r1, r2

        entry["pose_lite_plus_hand_video_tracking"] = bench(combined)
        vp.close()
        vh.close()
        vp2.close()
        vh2.close()
        bench_report[label] = entry
        print(f"[bench] {label} done")
    report["benchmarks"] = bench_report

    out = os.path.join(HERE, "measurements.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("[checkpoint] wrote", out)

    # GPU delegate availability (Windows Python Tasks).
    try:
        gpu_opts = vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(
                model_asset_buffer=read_model_buffer(POSE_MODEL_LITE),
                delegate=mp_tasks.BaseOptions.Delegate.GPU,
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
        )
        gpu_lm = vision.PoseLandmarker.create_from_options(gpu_opts)
        bgr = read_image_bgr(os.path.join(HERE, "pose.jpg"))
        r = gpu_lm.detect(to_mp_image(bgr))
        report["gpu_delegate_test"] = {
            "available": True,
            "num_poses": len(r.pose_landmarks),
        }
        gpu_lm.close()
    except Exception as e:  # noqa: BLE001
        report["gpu_delegate_test"] = {
            "available": False,
            "error": f"{type(e).__name__}: {e}",
        }
    print("[gpu]", json.dumps(report["gpu_delegate_test"], ensure_ascii=False)[:500])

    pose_lite.close()
    pose_full.close()
    hand.close()

    out = os.path.join(HERE, "measurements.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("[done] wrote", out)


if __name__ == "__main__":
    main()
