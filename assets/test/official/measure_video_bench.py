
"""VIDEO-mode steady-state benchmarks + GPU delegate probe. os._exit avoids telemetry shutdown stall."""
from __future__ import annotations
import json, os, statistics, time
import cv2, numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.abspath(os.path.join(HERE, "..", "..", "..", "models"))
OUT = os.path.join(HERE, "measurements_video.json")
PL = os.path.join(MODELS, "pose_landmarker_lite.task")
PF = os.path.join(MODELS, "pose_landmarker_full.task")
HM = os.path.join(MODELS, "hand_landmarker.task")


def log(*a):
    print(*a, flush=True)


def mbuf(p):
    with open(p, "rb") as f:
        return f.read()


def read_img(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)


def mpimg(bgr):
    return mp.Image(image_format=mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))


def mk_pose(path, mode, delegate=None):
    kw = {}
    if delegate is not None:
        kw["delegate"] = delegate
    return vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_buffer=mbuf(path), **kw),
        running_mode=mode, num_poses=1))


def mk_hand(mode):
    return vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_buffer=mbuf(HM)),
        running_mode=mode, num_hands=2))


def bench(fn, n=20, warmup=5):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter()-t0)*1000.0)
    s = sorted(ts)
    return {"n": n, "mean_ms": round(statistics.mean(ts),2), "median_ms": round(statistics.median(ts),2),
            "std_ms": round(statistics.stdev(ts),2), "p95_ms": round(s[int(round(0.95*(n-1)))],2),
            "min_ms": round(min(ts),2), "max_ms": round(max(ts),2)}


report = {"environment": {"mediapipe": mp.__version__, "opencv": cv2.__version__, "cpu_logical": os.cpu_count()},
          "video_mode": {}, "gpu_delegate": None}
for label, size in (("640x480", (640,480)), ("1280x720", (1280,720))):
    bgr = cv2.resize(read_img(os.path.join(HERE, "male_full_height_hands.jpg")), size)
    mi = mpimg(bgr)
    t0 = time.perf_counter(); vp = mk_pose(PL, vision.RunningMode.VIDEO); t_create = (time.perf_counter()-t0)*1000
    vf = mk_pose(PF, vision.RunningMode.VIDEO)
    vh = mk_hand(vision.RunningMode.VIDEO)
    log(f"[video-created] {label} pose_lite {t_create:.1f} ms")
    st = {"p": 0, "h": 0}

    def pose_call():
        st["p"] += 33
        return vp.detect_for_video(mi, st["p"])

    def hand_call():
        st["h"] += 33
        return vh.detect_for_video(mi, st["h"])

    e = {"input": label}
    e["pose_lite_video_tracking"] = bench(pose_call)
    log("[video-bench]", label, "pose_lite", json.dumps(e["pose_lite_video_tracking"]))
    e["hand_video_tracking"] = bench(hand_call)
    log("[video-bench]", label, "hand", json.dumps(e["hand_video_tracking"]))
    # fresh video pair for combined (each landmarker owns its last timestamp)
    vp2 = mk_pose(PL, vision.RunningMode.VIDEO)
    vh2 = mk_hand(vision.RunningMode.VIDEO)
    sc = {"p": 0, "h": 0}

    def combined():
        sc["p"] += 33; sc["h"] += 33
        return vp2.detect_for_video(mi, sc["p"]), vh2.detect_for_video(mi, sc["h"])

    e["pose_lite_plus_hand_video_tracking"] = bench(combined)
    log("[video-bench]", label, "combined", json.dumps(e["pose_lite_plus_hand_video_tracking"]))
    report["video_mode"][label] = e

# GPU delegate probe in same process (last, isolated try)
try:
    t0 = time.perf_counter()
    g = mk_pose(PL, vision.RunningMode.IMAGE, delegate=mp_tasks.BaseOptions.Delegate.GPU)
    t_create = (time.perf_counter()-t0)*1000
    mi = mpimg(read_img(os.path.join(HERE, "pose.jpg")))
    t0 = time.perf_counter()
    r = g.detect(mi)
    t_det = (time.perf_counter()-t0)*1000
    report["gpu_delegate"] = {"available": True, "create_ms": round(t_create,1),
                              "first_detect_ms": round(t_det,1), "num_poses": len(r.pose_landmarks)}
except Exception as e:
    report["gpu_delegate"] = {"available": False, "error": f"{type(e).__name__}: {e}"}
log("[gpu]", json.dumps(report["gpu_delegate"], ensure_ascii=False)[:400])

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
log("[save]", OUT)
os._exit(0)
