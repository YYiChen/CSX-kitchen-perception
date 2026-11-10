
"""v2 staged measurement: images + video timestamp + IMAGE-mode benchmarks.

Avoids Task.close()/normal interpreter shutdown because MediaPipe 0.10.35
Clearcut telemetry blocks ~40 s per landmarker on this restricted network.
Writes JSON, then os._exit(0).
"""
from __future__ import annotations
import json, math, os, statistics, time
import cv2, numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
MODELS = os.path.join(PROJ, "models")
IMAGES = ["pose.jpg", "male_full_height_hands.jpg", "right_hands.jpg",
          "left_hands.jpg", "fist.png", "woman_hands.jpg"]
PL = os.path.join(MODELS, "pose_landmarker_lite.task")
PF = os.path.join(MODELS, "pose_landmarker_full.task")
HM = os.path.join(MODELS, "hand_landmarker.task")
OUT = os.path.join(HERE, "measurements.json")


def log(*a):
    print(*a, flush=True)


def mbuf(p):
    with open(p, "rb") as f:
        return f.read()


def read_img(path):
    d = np.fromfile(path, dtype=np.uint8)
    b = cv2.imdecode(d, cv2.IMREAD_COLOR)
    return b


def mpimg(bgr):
    return mp.Image(image_format=mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))


def mk_pose(path, mode):
    return vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_buffer=mbuf(path)),
        running_mode=mode, num_poses=1,
        min_pose_detection_confidence=0.5, min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5))


def mk_hand(mode, num_hands=2):
    return vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_buffer=mbuf(HM)),
        running_mode=mode, num_hands=num_hands,
        min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5))


def interior_angle(a, b, c, dims):
    if dims == 3:
        v1 = np.array([a.x-b.x, a.y-b.y, a.z-b.z]); v2 = np.array([c.x-b.x, c.y-b.y, c.z-b.z])
    else:
        v1 = np.array([a.x-b.x, a.y-b.y]); v2 = np.array([c.x-b.x, c.y-b.y])
    n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
    if n1 < 1e-9 or n2 < 1e-9:
        return float("nan")
    return math.degrees(math.acos(float(np.clip(np.dot(v1, v2)/(n1*n2), -1.0, 1.0))))


FINGERS = {
    "thumb": [("mcp", 1, 2, 3), ("ip", 2, 3, 4)],
    "index": [("pip", 5, 6, 7), ("dip", 6, 7, 8)],
    "middle": [("pip", 9, 10, 11), ("dip", 10, 11, 12)],
    "ring": [("pip", 13, 14, 15), ("dip", 14, 15, 16)],
    "pinky": [("pip", 17, 18, 19), ("dip", 18, 19, 20)],
}


def curls(lms, dims=3):
    out = {}
    for name, joints in FINGERS.items():
        js, total = {}, 0.0
        for jname, a, b, c in joints:
            ang = interior_angle(lms[a], lms[b], lms[c], dims)
            js[jname] = round(ang, 1) if not math.isnan(ang) else None
            if not math.isnan(ang):
                total += 180.0 - ang
        out[name] = {"total_flexion_deg": round(total, 1), "joint_interior_deg": js,
                     "2d_total_flexion_deg": None}
    return out


def classify(t):
    if t is None:
        return "unknown"
    return "extended" if t < 30.0 else ("partial" if t <= 110.0 else "curled")


def hand_entry(res, i):
    lm = res.hand_landmarks[i]
    wl = res.hand_world_landmarks[i] if res.hand_world_landmarks else lm
    hd = res.handedness[i][0] if res.handedness else None
    cw = curls(wl, 3)
    cn = curls(lm, 2)
    fingers = {}
    for n, c in cw.items():
        fingers[n] = {"total_flexion_deg": c["total_flexion_deg"],
                      "joint_interior_deg": c["joint_interior_deg"],
                      "2d_total_flexion_deg": cn[n]["total_flexion_deg"],
                      "state_rule_30_110": classify(c["total_flexion_deg"])}
    return {"hand_index": i,
            "handedness_label": hd.category_name if hd else None,
            "handedness_score": round(hd.score, 4) if hd else None,
            "landmark_count": len(lm), "fingers": fingers,
            "world_xyz_sample_m": {
                "wrist_0": [round(wl[0].x, 4), round(wl[0].y, 4), round(wl[0].z, 4)],
                "middle_mcp_9": [round(wl[9].x, 4), round(wl[9].y, 4), round(wl[9].z, 4)],
                "middle_tip_12": [round(wl[12].x, 4), round(wl[12].y, 4), round(wl[12].z, 4)]}}


def pose_entry(res, model):
    if not res.pose_landmarks:
        return {"pose_model": os.path.basename(model), "num_poses": 0, "detected": False}
    lm = res.pose_landmarks[0]
    wl = res.pose_world_landmarks[0] if res.pose_world_landmarks else []
    e = {"pose_model": os.path.basename(model), "num_poses": len(res.pose_landmarks),
         "detected": True, "landmark_count": len(lm),
         "normalized_xyz_sample": {str(i): [round(lm[i].x,4), round(lm[i].y,4), round(lm[i].z,4)] for i in (0,11,12,23,24)}}
    if wl:
        e["world_xyz_sample_m"] = {str(i): [round(wl[i].x,4), round(wl[i].y,4), round(wl[i].z,4)] for i in (0,11,12,23,24)}
        e["nose_z_minus_shoulder_mid_z_m"] = round(wl[0].z - 0.5*(wl[11].z+wl[12].z), 4)
        e["shoulder_mid_y_minus_hip_mid_y_m"] = round(0.5*(wl[11].y+wl[12].y) - 0.5*(wl[23].y+wl[24].y), 4)
        e["shoulder_mid_y_minus_nose_y_m"] = round(0.5*(wl[11].y+wl[12].y) - wl[0].y, 4)
    return e


def bench(fn, n=15, warmup=3):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter()-t0)*1000.0)
    s = sorted(ts)
    return {"n": n, "mean_ms": round(statistics.mean(ts),2), "median_ms": round(statistics.median(ts),2),
            "std_ms": round(statistics.stdev(ts),2), "p95_ms": round(s[int(round(0.95*(n-1)))],2),
            "min_ms": round(min(ts),2), "max_ms": round(max(ts),2)}


def save(report, tag):
    report["_last_stage"] = tag
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    log("[save]", tag, "->", OUT)


def main():
    report = {"environment": {"python": ".".join(map(str, __import__("sys").version_info[:3])),
                              "mediapipe": mp.__version__, "numpy": np.__version__,
                              "opencv": cv2.__version__, "cpu_logical": os.cpu_count()},
              "stage": "init"}
    pl = mk_pose(PL, vision.RunningMode.IMAGE)
    pf = mk_pose(PF, vision.RunningMode.IMAGE)
    hd = mk_hand(vision.RunningMode.IMAGE)
    log("[created] all image-mode landmarkers")

    imgs = {}
    for name in IMAGES:
        p = os.path.join(HERE, name)
        bgr = read_img(p); h, w = bgr.shape[:2]; mi = mpimg(bgr)
        t0 = time.perf_counter(); rl = pl.detect(mi); t_l = (time.perf_counter()-t0)*1000
        t0 = time.perf_counter(); rf = pf.detect(mi); t_f = (time.perf_counter()-t0)*1000
        t0 = time.perf_counter(); rh = hd.detect(mi); t_h = (time.perf_counter()-t0)*1000
        e = {"file": name, "file_bytes": os.path.getsize(p), "width": w, "height": h,
             "pose_lite": pose_entry(rl, PL), "pose_full": pose_entry(rf, PF),
             "pose_first_frame_ms": {"lite": round(t_l,1), "full": round(t_f,1)},
             "num_hands": len(rh.hand_landmarks),
             "hand_first_frame_ms": round(t_h,1),
             "hands": [hand_entry(rh, i) for i in range(len(rh.hand_landmarks))]}
        imgs[name] = e
        log(f"[image] {name} {w}x{h} pose_lite={e['pose_lite']['num_poses']} pose_full={e['pose_full']['num_poses']} hands={e['num_hands']} lite_ms={t_l:.1f} full_ms={t_f:.1f} hand_ms={t_h:.1f}")
    report["images"] = imgs
    report["stage"] = "images_done"
    save(report, "images")

    # VIDEO-mode timestamp rule
    vp = mk_pose(PL, vision.RunningMode.VIDEO)
    mi = mpimg(read_img(os.path.join(HERE, "pose.jpg")))
    vt = {"frames": [], "violations": []}
    def try_ts(ts):
        try:
            r = vp.detect_for_video(mi, ts)
            vt["frames"].append({"ts_ms": ts, "num_poses": len(r.pose_landmarks)})
        except Exception as e:
            vt["violations"].append({"ts_ms": ts, "error": f"{type(e).__name__}: {e}"})
    try_ts(0); try_ts(33); try_ts(33); try_ts(10); try_ts(66)
    report["video_mode_timestamp_test"] = vt
    log("[video]", json.dumps(vt, ensure_ascii=False)[:400])
    report["stage"] = "video_ts_done"
    save(report, "video_ts")

    # IMAGE-mode benchmarks (detector re-runs every call)
    bm = {}
    for label, size in (("640x480", (640,480)), ("1280x720", (1280,720))):
        bgr = cv2.resize(read_img(os.path.join(HERE, "male_full_height_hands.jpg")), size)
        mi = mpimg(bgr)
        bm[label] = {"input": label,
                     "pose_lite_image": bench(lambda: pl.detect(mi)),
                     "pose_full_image": bench(lambda: pf.detect(mi)),
                     "hand_image": bench(lambda: hd.detect(mi))}
        log("[bench-image]", label, json.dumps(bm[label], ensure_ascii=False))
        report["benchmarks_image_mode"] = bm
        save(report, "bench_" + label)
    report["stage"] = "done"
    save(report, "done")
    os._exit(0)


if __name__ == "__main__":
    main()
