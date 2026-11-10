"""命令行入口。

用法示例：
  python -m kitchen_perception verify-models
  python -m kitchen_perception probe-cameras
  python -m kitchen_perception ui
  python -m kitchen_perception run --preview
  python -m kitchen_perception benchmark --frames 300 --json outputs/benchmark_camera.json
  python -m kitchen_perception snapshot --image assets/test/official/pose.jpg --json outputs/pose.json
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from pathlib import Path
from typing import Optional

import cv2

from .budget import FpsMeter
from .config import Config, load_config
from .engines import ModelError, verify_models
from .media_io import ensure_dir, imread_u, imwrite_u
from .activity import ActivityObservation, ActivityStateMachine
from .motion import MotionFeatures
from .pipeline import Pipeline
from .sources import create_source, probe_cameras

LOGGER = logging.getLogger("kitchen_perception")


# 合成观测序列：不依赖摄像头，用于演示/回归活动状态机。每项是（阶段, 持续毫秒）。
_SIM_PHASES: dict[str, list[tuple[str, float]]] = {
    "chop_a": [("chop_a", 3000.0)],
    "chop_b": [("chop_b", 3000.0)],
    "a_to_b": [("chop_a", 3000.0), ("gap", 400.0), ("chop_b", 3000.0)],
    "pause": [("chop_a", 3000.0), ("pause", 3000.0)],
    "moving": [("moving", 3000.0)],
    "occluded": [("chop_a", 2000.0), ("occluded", 2500.0)],
    "away_return": [("chop_a", 2000.0), ("away", 6000.0), ("pause", 2400.0)],
    "no_person": [("away", 6000.0)],
}

SIM_SCENARIOS = tuple(sorted(_SIM_PHASES))


def _sim_chopping(ts_ms: float, zone: str) -> MotionFeatures:
    return MotionFeatures(
        key=f"sim:right:{zone}",
        label="right",
        valid=True,
        source="palm",
        view_y=0.5,
        speed_norm_per_s=0.3,
        movement_level="HIGH",
        reversals=5,
        rhythm_hz=2.0,
        amplitude_ratio=0.18,
        period_cv=0.08,
        amplitude_cv=0.10,
        score=0.95,
        detected=True,
        stable=True,
        last_chop_ms=ts_ms,
        sampling_gap_ms=0.0,
    )


def _sim_observation(phase: str, ts_ms: float, frame_index: int) -> ActivityObservation:
    obs = ActivityObservation(
        ts_ms=float(ts_ms),
        frame_index=int(frame_index),
        person_visible=True,
        hands_valid=True,
        hands_valid_count=1,
        body_zone="A",
        chopping_zone="A",
        movement_level="LOW",
    )
    if phase in ("chop_a", "chop_b"):
        zone = "A" if phase == "chop_a" else "B"
        obs.body_zone = zone
        obs.chopping_zone = zone
        obs.movement_level = "HIGH"
        obs.chopping = _sim_chopping(ts_ms, zone)
    elif phase == "moving":
        obs.movement_level = "HIGH"
    elif phase == "occluded":
        obs.hands_valid = False
        obs.hands_valid_count = 0
        obs.chopping = None
        obs.movement_level = "unknown"
    elif phase == "away":
        obs.person_visible = False
        obs.hands_valid = False
        obs.hands_valid_count = 0
        obs.body_zone = "unknown"
        obs.chopping_zone = "unknown"
        obs.chopping = None
        obs.movement_level = "unknown"
    elif phase == "gap":
        obs.chopping = None
        obs.movement_level = "LOW"
    elif phase == "pause":
        obs.chopping = None
        obs.movement_level = "LOW"
    return obs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kitchen_perception",
        description="厨房情境感知 · 阶段一：默认摄像头躯干状态 + 五指曲张识别",
    )
    parser.add_argument("--config", default=None, help="用户配置文件路径（与环境变量二选一）")
    parser.add_argument("--pose-model", choices=["lite", "full"], default=None, help="姿态模型变体")
    parser.add_argument("--log-level", default=None, help="DEBUG/INFO/WARNING/ERROR")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("verify-models", help="校验 models/ 下模型文件大小与 sha256")
    sub.add_parser("probe-cameras", help="列出可用摄像头 index").add_argument("--max-index", type=int, default=5)

    p_ui = sub.add_parser("ui", help="启动桌面界面")
    p_ui.add_argument("--video", default=None, help="用视频文件代替摄像头")
    p_ui.add_argument("--index", type=int, default=0)
    p_ui.add_argument("--width", type=int, default=None)
    p_ui.add_argument("--height", type=int, default=None)

    p_run = sub.add_parser("run", help="无头运行（可预览、写 JSONL）")
    p_run.add_argument("--video", default=None)
    p_run.add_argument("--index", type=int, default=0)
    p_run.add_argument("--max-frames", type=int, default=None)
    p_run.add_argument("--preview", action="store_true", help="弹出 OpenCV 预览窗口（按 q 退出）")
    p_run.add_argument("--jsonl", default=None, help="把每帧快照写成 JSONL")
    p_run.add_argument("--save-overlay", default=None, help="保存带标注帧的目录（每 10 帧一张 jpg）")

    p_bench = sub.add_parser("benchmark", help="测延迟并输出 JSON 报告")
    p_bench.add_argument("--video", default=None)
    p_bench.add_argument("--index", type=int, default=0)
    p_bench.add_argument("--frames", type=int, default=300)
    p_bench.add_argument("--json", dest="json_out", default="outputs/benchmark.json")
    p_bench.add_argument("--no-warmup", action="store_true", help="不跑前 15 帧热身")

    p_snap = sub.add_parser("snapshot", help="对单张图片或视频文件跑一次推理并导出 JSON")
    p_snap.add_argument("--image", default=None)
    p_snap.add_argument("--video", default=None)
    p_snap.add_argument("--json", dest="json_out", default=None)
    p_snap.add_argument("--overlay", dest="overlay_out", default=None)

    p_sim = sub.add_parser("simulate", help="用合成观测序列演示活动状态机（无需摄像头）")
    p_sim.add_argument("--scenario", choices=SIM_SCENARIOS, default="a_to_b")
    p_sim.add_argument("--step-ms", type=float, default=50.0, help="合成帧间隔（毫秒）")
    p_sim.add_argument("--json", dest="json_out", default=None, help="输出 JSON 路径")
    return parser


def _apply_overrides(config: Config, args: argparse.Namespace) -> Config:
    overrides: dict = {}
    if args.pose_model:
        overrides.setdefault("models", {})["pose"] = args.pose_model
    if overrides:
        # 复用 Config 的深合并逻辑
        from .config import _deep_merge

        return Config(_deep_merge(config.to_dict(), overrides), config.path)
    return config


def _setup_logging(args: argparse.Namespace, config: Config) -> None:
    level_name = args.log_level or str(config.get("logging.level", "INFO"))
    logging.basicConfig(
        level=getattr(logging, level_name.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


# --------------------------------------------------------------------------- 子命令
def cmd_verify_models(config: Config) -> int:
    errors = verify_models(config)
    if errors:
        print("模型校验失败：")
        for error in errors:
            print("  -", error)
        return 1
    print("模型校验通过（大小与 sha256 一致）")
    return 0


def cmd_probe_cameras(args: argparse.Namespace) -> int:
    indexes = probe_cameras(max_index=args.max_index)
    print("可用摄像头 index:", indexes if indexes else "未发现")
    return 0 if indexes else 1


def _open_source(args, config: Config):
    video = getattr(args, "video", None)
    if video:
        return create_source("video", path=video, loop=False)
    return create_source(
        "camera",
        index=int(getattr(args, "index", 0)),
        width=int(getattr(args, "width", None) or config.get("camera.width", 640)),
        height=int(getattr(args, "height", None) or config.get("camera.height", 480)),
        fps=int(config.get("camera.fps", 30)),
    )


def cmd_run(args: argparse.Namespace, config: Config) -> int:
    source = _open_source(args, config)
    jsonl_fp = None
    overlay_dir: Optional[Path] = None
    if args.jsonl:
        path = config.resolve_path(args.jsonl)
        ensure_dir(path.parent)
        jsonl_fp = path.open("w", encoding="utf-8")
    if args.save_overlay:
        overlay_dir = ensure_dir(config.resolve_path(args.save_overlay))
    pipeline = Pipeline(config, source=source, enable_overlay=bool(args.preview or args.save_overlay))
    saved = 0

    def on_snapshot(snapshot: dict, frame) -> None:
        nonlocal saved
        if jsonl_fp is not None:
            jsonl_fp.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
        if args.preview and pipeline.last_overlay is not None:
            cv2.imshow("kitchen-perception (q to quit)", pipeline.last_overlay)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                raise KeyboardInterrupt
        if overlay_dir is not None and pipeline.last_overlay is not None and snapshot["frame"]["index"] % 10 == 0:
            imwrite_u(overlay_dir / f"frame_{snapshot['frame']['index']:06d}.jpg", pipeline.last_overlay)
            saved += 1

    print(f"开始运行：{getattr(source, 'description', '')}（Ctrl+C 退出）")
    started = time.time()
    try:
        pipeline.run(max_frames=args.max_frames, on_snapshot=on_snapshot)
    except KeyboardInterrupt:
        print("已中断")
    finally:
        pipeline.close()
        if jsonl_fp is not None:
            jsonl_fp.close()
        if args.preview:
            cv2.destroyAllWindows()
    elapsed = time.time() - started
    stats = pipeline.budget.stats()
    print(
        f"处理 {pipeline.frame_index} 帧，用时 {elapsed:.1f}s，"
        f"p50 {stats['p50_ms']}ms，p95 {stats['p95_ms']}ms，max {stats['max_ms']}ms，降级 {stats['level']}"
    )
    if overlay_dir is not None:
        print(f"已保存 {saved} 张标注帧到 {overlay_dir}")
    return 0


def cmd_benchmark(args: argparse.Namespace, config: Config) -> int:
    source = _open_source(args, config)
    pipeline = Pipeline(config, source=source, enable_overlay=False)
    samples: list[float] = []
    snapshots = 0
    warmup = 0 if args.no_warmup else 15

    def on_snapshot(snapshot: dict, frame) -> None:
        nonlocal snapshots
        snapshots += 1
        if snapshots > warmup:
            samples.append(float(snapshot["perf"]["total_ms"]))

    print(f"benchmark：{getattr(source, 'description', '')}，目标 {args.frames} 帧（热身 {warmup} 帧）")
    started = time.time()
    try:
        pipeline.run(max_frames=args.frames + warmup, on_snapshot=on_snapshot)
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.close()
    elapsed = time.time() - started
    if not samples:
        print("没有采到样本")
        return 1
    samples_sorted = sorted(samples)
    p50 = _percentile(samples_sorted, 50)
    p95 = _percentile(samples_sorted, 95)
    hard_cap = float(config.get("latency.hard_cap_ms", 500.0))
    target = float(config.get("latency.target_p95_ms", 250.0))
    verdict = "PASS" if p95 <= target else ("WARN" if p95 <= hard_cap else "FAIL")
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": getattr(source, "description", ""),
        "pose_model": config.get("models.pose"),
        "frames": len(samples),
        "elapsed_s": round(elapsed, 2),
        "fps": round(len(samples) / elapsed, 2) if elapsed > 0 else 0,
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "max_ms": round(max(samples), 2),
        "target_p95_ms": target,
        "hard_cap_ms": hard_cap,
        "verdict": verdict,
        "level": pipeline.budget.level_name,
        "machine": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": __import__("os").cpu_count(),
        },
    }
    out_path = config.resolve_path(args.json_out)
    ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8") as fp:
        json.dump(report, fp, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"报告已写入 {out_path}")
    return 0 if verdict != "FAIL" else 2


def cmd_snapshot(args: argparse.Namespace, config: Config) -> int:
    pipeline = Pipeline(config, source=None, enable_overlay=bool(args.overlay_out))
    if args.image:
        frame = imread_u(args.image)
        if frame is None:
            print(f"图片读取失败: {args.image}")
            pipeline.close()
            return 1
        snapshots = [pipeline.process(frame, 0.0)]
    elif args.video:
        source = create_source("video", path=args.video)
        pipeline.source = source
        snapshots = []
        max_frames = 1
        ok, frame, ts = source.read()
        if not ok:
            print("视频读取失败")
            return 1
        snapshots.append(pipeline.process(frame, ts))
    else:
        print("需要 --image 或 --video")
        return 1
    snapshot = snapshots[0]
    text = json.dumps(snapshot, ensure_ascii=False, indent=2)
    print(text)
    if args.json_out:
        out_path = config.resolve_path(args.json_out)
        ensure_dir(out_path.parent)
        out_path.write_text(text, encoding="utf-8")
        print(f"JSON 已写入 {out_path}")
    if args.overlay_out and pipeline.last_overlay is not None:
        out_path = config.resolve_path(args.overlay_out)
        ensure_dir(out_path.parent)
        imwrite_u(out_path, pipeline.last_overlay)
        print(f"标注图已写入 {out_path}")
    pipeline.close()
    return 0


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * max(0.0, min(100.0, q)) / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] * (hi - pos) + sorted_values[hi] * (pos - lo)


def cmd_simulate(args: argparse.Namespace, config: Config) -> int:
    """把合成观测喂给活动状态机，输出状态时间线与事件。"""
    scenario = str(args.scenario)
    phases = _SIM_PHASES.get(scenario)
    if not phases:
        print(f"未知场景: {scenario}")
        return 1
    step_ms = max(10.0, float(args.step_ms))
    machine = ActivityStateMachine(config)
    events: list[dict] = []
    timeline: list[dict] = []
    result: dict = {}
    ts_ms = 0.0
    frame_index = 0
    for phase, duration_ms in phases:
        steps = max(1, int(round(float(duration_ms) / step_ms)))
        for _ in range(steps):
            obs = _sim_observation(phase, ts_ms, frame_index)
            result = machine.update(obs)
            for event in machine.take_events():
                events.append(event.to_dict())
            if not timeline or timeline[-1]["state"] != result["state"]:
                timeline.append(
                    {
                        "ts_ms": round(ts_ms, 1),
                        "state": result["state"],
                        "state_zh": result["state_zh"],
                        "zone": result["zone"],
                        "quality": result["observation_quality"],
                    }
                )
            ts_ms += step_ms
            frame_index += 1
    summary = {
        "scenario": scenario,
        "duration_ms": round(ts_ms, 1),
        "frames": frame_index,
        "final_state": result.get("state", "UNKNOWN"),
        "final_state_zh": result.get("state_zh", "UNKNOWN"),
        "states_seen": sorted({item["state"] for item in timeline}),
        "event_count": len(events),
    }
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "step_ms": step_ms,
        "summary": summary,
        "timeline": timeline,
        "events": events,
    }
    out_path = config.resolve_path(args.json_out or f"outputs/simulate_{scenario}.json")
    ensure_dir(out_path.parent)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"场景 {scenario}：{frame_index} 帧，{ts_ms / 1000.0:.1f}s，"
        f"最终 {summary['final_state']}（{summary['final_state_zh']}），事件 {len(events)} 条"
    )
    for item in timeline:
        print(f"  {item['ts_ms'] / 1000.0:6.2f}s  {item['state']:<18} zone={item['zone']}  quality={item['quality']}")
    for event in events:
        print(
            f"  {float(event.get('ts_ms', 0.0)) / 1000.0:6.2f}s  EVENT {event.get('event')}  "
            f"{event.get('previous_state')}→{event.get('state')}"
        )
    print(f"JSON 已写入 {out_path}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    config = _apply_overrides(config, args)
    _setup_logging(args, config)
    try:
        if args.command == "verify-models":
            return cmd_verify_models(config)
        if args.command == "probe-cameras":
            return cmd_probe_cameras(args)
        if args.command == "run":
            return cmd_run(args, config)
        if args.command == "benchmark":
            return cmd_benchmark(args, config)
        if args.command == "snapshot":
            return cmd_snapshot(args, config)
        if args.command == "simulate":
            return cmd_simulate(args, config)
        if args.command == "ui":
            try:
                from .ui import run_ui
            except ImportError as exc:
                print(f"界面模块未就绪: {exc}")
                return 1
            return run_ui(config, video=args.video, index=args.index, width=args.width, height=args.height)
    except ModelError as exc:
        print(f"模型错误: {exc}")
        return 1
    except FileNotFoundError as exc:
        print(f"文件不存在: {exc}")
        return 1
    parser.print_help()
    return 1
