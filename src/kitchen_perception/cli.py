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
from .pipeline import Pipeline
from .sources import create_source, probe_cameras

LOGGER = logging.getLogger("kitchen_perception")


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
