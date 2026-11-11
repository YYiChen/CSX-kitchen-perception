"""环境自检：Python/依赖/模型/摄像头/配置。

用法：
  py -3.12 scripts/check_env.py            # 含摄像头探测
  py -3.12 scripts/check_env.py --no-camera
"""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="环境自检")
    parser.add_argument("--no-camera", action="store_true", help="跳过摄像头探测")
    args = parser.parse_args(argv)

    print("=== 环境 ===")
    print(f"系统: {platform.platform()}")
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    ok = True

    print("\n=== 依赖 ===")
    try:
        import cv2
        import mediapipe
        import numpy
        from PIL import Image
    except ImportError as exc:
        print(f"缺少依赖: {exc}")
        print("请执行: py -3.12 -m pip install -r requirements.txt")
        return 1
    print(f"mediapipe {mediapipe.__version__}")
    print(f"opencv {cv2.__version__}")
    print(f"numpy {numpy.__version__}")
    from PIL import __version__ as pillow_version

    print(f"Pillow {pillow_version}")

    print("\n=== 配置 ===")
    try:
        from kitchen_perception.config import load_config

        config = load_config()
        print(f"项目: {config.get('project.name')}")
        print(f"姿态模型: {config.get('models.pose')}，摄像头 index: {config.get('camera.index')}")
    except Exception as exc:
        print(f"配置加载失败: {exc}")
        ok = False

    print("\n=== 模型 ===")
    try:
        from kitchen_perception.engines import verify_models

        errors = verify_models(load_config())
        if errors:
            ok = False
            for error in errors:
                print(f"[错误] {error}")
        else:
            print("模型大小与 sha256 校验通过")
    except Exception as exc:
        ok = False
        print(f"模型校验失败: {exc}")

    if not args.no_camera:
        print("\n=== 摄像头 ===")
        try:
            from kitchen_perception.sources import create_source

            source = create_source("camera", index=int(load_config().get("camera.index", 0)), width=640, height=480)
            frame_ok = False
            frame = None
            for _ in range(10):
                good, frame, _ = source.read()
                if good and frame is not None:
                    frame_ok = True
                    break
            height, width = frame.shape[:2] if frame_ok and frame is not None else (0, 0)
            print(f"{getattr(source, 'description', 'camera')} 读取: {'成功' if frame_ok else '失败'} ({width}x{height})")
            source.close()
            if not frame_ok:
                ok = False
                print("提示：关闭其它占用摄像头的程序，或改 config.camera.index")
        except Exception as exc:
            ok = False
            print(f"摄像头异常: {exc}")

    print("\n=== 结果 ===")
    print("环境就绪" if ok else "存在问题，请按上面提示处理")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
