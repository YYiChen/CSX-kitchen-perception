"""运行全部测试，并用 os._exit 跳过 MediaPipe 遥测析构阻塞。

用法：
  py -3.12 -X utf8 tests/run_tests.py            # 全部
  py -3.12 -X utf8 tests/run_tests.py -p "test_torso.py"

直接 `python -m unittest discover -s tests` 也能跑，但进程退出时每个 MediaPipe
landmarker 可能要等约 42s 的 Clearcut 上传超时，整体会慢几分钟。
"""

from __future__ import annotations

import argparse
import os
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="运行测试（MediaPipe 安全退出版）")
    parser.add_argument("-p", "--pattern", default="test_*.py", help="测试文件模式")
    parser.add_argument("-v", "--verbose", action="store_true", help="逐条输出")
    args = parser.parse_args(argv)

    loader = unittest.TestLoader()
    suite = loader.discover(str(HERE), pattern=args.pattern)
    if suite.countTestCases() == 0:
        print(f"没有找到匹配 {args.pattern} 的测试")
        return 2
    result = unittest.TextTestRunner(verbosity=2 if args.verbose else 1).run(suite)
    sys.stdout.flush()
    sys.stderr.flush()
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    os._exit(main())
