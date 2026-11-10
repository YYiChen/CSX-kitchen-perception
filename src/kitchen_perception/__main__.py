"""python -m kitchen_perception 入口。

注意：MediaPipe 原生库在进程退出时会同步上传 Clearcut 遥测，离线环境下每个
landmarker 析构约阻塞 42s（本机实测）。这里在命令完成后直接用 os._exit 终止，
跳过解释器清理与原生析构；已完成的输出在退出前显式 flush。
"""

import os
import sys

from .cli import main

if __name__ == "__main__":
    code = main()
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        os._exit(int(code or 0))
