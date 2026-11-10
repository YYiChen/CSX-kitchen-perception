"""下载/修复 MediaPipe 模型文件，按 models/MANIFEST.json 校验 sha256。

用法：
  py -3.12 scripts/download_models.py            # 缺失才下载
  py -3.12 scripts/download_models.py --force    # 全部重新下载
  py -3.12 scripts/download_models.py --check    # 只校验不下载
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT_ROOT / "models" / "MANIFEST.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: Path, timeout: float = 60.0) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "csx-kitchen-perception/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response, tmp.open("wb") as fp:
        while True:
            chunk = response.read(1024 * 256)
            if not chunk:
                break
            fp.write(chunk)
    tmp.replace(target)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="下载并校验 MediaPipe 模型")
    parser.add_argument("--force", action="store_true", help="忽略已有文件，全部重新下载")
    parser.add_argument("--check", action="store_true", help="只校验，不下载")
    args = parser.parse_args(argv)

    if not MANIFEST.exists():
        print(f"找不到 {MANIFEST}")
        return 1
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    errors = 0
    for item in data.get("models", []):
        path = PROJECT_ROOT / item["path"]
        expected = item["sha256"]
        if path.exists() and not args.force:
            actual = sha256_file(path)
            if actual == expected:
                print(f"[OK] {path.name} ({path.stat().st_size} bytes)")
                continue
            print(f"[坏] {path.name} sha256 不符，准备重新下载")
        elif args.check:
            print(f"[缺失] {path}")
            errors += 1
            continue
        if args.check:
            errors += 1
            continue
        print(f"[下载] {item['url']}")
        try:
            download(item["url"], path)
        except Exception as exc:  # 网络问题要给出可操作提示
            print(f"  下载失败: {exc}")
            errors += 1
            continue
        actual = sha256_file(path)
        if actual != expected:
            print(f"  校验失败: 期望 {expected} 实际 {actual}")
            errors += 1
        else:
            print(f"  完成 {path.name} ({path.stat().st_size} bytes)")
    if errors:
        print(f"\n有 {errors} 个模型需要处理（检查网络/代理后重试）")
        return 1
    print("\n全部模型就绪")
    return 0


if __name__ == "__main__":
    sys.exit(main())
