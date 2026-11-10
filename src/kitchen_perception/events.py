"""事件日志：JSONL 追加写 + 内存最近事件。

日志只记录确认过的状态/事件和证据，不保存原始视频。
写失败不能影响检测：捕获 OSError，记录 error，UI 显示提示，后续事件仍进内存。
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional

from .config import Config

LOGGER = logging.getLogger(__name__)


def _wall_time() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()) + f".{int((time.time() % 1) * 1000):03d}+08:00"


class EventLogger:
    def __init__(self, config: Config) -> None:
        self.enabled = bool(config.get("logging.events_enabled", True))
        self.recent_limit = max(1, int(config.get("logging.recent_events_in_memory", 50)))
        self.dir = config.resolve_path(config.get("logging.events_dir", "outputs/events"))
        self.path: Optional[Path] = None
        self.error: Optional[str] = None
        self._lock = threading.Lock()
        self._recent: deque[dict] = deque(maxlen=self.recent_limit)
        self._fp = None
        if self.enabled:
            self._open()

    def _open(self) -> None:
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            millis = int((time.time() % 1) * 1000)
            self.path = self.dir / f"events_{stamp}_{millis:03d}.jsonl"
            self._fp = self.path.open("a", encoding="utf-8")
            self.error = None
        except OSError as exc:
            self.error = f"事件日志无法创建：{exc}"
            LOGGER.warning(self.error)
            self._fp = None

    def write(self, event: dict) -> bool:
        """写入一条事件；返回是否成功落盘（禁用/失败返回 False）。"""
        record = dict(event)
        record.setdefault("wall_time", _wall_time())
        record.setdefault("event_source", "AUTOMATIC")
        with self._lock:
            self._recent.append(record)
            if not self.enabled or self._fp is None:
                return False
            try:
                self._fp.write(json.dumps(record, ensure_ascii=False) + "\n")
                self._fp.flush()
                return True
            except (OSError, TypeError, ValueError) as exc:
                self.error = f"事件日志写入失败：{exc}"
                LOGGER.warning(self.error)
                try:
                    self._fp.close()
                except OSError:
                    pass
                self._fp = None
                return False

    def recent(self, limit: Optional[int] = None) -> list[dict]:
        with self._lock:
            items = list(self._recent)
        if limit is not None:
            items = items[-max(0, int(limit)):]
        return [dict(item) for item in items]

    def close(self) -> None:
        with self._lock:
            if self._fp is not None:
                try:
                    self._fp.close()
                except OSError:
                    pass
                self._fp = None
