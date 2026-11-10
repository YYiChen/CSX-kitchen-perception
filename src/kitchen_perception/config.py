"""配置加载。

默认配置在 config/default.json，支持三层覆盖：
1. 默认值（config/default.json）
2. 用户配置文件：构造函数传入，或环境变量 KITCHEN_PERCEPTION_CONFIG
3. 运行时 overrides（例如 CLI 的 --pose-model / --width）

所有阈值、中文标签、降级级别都在配置里，方便按炒菜环节/个人机位调整，
不需要改代码。
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.json"
ENV_CONFIG = "KITCHEN_PERCEPTION_CONFIG"


class ConfigError(RuntimeError):
    """配置错误。"""


def _deep_merge(base: dict, patch: dict) -> dict:
    """递归合并 dict，返回新对象；list 直接替换。"""
    out = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


class Config:
    """轻量配置容器：支持 cfg["torso"] 与 cfg.get("torso.ema_alpha", 0.3)。"""

    def __init__(self, data: dict, path: Path | None = None) -> None:
        self.data = data
        self.path = path

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def section(self, name: str) -> dict:
        value = self.data.get(name)
        if not isinstance(value, dict):
            raise ConfigError(f"配置缺少节: {name}")
        return value

    def resolve_path(self, value: str | os.PathLike) -> Path:
        """相对路径按项目根目录解析。"""
        p = Path(value)
        return p if p.is_absolute() else PROJECT_ROOT / p

    def to_dict(self) -> dict:
        return copy.deepcopy(self.data)


def load_config(path: str | os.PathLike | None = None, overrides: dict | None = None) -> Config:
    """加载配置；path 为空时依次看环境变量与默认文件。"""
    if not DEFAULT_CONFIG_PATH.exists():
        raise ConfigError(f"默认配置不存在: {DEFAULT_CONFIG_PATH}")
    with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    user_path = Path(path) if path else (Path(os.environ[ENV_CONFIG]) if os.environ.get(ENV_CONFIG) else None)
    if user_path is not None:
        if not user_path.exists():
            raise ConfigError(f"配置文件不存在: {user_path}")
        with user_path.open("r", encoding="utf-8") as fp:
            data = _deep_merge(data, json.load(fp))
    if overrides:
        data = _deep_merge(data, overrides)
    return Config(data, user_path)


def resolve_project_path(config: Config, value: str | os.PathLike) -> Path:
    return config.resolve_path(value)
