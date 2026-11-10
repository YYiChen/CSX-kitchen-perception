"""按炒菜环节定制“躯干 / 手指”定义的扩展点。

默认实现返回 None，表示沿用 config 里的通用规则。
想按某个环节（切菜/备菜/颠勺）改判定，只要改这个文件，或在 config 里把
torso.custom_rule_module / hands.custom_rule_module 指向你自己的模块。

约定：
- torso_state(features: dict, config) -> dict | None
  返回 dict 可包含 posture / confidence / note；posture 必须在
  ("no_person","not_visible","upright","leaning_forward","leaning_backward",
   "leaning_side","bent_over","uncertain") 里。
- grip_state(fingers: dict, metrics: dict, config) -> str | dict | None
  返回手势名（见 config.hands.grip_labels_zh 的 key）或 dict。
- 两个函数都必须快速、无副作用、不能抛异常（核心会兜底捕获，但抛异常会导致
  这一帧回退到默认规则）。
"""

from __future__ import annotations

from .config import Config


def torso_state(features: dict, config: Config) -> dict | None:
    """示例：切菜时“俯身”可以放宽到 35° 就算，而不是默认 45°。

    真的需要时取消注释即可：
    """
    # th = config.section("torso")["thresholds"]
    # pitch = features.get("pitch_deg")
    # if pitch is not None and pitch > 35.0:
    #     return {"posture": "bent_over", "confidence": 0.9, "note": "切菜环节自定义阈值"}
    return None


def grip_state(fingers: dict, metrics: dict, config: Config) -> str | dict | None:
    """示例：某环节里“食指弯曲 + 其余伸直”定义为 custom_grip。

    自定义手势名要求同时在 config.hands.grip_labels_zh 里加中文标签。
    """
    return None

# ---------------------------------------------------------------------------
# 加载器：核心模块通过 apply_torso_rule / apply_grip_rule 调用，
# 用户模块只需提供同名函数（模块名/函数名在 config 里配置）。
# ---------------------------------------------------------------------------

import importlib
import logging
import sys
from typing import Any

from .config import PROJECT_ROOT

LOGGER = logging.getLogger(__name__)
_CACHE: dict[tuple[str, str], Any] = {}


def _resolve(module_name: str, function_name: str):
    key = (module_name, function_name)
    if key in _CACHE:
        return _CACHE[key]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    try:
        module = importlib.import_module(module_name)
        func = getattr(module, function_name, None)
    except Exception as exc:  # 用户模块写错不应该拖垮主程序
        LOGGER.warning("自定义规则模块加载失败 %s.%s: %s", module_name, function_name, exc)
        func = None
    _CACHE[key] = func
    return func


def apply_torso_rule(features: dict, config: Config) -> dict | None:
    module_name = config.get("torso.custom_rule_module") or ""
    function_name = config.get("torso.custom_rule_function") or "torso_state"
    if not module_name:
        return None
    func = _resolve(str(module_name), str(function_name))
    if func is None:
        return None
    result = func(features, config)
    return result if isinstance(result, dict) else None


def apply_grip_rule(fingers: dict, metrics: dict, config: Config):
    module_name = config.get("hands.custom_rule_module") or ""
    function_name = config.get("hands.custom_rule_function") or "grip_state"
    if not module_name:
        return None
    func = _resolve(str(module_name), str(function_name))
    if func is None:
        return None
    try:
        result = func(fingers, metrics, config)
    except TypeError:
        # 兼容只接收两个参数的实现
        result = func(fingers, config)
    if isinstance(result, (str, dict)):
        return result
    return None
