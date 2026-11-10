"""Tkinter 桌面界面：左侧叠加画面 + 右侧中文状态面板。

与核心模块的分工：
- 检测全部在 PerceptionWorker 后台线程完成；UI 主线程只按 poll_ms（默认 30ms）
  轮询 worker.get_latest()，不直接调用 MediaPipe，避免界面卡顿。
- 叠加层已由 overlay.OverlayRenderer 画好骨架、五指颜色和中文 HUD，这里只做
  BGR ndarray -> PIL -> ImageTk 的显示缩放。
- 界面不重复实现检测/阈值逻辑；中文标签、窗口标题、轮询间隔都来自 config 与 snapshot。

暂停语义：
- 「暂停」调用 worker.set_paused(True)：后台线程退出并释放摄像头，画面停在最后一帧；
- 「继续」调用 worker.start()：重新打开采集（摄像头重新打开，视频从头播放）。
"""

from __future__ import annotations

import json
import logging
import math
import time
import traceback
from pathlib import Path
from typing import Optional

import tkinter as tk

from PIL import Image, ImageTk

from .config import Config
from .fingers import FINGER_ORDER, FINGER_ZH
from .media_io import bgr_to_pil, ensure_dir, imwrite_u
from .overlay import FACING_ZH
from .worker import PerceptionWorker
from .zones import ZoneSettings, save_settings, zones_file

LOGGER = logging.getLogger(__name__)

# 状态色：绿=伸直/直立，橙=半屈/轻倾，红=弯曲/俯身，灰=未知
COLOR_OK = "#1a7f37"
COLOR_PARTIAL = "#b35900"
COLOR_BAD = "#c62828"
COLOR_UNKNOWN = "#7a7a7a"
COLOR_ERROR = "#c62828"
COLOR_MUTED = "#666666"
BG = "#f4f4f4"
BG_PANEL = "#ffffff"
BG_PREVIEW = "#111111"

FINGER_STATE_ZH = {"extended": "伸直", "partial": "半屈", "flexed": "弯曲", "unknown": "未知"}
FINGER_STATE_COLOR = {
    "extended": COLOR_OK,
    "partial": COLOR_PARTIAL,
    "flexed": COLOR_BAD,
    "unknown": COLOR_UNKNOWN,
}
HAND_LABEL_ZH = {"left": "左手", "right": "右手"}
POSTURE_COLORS = {
    "upright": COLOR_OK,
    "leaning_forward": COLOR_PARTIAL,
    "leaning_backward": COLOR_PARTIAL,
    "leaning_side": COLOR_PARTIAL,
    "bent_over": COLOR_BAD,
    "uncertain": "#8a6d00",
    "no_person": COLOR_UNKNOWN,
    "not_visible": COLOR_UNKNOWN,
}

ACTIVITY_COLORS = {
    "ACTIVE_A": COLOR_OK,
    "ACTIVE_B": COLOR_OK,
    "MOVING": "#0b5cad",
    "PAUSED": COLOR_PARTIAL,
    "TRANSITION_A_TO_B": "#6a1b9a",
    "TRANSITION_B_TO_A": "#6a1b9a",
    "OCCLUDED": COLOR_PARTIAL,
    "AWAY": COLOR_UNKNOWN,
    "UNKNOWN": COLOR_UNKNOWN,
}
ACTIVITY_STATE_ZH = {
    "UNKNOWN": "等待观察",
    "ACTIVE_A": "A 区切菜中",
    "ACTIVE_B": "B 区切菜中",
    "MOVING": "手部移动（未见切菜节奏）",
    "PAUSED": "暂停（未见切菜节奏）",
    "TRANSITION_A_TO_B": "A→B 区域转移",
    "TRANSITION_B_TO_A": "B→A 区域转移",
    "OCCLUDED": "手部被遮挡/观察不足",
    "AWAY": "暂未观察到用户",
}
QUALITY_ZH = {
    "VALID": "有效",
    "LOW": "偏低",
    "OCCLUDED": "遮挡",
    "NO_PERSON": "无人",
    "UNKNOWN": "未知",
}
MOVEMENT_ZH = {"LOW": "低", "MEDIUM": "中", "HIGH": "高", "unknown": "未知"}
EVENT_ZH = {
    "SESSION_STARTED": "会话开始",
    "ACTIVITY_A_STARTED": "A 区开始切菜",
    "ACTIVITY_B_STARTED": "B 区开始切菜",
    "ACTIVITY_PAUSED": "切菜暂停",
    "HAND_MOVING": "手部移动",
    "ZONE_TRANSITION_A_TO_B": "A→B 转移",
    "ZONE_TRANSITION_B_TO_A": "B→A 转移",
    "CHOPPING_STARTED": "切菜节奏开始",
    "CHOPPING_STOPPED": "切菜节奏停止",
    "CHOPPING_STABLE": "切菜节奏稳定",
    "CHOPPING_UNSTABLE": "切菜节奏不稳",
    "HAND_OCCLUDED": "手部遮挡",
    "USER_AWAY": "暂未观察到用户",
    "USER_RETURNED": "用户回到画面",
    "CAMERA_DISCONNECTED": "摄像头断开",
    "ZONE_CONFIG_CHANGED": "分区配置变更",
}


class KitchenUI:
    """桌面主窗口。run_ui() 是 CLI 唯一入口；测试脚本可直接实例化后驱动 mainloop。"""

    def __init__(
        self,
        config: Config,
        video: Optional[str] = None,
        index: int = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> None:
        self.config = config
        self.video = str(video) if video else None
        self.index = int(index)
        self.width = width
        self.height = height
        self.poll_ms = max(10, int(config.get("ui.poll_ms", 30)))
        self.panel_every = max(1, int(config.get("ui.panel_update_every_frames", 2)))
        self.font_family = str(config.get("ui.font_family", "Microsoft YaHei UI"))
        self.snapshot_dir = config.resolve_path(config.get("output.snapshot_dir", "outputs/snapshots"))

        # 运行状态：自测脚本会读取 last_snapshot / preview_frames / panel_updates 等公开属性
        self.worker: Optional[PerceptionWorker] = None
        self.last_snapshot: Optional[dict] = None
        self.last_overlay = None
        self.preview_frames = 0
        self.panel_updates = 0
        self.hands_summary = "等待画面…"
        self.callback_errors: list[str] = []
        self._paused = False
        self._resuming = False
        self._resume_after_id: Optional[str] = None
        self._resume_attempts = 0
        self._video_ended = False
        self._closed = False
        self._after_id: Optional[str] = None
        self._preview_index = -1
        self._panel_index = -1
        self._photo = None
        self._last_pil = None
        self._preview_size: tuple[int, int] = (0, 0)
        self._last_error_text = ""
        self._last_status_text = ""
        self._startup_error: Optional[str] = None
        self._hand_blocks: list[dict] = []
        self._started_at = time.monotonic()
        self._zone_dialog: Optional[tk.Toplevel] = None
        self._zone_vars: dict[str, tk.StringVar] = {}
        self._event_log_sig: Optional[tuple] = None

        self.root = tk.Tk()
        self.root.title(str(config.get("ui.title", "厨房情境感知 · 阶段一")))
        # 演示机分辨率未知：窗口不超过屏幕可用区域，避免右侧面板/底部快捷键被裁掉。
        window_width = min(
            int(config.get("ui.width", 1280)), max(900, self.root.winfo_screenwidth() - 80)
        )
        window_height = min(
            int(config.get("ui.height", 900)), max(650, self.root.winfo_screenheight() - 120)
        )
        self.root.geometry(f"{window_width}x{window_height}")
        self.root.minsize(min(1000, window_width), min(700, window_height))
        self.root.configure(bg=BG)
        self.root.report_callback_exception = self._on_callback_exception
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self._build_widgets()
        self._bind_keys()

        # 视频路径先做一次存在性检查：worker 对视频打不开只会静默结束，界面必须自己给出原因
        if self.video and not Path(self.video).expanduser().exists():
            self._startup_error = f"视频文件不存在：{self.video}"
            self._label_error(self._startup_error)
            self._set_status("感知线程未启动（请检查 --video 路径）")
            self._set_buttons_enabled(False)
        else:
            self._start_worker()
        self._after_id = self.root.after(self.poll_ms, self._tick)

    # ------------------------------------------------------------------ 界面搭建
    def _font(self, size: int, weight: str = "normal") -> tuple[str, int, str]:
        return (self.font_family, size, weight)

    def _group(self, parent: tk.Misc, title: str) -> tk.LabelFrame:
        return tk.LabelFrame(
            parent,
            text=title,
            font=self._font(10, "bold"),
            fg="#333333",
            bg=BG_PANEL,
            bd=1,
            relief="groove",
            padx=6,
            pady=4,
        )

    def _build_widgets(self) -> None:
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill="both", expand=True, padx=8, pady=8)

        # 左：叠加画面（占满剩余空间，等比缩放）
        self.preview_label = tk.Label(
            main, bg=BG_PREVIEW, fg="#bbbbbb", text="等待画面…", font=self._font(16)
        )
        self.preview_label.pack(side="left", fill="both", expand=True)
        self.preview_label.bind("<Configure>", self._on_preview_resize)

        # 右：状态面板
        panel = tk.Frame(main, bg=BG)
        panel.pack(side="right", fill="y", padx=(8, 0))
        panel.grid_columnconfigure(0, weight=1)
        row = 0

        self.error_label = tk.Label(
            panel,
            text="",
            fg=COLOR_ERROR,
            bg="#ffecec",
            font=self._font(10),
            wraplength=330,
            justify="left",
            anchor="w",
            padx=8,
            pady=6,
        )
        self.error_label.grid(row=row, column=0, sticky="ew", pady=(0, 6))
        self.error_label.grid_remove()
        row += 1

        # 情境感知：活动状态 + 切菜节奏（阶段二核心面板）
        activity_frame = self._group(panel, "情境感知")
        activity_frame.grid(row=row, column=0, sticky="ew", pady=(0, 6))
        row += 1
        self.activity_label = tk.Label(
            activity_frame,
            text="等待观察",
            font=self._font(18, "bold"),
            fg=COLOR_UNKNOWN,
            bg=BG_PANEL,
            anchor="w",
        )
        self.activity_label.pack(fill="x", padx=6, pady=(2, 0))
        self.activity_info_label = tk.Label(
            activity_frame,
            text="区域 --   持续 --   移动 --   质量 --",
            font=self._font(10),
            fg="#333333",
            bg=BG_PANEL,
            anchor="w",
            justify="left",
        )
        self.activity_info_label.pack(fill="x", padx=6)
        self.chopping_label = tk.Label(
            activity_frame,
            text="切菜：未检测到节奏",
            font=self._font(10),
            fg="#444444",
            bg=BG_PANEL,
            anchor="w",
            justify="left",
            wraplength=310,
        )
        self.chopping_label.pack(fill="x", padx=6)
        self.last_event_label = tk.Label(
            activity_frame,
            text="最近事件：--",
            font=self._font(9),
            fg=COLOR_MUTED,
            bg=BG_PANEL,
            anchor="w",
            justify="left",
            wraplength=310,
        )
        self.last_event_label.pack(fill="x", padx=6, pady=(0, 3))

        # 躯干状态：大标签 + 置信度
        torso_frame = self._group(panel, "躯干状态")
        torso_frame.grid(row=row, column=0, sticky="ew", pady=(0, 6))
        row += 1
        self.torso_label = tk.Label(
            torso_frame,
            text="等待画面…",
            font=self._font(22, "bold"),
            fg=COLOR_UNKNOWN,
            bg=BG_PANEL,
            anchor="w",
        )
        self.torso_label.pack(fill="x", padx=6, pady=(2, 0))
        self.torso_conf_label = tk.Label(
            torso_frame,
            text="置信度：--",
            font=self._font(10),
            fg="#444444",
            bg=BG_PANEL,
            anchor="w",
        )
        self.torso_conf_label.pack(fill="x", padx=6, pady=(0, 4))

        # 躯干角度与朝向
        angles_frame = self._group(panel, "躯干角度")
        angles_frame.grid(row=row, column=0, sticky="ew", pady=(0, 6))
        row += 1
        self.angle_label = tk.Label(
            angles_frame,
            text="前倾 --   侧倾 --   转向 --\n朝向：--",
            font=self._font(10),
            fg="#333333",
            bg=BG_PANEL,
            anchor="w",
            justify="left",
        )
        self.angle_label.pack(fill="x", padx=6)

        # 手部：最多两块（左手/右手），每块一行五指色块
        hands_frame = self._group(panel, "手部")
        hands_frame.grid(row=row, column=0, sticky="ew", pady=(0, 6))
        row += 1
        self._hands_inner = tk.Frame(hands_frame, bg=BG_PANEL)
        self._hands_inner.pack(fill="x")
        self._hands_inner.grid_columnconfigure(0, weight=1)
        self.hand_placeholder = tk.Label(
            self._hands_inner,
            text="等待画面…",
            font=self._font(10),
            fg=COLOR_MUTED,
            bg=BG_PANEL,
            anchor="w",
            justify="left",
            wraplength=300,
        )
        self.hand_placeholder.grid(row=0, column=0, sticky="ew", padx=4, pady=4)

        # 性能
        perf_frame = self._group(panel, "性能")
        perf_frame.grid(row=row, column=0, sticky="ew", pady=(0, 6))
        row += 1
        self.fps_label = tk.Label(
            perf_frame, text="FPS：--", font=self._font(10), fg="#333333", bg=BG_PANEL, anchor="w"
        )
        self.fps_label.pack(fill="x", padx=6)
        self.p95_label = tk.Label(
            perf_frame, text="延迟 p95：--", font=self._font(10), fg="#333333", bg=BG_PANEL, anchor="w"
        )
        self.p95_label.pack(fill="x", padx=6)
        self.level_label = tk.Label(
            perf_frame, text="降级级别：--", font=self._font(10), fg="#333333", bg=BG_PANEL, anchor="w"
        )
        self.level_label.pack(fill="x", padx=6)

        # 事件日志：显示最近确认事件，写入文件由 EventLogger 负责
        log_frame = self._group(panel, "事件日志")
        log_frame.grid(row=row, column=0, sticky="ew", pady=(0, 6))
        row += 1
        log_inner = tk.Frame(log_frame, bg=BG_PANEL)
        log_inner.pack(fill="both", expand=True)
        self.event_text = tk.Text(
            log_inner,
            height=6,
            width=42,
            font=self._font(9),
            state="disabled",
            wrap="none",
            bg="#fbfbfb",
            fg="#333333",
            relief="flat",
            padx=4,
            pady=3,
        )
        event_scroll = tk.Scrollbar(log_inner, orient="vertical", command=self.event_text.yview)
        self.event_text.configure(yscrollcommand=event_scroll.set)
        self.event_text.pack(side="left", fill="both", expand=True)
        event_scroll.pack(side="right", fill="y")

        # 按钮
        buttons = tk.Frame(panel, bg=BG)
        buttons.grid(row=row, column=0, sticky="ew", pady=(4, 2))
        buttons.grid_columnconfigure(0, weight=1)
        buttons.grid_columnconfigure(1, weight=1)
        row += 1
        self.pause_button = tk.Button(
            buttons, text="暂停", command=self._toggle_pause, font=self._font(10)
        )
        self.pause_button.grid(row=0, column=0, sticky="ew", padx=(0, 3), pady=2)
        self.zone_button = tk.Button(
            buttons, text="分区设置…", command=self._open_zone_dialog, font=self._font(10)
        )
        self.zone_button.grid(row=0, column=1, sticky="ew", padx=(3, 0), pady=2)
        self.calibrate_button = tk.Button(
            buttons, text="标定直立基准", command=self._calibrate, font=self._font(10)
        )
        self.calibrate_button.grid(row=1, column=0, sticky="ew", padx=(0, 3), pady=2)
        self.clear_button = tk.Button(
            buttons, text="清除标定", command=self._clear_calibration, font=self._font(10)
        )
        self.clear_button.grid(row=1, column=1, sticky="ew", padx=(3, 0), pady=2)
        self.save_button = tk.Button(
            buttons, text="保存快照", command=self._save_snapshot, font=self._font(10)
        )
        self.save_button.grid(row=2, column=0, sticky="ew", padx=(0, 3), pady=2)
        self.save_zone_button = tk.Button(
            buttons, text="保存分区配置", command=self._save_zone_config, font=self._font(10)
        )
        self.save_zone_button.grid(row=2, column=1, sticky="ew", padx=(3, 0), pady=2)

        self.action_label = tk.Label(
            panel,
            text="快捷键：空格 暂停/继续 · Z 分区设置 · C 标定 · R 清除标定 · S 保存快照 · Q/Esc 退出",
            font=self._font(9),
            fg="#0b5cad",
            bg=BG,
            wraplength=340,
            justify="left",
            anchor="w",
        )
        self.action_label.grid(row=row, column=0, sticky="ew", pady=(2, 0))
        row += 1
        self.status_label = tk.Label(
            panel,
            text="正在启动…",
            font=self._font(9),
            fg=COLOR_MUTED,
            bg=BG,
            wraplength=340,
            justify="left",
            anchor="w",
        )
        self.status_label.grid(row=row, column=0, sticky="ew", pady=(2, 0))
        row += 1
        panel.grid_rowconfigure(row, weight=1)

    def _bind_keys(self) -> None:
        self.root.bind("<Escape>", lambda _event: self.close())
        self.root.bind("<KeyPress-q>", lambda _event: self.close())
        self.root.bind("<space>", lambda _event: self._toggle_pause())
        self.root.bind("<KeyPress-c>", lambda _event: self._calibrate())
        self.root.bind("<KeyPress-r>", lambda _event: self._clear_calibration())
        self.root.bind("<KeyPress-s>", lambda _event: self._save_snapshot())
        self.root.bind("<KeyPress-z>", lambda _event: self._open_zone_dialog())

    def _set_buttons_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for button in (self.pause_button, self.calibrate_button, self.clear_button, self.save_button):
            button.configure(state=state)

    def _create_hand_block(self, index: int) -> dict:
        frame = tk.Frame(self._hands_inner, bg=BG_PANEL)
        title = tk.Label(
            frame, text="手部", font=self._font(11, "bold"), fg="#333333", bg=BG_PANEL, anchor="w"
        )
        title.pack(fill="x", padx=2)
        cells_row = tk.Frame(frame, bg=BG_PANEL)
        cells_row.pack(fill="x", padx=2, pady=(0, 4))
        cells: dict[str, tk.Label] = {}
        for finger in FINGER_ORDER:
            cell = tk.Label(
                cells_row,
                text=f"{FINGER_ZH[finger]}\n--",
                font=self._font(9),
                fg=COLOR_UNKNOWN,
                bg=BG_PANEL,
                width=6,
                justify="center",
            )
            cell.pack(side="left", padx=1)
            cells[finger] = cell
        frame.grid(row=index + 1, column=0, sticky="ew", pady=(0, 4))
        return {"frame": frame, "title": title, "cells": cells}

    def _ensure_hand_blocks(self, count: int) -> None:
        while len(self._hand_blocks) < count:
            self._hand_blocks.append(self._create_hand_block(len(self._hand_blocks)))

    # ------------------------------------------------------------------ 启动/关闭
    def _start_worker(self) -> None:
        worker = PerceptionWorker(
            self.config,
            source_kind="video" if self.video else "camera",
            index=self.index,
            video_path=self.video,
            width=self.width,
            height=self.height,
        )
        self.worker = worker
        try:
            worker.start()
        except Exception as exc:
            self._startup_error = f"感知线程启动失败：{type(exc).__name__}: {exc}"
            self._label_error(self._startup_error)
            self._set_status("感知线程未启动")
            return
        self._set_status("正在启动感知线程（首次加载模型约 1–3 秒）…")

    def run(self) -> None:
        self.root.mainloop()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._zone_dialog is not None:
            try:
                self._zone_dialog.destroy()
            except tk.TclError:
                pass
            self._zone_dialog = None
        if self._after_id is not None:
            try:
                self.root.after_cancel(self._after_id)
            except tk.TclError:
                pass
            self._after_id = None
        if self._resume_after_id is not None:
            try:
                self.root.after_cancel(self._resume_after_id)
            except tk.TclError:
                pass
            self._resume_after_id = None
        self._resuming = False
        if self.worker is not None:
            try:
                self.worker.stop()
            except Exception:  # 关闭阶段不再抛出
                LOGGER.warning("停止感知线程失败", exc_info=True)
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    # ------------------------------------------------------------------ 轮询主循环
    def _tick(self) -> None:
        if self._closed:
            return
        try:
            self._refresh()
        except Exception as exc:  # 界面自身异常不能让窗口退出
            LOGGER.exception("界面刷新异常")
            self._set_action(f"界面刷新异常：{type(exc).__name__}: {exc}")
        finally:
            if not self._closed:
                self._after_id = self.root.after(self.poll_ms, self._tick)

    def _refresh(self) -> None:
        worker = self.worker
        if worker is None:
            return
        latest = worker.get_latest()
        if latest is not None:
            snapshot, overlay = latest
            self.last_snapshot = snapshot
            self.last_overlay = overlay
            frame_info = snapshot.get("frame") or {}
            try:
                frame_index = int(frame_info.get("index") or 0)
            except (TypeError, ValueError):
                frame_index = 0
            if frame_index < self._panel_index:
                # 恢复采集后新 Pipeline 的帧号从 1 重新计数，重置刷新游标
                self._panel_index = -1
                self._preview_index = -1
            if overlay is not None and frame_index != self._preview_index:
                self._preview_index = frame_index
                self._show_overlay(overlay)
            if frame_index != self._panel_index and (
                self._panel_index < 0 or frame_index - self._panel_index >= self.panel_every
            ):
                self._panel_index = frame_index
                self._update_panel(snapshot)
        error_text = (worker.error or "").strip()
        if error_text != self._last_error_text:
            self._last_error_text = error_text
            if error_text:
                self._label_error(f"错误：{error_text}")
            else:
                self._hide_error()
        # 视频文件播完时 worker 线程正常结束：自动进入「已暂停」，让按钮变成「继续」可直接重播
        if (
            not worker.is_alive()
            and not worker.error
            and self.video
            and self.last_snapshot is not None
            and not self._paused
            and not self._resuming
        ):
            self._video_ended = True
            self._paused = True
            self.pause_button.configure(text="继续")
        self._set_status(self._status_text(worker))

    def _status_text(self, worker: PerceptionWorker) -> str:
        if self._resuming:
            return "正在重新打开采集…"
        if self._paused:
            if self._video_ended:
                return "视频播放结束（显示最后一帧；点「继续」可重新播放）"
            return "已暂停：画面停在最后一帧，点「继续」重新打开采集"
        if self._startup_error:
            return "感知线程未启动（见上方错误）"
        if worker.error:
            return "感知线程已停止（原因见上方错误）"
        if worker.is_alive():
            if self.last_snapshot is None:
                elapsed = time.monotonic() - self._started_at
                if elapsed >= 5.0:
                    target = "视频" if self.video else "摄像头"
                    return (
                        f"已等待 {elapsed:.0f} 秒仍无画面：正在尝试打开{target}"
                        "（可能被其它程序占用，或 index 不正确）"
                    )
                return "正在启动感知线程（首次加载模型约 1–3 秒）…"
            description = (self.last_snapshot.get("frame") or {}).get("source_description") or ""
            return f"运行中 · {description}" if description else "运行中"
        if self.last_snapshot is None:
            return "感知线程已结束：没有收到任何画面（检查摄像头是否被占用，或视频路径是否正确）"
        if self.video:
            return "视频播放结束（显示最后一帧；点「继续」可重新开始）"
        return "感知线程已结束（显示最后一帧）"

    # ------------------------------------------------------------------ 画面显示
    def _on_preview_resize(self, event: tk.Event) -> None:
        if self._closed or self._last_pil is None:
            return
        size = (max(2, int(event.width)), max(2, int(event.height)))
        if size == self._preview_size:
            return
        self._render_preview(self._last_pil)

    def _show_overlay(self, overlay) -> None:
        try:
            image = bgr_to_pil(overlay)
        except Exception as exc:
            LOGGER.warning("叠加层转换为图像失败：%s", exc)
            return
        self._last_pil = image
        self._render_preview(image)
        self.preview_frames += 1

    def _render_preview(self, image: Image.Image) -> None:
        width = self.preview_label.winfo_width()
        height = self.preview_label.winfo_height()
        if width <= 1 or height <= 1:  # 窗口尚未完成布局，用配置尺寸兜底
            width = max(320, int(self.config.get("ui.width", 1180)) - 400)
            height = max(240, int(self.config.get("ui.height", 720)) - 60)
        self._preview_size = (int(width), int(height))
        copy = image.copy()
        copy.thumbnail((max(2, width - 4), max(2, height - 4)))
        try:
            self._photo = ImageTk.PhotoImage(copy)
            self.preview_label.configure(image=self._photo, text="")
        except tk.TclError:
            pass

    # ------------------------------------------------------------------ 面板更新
    def _update_panel(self, snapshot: dict) -> None:
        torso = snapshot.get("torso") or {}
        posture = str(torso.get("posture") or "unknown")
        posture_zh = str(torso.get("posture_zh") or posture)
        self.torso_label.configure(text=posture_zh, fg=POSTURE_COLORS.get(posture, COLOR_UNKNOWN))
        self.torso_conf_label.configure(text=f"置信度：{self._format_number(torso.get('confidence'), 2)}")

        pitch = self._format_deg(torso.get("pitch_deg"))
        roll = self._format_deg(torso.get("roll_deg"))
        yaw = self._format_deg(torso.get("yaw_deg"))
        facing = str(torso.get("facing") or "unknown")
        facing_zh = FACING_ZH.get(facing, facing)
        self.angle_label.configure(text=f"前倾 {pitch}   侧倾 {roll}   转向 {yaw}\n朝向：{facing_zh}")

        hands = snapshot.get("hands")
        self._render_hands(hands if isinstance(hands, list) else [])

        perf = snapshot.get("perf") or {}
        fps = self._format_number(perf.get("fps"), 1)
        self.fps_label.configure(text=f"FPS：{fps}")
        p95 = self._format_number(perf.get("p95_ms"), 0)
        self.p95_label.configure(text=f"延迟 p95：{p95} ms")
        level = perf.get("level") or "--"
        total = self._format_number(perf.get("total_ms"), 0)
        self.level_label.configure(text=f"降级级别：{level}   当前帧 {total} ms")
        self._refresh_activity(snapshot)
        self._refresh_event_log()
        self.panel_updates += 1

    def _render_hands(self, hands: list) -> None:
        count = len(hands)
        if count:
            self.hand_placeholder.grid_remove()
        else:
            self.hand_placeholder.configure(text="未检测到手（把手放进画面）")
            self.hand_placeholder.grid()
            self.hands_summary = "未检测到手"
        self._ensure_hand_blocks(max(2, count))
        summaries: list[str] = []
        for index, block in enumerate(self._hand_blocks):
            if index >= count:
                block["frame"].grid_remove()
                continue
            hand = hands[index] if isinstance(hands[index], dict) else {}
            label = str(hand.get("label") or "unknown").lower()
            label_zh = HAND_LABEL_ZH.get(label, "手部")
            grip = str(hand.get("grip_zh") or hand.get("grip") or "未知")
            notes: list[str] = []
            zone = str(hand.get("zone") or "unknown")
            if zone in ("A", "B"):
                notes.append(f"{zone} 区")
            if hand.get("in_workspace"):
                notes.append("在操作区")
            if hand.get("busy"):
                notes.append("移动中")
            title = f"{label_zh} · {grip}"
            if notes:
                title += f"（{'、'.join(notes)}）"
            block["title"].configure(text=title)
            fingers = hand.get("fingers") or {}
            for finger in FINGER_ORDER:
                state = str(fingers.get(finger) or "unknown").lower()
                block["cells"][finger].configure(
                    text=f"{FINGER_ZH[finger]}\n{FINGER_STATE_ZH.get(state, state)}",
                    fg=FINGER_STATE_COLOR.get(state, COLOR_UNKNOWN),
                )
            if not block["frame"].winfo_manager():
                block["frame"].grid()
            summaries.append(f"{label_zh}：{grip}")
        if summaries:
            self.hands_summary = " | ".join(summaries)

    # ------------------------------------------------------------------ 情境感知面板
    def _refresh_activity(self, snapshot: dict) -> None:
        activity = snapshot.get("activity") or {}
        if not isinstance(activity, dict):
            activity = {}
        state = str(activity.get("state") or "UNKNOWN").upper()
        state_zh = str(activity.get("state_zh") or ACTIVITY_STATE_ZH.get(state, state))
        self.activity_label.configure(text=state_zh, fg=ACTIVITY_COLORS.get(state, COLOR_UNKNOWN))

        zone = str(activity.get("zone") or "--")
        duration_ms = activity.get("duration_ms")
        duration_text = "--"
        if isinstance(duration_ms, (int, float)):
            seconds = max(0.0, float(duration_ms)) / 1000.0
            duration_text = f"{seconds:.1f}s" if seconds < 60.0 else f"{seconds / 60.0:.1f}min"
        movement = str(activity.get("movement_level") or "unknown").upper()
        movement_zh = MOVEMENT_ZH.get(movement, movement)
        quality = str(activity.get("observation_quality") or "UNKNOWN").upper()
        quality_zh = QUALITY_ZH.get(quality, quality)
        self.activity_info_label.configure(
            text=f"区域 {zone}   持续 {duration_text}   移动 {movement_zh}   质量 {quality_zh}"
        )

        chopping = activity.get("chopping") or {}
        if isinstance(chopping, dict) and chopping.get("detected"):
            rhythm = self._format_number(chopping.get("rhythm_hz"), 1)
            score = self._format_number(chopping.get("score"), 2)
            stable_text = "稳定" if chopping.get("stable") else "未稳定"
            self.chopping_label.configure(text=f"切菜节奏 {rhythm} Hz · {stable_text} · 分数 {score}")
        else:
            self.chopping_label.configure(text="切菜：未检测到节奏")

        last_event = activity.get("event")
        if last_event:
            last_text = EVENT_ZH.get(str(last_event), str(last_event))
        else:
            last_text = "--"
        self.last_event_label.configure(text=f"最近事件：{last_text}")

    def _refresh_event_log(self) -> None:
        worker = self.worker
        if worker is None or self.event_text is None:
            return
        events = worker.recent_events(30)
        signature = tuple((str(item.get("wall_time")), str(item.get("event"))) for item in events)
        if signature == self._event_log_sig:
            return
        self._event_log_sig = signature
        if events:
            lines = [self._event_line(item) for item in reversed(events)]
            text = "\n".join(lines)
        else:
            text = "（暂无事件；有活动变化时写入 outputs/events）"
        self.event_text.configure(state="normal")
        self.event_text.delete("1.0", "end")
        self.event_text.insert("end", text)
        self.event_text.configure(state="disabled")
        self.event_text.yview_moveto(0.0)

    def _event_line(self, event: dict) -> str:
        wall = str(event.get("wall_time") or "")
        if len(wall) >= 19 and "T" in wall:
            stamp = wall[11:19]
        else:
            stamp = wall[:8] or "--:--:--"
        name = str(event.get("event") or "?")
        name_zh = EVENT_ZH.get(name, name)
        zone = str(event.get("zone") or "")
        evidence = event.get("evidence") if isinstance(event.get("evidence"), dict) else {}
        detail = ""
        if evidence.get("rhythm_hz") is not None:
            try:
                detail = f" {float(evidence['rhythm_hz']):.1f}Hz"
            except (TypeError, ValueError):
                detail = ""
        elif evidence.get("from") and evidence.get("to"):
            detail = f" {evidence['from']}→{evidence['to']}"
        zone_text = f" {zone}" if zone else ""
        return f"{stamp}  {name_zh}{zone_text}{detail}"

    # ------------------------------------------------------------------ 分区设置
    def _zone_data_from_snapshot(self) -> dict:
        zones = (self.last_snapshot or {}).get("zones") or {}
        labels = zones.get("labels") if isinstance(zones.get("labels"), dict) else {}
        return {
            "split_x": zones.get("split_x", self.config.get("zones.split_x", 0.5)),
            "deadband": zones.get("deadband", self.config.get("zones.deadband", 0.02)),
            "ema_alpha": self.config.get("zones.ema_alpha", 0.35),
            "enter_ms": self.config.get("zones.enter_ms", 300),
            "labels": {
                "A": labels.get("A", self.config.get("zones.labels.A", "备菜区")),
                "B": labels.get("B", self.config.get("zones.labels.B", "处理区")),
            },
        }

    def _open_zone_dialog(self) -> None:
        if self._closed:
            return
        if self._zone_dialog is not None and self._zone_dialog.winfo_exists():
            self._zone_dialog.lift()
            return

        current: dict = {}
        worker = self.worker
        if worker is not None:
            try:
                current = worker.zone_settings() or {}
            except Exception:
                current = {}
        if not current:
            current = self._zone_data_from_snapshot()
        labels = current.get("labels") if isinstance(current.get("labels"), dict) else {}
        self._zone_vars = {
            "split_x": tk.StringVar(value=f"{float(current.get('split_x', 0.5)):.2f}"),
            "deadband": tk.StringVar(value=f"{float(current.get('deadband', 0.02)):.2f}"),
            "label_a": tk.StringVar(value=str(labels.get("A") or "备菜区")),
            "label_b": tk.StringVar(value=str(labels.get("B") or "处理区")),
        }

        win = tk.Toplevel(self.root)
        win.title("A/B 分区设置")
        win.configure(bg=BG_PANEL)
        win.resizable(False, False)
        win.transient(self.root)
        self._zone_dialog = win
        win.protocol("WM_DELETE_WINDOW", lambda: self._close_zone_dialog(win))

        body = tk.Frame(win, bg=BG_PANEL, padx=14, pady=12)
        body.pack(fill="both", expand=True)
        tk.Label(
            body,
            text="A/B 分区设置",
            font=self._font(13, "bold"),
            fg="#222222",
            bg=BG_PANEL,
            anchor="w",
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))

        tk.Label(body, text="分割线位置（0.10–0.90）", font=self._font(10), fg="#333333", bg=BG_PANEL).grid(
            row=1, column=0, sticky="w", pady=(0, 4)
        )
        split_row = tk.Frame(body, bg=BG_PANEL)
        split_row.grid(row=1, column=1, sticky="w", pady=(0, 4))
        tk.Scale(
            split_row,
            from_=0.10,
            to=0.90,
            resolution=0.01,
            orient="horizontal",
            length=230,
            showvalue=False,
            variable=self._zone_vars["split_x"],
            bg=BG_PANEL,
            highlightthickness=0,
            troughcolor="#d8d8d8",
            sliderlength=18,
        ).pack(side="left")
        tk.Entry(split_row, textvariable=self._zone_vars["split_x"], width=6, font=self._font(10)).pack(
            side="left", padx=(8, 0)
        )

        tk.Label(body, text="死区（0.00–0.10）", font=self._font(10), fg="#333333", bg=BG_PANEL).grid(
            row=2, column=0, sticky="w", pady=(0, 4)
        )
        tk.Entry(body, textvariable=self._zone_vars["deadband"], width=8, font=self._font(10)).grid(
            row=2, column=1, sticky="w", pady=(0, 4)
        )

        tk.Label(body, text="A 区名称", font=self._font(10), fg="#333333", bg=BG_PANEL).grid(
            row=3, column=0, sticky="w", pady=(0, 4)
        )
        tk.Entry(body, textvariable=self._zone_vars["label_a"], width=18, font=self._font(10)).grid(
            row=3, column=1, sticky="w", pady=(0, 4)
        )
        tk.Label(body, text="B 区名称", font=self._font(10), fg="#333333", bg=BG_PANEL).grid(
            row=4, column=0, sticky="w", pady=(0, 4)
        )
        tk.Entry(body, textvariable=self._zone_vars["label_b"], width=18, font=self._font(10)).grid(
            row=4, column=1, sticky="w", pady=(0, 4)
        )

        hint = tk.Label(
            body,
            text="分割线按你看到的画面计算（已包含镜像）；A 在左、B 在右。\n应用只对本次运行生效；保存会写入 config/zones.json，下次启动仍生效。",
            font=self._font(9),
            fg=COLOR_MUTED,
            bg=BG_PANEL,
            justify="left",
            anchor="w",
            wraplength=390,
        )
        hint.grid(row=5, column=0, columnspan=2, sticky="w", pady=(2, 4))

        error_label = tk.Label(
            body, text="", font=self._font(9), fg=COLOR_ERROR, bg=BG_PANEL, anchor="w", wraplength=390
        )
        error_label.grid(row=6, column=0, columnspan=2, sticky="w")

        buttons = tk.Frame(body, bg=BG_PANEL)
        buttons.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        tk.Button(
            buttons,
            text="应用",
            command=lambda: self._zone_dialog_apply(win, error_label, persist=False),
            font=self._font(10),
        ).pack(side="left", padx=(0, 4))
        tk.Button(
            buttons,
            text="保存到配置文件",
            command=lambda: self._zone_dialog_apply(win, error_label, persist=True),
            font=self._font(10),
        ).pack(side="left", padx=(0, 4))
        tk.Button(
            buttons, text="恢复默认", command=self._zone_dialog_reset, font=self._font(10)
        ).pack(side="left", padx=(0, 4))
        tk.Button(buttons, text="关闭", command=lambda: self._close_zone_dialog(win), font=self._font(10)).pack(
            side="left"
        )

        win.update_idletasks()
        x = self.root.winfo_rootx() + max(0, (self.root.winfo_width() - win.winfo_width()) // 2)
        y = self.root.winfo_rooty() + 80
        win.geometry(f"+{x}+{y}")

    def _zone_dialog_reset(self) -> None:
        for key, value in (("split_x", "0.50"), ("deadband", "0.02"), ("label_a", "备菜区"), ("label_b", "处理区")):
            var = self._zone_vars.get(key)
            if var is not None:
                var.set(value)

    def _zone_dialog_apply(self, win: tk.Toplevel, error_label: tk.Label, persist: bool) -> None:
        try:
            split = float(self._zone_vars["split_x"].get())
            deadband = float(self._zone_vars["deadband"].get())
            label_a = self._zone_vars["label_a"].get().strip() or "备菜区"
            label_b = self._zone_vars["label_b"].get().strip() or "处理区"
            if not 0.10 <= split <= 0.90:
                raise ValueError("分割线位置需在 0.10–0.90 之间")
            if not 0.0 <= deadband <= 0.10:
                raise ValueError("死区需在 0.00–0.10 之间")

            worker = self.worker
            base = {}
            if worker is not None:
                try:
                    base = worker.zone_settings() or {}
                except Exception:
                    base = {}
            data = {
                "split_x": split,
                "deadband": deadband,
                "ema_alpha": float(base.get("ema_alpha", self.config.get("zones.ema_alpha", 0.35))),
                "enter_ms": float(base.get("enter_ms", self.config.get("zones.enter_ms", 300))),
                "labels": {"A": label_a, "B": label_b},
            }
            if worker is not None and worker.pipeline is not None:
                worker.apply_zone_settings(data, persist=persist)
                action = "已保存分区配置（config/zones.json）" if persist else "已应用分区配置（下一帧生效）"
            elif persist:
                path = zones_file(self.config)
                save_settings(path, ZoneSettings.from_dict(data))
                action = f"已保存分区配置（{path.name}）"
            else:
                raise RuntimeError("感知线程尚未就绪，只能先保存到配置文件")
            error_label.configure(text="")
            self._set_action(action)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            error_label.configure(text=message)
            self._set_action(f"分区设置失败：{message}")

    def _save_zone_config(self) -> None:
        worker = self.worker
        try:
            if worker is not None and worker.pipeline is not None:
                data = worker.zone_settings()
                if not data:
                    raise RuntimeError("读取当前分区配置失败")
                worker.apply_zone_settings(data, persist=True)
            else:
                data = self._zone_data_from_snapshot()
                save_settings(zones_file(self.config), ZoneSettings.from_dict(data))
            self._set_action("已保存当前分区配置到 config/zones.json")
        except Exception as exc:
            self._set_action(f"保存分区配置失败：{type(exc).__name__}: {exc}")

    def _close_zone_dialog(self, win: tk.Toplevel) -> None:
        if self._zone_dialog is win:
            self._zone_dialog = None
        try:
            win.destroy()
        except tk.TclError:
            pass

    # ------------------------------------------------------------------ 按钮动作
    def _toggle_pause(self) -> None:
        worker = self.worker
        if worker is None:
            self._set_action("感知线程未启动，无法暂停")
            return
        try:
            if not self._paused:
                worker.set_paused(True)
                self._paused = True
                self.pause_button.configure(text="继续")
                self._set_action("已暂停采集（摄像头已释放，继续时会重新打开）")
                return
            if self._resuming:
                return
            # set_paused(True) 后旧线程还要花 1–3 秒关闭 MediaPipe 图；
            # 若立刻 worker.start()，会因旧线程仍存活而静默早退，表现为「继续」后不再出新帧。
            self._resuming = True
            self._resume_attempts = 0
            self.pause_button.configure(text="启动中…", state="disabled")
            self._set_action("正在重新打开采集…")
            self._resume_after_id = self.root.after(0, self._resume_worker)
        except Exception as exc:
            self._set_action(f"暂停/继续失败：{type(exc).__name__}: {exc}")

    def _resume_worker(self) -> None:
        self._resume_after_id = None
        worker = self.worker
        if worker is None or not self._resuming:
            return
        if worker.is_alive():
            self._resume_attempts += 1
            if self._resume_attempts > 100:  # 最多等 10 秒
                self._resuming = False
                self._paused = True
                self.pause_button.configure(text="继续", state="normal")
                self._set_action("继续失败：上一轮采集线程仍未退出，请稍后再试")
                return
            self._resume_after_id = self.root.after(100, self._resume_worker)
            return
        try:
            worker.start()
        except Exception as exc:
            self._resuming = False
            self._paused = True
            self.pause_button.configure(text="继续", state="normal")
            self._set_action(f"继续失败：{type(exc).__name__}: {exc}")
            return
        self._resuming = False
        self._paused = False
        self._video_ended = False
        self.pause_button.configure(text="暂停", state="normal")
        self._set_action("已继续采集")

    def _calibrate(self) -> None:
        worker = self.worker
        if worker is None or worker.pipeline is None:
            self._set_action("标定失败：感知线程尚未就绪")
            return
        torso = (self.last_snapshot or {}).get("torso") or {}
        if not torso.get("present"):
            self._set_action("标定失败：画面里还没有可用的躯干（请先让人站到画面里）")
            return
        try:
            offsets = worker.calibrate()
        except Exception as exc:
            self._set_action(f"标定失败：{type(exc).__name__}: {exc}")
            return
        if not offsets:
            self._set_action("标定失败：没有可用的躯干角度")
            return
        calibration_path = self.config.resolve_path(
            self.config.get("torso.calibration_file", "outputs/calibration.json")
        )
        self._set_action(
            "已标定直立基准："
            f"前倾 {self._format_deg(offsets.get('pitch'))}，"
            f"侧倾 {self._format_deg(offsets.get('roll'))}，"
            f"转向 {self._format_deg(offsets.get('yaw'))}"
            f"（已写入 {calibration_path}）"
        )

    def _clear_calibration(self) -> None:
        worker = self.worker
        if worker is None or worker.pipeline is None:
            self._set_action("清除标定失败：感知线程尚未就绪")
            return
        try:
            worker.clear_calibration()
        except Exception as exc:
            self._set_action(f"清除标定失败：{type(exc).__name__}: {exc}")
            return
        self._set_action("已清除直立基准（角度恢复为绝对测量）")

    def _save_snapshot(self) -> None:
        if self.last_snapshot is None:
            self._set_action("保存失败：还没有收到画面")
            return
        try:
            ensure_dir(self.snapshot_dir)
        except OSError as exc:
            self._set_action(f"保存失败：无法创建目录 {self.snapshot_dir}（{exc}）")
            return
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        milliseconds = int((time.time() % 1) * 1000)
        base = f"snapshot_{stamp}_{milliseconds:03d}"
        json_path = self.snapshot_dir / f"{base}.json"
        jpg_path = self.snapshot_dir / f"{base}.jpg"
        try:
            json_path.write_text(
                json.dumps(self.last_snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            jpg_saved = False
            if self.last_overlay is not None:
                jpg_saved = bool(imwrite_u(jpg_path, self.last_overlay))
        except (OSError, TypeError, ValueError) as exc:
            self._set_action(f"保存失败：{type(exc).__name__}: {exc}")
            return
        message = f"已保存 {json_path.name}"
        if self.last_overlay is not None:
            message += f" + {jpg_path.name}" if jpg_saved else "（标注图写入失败）"
        self._set_action(message)
        LOGGER.info("保存快照：%s", json_path)

    # ------------------------------------------------------------------ 状态标签
    def _set_status(self, text: str) -> None:
        if text != self._last_status_text:
            self._last_status_text = text
            self.status_label.configure(text=text)

    def _set_action(self, text: str) -> None:
        self.action_label.configure(text=text)

    def _label_error(self, text: str) -> None:
        self.error_label.configure(text=text)
        self.error_label.grid()

    def _hide_error(self) -> None:
        self.error_label.grid_remove()
        self.error_label.configure(text="")

    # ------------------------------------------------------------------ 工具
    @staticmethod
    def _format_number(value: object, digits: int) -> str:
        if value is None:
            return "--"
        try:
            number = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return "--"
        if math.isnan(number) or math.isinf(number):
            return "--"
        return f"{number:.{digits}f}"

    @staticmethod
    def _format_deg(value: object) -> str:
        text = KitchenUI._format_number(value, 1)
        return "--" if text == "--" else f"{float(text):+.1f}°"

    def _on_callback_exception(self, exc: type, value: BaseException, tb) -> None:
        text = "".join(traceback.format_exception(exc, value, tb))
        self.callback_errors.append(text)
        LOGGER.error("界面回调异常：\n%s", text)


def run_ui(
    config: Config,
    video: Optional[str] = None,
    index: int = 0,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> int:
    """启动桌面界面并阻塞到窗口关闭；返回进程退出码（0=正常关闭）。"""
    try:
        app = KitchenUI(config, video=video, index=index, width=width, height=height)
    except tk.TclError as exc:
        print(f"无法启动图形界面（Tkinter/Tcl 错误）：{exc}")
        return 1
    except Exception as exc:  # 初始化失败要给出可读原因而不是堆栈
        LOGGER.exception("界面初始化失败")
        print(f"界面初始化失败：{type(exc).__name__}: {exc}")
        return 1
    try:
        app.run()
    except KeyboardInterrupt:
        pass
    finally:
        app.close()
    return 0
