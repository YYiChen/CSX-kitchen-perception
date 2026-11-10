# 桌面界面说明（ui.py）

本文说明 T3 桌面界面的结构、交互语义、自测记录与遗留问题。实现只涉及
src/kitchen_perception/ui.py 与本文档，未修改核心模块、配置与模型。

## 1. 启动方式

在项目根目录、已设置 PYTHONPATH 的环境下：

```powershell
$env:PYTHONPATH="$PWD\src"

# 默认摄像头 index 0
py -3.12 -X utf8 -m kitchen_perception ui

# 指定摄像头 index / 分辨率
py -3.12 -X utf8 -m kitchen_perception ui --index 1 --width 1280 --height 720

# 用视频文件代替摄像头（便于无摄像头联调与重复演示）
py -3.12 -X utf8 -m kitchen_perception ui --video assets/test/official/pose.mp4
```

cli.py 的 ui 子命令调用 ui.run_ui(config, video=..., index=..., width=..., height=...)；
正常情况下窗口关闭后 run_ui 返回 0，初始化失败（例如 Tk 不可用）返回 1 并打印可读原因。

## 2. 界面布局

```text
┌──────────────────────────────────────────────┬──────────────────────┐
│                                              │ [错误提示（红，默认隐藏）] │
│   左侧：摄像头/视频画面 + 叠加层              │ 躯干状态             │
│   - 33 点骨架（由 overlay.py 绘制）           │   前倾（大号中文标签） │
│   - 五指按 伸直/半屈/弯曲 着色                │   置信度 0.86        │
│   - 中文 HUD（overlay.py）                    │ 躯干角度             │
│   - 等比缩放，随窗口大小自适应                │   前倾 +24.2° …      │
│                                              │   朝向：面向镜头      │
│                                              │ 手部                 │
│                                              │   左手 · 张开        │
│                                              │     五指状态两行      │
│                                              │ 性能                 │
│                                              │   FPS / 延迟 p95 / L0 │
│                                              │ [暂停][标定][清除][保存]│
│                                              │ 状态行 / 操作反馈      │
└──────────────────────────────────────────────┴──────────────────────┘
```

右侧面板字段：

- **躯干状态**：躯干中文标签（按状态着色：直立绿、前倾/后仰/侧倾橙、俯身红、不确定黄、
  不可见/无人灰）+ 置信度（2 位小数）。
- **躯干角度**：前倾、侧倾、转向（正值角度、`--` 表示未知）+ 朝向（面向镜头/背对镜头/侧身/未知）。
  标定直立基准后显示相对角度。
- **手部**：每只手一行标题（左手/右手 · 手势）+ 五指状态（伸直/半屈/弯曲/未知，按状态着色）；
  标注是否在操作区/移动中；未检测到手时显示引导文字。
- **性能**：FPS、延迟 p95、降级级别（L0–L3）、当前帧处理耗时。
- **状态行**：启动中 / 运行中（显示视频来源描述）/ 暂停 / 视频播放结束 / 等待画面提示 /
  线程错误；**操作反馈行**显示标定偏移量、保存文件名等。

## 3. 按钮与快捷键

| 操作 | 按钮 | 快捷键 | 行为 |
| --- | --- | --- | --- |
| 暂停/继续 | 暂停 ↔ 继续（启动中…） | 空格 | 暂停会停掉后台采集线程（释放摄像头）；继续会重新打开 |
| 标定直立基准 | 标定 | C | 把当前 pitch/roll/yaw 记为 0°，写入 outputs/calibration.json |
| 清除标定 | 清除 | R | 删除校准文件，恢复绝对角度 |
| 保存快照 | 保存 | S | 把当前 JSON 快照与叠加图写入 outputs/snapshots/ |
| 退出 | 关闭窗口 | Q / Esc | 停止线程并关闭窗口 |

保存文件命名：`snapshot_YYYYmmdd_HHMMSS_mmm.json` 与同名 `.jpg`；JSON 为 indent=2 的
完整感知快照，图片为带叠加层的当前帧（中文路径安全写入）。

## 4. 交互语义与边界状态

- **线程模型**：Tkinter 主线程只做绘制与按钮响应；PerceptionWorker 后台线程独占摄像头与
  MediaPipe 对象。UI 每 `ui.poll_ms`（默认 30 ms）轮询 worker.get_latest()，有新帧才重绘预览，
  面板每 `ui.panel_update_every_frames`（默认 2）帧更新一次，避免无谓刷新。
- **暂停/继续**：`worker.set_paused(True)` 后旧线程还要 1–3 秒关闭 MediaPipe 图；
  若立刻 `worker.start()`，start() 会因旧线程仍存活而静默早退（表现为「继续」后不再出新帧）。
  UI 因此用异步等待：点「继续」后按钮显示「启动中…」并禁用，每 100 ms 检查旧线程是否退出，
  退出后再启动，最长等 10 秒；超时则恢复「继续」按钮并提示稍后再试。
- **视频播完**：worker 线程正常结束 → 自动进入暂停态，按钮变「继续」，状态行显示
  「视频播放结束（显示最后一帧；点「继续」可重新播放）」，点击即从头重播。
- **无画面等待**：摄像头打开在开发机上需要 5–7 秒，5 秒内状态行显示「正在启动感知线程」，
  超过 5 秒显示「已等待 N 秒仍无画面：正在尝试打开摄像头…（可能被其它程序占用，或 index 不正确）」。
- **错误显示**：worker.error 非空时在面板顶部显示红色错误标签，状态行提示线程已停止；
  错误消失（如重启后）自动隐藏。`--video` 路径不存在时启动即显示「视频文件不存在」，
  不创建后台线程，窗口仍可正常关闭。
- **窗口缩放**：左侧画面按控件实际尺寸等比缩放（Pillow thumbnail），窗口未完成布局时用
  config 的 ui.width/ui.height 兜底；保留 PhotoImage 引用避免被 GC。

## 5. 自测记录

测试脚本位于 scratch/（已被 .gitignore 忽略，不进入仓库）。环境：Python 3.12.10、
Pillow 12.3.0、OpenCV 5.0.0、Windows；摄像头为 Integrated Camera（index 0）。

### 5.1 导入与编译

```powershell
$env:PYTHONPATH="$PWD\src"
py -3.12 -X utf8 -c "import kitchen_perception.ui"   # ui import ok
py -3.12 -m py_compile src/kitchen_perception/ui.py  # 通过
```

### 5.2 摄像头显示（index 0，15 秒）

`py -3.12 -X utf8 scratch/ui_selftest.py --seconds 15`

结果：窗口建立；感知帧 155、预览刷新 152、面板更新 77；躯干「前倾」置信度 0.61；
角度与朝向正常；FPS 16.2、延迟 p95 52 ms、L0；error 为空、Tk 回调异常为空；窗口由脚本关闭。
（摄像头打开本身约需 5–7 秒，属 sources/OpenCV 行为，与 UI 无关。）

### 5.3 视频源与全部按钮（交互时间线自测）

`py -3.12 -X utf8 scratch/ui_exercise.py`（视频源 scratch/ui_test.mp4：640×480、240 帧、30 fps，
由 assets/test/official/pose.jpg 生成）

结果（最后一次通过）：

- 标定：写入 outputs/calibration.json，偏移 pitch +5.4°、roll +1.9°、yaw +22.1°，操作反馈显示路径；
- 保存快照：生成同名 .json/.jpg，JSON 含 torso，帧号 28；
- 暂停：按钮变「继续」、paused=True、暂停期间只多 1 帧；
- 继续：点击瞬间按钮「启动中…」；恢复后新增预览 27 帧、面板 14 次，按钮回「暂停」；
- 清除标定：文件删除，角度恢复绝对测量；
- 测试结束自动清理本次生成的校准文件与快照；Tk 回调异常为空；窗口关闭。

### 5.4 视频播放结束分支

`py -3.12 -X utf8 scratch/ui_selftest.py --video scratch/ui_short.mp4 --seconds 10`
（scratch/ui_short.mp4：60 帧、2 秒）

结果：播放到第 60 帧后线程结束；状态行「视频播放结束（显示最后一帧；点「继续」可重新播放）」；
按钮变「继续」、paused=True、video_ended=True；无异常。

### 5.5 错误分支

- 视频路径不存在：`--video scratch/missing_file.mp4 --seconds 3` → 错误标签「视频文件不存在：
  scratch/missing_file.mp4」、状态「感知线程未启动（请检查 --video 路径）」、无异常、窗口正常关闭。
- 坏摄像头 index：`--index 99 --seconds 75` → 错误标签「错误：无法打开摄像头 index=99
  （已尝试 ['dshow', 'msmf', 'any']）」、状态「感知线程已停止（原因见上方错误）」、无崩溃。
  注意：核心 worker 对坏 index 按帧重试、约 40–60 秒才置 error，此前 UI 显示等待提示（见 6.2）。

### 5.6 进程退出说明

本机（受限网络）MediaPipe 原生库在解释器退出时会尝试上传 Clearcut 遥测，导致进程延迟约
40–45 秒退出。Lead 已修复：`engines.close()` 改为非阻塞（最多等 1.5 s），`__main__.py`
在命令结束后 `os._exit`，`tests/run_tests.py` 同样处理；实测真实 `python -m kitchen_perception ui`
窗口关闭后进程约 4.7 秒退出（等 worker 收尾），不再出现 40 秒残留。

## 6. 已知限制与遗留问题

1. **暂停→继续有 1–3 秒重启延迟**：worker.set_paused(True) 后旧线程关闭 MediaPipe 图较慢；
   UI 用「启动中…」提示并异步等待，属预期行为。若要瞬时恢复需改 worker 的暂停实现（如不关闭图），
   超出本任务文件范围。
2. **坏摄像头 index 报错慢**：worker 对打开失败按帧重试，约 40–60 秒才写入 error；UI 只能在
   5 秒后给出「正在尝试打开摄像头」提示。若要更快报错需改 worker.py 的重试策略（不在本任务范围）。
3. **核心模块 bug（Lead 已修复）**：media_io.write_video_u 原先在 OpenCV 5/Windows 上先
   shutil.move 再 writer.release()，触发 PermissionError [WinError 32]；现改为先 release
   再移动，并加退避重试（中文路径实测通过）。
4. **未实现（阶段一不需要）**：全屏/画中画、录制视频、多摄像头切换、设置界面、鼠标框选操作区；
   窗口最小尺寸由布局固定为约 900×560，更小会滚动/截断面板。
5. **视频源不循环**：播完停留在最后一帧，点「继续」从头重播；需要自动循环时用
   PerceptionWorker(loop_video=True) 的现成能力，UI 暂未暴露开关。

## 7. 改动范围

- 新增/修改：src/kitchen_perception/ui.py（约 780 行，Tkinter 界面）、docs/ui_notes.md（本文）。
- 未改动：config/、src/kitchen_perception 其它模块、models/、tests/、README。
- 临时自测脚本与测试视频在 scratch/（.gitignore 忽略），不随仓库提交。
