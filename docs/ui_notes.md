# 桌面界面说明（ui.py）

本文说明桌面界面的结构、交互语义、自测记录与遗留问题。当前版本在阶段一（躯干/五指）和阶段二
（A/B 分区、活动状态）之上增加了“两状态演示层”，主面板只回答“正在切菜 / 切菜完成”。代码集中在
`src/kitchen_perception/ui.py`，配套改动在 `worker.py`（快照/事件/分区接口）、
`overlay.py`（分区与演示大标签）与 `pipeline.py`（演示状态、活动、事件）。

## 1. 启动方式

在项目根目录、已设置 PYTHONPATH 的环境下：

```powershell
$env:PYTHONPATH="$PWD\src"

# 默认摄像头 index 0（演示主用法）
py -3.12 -X utf8 -m kitchen_perception ui

# 指定摄像头 index / 分辨率
py -3.12 -X utf8 -m kitchen_perception ui --index 1 --width 1280 --height 720

# 用视频文件代替摄像头（便于重复演示与无摄像头联调）
py -3.12 -X utf8 -m kitchen_perception ui --video outputs/video_review/chopping_720p_30fps.mp4
```

cli.py 的 ui 子命令调用 ui.run_ui(config, video=..., index=..., width=..., height=...)；
正常情况下窗口关闭后 run_ui 返回 0，初始化失败（例如 Tk 不可用）返回 1 并打印可读原因。
视频播完后窗口停在最后一帧，点“继续”从头重播。

## 2. 界面布局

```text
┌──────────────────────────────────────────────┬──────────────────────┐
│                                              │ 切菜状态             │
│   左侧：摄像头/视频画面 + 叠加层              │   正在切菜（30 号）  │
│   - A/B 半透明分区 + 白色分割线 + 重心十字    │   节奏/持续/最近抬放 │
│   - 33 点骨架（overlay.py）                    │ 躯干状态             │
│   - 五指按 伸直/半屈/弯曲 着色                │ 躯干角度             │
│   - 中文 HUD + 底部中央“正在切菜/切菜完成”    │ 手部（含所在区）     │
│   - 等比缩放，随窗口大小自适应                │ 性能                 │
│                                              │ 状态变化（两状态）   │
│                                              │ [暂停][分区设置…]    │
│                                              │ [标定][清除]         │
│                                              │ [保存快照][保存分区] │
│                                              │ 状态行/操作反馈      │
└──────────────────────────────────────────────┴──────────────────────┘
```

右侧面板字段（自上而下）：

- **切菜状态（演示主面板）**：30 号大字显示“等待切菜 / 正在切菜 / 切菜完成”，颜色分别灰/绿/蓝。
  下方：切菜中显示“切菜节奏 1.1 Hz   已持续 0.3s”和“最近抬放 0.0s 前”；完成显示
  “切菜完成   已保持 0.2s”和“最后一次切菜 1.1s 前”；等待显示操作提示；底部“最近状态”。
- **躯干状态**：躯干中文标签（直立绿、前倾/后仰/侧倾橙、俯身红、不确定黄、不可见/无人灰）+ 置信度。
  演示视频中躯干通常不入镜，面板会显示“躯干未入镜（手部可见）”，不再用旧文案报错。
- **躯干角度**：前倾、侧倾、转向（`--` 表示未知）+ 朝向（面向镜头/背对镜头/侧身/未知）；标定后为相对角度。
- **手部**：每只手标题（左手/右手 · 手势 · A 区/B 区）+ 五指状态两行；未检测到手时显示引导文字。
- **性能**：FPS、延迟 p95、降级级别（L0–L3）、当前帧耗时。
- **状态变化**：只记录演示状态切换（最多 12 条，最新在上），格式“20:11:23  正在切菜”；
  尚未切换时显示“（等待状态变化）”。旧阶段二的 ACTIVE_B/TRANSITION 等事件仍写入 JSONL，但不在此面板展示。
- **状态行**：启动中 / 运行中（含视频来源）/ 暂停 / 视频播放结束 / 等待画面 / 线程错误；
  **操作反馈行**显示标定偏移、保存文件名、分区保存结果等。

## 3. 按钮与快捷键

| 操作 | 按钮 | 快捷键 | 行为 |
| --- | --- | --- | --- |
| 暂停/继续 | 暂停 ↔ 继续（启动中…） | 空格 | 暂停停掉采集线程并释放摄像头；继续重新打开 |
| 分区设置 | 分区设置… | Z | 打开 A/B 分区对话框（见 4；演示不强制使用） |
| 标定直立基准 | 标定 | C | 当前 pitch/roll/yaw 记为 0°，写 outputs/calibration.json |
| 清除标定 | 清除 | R | 删除校准文件，恢复绝对角度 |
| 保存快照 | 保存快照 | S | 当前 JSON 快照与叠加图写入 outputs/snapshots/ |
| 保存分区配置 | 保存分区配置 | — | 把当前界面里的分区设置持久化到 config/zones.json |
| 退出 | 关闭窗口 | Q / Esc | 停止线程并关闭窗口 |

保存快照命名：`snapshot_YYYYmmdd_HHMMSS_mmm.json` 与同名 `.jpg`；JSON 为 indent=2 的完整
感知快照（schema 1.1，含 camera_status/zones/activity/demo）。

## 4. 分区设置对话框（阶段二保留）

- **分割线**：Scale + 数值输入，范围 0.10–0.90，表示显示画面中 A/B 分界的归一化 x；
  预览镜像时按“用户看到的画面”计算，向左拖就是把 A 区缩小。
- **死区**：默认 0.02；身体重心落在 split_x ± deadband 内时保持上一次区域，防止边界抖动。
- **A/B 名称**：默认“备菜区/处理区”，可改成“左灶台/右灶台”等。
- **应用**：立即作用于后台 pipeline 并刷新叠加层，不写文件；**保存到配置文件**额外写 config/zones.json，
  下次启动由 load_settings 覆盖默认值；**恢复默认**把界面值恢复为 0.5/0.02。
- 校验：分割线必须在 0.10–0.90、死区在 0–0.10；非法值弹提示且不应用。
- 无后台 worker（例如视频已结束）时，保存按钮直接写配置文件，保证设置不丢。
- 两状态演示只看手的切菜节奏，不要求分区正确；A/B 叠加层和该对话框用于后续阶段二演示。

## 5. 交互语义与边界状态

- **线程模型**：Tkinter 主线程只做绘制与按钮响应；PerceptionWorker 后台线程独占摄像头与
  MediaPipe 对象。UI 每 ui.poll_ms（默认 30 ms）轮询 worker.get_latest()，有新帧才重绘预览，
  面板每 ui.panel_update_every_frames（默认 2）帧更新一次。
- **两状态判定**：DemoStateTracker 只读最近 3 秒的手部运动特征；当 reversals、节奏频率、振幅比、
  分数同时达到 demo 阈值并连续满足 start_confirm_ms（300 ms）才进入“正在切菜”。“正在切菜”期间
  检测短暂丢失时靠 900 ms 的完成确认保持，超过 done_after_ms（900 ms）无有效节奏才转“切菜完成”；
  完成后再次连续切菜可回到“正在切菜”。阈值在 config/default.json 的 demo 节，与 activity 独立。
- **暂停/继续**：worker.set_paused(True) 后旧线程还要 1–3 秒关闭 MediaPipe 图；
  若立刻 worker.start()，start() 会因旧线程仍存活而静默早退。UI 因此用异步等待：
  点“继续”后按钮显示“启动中…”并禁用，每 100 ms 检查旧线程是否退出，退出后再启动，最长等 10 秒。
- **视频播完**：worker 线程正常结束 → 自动进入暂停态，按钮变“继续”，状态行提示“视频播放结束”；
  点击从头重播。
- **无画面等待**：摄像头打开在开发机上需要 5–7 秒，5 秒内显示“正在启动感知线程”，
  超过 5 秒显示“已等待 N 秒仍无画面…（可能被其它程序占用，或 index 不正确）”。
- **错误显示**：worker.error 非空时在面板顶部显示红色错误标签，状态行提示线程已停止；
  --video 路径不存在时启动即提示，不创建后台线程。
- **窗口自适应**：窗口宽高取 min(config.ui.width, 屏幕宽-80) / min(900, 屏幕高-120)，
  最小 900×650；左侧画面按控件实际尺寸等比缩放（Pillow thumbnail），保留 PhotoImage 引用避免被 GC。
  这样在 1366×768 演示机上右侧面板不会被裁掉。
- **叠加层语义**：左侧底部中央大标签显示与右面板相同的两状态；已完成/等待时抑制旧的
  “未检测到人物/把手放进画面”警告；躯干不可见但双手稳定可见时 HUD 显示“躯干未入镜（手部可见）”。

## 6. 自测记录

环境：Python 3.12.10、Pillow 12.3.0、OpenCV 5.0.0、Windows；摄像头为 Integrated Camera（index 0）。
临时脚本在 outputs/ui_smoke/、outputs/video_demo/ 或 scratch/（后两者被 .gitignore 忽略）。

### 6.1 导入与编译

```powershell
$env:PYTHONPATH="$PWD\src"
py -3.12 -X utf8 -c "import kitchen_perception.ui"
py -3.12 -m py_compile src/kitchen_perception/ui.py
```

### 6.2 摄像头显示（阶段一，index 0，15 秒）

py -3.12 -X utf8 scratch/ui_selftest.py --seconds 15

结果：窗口建立；感知帧 155、预览刷新 152、面板更新 77；躯干“前倾”置信度 0.61；
FPS 16.2、延迟 p95 52 ms、L0；error 为空、Tk 回调异常为空；窗口由脚本关闭。

### 6.3 视频源与全部按钮（阶段一）

py -3.12 -X utf8 scratch/ui_exercise.py（视频源 640×480、240 帧、30 fps）

结果：标定写入 outputs/calibration.json；保存快照生成 .json/.jpg；暂停/继续正常；
清除标定删除文件；Tk 回调异常为空；窗口关闭。

### 6.4 阶段二 UI smoke（空场景，2026-11-11）

py -3.12 -X utf8 outputs/ui_smoke/run_ui_smoke.py

脚本启动 KitchenUI(index=0)，8 秒后截图，9 秒打开分区设置对话框，11 秒关闭并退出。结果：
感知帧 42、面板更新 21 次、快照可生成、worker 存活、latest_age 15 ms、状态“运行中 camera:0 dshow”、
分区对话框 open/close 均为 True。截图 outputs/ui_smoke/ui_smoke.png 可见：左侧画面 + A/B 分区，
右侧切菜/躯干/手部/性能面板；空场景下活动为“等待切菜”、躯干为“未检测到人物”，属预期。

### 6.5 空场景误检修复

同一 smoke 首次运行时，画面里挂着的衣服被 MediaPipe 判成人体（pitch −169.5°、roll 137.7°、
confidence 1.0，显示“后仰”）。已在 torso.py 增加角度可信范围门限（|pitch|>110° 或 |roll|>80°
→ uncertain），并增加 plausible_pitch_deg/plausible_roll_deg 配置与单测。
修复后空场景躯干为“未检测到人物/uncertain”，活动 UNKNOWN。

### 6.6 进程退出说明

本机（受限网络）MediaPipe 原生库在解释器退出时会尝试上传 Clearcut 遥测，导致进程延迟约
40–45 秒退出。已修复：engines.close() 非阻塞（最多等 1.5 s），__main__.py 结束后
os._exit，tests/run_tests.py 同样处理；实测真实 UI 窗口关闭后进程约 4.7 秒退出。

### 6.7 两状态真实视频演示（最终证据，2026-11-11）

脚本：py -3.12 -X utf8 outputs/video_demo/run_ui_video.py

输入为侧身切菜固定机位视频（原始 3840×2160/59.82 fps，分析副本 1280×720/30 fps/551 帧）。
脚本用 KitchenUI(video=...) 启动窗口，在状态首次变为 chopping 和 done 后各等 800 ms 再截图，
同时保存 last_snapshot。最终结果：

- 无头 run：551 帧、55.0 s，p50 60.9 ms / p95 81.3 ms / max 103.3 ms，L0；
  waiting 0–2.800 s、chopping 2.833–17.333 s、done 17.367–18.333 s；只切换 2 次。
- UI 截图：outputs/video_demo/ui_video_chopping.png（感知帧 95，标签“正在切菜”，
  节奏 1.1 Hz，已持续 0.3 s，状态变化“20:11:23 正在切菜”）与 ui_video_done.png
  （感知帧 528，标签“切菜完成”，已保持 0.2 s，最后一次切菜 1.1 s 前；状态变化只有
  “20:12:07 切菜完成 / 20:11:23 正在切菜”两行）。
- 截图同名 JSON 与无头 run_snapshots_final.jsonl 中 demo.state 分别为 chopping/done；
  done 阶段叠加层只显示“切菜 切菜完成”，不再出现旧警告或“节奏 0.0Hz 未稳定”。
- 结论：在这段侧身只露手的视频上，界面能真实显示“正在切菜”与“切菜完成”，满足视频演示要求；
  不代表正面机位、全厨房、多人或不同光照下的准确率。

## 7. 已知限制与遗留问题

1. **暂停→继续有 1–3 秒重启延迟**：属 worker 关闭/新建 MediaPipe 图的预期行为，UI 用“启动中…”提示。
2. **坏摄像头 index 报错慢**：worker 对打开失败按帧重试，约 40–60 秒才写 error；UI 只能在 5 秒后提示。
3. **两状态演示只验证了 1 段侧身视频**：切菜阈值是规则判定，正面机位、全厨房、多人、不同光照/服装
   仍需复核；误报/漏报时优先调 config/default.json 的 demo 节。
4. **演示不判断切什么/切多少**：只回答正在切/切完，不识别食材、步骤或菜刀位置。
5. **A/B 只按身体重心**：不做鼠标框选矩形；两状态演示不依赖分区，但叠加层仍显示 A/B 和重心。
6. **事件日志最多显示 12 条两状态切换**：完整活动事件仍在 outputs/events/*.jsonl；UI 不做历史翻页。
7. **未实现**：全屏/画中画、录制视频、多摄像头切换、深色主题、活动时间轴图表。
8. **视频源不循环**：播完停留在最后一帧，点“继续”从头重播。
9. **窗口下限**：900×650；更小屏幕会压缩左侧预览，右侧面板仍完整但可能拥挤。

## 8. 改动范围

- 阶段一：ui.py、docs/ui_notes.md。
- 阶段二：ui.py（活动面板/事件日志/分区对话框/窗口自适应）、worker.py（recent_events/apply_zone_settings）、
  overlay.py（A/B 分区与重心）、pipeline.py（活动/事件/分区接入）、cli.py（simulate）。
- 两状态演示（2026-11-11）：新增 demo.py 与 tests/test_demo.py；ui.py（切菜状态主面板、两状态历史、
  视频证据脚本）、pipeline.py（demo 接入、手可见即人在）、overlay.py（底部大标签/警告抑制）、
  types.py（snapshot.demo）、config/default.json（demo 节）、README/docs、tests。
- 未改动：models/、tests/ 中的既有断言（只增不删）。