# 架构与设计说明（阶段一 + 阶段二）

## 1. 目标与边界

阶段一目标：默认摄像头前的**躯干状态**与**五指曲张状态**实时识别，延迟硬上限 500 ms。
阶段二目标：在固定机位下增加**A/B 操作区**与**切菜节奏/活动状态**，为后续决策/反馈模型提供结构化输入。
演示版目标（2026-11-11）：在阶段二切菜节奏的基础上，只输出**等待切菜 / 正在切菜 / 切菜完成**三个状态，供侧身切菜视频/固定机位演示稳定显示。

明确不做：菜刀/食材等物品识别、多摄像头、训练自定义模型、鼠标框选矩形操作区、完整活动分类（炒、洗、盛等）。活动状态只回答“在哪个区、是否在切、是否暂停/离开/遮挡”。

## 2. 数据流

```text
CameraSource/VideoFileSource
        │  frame(BGR) + ts_ms
        ▼
Pipeline.process
        ├─ PoseEngine.detect  (VIDEO 模式, 33 landmarks + world landmarks)
        ├─ HandEngine.detect  (VIDEO 模式, 每手 21 landmarks, 最多 2 手)
        ├─ TorsoAnalyzer.analyze   → torso 状态（pitch/roll/yaw/facing/bbox/置信度）
        ├─ HandAnalyzer.analyze    → 每手五指状态 + 手势
        ├─ hands_in_workspace / busy / hands_state
        ├─ ZoneManager.update      → 身体重心、A/B 区、手部 zone、警告
        ├─ HandMotionTracker.update→ 每只手/手腕的切菜节奏特征
        ├─ ActivityStateMachine    → activity 状态 + 确认事件
        ├─ EventLogger.write       → outputs/events/*.jsonl（失败不影响检测）
        ├─ LatencyBudget.observe   → L0–L3 自适应降级
        └─ PerceptionSnapshot.to_dict() → JSON 快照（schema 1.1）
                 │
                 ├─ OverlayRenderer.render → 叠加帧（骨架+手部+A/B 分区+HUD）
                 └─ UI（PerceptionWorker 线程） / CLI / JSONL / 快照文件
                          │
                          └─ VideoExportJob（独立后台线程，导出时暂停预览）
                               └─ 逐帧标注 → mp4v MP4（无音轨）
```

线程模型：UI 主线程只做 Tkinter 绘制与按钮；`PerceptionWorker` 后台线程独占摄像头与 MediaPipe 对象（MediaPipe Tasks 非线程安全）。UI 通过 `worker.apply_zone_settings()` 把分区配置交给后台线程，不直接碰 MediaPipe。视频导出时 UI 先停掉预览 worker，再由 `VideoExportJob` 线程创建独立 `Pipeline`，避免两份 MediaPipe 图同时运行。

## 3. 模块职责

| 模块 | 职责 |
| --- | --- |
| config.py | 默认配置 + 用户配置 + overrides 深合并；路径解析 |
| sources.py | 摄像头枚举/打开/重连；视频文件中文路径处理；时间戳 |
| engines.py | 模型校验（sha256）；Pose/Hand Landmarker 封装；时间戳单调保护 |
| torso.py | world/2D 几何特征、状态分类、EMA 平滑、滞回、直立基准标定 |
| fingers.py | 五指关节角/曲率/距离比，曲张分类，手势分类 |
| zones.py | A/B 分区：身体重心/掌心投影、镜像感知、死区、设置持久化 |
| motion.py | 切菜节奏：轨迹采样、方向反转、频率/振幅/稳定性评分 |
| activity.py | 活动状态机：状态、延时、事件去重、质量标记 |
| demo.py | 两状态演示层：把切菜节奏压成等待/正在切菜/切菜完成；确认/保持时间可配 |
| events.py | 事件 JSONL 追加写 + 内存最近事件 |
| overlay.py | 骨架绘制、五指状态着色、A/B 分区、中文 HUD、镜像预览 |
| pipeline.py | 组合推理、分区/活动/事件、降级调度、快照、标定持久化 |
| worker.py | 后台采集/推理线程，供 UI 轮询最新结果与读取事件 |
| exporter.py | 后台视频导出：逐帧跑 Pipeline、写带标注 MP4、进度/取消/错误状态 |
| ui.py | Tkinter 桌面界面（摄像头/视频源切换、情境感知、事件日志、分区设置与开关、视频导出） |
| cli.py | verify-models / probe-cameras / ui / run / benchmark / snapshot / simulate / export-video |
| custom_rules.py | 按炒菜环节覆盖躯干/手指定义的扩展点 |

## 4. 躯干状态定义

关键点（MediaPipe Pose 33 点）：鼻尖 0、双肩 11/12、双髋 23/24、双膝 25/26。

- **world landmarks**：原点在髋中心、单位约米；用“髋中点 → 肩中点”向量 `up` 计算：
  - pitch = atan2(up.z × front, −up.y)，front = sign(鼻尖.z − 肩中点.z)；正值 = 前倾
  - roll = atan2(up.x, −up.y)；正值 = 上身向图像右侧倾斜
  - yaw = atan2(肩线.z, |肩线.x|)；0 = 正对/正背，±90 = 侧身
- **2D 兜底**：world 缺失但髋部可见时，只用肩-髋连线在图像内的倾角判断侧倾
- **可见性策略**：双肩可见即可分析；髋部不在画面内时用 world 髋位置估算 pitch，置信度乘 `hip_visibility_penalty`（默认 0.85）并标注“估算”
- **可信范围门限**：|pitch| > 110° 或 |roll| > 80° 时输出 `uncertain`（低置信度），避免把衣架/椅背等误检成“倒置人体”
- 状态优先级：bent_over(≥45°) > leaning_forward(≥15°) > leaning_backward(≤−15°) > leaning_side(|roll|>10°) > upright
- 平滑：EMA(α=0.3)；滞回：连续 5 帧同候选才切换显示状态
- 标定：点“标定直立基准”把当前 pitch/roll/yaw 记为 0°，保存到 `outputs/calibration.json`

阈值全部在 `config/default.json → torso` 与 `torso.thresholds`。

## 5. 手指状态定义

每根手指取 MCP/PIP/DIP（拇指为 MCP/IP/TIP）：

- pip_deg：PIP 关节内角（180° = 伸直）；dip_deg：DIP 关节内角
- curl_deg = (180 − pip_deg) + (180 − dip_deg)
- 非拇指：curl ≤ 40° 且 pip ≥ 150° → extended；curl ≥ 100° 或 pip ≤ 110° → flexed；否则 partial
- 拇指：IP 内角 ≥ 150° 且 拇指尖→食指 MCP 距离/掌长 ≥ 0.65 → extended；IP ≤ 110° 或距离比 ≤ 0.35 → flexed；否则 partial
- 手势：五指全伸 = open；四指全屈且拇指屈 = fist；拇指尖-食指尖距离/掌长 ≤ 0.4 且食指非屈 = pinch；仅食指伸 = point；其余 = partial；有效手指 <3 = unknown

阈值在 `config/default.json → hands.finger_thresholds`。默认值用官方测试图实测标定（open 手 curl 1–6°、fist curl 190–200°、拇指距离比 open 0.7–0.9 / fist 0.1）。

## 6. A/B 分区（阶段二）

坐标约定：`split_x` / `body_center_norm` / `palm_norm` 都按**显示画面**的归一化 x（0 左、1 右）；预览镜像开启时，原始 landmark x 会先 `1 - x` 再参与判定，所以用户拖分割线看到的方向与画面一致。

- **身体重心来源**（`estimate_body_center`，按优先级）：
  1. 双髋 23/24，可见度 ≥ `body_visibility_min`（默认 0.3）：取中点，`source=hips`
  2. 双肩 11/12：中点并向画面下方偏移，`source=shoulders`
  3. 躯干 bbox 中心，`source=bbox`
  4. 都不可用：`None`，zone 保持上一次/unknown
- **掌心中心**：手部关键点 0/5/9/13/17 的均值；用于手部 zone 和“手与身体不同区”提示
- **判定**：`classify_x(view_x, settings, previous)`：`view_x < split_x - deadband` → A，`> split_x + deadband` → B，死区内保持 `previous`
- **平滑/确认**：身体重心 EMA(`ema_alpha=0.35`)；候选区连续 `enter_ms=300` 才提交，避免边界抖动
- **配置**：`config/default.json → zones`；界面保存后写 `config/zones.json`（字段 split_x / deadband / ema_alpha / enter_ms / labels），下次启动由 `load_settings` 覆盖默认值
- **输出**：snapshot.zones 含 split_x/deadband/view_mirrored/labels/body_center_norm/body_zone/zone_source/zone_confidence/hand_zone_mismatch/hands/warnings；HandState 增加 palm_norm/zone/zone_code

## 6.5 视频导入、显示开关与导出（2026-11-11）

界面支持两种视频源：摄像头（默认）与本地视频。视频源切换由 UI 先停掉旧 `PerceptionWorker`，再按新源重建；视频帧按 `video.max_width` 等比缩放（默认 1280），并按 `video.max_fps`（默认 30）抽帧，源 60 fps 时步长为 2。视频结束不会自动循环，界面进入“已暂停”，点「继续」从头播放。

`VideoExportJob` 是独立后台任务，构造参数为 config、输入视频、输出路径、`max_width`、`max_fps`、`zone_settings`、`max_frames`；其 `status()` 返回 state/frames/total_frames/progress/fps/stride/elapsed_s/output_path/error。导出流程：

1. UI 停预览 worker，读取当前分区设置（含 enabled）传给导出任务；
2. 导出线程创建 `VideoFileSource` + `Pipeline(enable_overlay=True)`，与预览共用同一套标注/状态逻辑；
3. 逐帧抽帧、缩放、处理，写 mp4v MP4；输出 fps = 源 fps / 抽帧步长，宽高取偶；
4. 先写 ASCII 临时文件，完成后移动到目标路径（支持中文路径）；取消或失败时清理临时目录；
5. 输出无音轨，不依赖 ffmpeg/额外编码器。

分区显示开关是 `ZoneSettings.enabled`（默认 true），随 `config/zones.json` 持久化；`ZoneManager.update()` 会把 enabled 写进 snapshot.zones，`OverlayRenderer.render()` 在 enabled=false 时跳过分割线、A/B 底色和手部区域标记，但后台判区仍继续计算，便于以后只隐藏视觉提示而不影响活动状态。

## 7. 切菜节奏（阶段二）

`HandMotionTracker` 为每只手维护一条轨迹；输入是 `MotionObservation`（掌心优先，无手时用姿态手腕 15/16 兜底，`source=palm|wrist`）：

- **速度**：相邻帧位移 / 真实 dt，EMA 平滑；`movement_level` 按 `motion.low_speed=0.08` / `high_speed=0.35` 分 LOW/MEDIUM/HIGH
- **反转**：y 方向（归一化画面，向下为正）连续运动超过 `reversal_deadband_norm=0.0015` 才计一次反转，记录反转时刻与行程
- **窗口**：最近 `window_s=2.5` 秒；`rhythm_hz` = 反转次数 / 窗口时长 / 2；`amplitude_ratio` = 平均单程行程 / 画面高度；`period_cv`、`amplitude_cv` 衡量节奏稳定性
- **判定 detected**：`reversals >= min_reversals=4`、频率在 `0.8–4.0 Hz`、`amplitude_ratio >= 0.04`、`period_cv <= 0.35`、`amplitude_cv <= 0.45`，且手部 score ≥ `min_hand_score=0.5`；`stable` 另要求稳定窗口
- **启动/停止**：need `start_confirm_ms=600` 连续满足才 confirmed；停止 `stop_after_ms=1500` 无节奏才判停；手切换/断手超过 `reset_gap_ms=800` 重置轨迹，防止把左右手拼接成假节奏
- **输出**：`chopping` 字典（key/label/valid/source/speed/movement_level/reversals/rhythm_hz/amplitude_ratio/period_cv/amplitude_cv/score/detected/stable/last_chop_ms/sampling_gap_ms）

阈值全部在 `config/default.json → chopping`。

## 8. 活动状态机（阶段二）

每帧把视觉/运动结果汇总成 `ActivityObservation`，交给 `ActivityStateMachine.update()`；输出 snapshot.activity，并用 `take_events()` 取走本帧确认事件。

| 状态 | 含义 | 进入条件（简述） |
| --- | --- | --- |
| UNKNOWN | 等待观察/信息不足 | 初始；无人但未满 AWAY 阈值；角度不可信等 |
| ACTIVE_A / ACTIVE_B | 在 A/B 区切菜 | 切菜 detected + 身体/切菜区在 A/B，持续 `start_confirm_ms` |
| MOVING | 手在动但不像切菜 | 手可见、无切菜节奏、movement HIGH/MEDIUM，持续 `enter_ms` |
| PAUSED | 手可见但停止切菜 | 手可见、无切菜、movement LOW，持续 `pause_after_ms=2000` |
| TRANSITION_A_TO_B / B_TO_A | 从一区切到另一区 | 之前 ACTIVE 离开 ≤ `transition_window_ms=8000`，且切入另一区；保持 `transition_hold_ms=1200` 后进 ACTIVE |
| OCCLUDED | 手部遮挡/观察不足 | 人可见但连续 `occluded_after_ms=800` 无有效手 |
| AWAY | 暂未观察到用户 | 连续 `away_after_ms=5000` 无人/无活动 |

事件：`SESSION_STARTED / ACTIVITY_A_STARTED / ACTIVITY_B_STARTED / ACTIVITY_PAUSED / HAND_MOVING / ZONE_TRANSITION_A_TO_B / ZONE_TRANSITION_B_TO_A / CHOPPING_STARTED / CHOPPING_STOPPED / HAND_OCCLUDED / USER_AWAY / USER_RETURNED / CAMERA_DISCONNECTED / ZONE_CONFIG_CHANGED`。

防抖：`min_duration_ms=500` 内不因抖动离开新状态；事件冷却 `event_cooldown_ms=3000`；OCCLUDED/AWAY 例外，安全优先。质量 `observation_quality`：VALID / LOW / OCCLUDED / NO_PERSON / UNKNOWN。

## 8.5 两状态演示层（DemoStateTracker，2026-11-11）

演示版不替换阶段二活动状态机，而是并行增加一个很薄的 DemoStateTracker，专门回答“现在是不是在切菜、切完没有”：

| 状态 | 中文 | 进入条件 |
| --- | --- | --- |
| waiting | 等待切菜 | 初始；尚未连续确认切菜节奏 |
| chopping | 正在切菜 | Detected(best) 连续满足 demo.start_confirm_ms=300 |
| done | 切菜完成 | 进入 chopping 后，连续 demo.done_after_ms=900 未再检测到切菜节奏 |

Detected 直接复用 HandMotionTracker 的 MotionFeatures，判定条件为：reversals >= min_reversals=2、min_rhythm_hz=0.5 <= rhythm_hz <= max_rhythm_hz=4.0、amplitude_ratio >= min_amplitude_ratio=0.02、score >= min_score=0.5。不训练模型、不引入新依赖。

输出 snapshot.demo：state / state_zh / since_ms / duration_ms / last_detected_ms / last_detected_age_ms / raw_detected / rhythm_hz / reversals / amplitude_ratio / score / hand。界面用 state 显示大标签；raw_detected=false 且 state=chopping 时显示“节奏保持中”，避免把 0 Hz 当成正在切菜的证据。

演示界面只把两状态切换写进右侧“状态变化”记录；阶段二 activity 仍写入 snapshot/事件文件，但不再占据演示主面板。切菜中/完成后手离开画面属正常，叠加层不再弹“未检测到人物/把手放进画面”的旧提示。

真实侧身视频验证（2026-11-11）：551 帧、1280x720、30 fps；2.833 s 进入 chopping，17.367 s 进入 done，视频 18.333 s 结束前保持 done；状态只切换 2 次，无中途抖动。该结果只证明这一机位/动作的可演示性，不等同于全厨房准确率。

## 9. 延迟与降级

- 统计对象：单帧处理耗时（读帧之后到分析/快照结束），不含摄像头等待
- 每 30 帧评估最近 120 帧 p95；级别切换时清空统计窗口：
  - p95 > 500 ms → 跳 2 级
  - p95 > 250 ms → 升 1 级
  - p95 < 137.5 ms → 降 1 级
- 级别（config.latency.levels）：
  - L0：每帧姿态+手部
  - L1：手部隔帧
  - L2：姿态隔帧
  - L3：姿态隔 2 帧、手部隔 3 帧、分析缩放 0.75（预留字段，阶段一/二尚未接入分辨率缩放，实际只隔帧生效）
- 跳过的帧复用上次推理结果，快照标记 `stale_pose` / `stale_hands`

## 10. JSON schema（schema_version 1.1）

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| schema_version | str | 数据契约版本（阶段二为 1.1） |
| camera_status | str | CONNECTED / DISCONNECTED / UNKNOWN |
| frame.index | int | 帧序号（从 1 开始） |
| frame.ts_ms | float | 帧时间戳（摄像头为开机后毫秒；视频为 帧号/fps） |
| frame.width/height | int | 画面尺寸 |
| frame.source | str | camera / video / none |
| frame.calibration | dict | 当前直立基准偏移 |
| perf.total_ms | float | 本帧总处理耗时 |
| perf.pose_ms / hand_ms / analyze_ms / activity_ms | float | 分阶段耗时 |
| perf.p50_ms / p95_ms / max_ms | float? | 最近窗口统计 |
| perf.level / level_index | str/int | 当前降级级别 |
| perf.fps | float | 估计帧率 |
| perf.stale_pose / stale_hands | bool | 本帧是否复用了上次推理 |
| zones.split_x / deadband | float | 显示画面归一化分割线/死区 |
| zones.enabled | bool | 是否在叠加层显示分区线和标记（不影响后台判区） |
| zones.view_mirrored | bool | 当前预览是否镜像 |
| zones.labels | dict | A/B 中文标签 |
| zones.body_center_norm | list[2]? | 平滑后的身体重心（显示坐标） |
| zones.body_zone | str | A / B / unknown |
| zones.zone_source | str | hips / shoulders / bbox / none |
| zones.zone_confidence | float | 0–1 |
| zones.hand_zone_mismatch | bool | 手与身体是否在不同区 |
| zones.hands | list | 每只手 zone 摘要 |
| zones.warnings | list[str] | 配置/可见性警告 |
| torso.present | bool | 是否检测到人 |
| torso.posture | str | no_person/not_visible/upright/leaning_forward/leaning_backward/leaning_side/bent_over/uncertain |
| torso.posture_zh | str | 中文标签 |
| torso.confidence | float | 0–1 |
| torso.features.pitch_deg/roll_deg/yaw_deg | float? | 度，NaN 转 null |
| torso.features.facing | str | toward_camera/away_from_camera/sideways/unknown |
| torso.features.visibility | float | 双肩可见度 |
| torso.features.bbox_norm | list[4] | 躯干包围盒 [x0,y0,x1,y1] |
| torso.features.source | str | world/2d/none |
| torso.features.extra | dict | 诊断信息（肩/髋可见度、鼻尖 z 偏移等） |
| torso.hands_in_workspace | bool | 是否有手在躯干附近 |
| hands[] | list | 每只手：label、label_source、score、fingers{5}、finger_metrics{5}、grip、grip_zh、in_workspace、busy、bbox_norm、wrist_norm、palm_norm、zone、zone_code |
| hands_state | str | no_hands/hands_visible/in_workspace/hands_busy |
| activity.state | str | UNKNOWN/ACTIVE_A/ACTIVE_B/MOVING/PAUSED/TRANSITION_A_TO_B/TRANSITION_B_TO_A/OCCLUDED/AWAY |
| activity.state_zh | str | 中文标签 |
| activity.zone | str | A/B/unknown |
| activity.duration_ms | float | 当前状态持续 |
| activity.movement_level | str | LOW/MEDIUM/HIGH/unknown |
| activity.observation_quality | str | VALID/LOW/OCCLUDED/NO_PERSON/UNKNOWN |
| activity.chopping | dict | 切菜节奏特征（detected/stable/rhythm_hz/score/…） |
| activity.event | str | 本帧最后一个事件名（无则空串） |
| demo.state | str | waiting / chopping / done |
| demo.state_zh | str | 等待切菜 / 正在切菜 / 切菜完成 |
| demo.duration_ms | float | 当前演示状态已持续毫秒 |
| demo.raw_detected | bool | 本帧是否检测到切菜节奏（false 仍可能处于 chopping 保持期） |
| demo.rhythm_hz / reversals / amplitude_ratio / score | float/int | 当前帧切菜特征（供调阈值/诊断） |
| demo.last_detected_age_ms | float? | 距上次检测到切菜节奏的毫秒数 |
| planned_step / observed_step | null | 预留给后续阶段 |
| events | list | 本帧状态变化（旧 posture/hands_state + 新活动事件） |

## 11. 关键设计决策

1. **MediaPipe Tasks 而非旧版 solutions**：本机 mediapipe 0.10.35 已移除 `mp.solutions`，只能用 `mediapipe.tasks.python.vision`。
2. **不训练模型**：需求要求“成熟开源模型优先，不自己写模型”；姿态与手部都是官方 float16 task 模型。
3. **模型用 memory buffer 加载**：Windows 中文路径下 `model_asset_path` 会 FileNotFoundError（C++ 侧按窄字符路径读取），改用 `model_asset_buffer=path.read_bytes()`。
4. **时间戳强制单调**：VIDEO 模式对重复/回退时间戳抛 `ValueError: Input timestamp must be monotonically increasing`；`_TimestampGuard` 兜底。
5. **中文路径 I/O**：`cv2.imread/imwrite` 在中文路径下失败，统一走 `np.fromfile+imdecode` / `imencode+tofile`；视频写 ASCII 临时路径再 move。
6. **可见性口径放宽**：笔记本摄像头常只拍到上半身，若要求髋部图像可见会永远 not_visible；改用双肩可见 + world 髋估算，并降低置信度。
7. **规则化而非学习**：手指曲张、分区、切菜节奏都用几何/时序规则，便于按环节调整，也避免训练数据与算力开销。
8. **配置驱动 + 插件钩子**：所有阈值/标签在 JSON，`custom_rules.py` 可覆盖判定，满足“定义不要定太死、按炒菜环节定制”。
9. **非阻塞关闭 + 入口 os._exit**：MediaPipe 0.10.35 的 Clearcut 遥测在离线网络下会让 landmarker 的 `close()` 同步等待约 42 s/实例。`engines._release_landmarker` 把关闭动作丢进守护线程、最多等 1.5 s 就放弃；`__main__.py` 与 `tests/run_tests.py` 在命令/测试结束后 flush 并 `os._exit`。
10. **二次标定可重复**：`TorsoAnalyzer.calibrate()` 会把旧 offset 加回当前角度后再记基准（new = old + current），连续点两次“标定直立基准”不会清掉基准（T5 报告 S2-1，已补单测）。
11. **A/B 用一条可配置分割线、不做矩形框选**：用户明确简化；身体重心判区对固定机位足够，且配置/保存成本低。
12. **ACTIVE 只由切菜节奏触发**：普通移动不算活动；这样“活动状态”能直接对应到切菜行为，而不是任何手部运动。
13. **切菜节奏用掌心/手腕的纵向往复**：不依赖菜刀检测；固定摄像头下手臂反复抬起/放下就是最稳定信号；手部不可靠时用姿态手腕兜底。
14. **事件双写（内存+文件）**：UI 立即显示，文件留证据；写失败只记录 error，不阻塞检测。
15. **无摄像头验证路径**：`simulate` 用合成观测序列覆盖状态机分支；真实摄像头只做空场景/单人 smoke。真人切菜准确率标“待复核”，不伪造。

16. **演示层与活动状态机解耦**：需求收窄为“只显示正在切菜/切菜完成”，因此新增独立 DemoStateTracker，而不是改 activity 的状态枚举；旧状态机保留给后续阶段，演示面板只取两状态，且复用 motion 特征避免重复造轮子。

17. **视频导出用独立 Pipeline、暂停预览**：MediaPipe Tasks 对象不能跨线程共用，两个 Pipeline 同时跑也会争 CPU；导出前停掉预览线程，导出线程从头跑一遍，保证导出的每一帧与实时预览走完全相同的标注逻辑。
18. **mp4v 静音导出、不引入 ffmpeg**：需求只要求导出 MP4；OpenCV 自带 mp4v 在 Windows 上可用，避免演示机额外安装编码器。代价是无音轨，已在文档标注。
19. **分区线开关与判区解耦**：`ZoneSettings.enabled` 只控制叠加层是否绘制，活动状态/手部 zone 仍照常计算；这样演示时可以隐藏干扰线，而不改变识别结果。

## 12. 踩坑记录

- 中文路径：见决策 3、5
- MediaPipe VIDEO 模式时间戳：见决策 4
- 手部 landmark 的 visibility/presence 常为 0（不是不可见）：不能拿它当可见度门槛，实测 fist.png 全为 0 仍能正确分类
- 摄像头抢占：两个进程同时打开同一摄像头时，后打开者 `read()` 会阻塞约 1 s/帧；benchmark 与 UI 不要同时跑
- `cv2.VideoWriter` 的 mp4v 编码在部分机器不可用：写视频失败时返回 False，不抛异常
- 世界坐标 z 方向：朝向摄像头时鼻尖 z 比肩更负，代码里用 `front=sign(nose.z-shoulder.z)` 处理
- Clearcut 遥测阻塞：见决策 9。实测单实例 `close()` 约 44 s；加 1.5 s 超时后全套测试由 254 s 降到约 11 s
- 空场景误检：挂着的衣服可能被 MediaPipe 当成人体；仅靠 confidence 不可靠，必须加角度可信范围门限
- 候选计时不能用 `candidate_since or now`：0.0 是合法时间戳但 falsy，会让候选永不确认（阶段二修）
- 手切换会制造假节奏：左右手交替可能被拼成高频反转；按 key 分轨迹并在断手超时后 reset
- 无人但未到 AWAY 阈值时不要报 OCCLUDED：阶段二改为先回 UNKNOWN，避免把“无人”显示成“手部遮挡”
