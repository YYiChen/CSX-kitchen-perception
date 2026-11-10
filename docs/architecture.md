# 架构与设计说明

## 1. 目标与边界

阶段一目标：默认摄像头前的**躯干状态**与**五指曲张状态**实时识别，延迟硬上限 500 ms。
明确不做：菜刀/食材等物品识别、活动识别（activity 字段保留，值为 unknown）、多摄像头、训练自定义模型。

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
        ├─ LatencyBudget.observe   → L0–L3 自适应降级
        └─ PerceptionSnapshot.to_dict() → JSON 快照
                 │
                 ├─ OverlayRenderer.render → 叠加帧（骨架+五指着色+中文 HUD）
                 └─ UI（PerceptionWorker 线程） / CLI / JSONL / 快照文件
```

线程模型：UI 主线程只做 Tkinter 绘制与按钮；`PerceptionWorker` 后台线程独占摄像头与 MediaPipe 对象（MediaPipe Tasks 非线程安全）。

## 3. 模块职责

| 模块 | 职责 |
| --- | --- |
| config.py | 默认配置 + 用户配置 + overrides 深合并；路径解析 |
| sources.py | 摄像头枚举/打开/重连；视频文件中文路径处理；时间戳 |
| engines.py | 模型校验（sha256）；Pose/Hand Landmarker 封装；时间戳单调保护 |
| torso.py | world/2D 几何特征、状态分类、EMA 平滑、滞回、直立基准标定 |
| fingers.py | 五指关节角/曲率/距离比，曲张分类，手势分类 |
| overlay.py | 骨架绘制、五指状态着色、中文 HUD、镜像预览 |
| pipeline.py | 组合推理、降级调度、快照与事件、标定持久化 |
| worker.py | 后台采集/推理线程，供 UI 轮询最新结果 |
| ui.py | Tkinter 桌面界面（见 docs/ui_notes.md） |
| cli.py | verify-models / probe-cameras / ui / run / benchmark / snapshot |
| custom_rules.py | 按炒菜环节覆盖躯干/手指定义的扩展点 |

## 4. 躯干状态定义

关键点（MediaPipe Pose 33 点）：鼻尖 0、双肩 11/12、双髋 23/24、双膝 25/26。

- **world landmarks**：原点在髋中心、单位约米；用“髋中点 → 肩中点”向量 `up` 计算：
  - pitch = atan2(up.z × front, −up.y)，front = sign(鼻尖.z − 肩中点.z)；正值 = 前倾
  - roll = atan2(up.x, −up.y)；正值 = 上身向图像右侧倾斜
  - yaw = atan2(肩线.z, |肩线.x|)；0 = 正对/正背，±90 = 侧身
- **2D 兜底**：world 缺失但髋部可见时，只用肩-髋连线在图像内的倾角判断侧倾
- **可见性策略**：双肩可见即可分析；髋部不在画面内时用 world 髋位置估算 pitch，置信度乘 `hip_visibility_penalty`（默认 0.85）并标注“估算”
- 状态优先级：bent_over(≥45°) > leaning_forward(≥15°) > leaning_backward(≤−15°) > leaning_side(|roll|>10°) > upright
- 平滑：EMA(α=0.3)；滞回：连续 5 帧同候选才切换显示状态
- 标定：点“标定直立基准”把当前 pitch/roll/yaw 记为 0°，保存到 `outputs/calibration.json`

阈值全部在 `config/default.json → torso.thresholds`。

## 5. 手指状态定义

每根手指取 MCP/PIP/DIP（拇指为 MCP/IP/TIP）：

- pip_deg：PIP 关节内角（180° = 伸直）；dip_deg：DIP 关节内角
- curl_deg = (180 − pip_deg) + (180 − dip_deg)
- 非拇指：curl ≤ 40° 且 pip ≥ 150° → extended；curl ≥ 100° 或 pip ≤ 110° → flexed；否则 partial
- 拇指：IP 内角 ≥ 150° 且 拇指尖→食指 MCP 距离/掌长 ≥ 0.65 → extended；IP ≤ 110° 或距离比 ≤ 0.35 → flexed；否则 partial
- 手势：五指全伸 = open；四指全屈且拇指屈 = fist；拇指尖-食指尖距离/掌长 ≤ 0.4 且食指非屈 = pinch；仅食指伸 = point；其余 = partial；有效手指 <3 = unknown

阈值在 `config/default.json → hands.finger_thresholds`。默认值用官方测试图实测标定（open 手 curl 1–6°、fist curl 190–200°、拇指距离比 open 0.7–0.9 / fist 0.1）。

## 6. 延迟与降级

- 统计对象：单帧处理耗时（读帧之后到分析/快照结束），不含摄像头等待
- 每 30 帧评估最近 120 帧 p95：
  - p95 > 500 ms → 跳 2 级
  - p95 > 250 ms → 升 1 级
  - p95 < 137.5 ms → 降 1 级
- 级别（config.latency.levels）：
  - L0：每帧姿态+手部
  - L1：手部隔帧
  - L2：姿态隔帧
  - L3：姿态隔2帧、手部隔2帧、分析缩放 0.75
- 跳过的帧复用上次推理结果，快照标记 `stale_pose` / `stale_hands`

## 7. JSON schema（schema_version 1.0）

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| schema_version | str | 数据契约版本 |
| frame.index | int | 帧序号（从 1 开始） |
| frame.ts_ms | float | 帧时间戳（摄像头为开机后毫秒；视频为 帧号/fps） |
| frame.width/height | int | 画面尺寸 |
| frame.source | str | camera / video / none |
| frame.calibration | dict | 当前直立基准偏移 |
| perf.total_ms | float | 本帧总处理耗时 |
| perf.pose_ms / hand_ms / analyze_ms | float | 分阶段耗时 |
| perf.p50_ms / p95_ms / max_ms | float? | 最近窗口统计 |
| perf.level / level_index | str/int | 当前降级级别 |
| perf.fps | float | 估计帧率 |
| perf.stale_pose / stale_hands | bool | 本帧是否复用了上次推理 |
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
| hands[] | list | 每只手：label(left/right/unknown)、label_source、score、fingers{5}、finger_metrics{5}、grip、grip_zh、in_workspace、busy、bbox_norm、wrist_norm |
| hands_state | str | no_hands/hands_visible/in_workspace/hands_busy |
| activity | dict | 阶段一固定 unknown |
| planned_step / observed_step | null | 预留给后续阶段 |
| events | list | 本帧状态变化（posture_change / hands_state_change） |

## 8. 关键设计决策

1. **MediaPipe Tasks 而非旧版 solutions**：本机 mediapipe 0.10.35 已移除 `mp.solutions`，只能用 `mediapipe.tasks.python.vision`。
2. **不训练模型**：需求要求“成熟开源模型优先，不自己写模型”；姿态与手部都是官方 float16 task 模型。
3. **模型用 memory buffer 加载**：Windows 中文路径下 `model_asset_path` 会 FileNotFoundError（C++ 侧按窄字符路径读取），改用 `model_asset_buffer=path.read_bytes()`。
4. **时间戳强制单调**：VIDEO 模式对重复/回退时间戳抛 `ValueError: Input timestamp must be monotonically increasing`；`_TimestampGuard` 兜底。
5. **中文路径 I/O**：`cv2.imread/imwrite` 在中文路径下失败，统一走 `np.fromfile+imdecode` / `imencode+tofile`；视频写 ASCII 临时路径再 move。
6. **可见性口径放宽**：笔记本摄像头常只拍到上半身，若要求髋部图像可见会永远 not_visible；改用双肩可见 + world 髋估算，并降低置信度。
7. **规则化而非学习**：手指曲张用关节角规则，便于按环节调整，也避免训练数据与算力开销。
8. **配置驱动 + 插件钩子**：所有阈值/标签在 JSON，`custom_rules.py` 可覆盖判定，满足“定义不要定太死、按炒菜环节定制”。
9. **非阻塞关闭 + 入口 os._exit**：MediaPipe 0.10.35 的 Clearcut 遥测在离线网络下会让 landmarker 的
   `close()`（以及解释器正常退出）同步等待约 42 s/实例。`engines._release_landmarker` 把关闭动作
   丢进守护线程、最多等 1.5 s 就放弃，引用存入进程级列表避免提前析构；`__main__.py` 与
   `tests/run_tests.py` 在命令/测试结束后 flush 并 `os._exit`，因此 CLI 与测试不会被退出拖慢。

## 9. 踩坑记录

- 中文路径：见决策 3、5
- MediaPipe VIDEO 模式时间戳：见决策 4
- 手部 landmark 的 visibility/presence 常为 0（不是不可见）：不能拿它当可见度门槛，实测 fist.png 全为 0 仍能正确分类
- 摄像头抢占：两个进程同时打开同一摄像头时，后打开者 `read()` 会阻塞约 1 s/帧；benchmark 与 UI 不要同时跑
- `cv2.VideoWriter` 的 mp4v 编码在部分机器不可用：写视频失败时返回 False，不抛异常
- 世界坐标 z 方向：朝向摄像头时鼻尖 z 比肩更负，代码里用 `front=sign(nose.z-shoulder.z)` 处理
- Clearcut 遥测阻塞：见决策 9。实测单实例 `close()` 约 44 s；加 1.5 s 超时后全套测试由 254 s 降到约 11 s。
