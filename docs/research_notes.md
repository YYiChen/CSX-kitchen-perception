# T1 技术调研笔记 — 躯干/手部感知选型与算法参考

> 项目：厨房情境感知（CSX_人体切菜检测）｜文档版本 0.1｜整理：researcher｜日期：2026-11-10
> 配套实测数据与素材清单：`docs/test_assets.md`、`assets/test/official/measurements.json`、`assets/test/official/sources/`。
> 标记约定：【已实测】= 本机跑过或从已安装包/已下载文件中直接读取；【文献/文档结论，未在本机复现】= 来自官方模型卡、README、PyPI 元数据或公开文档，本文未做独立复现。

## 0. 结论速览（给开发/文档直接引用）

1. **基线选型：MediaPipe Tasks 0.10.35（Pose Landmarker lite/full + Hand Landmarker）**。同一套 Python API 同时给出 33 点人体 + 每手 21 点 + 左右手标签；Windows + Python 3.12 的 pip wheel 可直接用，Apache 2.0，模型随项目 `models/` 本地加载，完全离线【已实测】。
2. **性能（CPU，本机 32 逻辑核）**：640×480 视频跟踪模式下 pose_lite 平均 22.6 ms、hand 平均 32.6 ms，串联 58.4 ms/帧（≈17 FPS 上限，且未计摄像头采集、绘制、业务逻辑）【已实测】。IMAGE 模式单帧约 45 ms / 53 ms【已实测】。1280×720 与 640×480 的检测耗时几乎相同（模型内部会 resize 到固定输入）【已实测】——**不要靠降分辨率省检测时间**，应在“每 N 帧检测一次 + 中间帧复用”上做优化。
3. **五指曲张判定必须优先用 2D 归一化坐标角度**；3D world landmarks 的手指角度在张开手上会被明显放大（同图 2D 0–24°，3D 17–91°），不能直接套阈值【已实测】。握拳样本 2D 四指 189–204° 能稳定判 curled，但**拇指必须特判**（握拳时 3D 拇指总弯曲 109°，落在 partial 区间）【已实测】。
4. **handedness 未在真实摄像头下验证**：官方非镜像图片上标签与文件名一致，但 MediaPipe 文档默认假设自拍镜像输入；接入本项目的 cv2 摄像头后需实测一次，必要时交换标签【文献/文档结论，未在本机复现；图片标签部分已实测】。
5. **GPU delegate 不可用**：0.10.35 Windows wheel 是 CPU-only，`delegate=GPU` 创建即抛 `NotImplementedError: ValidatedGraphConfig Initialization failed.`，日志含 `ImageCloneCalculator: GPU processing is disabled in build flags`【已实测】→ 性能规划按 CPU 做。
6. **旧版 `mp.solutions` 已从 0.10.35 移除**（`from mediapipe import solutions` → `ImportError: cannot import name 'solutions' from 'mediapipe'`）【已实测】，网上旧教程的 `mp.solutions.pose/hands` 代码不能直接用，必须走 Tasks API。
7. **其他路线**：YOLO11-pose（COCO 17 点，无手部关键点，AGPL-3.0）、MoveNet（17 点，无手）、OpenPose（非商用许可、C++ 部署重）、MMPose/RTMPose（能力足够但 mmcv 编译依赖在 Windows+Py3.12 上成本高）。**综合许可、手部能力、部署成本，MediaPipe Tasks 是当前唯一低风险满足“躯干+双手”需求的现成方案**【文献/文档结论，未在本机复现；MediaPipe 可用性部分已实测】。

---

## 1. 环境与实测方法

| 项 | 值 | 来源 |
|---|---|---|
| 系统 / Python | Windows，Python 3.12.10（`py -3.12`） | 【已实测】 |
| mediapipe | 0.10.35（PyPI win_amd64 wheel，py3-none，10,905,503 B；另有 win_arm64 轮 9,992,331 B） | 【已实测】 |
| numpy / opencv-python | 2.3.5 / 5.0.0.93（另装 opencv-contrib-python 4.10.0.84） | 【已实测】 |
| CPU | 32 逻辑核（本机） | 【已实测】 |
| 模型 | `models/pose_landmarker_lite.task`（5,777,746 B，sha256 59929e1d…）、`pose_landmarker_full.task`（9,398,198 B，sha256 5134a3aa…）、`hand_landmarker.task`（7,819,105 B，sha256 fbc2a300…） | 【已实测】 |
| 测试图 | MediaPipe 官方 6 张（1000×667 ~ 358×354），sha256/尺寸见 `assets/test/official/sources/assets_manifest.json` | 【已实测】 |
| 主脚本 | `assets/test/official/measure_v2.py`（图像检测 + VIDEO 时间戳 + IMAGE 基准）；`measure_video_bench.py`（VIDEO 跟踪基准） | 【已实测】 |

说明：所有模型用 `BaseOptions(model_asset_buffer=open(path,'rb').read())` 加载，避免 Windows 中文路径问题；脚本以 `os._exit(0)` 收尾，原因见 §2.8。

---

## 2. MediaPipe Tasks 0.10.35 实测要点

### 2.1 可用 API

- `mediapipe.tasks.python.vision` 提供：`FaceLandmarker`、`GestureRecognizer`、`HandLandmarker`、`HolisticLandmarker`、`ImageClassifier`、`PoseLandmarker`【已实测】。
- `RunningMode`：`IMAGE` / `VIDEO` / `LIVE_STREAM`【已实测】。
- **没有** `mediapipe.solutions`；`hasattr(mp, "solutions") == False`；`from mediapipe import solutions` 直接 ImportError【已实测】。
- `GestureRecognizer` 类存在（可识别预定义手势），但本轮未下载 `gesture_recognizer.task`，其“剪刀/握拳”等 canned gestures 未实测；若采用需另测【已实测：类存在；未实测：识别效果】。

### 2.2 关键参数（读取本机 site-packages 源码确认）

- `PoseLandmarkerOptions` 默认：`running_mode=IMAGE`、`num_poses=1`、`min_pose_detection_confidence=0.5`、`min_pose_presence_confidence=0.5`、`min_tracking_confidence=0.5`、`output_segmentation_masks=False`【已实测】。
- `HandLandmarkerOptions` 默认：`num_hands=1`、`min_hand_detection_confidence=0.5`、`min_hand_presence_confidence=0.5`、`min_tracking_confidence=0.5`【已实测】。
- 本项目建议：`num_poses=1`、`num_hands=2`（切菜双手）、`running_mode=VIDEO`（摄像头流）；置信度先用默认 0.5，用真实素材再调。

### 2.3 两种运行模式的行为差异

- `IMAGE`：每帧独立检测，无跨帧跟踪；适合单张图片/离线批处理。另：1280×720 与 640×480 单帧耗时几乎一样【已实测】。
- `VIDEO`：`detect_for_video(image, timestamp_ms)`，内部做跨帧跟踪，640×480 下 pose 22.6 ms / hand 32.6 ms，明显快于 IMAGE 模式【已实测】。**摄像头主循环应使用 VIDEO 模式**。
- `LIVE_STREAM`：异步回调模式，本轮未测（时间盒）【未实测】。

### 2.4 VIDEO 时间戳约束（易踩）

- 同一 landmarker 上时间戳必须**严格单调递增**。实测 `ts=0,33,66` 正常；重复 `ts=33` 或回退 `ts=10` 均抛 `ValueError: Input timestamp must be monotonically increasing.`【已实测】。
- 建议用 `int(time.perf_counter()*1000)` 或视频帧号×1000/fps 作为时间戳，并保证递增；不要用每帧固定 0。

### 2.5 world landmarks 坐标系（躯干算法的基础）

以 `pose.jpg`（lite）为例的实测：

- 原点在**两髋中点**：`hip_mid ≈ [-0.0003, -0.0001, 0.0002]`（m）【已实测】。
- **y 轴向下**：肩中点 y = −0.4891（髋中点 y ≈ 0），即肩在上方为负【已实测】。
- **z 负方向朝相机**：鼻尖 z − 肩中点 z = −0.1161（< 0），与 BlazePose 模型卡“negative values are between the hips and the camera”一致【已实测 + 模型卡】。
- 尺度为米（肩线长 0.3396 m，符合成人肩宽量级）【已实测】。
- 模型卡补充：z 是**相对髋部平面的非 metric 深度**，正值在髋后；不是精确相机深度【文献/文档结论，未在本机复现】。
- 手部 world landmarks 原点在手掌附近，量级 ≤0.09 m（`fist.png` hand0：wrist(0) ≈ [-0.0083, 0.0815, 0.0034]，middle MCP(9) ≈ [0.0015, -0.0047, 0.0058]，middle tip(12) ≈ [0.0143, 0.0116, -0.0038]）【已实测】。

### 2.6 手部输出与 handedness

- 每手 21 个 landmark（索引见 §3.1），输出 `handedness`（`label + score`）与 world landmarks【已实测】。
- 官方图实测：`right_hands.jpg` 两手 Right（0.9433/0.9606），`left_hands.jpg` 两手 Left（0.9679/0.9562），`fist.png` Right（0.989），`woman_hands.jpg` Left（0.936）/Right（0.9622），`male_full_height_hands.jpg` Left（0.9577）/Right（0.9365）【已实测】。
- **注意**：MediaPipe 文档说明 handedness 默认假设输入为**镜像（自拍）**画面；若摄像头未水平翻转，标签可能相反。本轮官方图为非镜像照片，标签与图像实际左右一致，但**真实摄像头语义仍需实测**（本项目待办）【文献/文档结论，未在本机复现；官方图标签已实测】。

### 2.7 性能实测（CPU，n=15 图像 / n=20 视频，warmup 3–5）

IMAGE 模式（每图独立检测，单帧，ms）：

| 输入分辨率 | pose_lite mean/median/std/p95 | pose_full | hand |
|---|---|---|---|
| 640×480 | 45.16 / 45.08 / 1.71 / 48.23 | 52.5 / 52.25 / 1.44 / 55.09 | 53.21 / 53.22 / 1.53 / 55.54 |
| 1280×720 | 44.39 / 44.29 / 0.77 / 45.42 | 52.66 / 52.8 / 1.67 / 53.83 | 53.76 / 53.61 / 1.06 / 54.88 |

VIDEO 跟踪模式（`detect_for_video`，单帧，ms）：

| 输入分辨率 | pose_lite | hand | pose + hand 串联 |
|---|---|---|---|
| 640×480 | 22.6 / 22.28 / 0.79 / 24.41 | 32.56 / 32.73 / 1.3 / 34.02 | **58.36 / 58.2 / 1.37 / 60.35（≈17 FPS）** |
| 1280×720 | 24.06 / 23.6 / 1.02 / 25.88 | 32.64 / 32.33 / 0.89 / 34.08 | 未测（见 §8 失败尝试） |

- landmarker 创建耗时：IMAGE 每个约 190–194 ms；VIDEO 640×480 约 189.6 ms；VIDEO 1280×720 曾出现 **42,330.1 ms** 异常（遥测阻塞，见 §2.8）【已实测】。
- 图像首帧耗时含加载/初始化开销（lite 20–44 ms、full 23–53 ms、hand 39–54 ms），只作量级参考，性能以重复统计为准【已实测】。
- 解读：17 FPS 是“pose+hand 检测”本身的上限；若再加绘制/编码/业务判断，实时性会下降。可选策略：pose 每 2–3 帧一次（跟踪模式下漏帧可复用）、hand 每帧或每 2 帧一次；或降低 hand 检测频率+用跟踪结果插值。**“每帧都跑两个模型”在纯 CPU 上余量不大**。

### 2.8 遥测阻塞与 close() 陷阱（本机特有，必须写进文档）

- 日志反复出现：
  `E0000 portable_clearcut_uploader.cc:90] Failed to send to clearcut: INTERNAL: ResponseInfo[id: N, type: 0, code: 2, description: "Status_ConnectFailed: 12002", headers: ]`
  （Source Location Trace: wireless/android/play/playlog/cplusplus/ion_http_client.cc:44）【已实测】
- 后果：`landmarker.close()` / 正常解释器退出可能阻塞约 40 s/实例；一次 landmarker 创建被阻塞 42,330 ms；三个实例 close 累计 >100 s【已实测】。
- 规避：实测脚本用 `os._exit(0)` 跳过清理；**产品代码应在启动时一次性创建 landmarker 并全程复用，避免反复创建/关闭**；退出时不必强求 close()【建议，基于已实测现象】。
- 该行为与“Clearcut 后端不可达”有关，换网络/离线环境可能不同；不要据此认为 MediaPipe 本身很慢【推断，未单独验证】。

### 2.9 GPU

- `mp_tasks.BaseOptions.Delegate` = [CPU:0, GPU:1]【已实测】。
- `delegate=Delegate.GPU` 创建 PoseLandmarker 立即抛 `NotImplementedError: ValidatedGraphConfig Initialization failed.`，日志含 `ImageCloneCalculator: GPU processing is disabled in build flags`（`assets/test/official/sources/gpu_probe.log`）【已实测】。
- 结论：**本机 pip 安装的 mediapipe 0.10.35 Windows wheel 是 CPU-only**；要用 GPU 需自行编译 MediaPipe、或走 ONNX Runtime/TensorRT 等替代运行时，超出本项目时间盒【已实测 + 推断】。

---

## 3. 手指曲张（curl）算法参考

### 3.1 MediaPipe 手部 21 点索引（标准拓扑）

| 手指 | 关节链（MCP → PIP → DIP → TIP） |
|---|---|
| 拇指 | 1 (CMC) → 2 (MCP) → 3 (IP) → 4 (TIP) |
| 食指 | 5 → 6 → 7 → 8 |
| 中指 | 9 → 10 → 11 → 12 |
| 无名指 | 13 → 14 → 15 → 16 |
| 小指 | 17 → 18 → 19 → 20 |
| 腕 | 0 |

（0 = wrist；每指 4 点，共 21 点。）【已实测：每手均检出 21 点；拓扑为 MediaPipe 官方定义】

### 3.2 关节角与总弯曲公式

对每根手指，用相邻三点 `(a, b, c)`（b 为关节）计算内角：

```python
def angle_deg(a, b, c):
    v1 = a - b
    v2 = c - b
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

# 以食指为例（2D：landmark.x/y 乘图像宽高；3D：world landmarks x/y/z）
a1 = angle_deg(p[5], p[6], p[7])   # PIP 内角
a2 = angle_deg(p[6], p[7], p[8])   # DIP 内角
total_flexion = (180 - a1) + (180 - a2)   # 0°=完全伸直，~180°=完全弯曲
```

- 对中指/无名指/小指同理；拇指可用 `(1,2,3)` 与 `(2,3,4)` 两个角，但**阈值/权重需单独标定**（见 §3.4）。
- 也可加 MCP 角（`angle(0,5,6)` 等）辅助区分“掌指关节弯曲”和“指间关节弯曲”【算法建议】。

### 3.3 实测数值与阈值

本批官方图实测（`total_flexion`，度；顺序 拇/食/中/无/小；完整表见 `docs/test_assets.md` 表 B）：

- 张开手（2D）：大多 0–24°；同图 3D：17–91°【已实测】。
- 握拳 `fist.png`：四指 **2D 189/198/204/192°、3D 168/195/168/175°**；拇指 2D 127°、3D 109°【已实测】。
- 因此**2D 阈值余量很大**：`extended < 30°`、`partial 30–110°`、`curled > 110°` 在现有样本上能正确分开张开与握拳（本报告测试表的判定规则）【已实测】。
- 若用 3D：张开手会被误判 partial（例如 `male_full_height_hands.jpg` 3D 30–55°、2D 仅 1–14°），**不建议单独用 3D**【已实测】。
- 生产建议：阈值做成配置；先按距摄像头 0.5–1.5 m 的实测素材标定；小手掌（bbox 对角线 <80 px）直接输出 unknown，而不是硬判【建议，基于已实测 bbox 数据：§2.7 与 `sources/hand_bbox_px.json`】。

### 3.4 拇指特判

- 握拳时拇指横向压在其它手指上，其“弯曲方向”和四指不同：`fist.png` 中四指 109–175°（判 curled），拇指 109°（判 partial）【已实测】。
- 建议：①拇指用更宽的 curled 阈值（如 ≥90° 即视为 tucked/curled），或 ②结合四指状态：四指 curled 且拇指 flexion >60° → 判“拇指收拢/按压”；③用拇指尖到食指 MCP 的距离/朝向辅助【建议，基于已实测】。
- 若产品只关心“是否张开手掌/是否握刀”，可先只输出四指状态，把拇指作为附加字段【建议】。

### 3.5 遮挡 / 低置信度 → unknown

- 手部模型卡明确 out-of-scope：戴手套、手被遮挡（例如**手拿物体/菜刀**）、饰品/纹身/henna；训练数据有限，in-the-wild 条件未充分验证【文献/文档结论，未在本机复现】。
- 切菜场景恰恰是“手拿刀 + 部分遮挡”，因此必须设计 unknown：当 `handedness score` 低、landmark presence 低、手 bbox 太小、角度帧间抖动大时，输出 `unknown` 并保持上一次稳定状态（短时间内），不要用噪声硬判【建议，基于模型卡 + 实测抖动】。
- 建议状态机：每指输出 `extended / partial / curled / unknown`；连续 N 帧（如 3–5 帧）一致才更新；用少量滞回（进入/退出阈值不同）抑制抖动【建议】。

### 3.6 参考实现 / URL

- MediaPipe 旧版手部实现（算法对照，0.10.35 已无此模块）：
  https://raw.githubusercontent.com/google-ai-edge/mediapipe/master/mediapipe/python/solutions/hands.py
- Tasks 版 HandLandmarker（参数/结果结构）：
  https://raw.githubusercontent.com/google-ai-edge/mediapipe/master/mediapipe/tasks/python/vision/hand_landmarker.py
- 手部 landmark 模块说明：
  https://raw.githubusercontent.com/google-ai-edge/mediapipe/master/mediapipe/modules/hand_landmark/README.md
- 模型卡 PDF（含 GHUM 说明）：
  https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Hand%20Tracking%20(Lite_Full)%20with%20Fairness%20Oct%202021.pdf
（以上 URL 本轮均 HTTP 200 抓取过，见 `assets/test/official/sources/fetched_sources.json`。）

---

## 4. 躯干姿态算法参考

### 4.1 由 world landmarks 计算躯干角

取 Pose 33 点中的 11/12（左右肩）、23/24（左右髋）：

- `mid_sh = (lm[11] + lm[12]) / 2`，`mid_hip = (lm[23] + lm[24]) / 2`
- 肩线向量 `s = lm[11] - lm[12]`；躯干向量 `t = mid_sh - mid_hip`

建议公式（角度单位：度；坐标轴见 §2.5，y 向下、z 负朝相机）：

```python
roll   = degrees(atan2(s.y, s.x))            # 肩线在图像平面内的倾斜
yaw    = degrees(atan2(s.z, s.x))            # 肩线绕竖直轴的旋转（侧身程度）
lean_side    = degrees(atan2(t.x, -t.y))     # 躯干左右倾（t.y<0 表示躯干向上）
lean_forward = degrees(atan2(-t.z, -t.y))    # 躯干前后倾（z 负朝相机 → 前倾为正）
```

实测样例（`pose.jpg`，lite）：`roll = -3.41°`、`yaw = -21.45°`、`lean_side = 2.07°`、`lean_forward = 3.68°`；full 模型：`-0.8 / -6.7 / 0.6 / 8.29°`【已实测】。`male_full_height_hands.jpg` lite：`1.57 / -8.23 / 3.47 / 12.38°`【已实测】。

> 符号约定以本报告脚本为准；接入时应用“正面站立、已知姿态”的样本重新校准符号与零位。

### 4.2 坐/站/弯腰分类（可扩展，不写死状态）

建议不写死“站/坐”，而是输出**连续量 + 少量离散状态**，便于后续扩展（切菜可能是站姿、坐姿、弯腰）：

- 膝关节内角：`angle(lm[23], lm[25], lm[27])`（髋-膝-踝），站立接近 180°，坐姿约 90°【算法建议】。
- 髋部相对高度：站立时 `hip.y < knee.y`（髋更高，y 更负）；坐姿髋膝接近同高【算法建议】。
- 躯干前倾 `lean_forward` 超过约 20–30° 视为弯腰/前倾【算法建议，阈值待真实素材标定】。
- 建议状态：`standing` / `sitting` / `bending` / `unknown`；用 0.5–1 s 窗口投票平滑【建议】。

### 4.3 关键点不完整时的防护（实测教训）

- `woman_hands.jpg` 只有上半身：pose_lite 检出 1 个姿态，但推出的肩线 roll 71.7°、左右倾 75.3°（明显失真）；pose_full 则检出 0 个【已实测】→ **关键点缺失时角度不可信**。
- 防护：①检查 11/12/23/24 的 `visibility/presence`；②检查肩宽、肩-髋距离、躯干长度是否在合理范围（如肩线 0.25–0.5 m）；③检查左右肩/髋是否交叉；④不满足则输出 `unknown`，不参与业务判断【建议，基于已实测失真】。
- 模型卡限制：多人画面只跟踪一人；距离 >4 m、头部不可见、低光/运动模糊会显著降质；不用于需要精确 metric 深度的场景【文献/文档结论，未在本机复现】。

---

## 5. 技术路线对比（躯干 + 手部）

| 方案 | 许可 | 手部关键点 | 身体关键点 | 本机/Windows+Py3.12 | 结论 |
|---|---|---|---|---|---|
| **MediaPipe Tasks 0.10.35（Pose+Hand）** | Apache 2.0 | 每手 21 点 | 33 点 | 已装、wheel 可用、纯 CPU；实测 17 FPS 级 | **基线方案** |
| 旧版 mediapipe `mp.solutions` | Apache 2.0 | 每手 21 点 | 33 点 | 0.10.35 已移除（ImportError） | 不可用（旧教程不适用） |
| YOLO11-pose（ultralytics 8.4.106） | AGPL-3.0（商用需企业许可） | **无**（COCO 17 点仅身体） | 17 点 | 已装 ultralytics；torch 依赖重；CPU 单帧未实测 | 单模型不满足手部需求；AGPL 有合规风险 |
| MoveNet（tfjs / TF Hub） | Apache 2.0（TF 模型常规；未逐项核实 MoveNet 权重许可） | **无** | 17 点 COCO | 需 TensorFlow/TFLite 环境 | 不满足手部需求 |
| OpenPose | **非商用免费**（商用需授权） | 有（联合 135 点） | 有 | C++/CUDA 复杂，Windows 构建重，无 PyPI 轮 | 许可与部署均不适合 |
| MMPose / RTMPose | Apache 2.0 | 有（133 全身点含手） | 有 | mmcv/mmengine 编译依赖，Py3.12+Win 成本高；PyPI 仅 py2.py3/sdist | 能力强但集成成本高，可作二期备选 |
| MediaPipe HolisticLandmarker | Apache 2.0 | 有（类存在） | 有 | 0.10.35 类存在但未下载模型、未实测 | 可作后续统一方案的候选（未验证） |

补充事实：

- Ultralytics PyPI：latest 8.4.208，license AGPL-3.0，requires-python >=3.8；README 明确“商用需企业许可”【文献/文档结论，未在本机复现】。COCO-pose 是 17 点人体关键点，不含手指【文献/文档结论，未在本机复现】。
- MoveNet：tfjs pose-detection README 写“detects 17 keypoints of a body”，并说明 BlazePose 额外提供 face/hands/feet 关键点【文献/文档结论，未在本机复现】。
- OpenPose README 写“freely available for free non-commercial use… commercial license”，并宣称可联合检测 body/hand/face/foot 共 135 点【文献/文档结论，未在本机复现】。
- MMPose PyPI：latest 1.3.2，Apache-2.0，requires-python >=3.7，发布文件仅 py2.py3 + source（无 Windows 平台轮）【已实测：PyPI JSON】；README 提到 133 keypoint whole-body pose estimation 与 RTMPose 系列【文献/文档结论，未在本机复现】。
- MediaPipe PyPI：latest 1.2.2（查询时），0.10.35 上传于 2026-04-27，win_amd64 轮为 py3-none、10,905,503 B【已实测】。本项目固定在 0.10.35（本地验证过的版本），升级 1.x 前必须重跑本文所有实测。

---

## 6. 许可与来源（模型许可）

- **模型卡是许可的权威来源**：三个 `.task` 包均只是 zip，各含 2 个 `.tflite`，**包内无 manifest/metadata/许可文件**【已实测】。因此不能从模型文件推断许可。
- Pose（BlazePose GHUM 3D）模型卡：**Apache License 2.0**；Lite 3 MB / Full 6 MB / Heavy 26 MB；Pixel 3 CPU XNNPack 参考 Lite ~44 FPS、Full ~18 FPS、Heavy ~4 FPS；输入 256×256×3；输出 33×5（x, y, z, visibility, presence）【文献/文档结论，未在本机复现；PDF 已下载，见 `assets/test/official/model_cards/Model_Card_BlazePose_GHUM_3D.pdf`】。
- Hand 模型卡：**Apache License 2.0**；detector 输入 192×192×3，landmark/tracker 输入 224×224×3；输出每手 21 个 screen landmarks + handedness + 21 个 metric world landmarks（由 GHUM 合成数据拟合）；out-of-scope 含“手拿物体/遮挡、戴手套、饰品/纹身”等【文献/文档结论，未在本机复现；PDF 已下载：`Model_Card_Hand_Tracking_Lite_Full_Fairness_Oct2021.pdf`】。
- 已安装包的元数据许可：mediapipe Apache 2.0；opencv-python Apache 2.0；ultralytics AGPL-3.0；mmpose Apache-2.0；onnxruntime MIT【已实测：importlib.metadata / PyPI JSON】。
- 测试图片：来自 MediaPipe 官方示例素材（`https://storage.googleapis.com/mediapipe-assets/`），单张版权未逐张标注；本项目仅本地测试、不再分发【已实测下载；版权结论为推断】。

---

## 7. 调研素材缺口（需 Lead 决策）

1. **没有真实切菜素材**：官方 6 图只覆盖“张开手/握拳/全身/半身”，不含手拿刀、手部被砧板/食材遮挡、俯拍视角——而这正是模型卡明确 out-of-scope 的场景【已实测：素材清单】。建议用本机摄像头录制 10–20 s 切菜视频（或拍 10 张手部照片）作为二期验证集；录制前需与 Lead 协调摄像头占用（Lead 已提醒单摄像头资源竞争）。
2. **没有多角度/多光照样本**：无法评估低光、逆光、运动模糊对 handedness/角度的影响【未实测】。
3. **没有视频级跟踪指标**：只测了单帧耗时，未测帧间抖动/丢检率/ID 切换（手部有 handedness 但无 ID）【未实测】。
4. **公开数据集未尝试**：网络探测显示 ai.google.dev 不可用、web_search HTTP 402；为避免耗时，未尝试 Pexels/Wikimedia/HuggingFace 等【已实测网络限制】。

---

## 8. 复现命令、耗时、失败尝试

### 8.1 命令

```powershell
# 工作目录 = 项目根
py -3.12 -u assets/test/official/measure_v2.py           # 图像检测 + VIDEO 时间戳 + IMAGE 基准，约 10 s
py -3.12 -u assets/test/official/measure_video_bench.py  # VIDEO 跟踪基准（640×480 完整；1280×720 的 pose/hand 完整，串联未完成）
```

### 8.2 耗时

- 图像检测 6 张（3 个模型 × 2 个 pose 模型）约 10 s（含加载/初始化）【已实测】。
- VIDEO 基准 640×480 完整跑完；1280×720 在“再建 landmarker”阶段被遥测阻塞，脚本被杀，串联数据缺失【已实测】。
- GPU 探测脚本约 3 s（创建立即失败）【已实测】。

### 8.3 失败尝试（避免后来者重复）

1. 第一版 `measure_official_images.py` 跑通检测，但收尾 `close()` 阻塞 ~40 s/实例，且未及时落盘 JSON → 已弃用，改 `measure_v2.py`（分段写盘 + `os._exit(0)`）【已实测】。
2. 多次前台命令被 120 s 工具超时中断 → 改用后台任务 + 轮询输出【已实测】。
3. 1280×720 VIDEO 串联基准未完成（建 landmarker 时阻塞）【已实测】。
4. GPU delegate 创建失败（NotImplementedError，见 §2.9）【已实测】。
5. `from mediapipe import solutions` ImportError（见 §2.1）【已实测】。
6. 尝试用 pypdf/PyPDF2 抽模型卡文本 → 未安装；改用 PyMuPDF(fitz) 成功【已实测】。

---

## 9. 未核实 / 开放问题（交付给 Lead 的诚实清单）

1. handedness 在**本项目真实摄像头**（是否镜像）下的语义——必须实测一次。
2. 1280×720 VIDEO 模式下 pose+hand 串联耗时（未完成）。
3. `GestureRecognizer` 的预定义手势（握拳等）在切菜场景的可用性（模型未下载、未测）。
4. YOLO11-pose / MoveNet / MMPose 的**本机精度与 CPU 性能**均未实测（仅结构/许可/依赖层面对比），对比结论是文档级判断。
5. 真实切菜视频（手拿刀/遮挡）下的手部检出率、handedness 稳定性、角度抖动——无素材，未测。
6. 低光、多人、远距离（>4 m）等模型卡限制场景未测。
7. LiveStream 模式与异步回调未测。
8. 升级到 mediapipe 1.x 的兼容性未测（本项目固定 0.10.35）。

---

## 10. 关键 URL 清单（本轮实际访问，均 HTTP 200，除注明）

- MediaPipe 仓库 README：https://raw.githubusercontent.com/google-ai-edge/mediapipe/master/README.md
- Tasks HandLandmarker 源码：https://raw.githubusercontent.com/google-ai-edge/mediapipe/master/mediapipe/tasks/python/vision/hand_landmarker.py
- Tasks PoseLandmarker 源码：https://raw.githubusercontent.com/google-ai-edge/mediapipe/master/mediapipe/tasks/python/vision/pose_landmarker.py
- 旧版 solutions/hands.py：https://raw.githubusercontent.com/google-ai-edge/mediapipe/master/mediapipe/python/solutions/hands.py
- 手部模块 README：https://raw.githubusercontent.com/google-ai-edge/mediapipe/master/mediapipe/modules/hand_landmark/README.md
- Pose 模型卡 PDF：https://storage.googleapis.com/mediapipe-assets/Model%20Card%20BlazePose%20GHUM%203D.pdf
- Hand 模型卡 PDF：https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Hand%20Tracking%20(Lite_Full)%20with%20Fairness%20Oct%202021.pdf
- PyPI mediapipe：https://pypi.org/project/mediapipe/
- Ultralytics README：https://raw.githubusercontent.com/ultralytics/ultralytics/main/README.md
- OpenPose README：https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/README.md
- MMPose README：https://raw.githubusercontent.com/open-mmlab/mmpose/main/README.md
- TFJS pose-detection README（MoveNet/BlazePose）：https://raw.githubusercontent.com/tensorflow/tfjs-models/master/pose-detection/README.md
- 测试图片：https://storage.googleapis.com/mediapipe-assets/pose.jpg 等 6 张（完整 URL 见 `docs/test_assets.md` §1）

（未访问/不可用：https://ai.google.dev/ —— 按任务说明未抓取；web_search 返回 HTTP 402。）
