# 官方测试图与 MediaPipe 实测结果（assets/test/official/）

> 实测时间：2026-11-10 22:40–22:50（Asia/Shanghai）。数据由 `assets/test/official/measure_v2.py` 在本机生成，原始数据见同目录 `measurements.json`；图片 sha256/尺寸见 `sources/assets_manifest.json`。

## 1. 素材清单与来源

全部 6 张图片用 `py -3.12` + `urllib`（User-Agent 自设，非浏览器）从 Google MediaPipe 官方公共资源桶 `storage.googleapis.com/mediapipe-assets/` 下载，均 HTTP 200。

| 文件 | 来源 URL | 像素尺寸 | 字节数 | sha256（前 16 位） |
|---|---|---|---|---|
| `pose.jpg` | https://storage.googleapis.com/mediapipe-assets/pose.jpg | 1000×667 | 44100 | `c8a830ed683c0276` |
| `male_full_height_hands.jpg` | https://storage.googleapis.com/mediapipe-assets/male_full_height_hands.jpg | 638×1000 | 55658 | `8a7fe5be8b90d607` |
| `right_hands.jpg` | https://storage.googleapis.com/mediapipe-assets/right_hands.jpg | 720×382 | 54887 | `4b5134daa4cb6046` |
| `left_hands.jpg` | https://storage.googleapis.com/mediapipe-assets/left_hands.jpg | 720×382 | 54888 | `240c082e80128ff1` |
| `fist.png` | https://storage.googleapis.com/mediapipe-assets/fist.png | 358×354 | 207870 | `4397b3d3f590c88a` |
| `woman_hands.jpg` | https://storage.googleapis.com/mediapipe-assets/woman_hands.jpg | 640×960 | 67719 | `70cbeb38e198c986` |

**许可说明（重要）**：这些图片是 MediaPipe 官方示例/文档使用的测试素材；MediaPipe 代码仓库与官方模型均为 Apache License 2.0（见 research_notes.md §6），但单张图片的版权未逐张标注。本项目仅把图片用于本地离线测试/评估，不再分发、不用于训练或商用；若演示材料要对外发布，建议换成自拍素材。

## 2. 检测设置

- Python 3.12.10（`py -3.12`），mediapipe 0.10.35，numpy 2.3.5，opencv-python 5.0.0.93，CPU 逻辑核 32（本机无可用 GPU delegate，原因见 §6 第 6 条）。
- 模型：`models/pose_landmarker_lite.task`（5,777,746 B，sha256 `59929e1d1ee95287`…）、`models/pose_landmarker_full.task`（9,398,198 B）、`models/hand_landmarker.task`（7,819,105 B）。均用 `model_asset_buffer` 读取（规避仓库路径含中文）。
- PoseLandmarker：`num_poses=1`，detection/presence/tracking 阈值均 0.5，`output_segmentation_masks=False`，`RunningMode.IMAGE`（每图独立检测）。
- HandLandmarker：`num_hands=2`，三个阈值均 0.5，`RunningMode.IMAGE`。
- 五指判定规则（仅本测试表口径，不是最终产品阈值）：`total_flexion = Σ(180° − 关节内角)`（每指取 2 个关节，定义见 research_notes.md §3）；`<30°` → extended，`30–110°` → partial，`>110°` → curled。

## 3. 躯干姿态检出结果

| 图片 | 尺寸 | pose_lite | pose_full | 说明 |
|---|---|---|---|---|
| `pose.jpg` | 1000×667 | 1 | 1 | 全身可见，姿态可信（本轮唯一全身样本） |
| `male_full_height_hands.jpg` | 638×1000 | 1 | 1 | 全身可见，两者均检出 |
| `right_hands.jpg` | 720×382 | 0 | 0 | 只有手，预期无人体，0 正确 |
| `left_hands.jpg` | 720×382 | 0 | 0 | 只有手，预期无人体，0 正确 |
| `fist.png` | 358×354 | 0 | 0 | 只有手，预期无人体，0 正确 |
| `woman_hands.jpg` | 640×960 | 1 | 0 | 仅上半身；lite 检出 1（但躯干角失真），full 检出 0 |

`pose.jpg` 的 world landmarks 实测（单位 m，原点在两髋中点），用于核对坐标轴：

- 左髋(23) = [0.1018, 0.0102, -0.0464]，右髋(24) = [-0.1025, -0.0105, 0.0467]；两髋中点 ≈ [-0.0003, -0.0001, 0.0002] → 与“原点在髋部中心”一致【已实测】。
- 肩中点 y = -0.4891，髋中点 y ≈ 0 → 肩在上方，**y 轴向下**（y 越小越高）【已实测】。
- 鼻尖 z − 肩中点 z = -0.1161 m（< 0）→ 鼻尖比肩更靠近相机，**z 负方向朝相机**；与 BlazePose 模型卡“negative values are between the hips and the camera”一致【已实测 + 模型卡】。
- 由肩线/躯干向量算出的角度：肩线 roll ≈ -3.41°，yaw ≈ -21.45°（侧身约 21°），躯干左右倾 ≈ 2.07°、前后倾 ≈ 3.68°。

## 4. 手部检出与五指曲张实测

表 A：检出数与 handedness（模型原始输出，未做镜像交换）：

| 图片 | hand 检出数 | 左右手标签（置信度） |
|---|---|---|
| `pose.jpg` | 0 | — |
| `male_full_height_hands.jpg` | 2 | Left (0.9577)；Right (0.9365) |
| `right_hands.jpg` | 2 | Right (0.9433)；Right (0.9606) |
| `left_hands.jpg` | 2 | Left (0.9679)；Left (0.9562) |
| `fist.png` | 1 | Right (0.989) |
| `woman_hands.jpg` | 2 | Left (0.936)；Right (0.9622) |

表 B：每只手五指 `total_flexion`（度；顺序 拇/食/中/无/小）。3D 用 hand world landmarks，2D 用归一化图像坐标；末两列为规则判定与视觉预期：

| 图片 | 手# | 标签 | 3D 五指 | 2D 五指 | 规则判定（拇/食/中/无/小） | 视觉预期 |
|---|---|---|---|---|---|---|
| `male_full_height_hands.jpg` | 0 | Left | 17/51/52/42/37 | 11/3/3/3/1 | extended/partial/partial/partial/partial | 全伸 |
| `male_full_height_hands.jpg` | 1 | Right | 27/55/40/30/35 | 9/8/14/12/7 | extended/partial/partial/partial/partial | 全伸 |
| `right_hands.jpg` | 0 | Right | 33/29/26/21/42 | 21/4/1/1/1 | partial/extended/extended/extended/partial | 全伸 |
| `right_hands.jpg` | 1 | Right | 29/37/33/19/38 | 20/3/1/1/0 | extended/partial/partial/extended/partial | 全伸 |
| `left_hands.jpg` | 0 | Left | 40/89/34/53/33 | 24/3/3/2/6 | partial/partial/partial/partial/partial | 全伸 |
| `left_hands.jpg` | 1 | Left | 53/91/30/53/29 | 20/5/2/4/3 | partial/partial/extended/partial/extended | 全伸 |
| `fist.png` | 0 | Right | 109/168/195/168/175 | 127/189/198/204/192 | partial/curled/curled/curled/curled | 四指弯 + 拇指半弯 |
| `woman_hands.jpg` | 0 | Left | 18/56/38/35/57 | 14/16/16/19/19 | extended/partial/partial/partial/partial | 全伸 |
| `woman_hands.jpg` | 1 | Right | 74/58/45/44/38 | 64/8/4/6/15 | partial/partial/partial/partial/partial | 全伸 |

每只手的 landmark 数均为 21（索引 0–20）。world landmarks 示例（`fist.png` hand0，单位 m）：wrist(0) = [-0.0083,0.0815,0.0034]，middle MCP(9) = [0.0015,-0.0047,0.0058]，middle tip(12) = [0.0143,0.0116,-0.0038]；坐标量级 ≤ 0.09 m，说明原点在手掌附近、单位为米【已实测】。

## 5. 逐图结论 / 期望状态

### `pose.jpg`

- 视觉预期：画面中人双臂水平展开（瑜伽战士二式），手很小，预期 hand 检出 0；躯干/全身姿态是主要测试目标。视觉预期五指：全伸（张开），但此分辨率下手部不可用。
- 实测：pose_lite=1，pose_full=1，hands=0；首帧耗时 lite 43.9 / full 53 / hand 52.6 ms（含初始开销，仅量级参考）。
- 结论：手在画面中太小，hand 检出 0（预期内）；本图只用于验证躯干/全身 pose，不用于手指测试。

### `male_full_height_hands.jpg`

- 视觉预期：站立全身照，双手张开（一手在体侧、一手贴胸）。视觉预期：双手五指全伸（open hand）。
- 实测：pose_lite=1，pose_full=1，hands=2；首帧耗时 lite 43.6 / full 49.6 / hand 50.3 ms（含初始开销，仅量级参考）。
- 结论：全身站立、双手张开；hand 两个都检出；3D 五指 30–55° 判 partial，2D 仅 1–14° 判 extended。视觉为张开手掌，说明小/中尺寸手的 3D 角度被 GHUM 深度拟合放大（手 bbox 对角线约 126/100 px，见 `sources/hand_bbox_px.json`）→ 用 2D 或按手掌像素尺寸做置信分级。

### `right_hands.jpg`

- 视觉预期：两只右手：左侧为手背朝外、指尖向下；右侧为掌心朝外、指尖向上。视觉预期：两只手五指全伸（张开）。
- 实测：pose_lite=0，pose_full=0，hands=2；首帧耗时 lite 22.3 / full 22.9 / hand 51.1 ms（含初始开销，仅量级参考）。
- 结论：两手均检出，handedness 与文件名一致（均 Right）；2D 五指 0–24° 大多判 extended/partial；3D world 个别手指（hand0 拇指 32.5°）明显偏大，与视觉不符 → **平面手掌上 3D 角不可靠，2D 为主更稳**。

### `left_hands.jpg`

- 视觉预期：两只左手：左侧为掌心朝外、指尖向上；右侧为手背朝外、指尖向下。视觉预期：两只手五指全伸（张开）。
- 实测：pose_lite=0，pose_full=0，hands=2；首帧耗时 lite 20.6 / full 22.8 / hand 50.9 ms（含初始开销，仅量级参考）。
- 结论：两手均检出，handedness 与文件名一致（均 Left）；2D 五指 0–24° 大多判 extended/partial；3D world 个别手指（hand0 食指 88.9°）明显偏大，与视觉不符 → **平面手掌上 3D 角不可靠，2D 为主更稳**。

### `fist.png`

- 视觉预期：握拳（手背/指节朝向镜头）。视觉预期：食指/中指/无名指/小指弯曲（curled），拇指横向压在中指上、属部分弯曲。
- 实测：pose_lite=0，pose_full=0，hands=1；首帧耗时 lite 19.9 / full 23.2 / hand 38.6 ms（含初始开销，仅量级参考）。
- 结论：四指 3D flexion 约 168/195/168/175°（>110° → curled），与视觉一致；**拇指 109° 判 partial、未达 curled**——握拳时拇指横向压在手指上，与其它四指同向弯曲假设不同，必须拇指特判（见 research_notes.md §3）。2D 与 3D 在本图基本一致（四指 2D 189–204°）。

### `woman_hands.jpg`

- 视觉预期：半身照，双手张开挡在脸/额前。视觉预期：双手五指全伸（张开）；画面只有上半身，可作“躯干关键点不完整”测试。
- 实测：pose_lite=1，pose_full=0，hands=2；首帧耗时 lite 42.8 / full 49.8 / hand 53.6 ms（含初始开销，仅量级参考）。
- 结论：双手张开（前景手更大），hand 两个都检出；lite 在此半身图检出 1 个 pose（full 0 个），但由 landmark 推出的躯干角（肩线 roll 71.7°、左右倾 75.3°）明显失真 → **躯干关键点不全时必须输出 unknown，不能直接用角度**。手部 2D 角度多数 8–19°（extended 倾向），3D 37–74°（partial）。

## 6. 注意事项（给开发用）

1. **3D world landmarks 角度不可靠**：本测试中张开的手 3D 五指总弯曲 17–91°，同图 2D 仅 0–24°。MediaPipe 模型卡也说明 3D 坐标由 GHUM 合成数据拟合得到。手指曲张判定应优先用 2D 归一化坐标角度，3D 只作辅助（或仅对近距离大手掌使用）。
2. **handedness 语义未用真实摄像头验证**：官方双人测试图（非镜像照片）上模型输出与文件名左右一致；但 MediaPipe 文档称 handedness 默认假设自拍镜像输入。接入本项目摄像头（cv2 默认不镜像）后必须实测一次，若相反就在代码里交换标签。
3. **握拳拇指需要特判**：fist.png 中拇指 109° 被判 partial。建议拇指单独阈值 + 结合他指状态（四指 curled 时拇指判定为 tucked/按压）。
4. **pose_full 比 pose_lite 更保守**：woman_hands.jpg 上 lite 检出 1、full 检出 0；轻量模型抗部分遮挡更好，但也更容易在残缺画面上“编”出姿态，必须配 landmark presence/visibility 门槛与合理性检查（肩宽、肩-髋距离、躯干比例）。
5. **首次调用耗时明显高于稳定值**（模型加载 + TFLite delegate 初始化）；性能数字以 `measurements.json` 的 15 次重复统计为准。
6. **GPU delegate 不可用**：本机 mediapipe 0.10.35 Windows wheel 为 CPU-only。实测 `BaseOptions(delegate=Delegate.GPU)` 创建即抛 `NotImplementedError: ValidatedGraphConfig Initialization failed.` 且日志含 `ImageCloneCalculator: GPU processing is disabled in build flags`（见 `sources/gpu_probe.log`）。要做 GPU 需自行编译 MediaPipe/换 ONNX-TensorRT 等，超出本项目范围。
7. **`close()` 会阻塞（本机网络特有）**：MediaPipe Tasks 的 Clearcut 遥测在后端不可达时，`landmarker.close()`/进程正常退出会等待 ~40 s/实例（实测 42,330 ms 一次）。实测脚本因此用 `os._exit(0)` 收尾；产品代码应复用 landmarker 实例、避免反复创建/关闭。

## 7. 复现命令

```powershell
# 工作目录 = 项目根
# 下载：见本任务说明中的 6 个 URL，用 urllib 逐张保存到 assets/test/official/（脚本 measure_v2.py 假定文件已存在）
py -3.12 -u assets/test/official/measure_v2.py          # 图像检测 + VIDEO 时间戳 + IMAGE 基准（约 10 s）
py -3.12 -u assets/test/official/measure_video_bench.py  # VIDEO 跟踪基准（640x480 完整；1280x720 pose/hand 完整）
```

脚本：`measure_v2.py`（主实测）、`measure_video_bench.py`（VIDEO 跟踪基准）、`measure_official_images.py`（第一版，含 close()，在本机会卡住，保留作对照）。原始数据：`measurements.json`；VIDEO 基准输出见 `measure_video_bench.log`（未单独落盘 JSON）；抓取的源数据见 `sources/`。

