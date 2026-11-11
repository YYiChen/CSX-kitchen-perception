# CSX 人体切菜检测 · 演示版 + 阶段一/二

电脑端（Windows）实时感知程序：调用**默认摄像头**，识别**人体躯干状态**与**五指曲张状态**；阶段二增加**A/B 操作区**与**切菜节奏/活动状态**识别。2026-11-11 增加**两状态演示模式**（正在切菜 / 切菜完成），用于固定机位、侧身切菜视频的稳定演示；同一天增加**界面内视频导入、标注预览与 MP4 导出**，A/B 分区线可一键显示/隐藏。菜刀、食材等物品识别仍不在范围内。

技术路线是成熟的 MediaPipe Tasks（Pose Landmarker + Hand Landmarker）+ 规则化几何/时序后处理，**不训练模型、不自己写网络**。

## 功能

### 演示版（2026-11-11，当前视频演示目标）

- 只显示三个状态：**等待切菜 / 正在切菜 / 切菜完成**；画面底部中央与右侧面板均为大号中文标签
- demo.py 的 DemoStateTracker 独立于阶段二活动状态机，直接复用 HandMotionTracker 的切菜节奏特征
- 阈值集中在 config/default.json 的 demo 节：确认 start_confirm_ms=300、停止 done_after_ms=900、反转 >=2、频率 0.5-4 Hz、振幅比 >=0.02、score >=0.5
- 演示面板的“状态变化”只记录两状态切换，不再混入 ACTIVE_A/B、MOVING、TRANSITION 等旧活动状态
- 已用一段真实侧身切菜视频验证：2.833 s 进入“正在切菜”，17.367 s 进入“切菜完成”，视频结束前保持完成；551 帧 p95 81.3 ms，L0（详见 docs/verification.md）
- 阶段一/二的检测能力保留在 snapshot 与调试字段中，便于后续扩展；演示主面板只强调两状态

### 视频识别与导出（2026-11-11）

- **两种视频源**：启动时默认用摄像头；左侧工具栏「导入视频…」可切换为本地视频，画面与实时模式使用同一套标注/状态逻辑
- **应用内导出**：「导出标注视频…」把当前视频逐帧重新跑一遍 Pipeline，写出带骨架、五指、分区、HUD、两状态大标签的 MP4；导出进度显示在底部状态栏，可取消
- **4K 友好**：导入/导出默认缩放到最大宽度 1280、按最大 30 fps 抽帧（4K 60 fps 素材处理一半帧，输出约 30 fps）；缩放发生在模型之前，4K 源直接导入即可，不需要手工转码（4K 解码本身较慢，18 s 素材导出约 79 s）
- **A/B 分区线开关**：主面板复选框「显示 A/B 分区线和标记」控制画面上的分割线、A/B 底色、手部区域标记；开关状态保存到 `config/zones.json`，实时预览和导出视频都生效
- **导出不依赖 ffmpeg**：使用 OpenCV mp4v 静音输出；需要音频时可用原视频工具自行合成。导出比实时播放慢（本机约 1 帧/0.1 s），属正常现象

### 阶段一（已完成）

- 躯干状态：直立 / 前倾 / 后仰 / 侧倾 / 俯身 / 躯干不可见 / 未检测到人物（阈值与中文标签可配置）
- 五指状态：每根手指 伸直 / 半屈 / 弯曲 / 未知；总手势：张开 / 握拳 / 捏取 / 指向 / 半握
- 桌面窗口：左侧摄像头画面叠加骨架与五指颜色，右侧中文状态面板
- 实时性：自适应降级（L0–L3），目标单帧处理 p95 ≤ 250 ms，硬上限 500 ms
- 可扩展：`custom_rules.py` 提供按炒菜环节覆盖“躯干/手指”定义的钩子
- 每帧输出统一 JSON 快照

### 阶段二（2026-11-11 完成）

- **A/B 分区**：按人体重心（髋→肩→躯干框兜底）判断当前在画面左/右半区；分割线比例、死区、确认时间可配置，设置保存到 `config/zones.json`
- **切菜节奏**：优先用掌心、无手时用姿态手腕的纵向往复运动，计算速度、反转次数、频率、振幅和稳定性；只有“像切菜”的节奏才进入 ACTIVE，普通移动只算 MOVING
- **活动状态机**：`UNKNOWN / ACTIVE_A / ACTIVE_B / MOVING / PAUSED / TRANSITION_A_TO_B / TRANSITION_B_TO_A / OCCLUDED / AWAY`；带进入/退出延时、最短持续时间与事件冷却，避免抖动
- **事件日志**：确认事件写入 `outputs/events/events_*.jsonl`，同时保留最近 50 条供界面显示；写失败不影响检测
- **界面**：新增“情境感知”面板（活动状态/持续/移动/质量/切菜节奏）、事件日志、分区设置对话框（快捷键 Z）、保存分区配置按钮；演示版把主面板改为“切菜状态”，状态变化只显示两状态
- **无摄像头可验证**：`simulate` 子命令用合成观测序列演示/回归活动状态机

## 环境

- Windows 10/11，Python 3.12（开发机实测 3.12.10）
- 依赖：`mediapipe`、`opencv-python`、`numpy`、`Pillow`（见 requirements.txt，无其它第三方依赖）
- 摄像头：电脑自带或 USB 摄像头，默认 index 0

## 快速开始

```powershell
# 0) 进入项目根目录，按需创建虚拟环境（可选）
# 1) 安装依赖
py -3.12 -m pip install -r requirements.txt

# 2) 校验模型文件（大小 + sha256）
$env:PYTHONPATH="$PWD\src"
py -3.12 -X utf8 -m kitchen_perception verify-models

# 3) 启动桌面界面（演示主面板 + 阶段一/二细节）
py -3.12 -X utf8 -m kitchen_perception ui

# 3b) 用侧身切菜视频做两状态演示（按实际路径替换；UI 内按空格暂停/继续）
py -3.12 -X utf8 -m kitchen_perception ui --video outputs/video_review/chopping_720p_30fps.mp4

# 3c) 无界面导出带标注的 MP4（可选；不启动 Tk 界面）
py -3.12 -X utf8 -m kitchen_perception export-video --video outputs/video_demo/chopping_720p_input.mp4 --output outputs/exports/demo_annotated.mp4

# 4) 无头运行 + OpenCV 预览（按 q 退出）
py -3.12 -X utf8 -m kitchen_perception run --preview

# 5) 延迟实测（默认 300 帧，输出 outputs/benchmark.json）
py -3.12 -X utf8 -m kitchen_perception benchmark --frames 300

# 6) 对单张图片出 JSON + 标注图（不需要摄像头）
py -3.12 -X utf8 -m kitchen_perception snapshot --image assets/test/official/pose.jpg --json outputs/snapshots/pose.json --overlay outputs/debug/pose.jpg

# 7) 合成观测演示活动状态机（不需要摄像头）
py -3.12 -X utf8 -m kitchen_perception simulate --scenario a_to_b
py -3.12 -X utf8 -m kitchen_perception simulate --scenario occluded
```

可用合成场景：`chop_a`、`chop_b`、`a_to_b`、`pause`、`moving`、`occluded`、`away_return`、`no_person`。

运行测试：

```powershell
py -3.12 -X utf8 tests/run_tests.py -v
```

> 用 `tests/run_tests.py` 而不是直接 `python -m unittest`：MediaPipe 在离线/受限网络下
> 退出时会同步尝试上传遥测，每个 landmarker 多阻塞约 42 秒；该 runner 用 `os._exit`
> 跳过解释器清理，全套测试约 11 秒跑完。

## 目录

```text
config/default.json              全部阈值/标签/降级级别/UI/输出配置
config/zones.json                （可选）界面“保存分区配置”后生成；覆盖默认分割线/标签
models/                          3 个 MediaPipe task 模型 + MANIFEST.json（sha256）
src/kitchen_perception/
  sources.py                     摄像头/视频帧源（中文路径、断线重连、时间戳）
  engines.py                     PoseLandmarker / HandLandmarker 封装
  torso.py                       躯干几何特征 + 状态分类 + 平滑/滞回/直立标定
  fingers.py                     五指关节角 + 曲张分类 + 手势分类
  zones.py                       A/B 分区：身体重心/掌心投影、死区、设置持久化
  motion.py                      切菜节奏：手腕/掌心轨迹、反转、频率、振幅、稳定性
  activity.py                    活动状态机：ACTIVE/PAUSED/MOVING/TRANSITION/OCCLUDED/AWAY
  demo.py                        两状态演示：等待切菜/正在切菜/切菜完成
  events.py                      事件 JSONL 追加写 + 内存最近事件
  exporter.py                    后台视频导出：逐帧标注并写 MP4（与预览同一 Pipeline）
  overlay.py                     骨架/五指着色 + A/B 分区 + 中文 HUD
  pipeline.py                    推理流水线（降级、分区、活动、事件、快照）
  worker.py                      后台线程，供 Tkinter UI 轮询
  ui.py                          桌面界面（Tkinter）：摄像头/视频源切换、标注预览、导出、分区开关
  cli.py / __main__.py           命令行入口（含 simulate / export-video）
  custom_rules.py                按环节定制定义的扩展点
tests/                           unittest 测试（几何/躯干/手指/预算/视频源/流水线/分区/节奏/活动/事件/模拟/演示/导出）
docs/                            架构、验证、迁移、UI 文档
scripts/                         模型重下载、环境检查、一键验证、启动、便携包
outputs/                         运行产物（默认被 .gitignore 忽略）
```

## 输出 JSON（schema 1.1，每帧一个对象）

```json
{
  "schema_version": "1.1",
  "camera_status": "CONNECTED",
  "frame": {"index": 12, "ts_ms": 400.0, "width": 640, "height": 480, "source": "camera"},
  "perf": {"total_ms": 45.3, "p95_ms": 69.8, "level": "L0", "fps": 15.2},
  "zones": {
    "split_x": 0.5, "deadband": 0.02, "view_mirrored": true,
    "labels": {"A": "备菜区", "B": "处理区"},
    "body_center_norm": [0.72, 0.61], "body_zone": "B", "zone_source": "hips",
    "hand_zone_mismatch": false, "hands": []
  },
  "torso": {"present": true, "posture": "leaning_forward", "posture_zh": "前倾",
             "confidence": 0.86, "pitch_deg": 24.2, "roll_deg": 3.1, "facing": "toward_camera"},
  "hands": [{"label": "right", "grip": "open", "grip_zh": "张开", "zone": "B",
             "fingers": {"thumb": "extended", "index": "extended", "middle": "extended", "ring": "extended", "pinky": "extended"}}],
  "hands_state": "in_workspace",
  "activity": {
    "state": "ACTIVE_B", "state_zh": "B 区切菜中", "zone": "B",
    "duration_ms": 1200.0, "movement_level": "HIGH", "observation_quality": "VALID",
    "chopping": {"detected": true, "stable": true, "rhythm_hz": 2.0, "score": 0.92}
  },
  "demo": {
    "state": "chopping", "state_zh": "正在切菜", "duration_ms": 1200.0,
    "raw_detected": true, "rhythm_hz": 1.9, "reversals": 5, "score": 0.88
  },
  "planned_step": null,
  "observed_step": null,
  "events": []
}
```

事件日志（`outputs/events/events_*.jsonl`）每行字段：`event / ts_ms / frame / state / previous_state / zone / duration_ms / quality / event_source / wall_time / evidence`。

完整字段见 `docs/architecture.md`。

## 实测性能（开发机，640×480，lite 模型）

| 指标 | 数值 |
| --- | --- |
| 单帧处理 p50 | 45.5 ms |
| 单帧处理 p95 | 69.8 ms |
| 单帧处理 max | 72.3 ms |
| 摄像头帧率 | 约 12–18 FPS（受曝光/环境光影响） |

2026-11-11 benchmark：120 帧，verdict PASS，L0；含姿态 + 手部 + 躯干/手指分析 + 分区 + 活动 + 事件 + JSON 组装，未含 UI 绘制。低于 500 ms 硬上限，满足实时要求。测试套件 **103 项**（约 15 s，含视频导出合成测试）。

真实侧身切菜视频（1280×720、30 fps、551 帧）：p50 60.9 ms / p95 81.3 ms / max 103.3 ms，L0；2.833 s 进入“正在切菜”，17.367 s 进入“切菜完成”，视频结束前保持完成。该视频验证的是两状态演示逻辑与界面显示，不等同于全厨房场景准确率。

## A/B 分区与活动判定怎么调

- 分割线默认在画面中心 `zones.split_x=0.5`，死区 `deadband=0.02`；界面“分区设置…”（Z）可拖动/输入 0.10–0.90，A 在左、B 在右（按显示画面，已包含镜像）
- 主面板复选框「显示 A/B 分区线和标记」可随时隐藏/显示分割线、A/B 底色与手部区域标记；该开关与分区设置一起写入 `config/zones.json`，下次启动保持
- 人整体重心在左半区 = A，右半区 = B；重心来源优先髋部，其次肩部，最后躯干框
- 普通手部移动不会触发 ACTIVE：需要切菜节奏（默认 0.8–4.0 Hz、至少 4 次反转、振幅比 ≥ 0.04、周期/振幅变异系数上限 0.35/0.45，见 `chopping` 配置）
- 进入/退出延时、遮挡/离开阈值、转移窗口都在 `config/default.json` 的 `activity` / `chopping` / `zones` 节
- 想按具体炒菜环节改“什么算切菜/什么算 A 区”，可改配置或 `custom_rules.py`

## 两状态演示怎么调

- 演示状态在 config/default.json 的 demo 节；labels_zh 可改中文文案，start_confirm_ms 控制进入“正在切菜”的连续确认，done_after_ms 控制手停后多久显示“切菜完成”
- 误把普通移动判成切菜：提高 min_reversals / min_amplitude_ratio / min_score；漏检切菜：适当降低这些值或 min_rhythm_hz
- 演示界面启动：py -3.12 -X utf8 -m kitchen_perception ui --video <视频路径>；右侧“切菜状态”与画面底部标签同步
- 想回到阶段二完整活动状态，看 snapshot 的 activity 字段或运行 simulate 子命令；演示面板不再展示这些内部状态

## 迁移到演示机

代码全程 git 管理，模型（约 23 MB）随仓库提交；迁移步骤见 `docs/migration.md`：

1. `git clone` 或直接拷贝整个项目目录（含 models/）
2. 目标机 `pip install -r requirements.txt`
3. `$env:PYTHONPATH="$PWD\src"; py -3.12 -m kitchen_perception verify-models` 校验模型
4. `scripts/check_env.py` 检查 Python/依赖/摄像头
5. 启动 `scripts/run_demo.ps1`
6. 无网络时用 `scripts/pack_portable.ps1` 生成 `dist/CSX_kitchen_perception_portable.zip`（约 19 MB，含代码/模型/测试/文档，不含 outputs 与用户 `config/zones.json`）；迁移前建议把整个项目目录连同 `.git` 一起备份
7. 界面内视频导出只用 OpenCV，不需要 ffmpeg/额外编码器；导出的 MP4 无音轨，需要音频时用原视频重新合成

## 已知限制

- 只支持单摄像头、单人（num_poses=1）；多人场景只取置信度最高的人
- 手部像素过小、严重遮挡、快速运动模糊时，手指状态可能为“未知”
- 摄像头角度导致整体前倾角偏移时，用界面上的“标定直立基准”按钮归零
- 五指交叉或并拢时，基于 2D 关节角的规则可能把相邻手指判成半屈
- 笔记本摄像头常见只拍到上半身：此时髋部不可见，躯干姿态为**估算值**（界面会标注），置信度自动下调
- 近景/特写构图（脸和手占满画面）时，MediaPipe 髋部 landmark 可能明显错误，导致躯干误判且置信度偏高（官方图 woman_hands.jpg 可复现）；演示时保持上半身+胸腹入镜、距离 ≥1 m
- 空场景中挂着的衣服/椅背可能被 MediaPipe 误检成人体；已增加“角度超出可信范围 → uncertain”门限，但极端构图仍可能误检
- **真人切菜验证范围有限**：已用一段真实侧身、固定机位、只露出双手的切菜视频验证两状态演示（切菜开始/完成时间正确）；但全厨房、正面机位、不同人/光照尚未统计验证，正式演示前仍建议用目标机位彩排一次
- **两状态演示只回答“正在切/切完”**：不识别切什么、切了多少、是否换区；旧活动状态机仍在运行，但演示面板不展示
- **视频导出是离线处理**：比实时慢（本机约 56 s 处理 18 s 的 720p 视频），导出期间预览暂停；完成/取消后点「继续」重放视频
- **导出 MP4 无声**：mp4v 静音输出，不依赖 ffmpeg；需要保留现场音时另行合成
- A/B 分区按身体重心而非菜板位置；如果人站在画面中间但手伸到另一侧，可能出现“身体在 A、手在 B”的提示，需调 `split_x` 或让身体对准操作区
- 一个 Windows 摄像头同一时间只能被一个进程稳定读取；开发/验证时避免多个程序同时抢摄像头
- 离线/受限网络下，MediaPipe 原生库退出时会同步尝试上传 Clearcut 遥测，`close()` 或正常退出
  可能阻塞约 42 秒/实例（本机实测）。程序用“非阻塞 close + 入口 `os._exit`”规避，属上游库行为，
  不影响检测速度；演示机若联网正常则无此现象

## 文档索引

- `docs/architecture.md`：模块、数据流、状态定义、JSON schema、设计决策与踩坑
- `docs/verification.md`：验证方法与验收清单（阶段一 + 阶段二 + 两状态演示）
- `docs/verification_report.md`：T5 阶段一独立验证报告（历史记录）
- `docs/migration.md`：演示机迁移/离线部署步骤
- `docs/ui_notes.md`：界面结构、交互与自测记录
- `docs/research_notes.md`、`docs/test_assets.md`：调研素材与实测数据
