# CSX 人体切菜检测 · 阶段一

电脑端（Windows）实时感知程序：调用**默认摄像头**，识别人体**躯干状态**；当画面里能看到手时，识别**五根手指的曲张状态**。菜刀、食材等物品不在阶段一范围内。

技术路线是成熟的 MediaPipe Tasks（Pose Landmarker + Hand Landmarker）+ 规则化几何后处理，**不训练模型、不自己写网络**。

## 功能

- 躯干状态：直立 / 前倾 / 后仰 / 侧倾 / 俯身 / 躯干不可见 / 未检测到人物（全部阈值和中文标签可在 config 里改）
- 五指状态：每根手指 伸直 / 半屈 / 弯曲 / 未知；总手势：张开 / 握拳 / 捏取 / 指向 / 半握
- 桌面窗口：左侧摄像头画面叠加骨架与五指颜色，右侧中文状态面板
- 实时性：自适应降级（L0–L3），目标单帧处理 p95 ≤ 250 ms，硬上限 500 ms
- 可扩展：`src/kitchen_perception/custom_rules.py` 提供按炒菜环节覆盖“躯干/手指”定义的钩子
- 每帧输出统一 JSON 快照，便于之后接感知/决策/反馈模型

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

# 3) 启动桌面界面
py -3.12 -X utf8 -m kitchen_perception ui

# 4) 无头运行 + OpenCV 预览（按 q 退出）
py -3.12 -X utf8 -m kitchen_perception run --preview

# 5) 延迟实测（默认 300 帧，输出 outputs/benchmark.json）
py -3.12 -X utf8 -m kitchen_perception benchmark --frames 300

# 6) 对单张图片出 JSON + 标注图（不需要摄像头）
py -3.12 -X utf8 -m kitchen_perception snapshot --image assets/test/official/pose.jpg --json outputs/snapshots/pose.json --overlay outputs/debug/pose.jpg
```

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
models/                          3 个 MediaPipe task 模型 + MANIFEST.json（sha256）
src/kitchen_perception/
  sources.py                     摄像头/视频帧源（中文路径、断线重连、时间戳）
  engines.py                     PoseLandmarker / HandLandmarker 封装
  torso.py                       躯干几何特征 + 状态分类 + 平滑/滞回/直立标定
  fingers.py                     五指关节角 + 曲张分类 + 手势分类
  overlay.py                     骨架/五指着色 + 中文 HUD
  pipeline.py                    推理流水线（降级、快照、JSON 契约）
  worker.py                      后台线程，供 Tkinter UI 轮询
  ui.py                          桌面界面（Tkinter）
  cli.py / __main__.py           命令行入口
  custom_rules.py                按环节定制定义的扩展点
tests/                           unittest 测试（几何/躯干/手指/预算/视频源/流水线）
docs/                            架构、验证、迁移文档
scripts/                         模型重下载、环境检查、一键验证、启动脚本
outputs/                         运行产物（默认被 .gitignore 忽略）
```

## 输出 JSON（每帧一个对象）

```json
{
  "schema_version": "1.0",
  "frame": {"index": 12, "ts_ms": 400.0, "width": 640, "height": 480, "source": "camera"},
  "perf": {"total_ms": 58.2, "p95_ms": 71.3, "level": "L0", "fps": 17.5, "stale_pose": false},
  "torso": {"present": true, "posture": "leaning_forward", "posture_zh": "前倾",
             "confidence": 0.86, "pitch_deg": 24.2, "roll_deg": 3.1, "facing": "toward_camera"},
  "hands": [{"label": "right", "grip": "open", "grip_zh": "张开",
             "fingers": {"thumb": "extended", "index": "extended", "middle": "extended", "ring": "extended", "pinky": "extended"}}],
  "hands_state": "in_workspace",
  "events": [{"type": "posture_change", "from": "upright", "to": "leaning_forward", "ts_ms": 400.0}]
}
```

完整字段见 `docs/architecture.md`。

## 实测性能（开发机，640×480，lite 模型）

| 指标 | 数值 |
| --- | --- |
| 单帧处理 p50 | 48–64 ms |
| 单帧处理 p95 | 66–82 ms |
| 单帧处理 max | 71–91 ms |
| 摄像头帧率 | 约 15–18 FPS（受曝光/环境光影响） |

以上为不同时段实测区间（Lead 自测与 T5 独立验证）。含姿态检测 + 手部检测 + 躯干/手指分析 + JSON 组装；未含 UI 绘制。低于 500 ms 硬上限，满足实时要求。

## 迁移到演示机

代码全程 git 管理，模型（约 23 MB）随仓库提交；迁移步骤见 `docs/migration.md`：

1. `git clone` 或直接拷贝整个项目目录（含 models/）
2. 目标机 `pip install -r requirements.txt`
3. `$env:PYTHONPATH="$PWD\src"; py -3.12 -m kitchen_perception verify-models` 校验模型
4. `scripts/check_env.py` 检查 Python/依赖/摄像头
5. 启动 `scripts/run_demo.ps1`

## 已知限制

- 只支持单摄像头、单人（num_poses=1）；多人场景只取置信度最高的人
- 手部像素过小、严重遮挡、快速运动模糊时，手指状态可能为“未知”
- 摄像头角度导致整体前倾角偏移时，用界面上的“标定直立基准”按钮归零
- 五指交叉或并拢时，基于 2D 关节角的规则可能把相邻手指判成半屈
- 笔记本摄像头常见只拍到上半身：此时髋部不可见，躯干姿态为**估算值**（界面会标注），置信度自动下调
- 近景/特写构图（脸和手占满画面）时，MediaPipe 髋部 landmark 可能明显错误，导致躯干误判且置信度偏高（官方图 woman_hands.jpg 可复现）；演示时保持上半身+胸腹入镜、距离 ≥1 m
- 一个 Windows 摄像头同一时间只能被一个进程稳定读取；开发/验证时避免多个程序同时抢摄像头
- 离线/受限网络下，MediaPipe 原生库退出时会同步尝试上传 Clearcut 遥测，`close()` 或正常退出
  可能阻塞约 42 秒/实例（本机实测）。程序用“非阻塞 close + 入口 `os._exit`”规避，属上游库行为，
  不影响检测速度；演示机若联网正常则无此现象

## 文档索引

- `docs/architecture.md`：模块、数据流、状态定义、JSON schema、设计决策与踩坑
- `docs/verification.md`：验证方法与验收清单
- `docs/verification_report.md`：T5 独立验证报告（结论、A1–A6、问题清单、未验证项）
- `docs/migration.md`：演示机迁移/离线部署步骤
- `docs/research_notes.md`、`docs/test_assets.md`：调研素材与实测数据（researcher 产出）
