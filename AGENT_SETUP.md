# 另一台电脑的 Agent 配置说明

目标：在 Windows 电脑上从零配置本仓库，跑通环境自检、103 项测试和摄像头界面。
仓库已包含 3 个 MediaPipe 模型文件（约 23 MB），**不需要另外下载模型**。

## 0. 机器要求

- Windows 10/11 x64（开发机为 Windows 11）。
- Python **3.12.x 64-bit**，来自 <https://www.python.org/downloads/windows/>；
  安装时勾选 **tcl/tk and IDLE**，并确认 `py -3.12 --version` 可用。
- 默认摄像头（index 0）或 USB 摄像头。Windows 设置 → 隐私和安全性 → 相机 →
  允许桌面应用访问相机（或允许应用访问相机）要打开。
- 只使用 CPU；不需要 CUDA、GPU、Node.js、ffmpeg 或 Git LFS。
- 磁盘：仓库约 30 MB，依赖（mediapipe/opencv wheel）安装后约 500 MB。
- 网络：只在 `pip install` 时需要；检测运行本身不依赖网络。

> 不要用 32-bit Python，也不要用 Python 3.13：mediapipe 0.10.35 的 Windows wheel
> 按 64-bit Python 3.12 验证过。Tkinter 缺失时界面会报 `No module named tkinter`，
> 重新运行 python.org 安装器并勾选 tcl/tk 即可。

## 1. 获取代码

```powershell
git clone https://github.com/YYiChen/CSX-kitchen-perception.git
cd CSX-kitchen-perception
```

没有 GitHub 凭据时，也可以在网页上 Download ZIP 后解压；仓库是公开的。

## 2. 安装依赖

推荐直接用一键脚本（会检查 Python 位数/Tkinter、安装依赖、校验模型并探测摄像头）：

```powershell
pwsh -File scripts/setup_windows.ps1
```

如果脚本因执行策略被拦截，可手动执行：

```powershell
py -3.12 -m pip install --upgrade pip
py -3.12 -m pip install -r requirements.txt
```

`requirements.txt` 固定为开发机实测版本：
`mediapipe==0.10.35`、`opencv-contrib-python==5.0.0.93`、
`numpy==2.3.5`、`Pillow==12.3.0`。

注意：**不要同时安装 `opencv-python` 和 `opencv-contrib-python`**。
mediapipe 自带 opencv-contrib-python 依赖，两个包会写入同一个 `cv2` 目录，
导致 `cv2.__version__` 不可控；本仓库只要求 opencv-contrib-python。
如果机器上已经有 `opencv-python`，先 `py -3.12 -m pip uninstall opencv-python`。

## 3. 环境自检

```powershell
py -3.12 scripts/check_env.py
```

预期看到 `环境就绪`，并且：

- Python 3.12.10 或其它 3.12.x（64-bit）
- mediapipe / opencv / numpy / Pillow 可导入
- 3 个模型的大小与 sha256 校验通过
- 摄像头读取成功（例如 `Integrated Camera 读取: 成功 (640x480)`）

没有摄像头或只想做无头验证时：

```powershell
py -3.12 scripts/check_env.py --no-camera
```

## 4. 跑测试（不需要摄像头，约 15 秒）

```powershell
py -3.12 -X utf8 tests/run_tests.py
```

预期最后一行：`Ran 103 tests ... OK`。

必须用 `tests/run_tests.py`，不要直接 `python -m unittest`：MediaPipe 在受限网络下
退出时会同步尝试上传 Clearcut 遥测，每个 landmarker 可能多阻塞约 40 秒；
该 runner 和程序入口都做了 `os._exit` 规避。

## 5. 启动演示界面

```powershell
pwsh -File scripts/run_demo.ps1
# 默认摄像头不是 index 0 时：
pwsh -File scripts/run_demo.ps1 -Index 1
# 直接打开一个视频：
pwsh -File scripts/run_demo.ps1 -Video "D:\path\to\chopping.mp4"
```

窗口左侧是摄像头/视频画面、骨架、五指颜色、A/B 分区和底部大标签；
右侧是“切菜状态”、躯干/手部/性能面板。首次加载 MediaPipe 模型约 1–3 秒。

界面内操作：

- 「导入视频…」：选择 MP4/AVI/MOV/MKV/WMV，自动缩放到最大宽度 1280、最大 30 fps 再检测。
- 「回到摄像头」：切回默认摄像头。
- 「导出标注视频…」：后台逐帧跑同一套 Pipeline，导出带标注 MP4；无音轨，比实时慢。
- 「显示 A/B 分区线和标记」：一键隐藏/显示分割线、分区底色和手部区域标记，设置持久化。

## 6. 无界面验证（可选）

```powershell
$env:PYTHONPATH = "$PWD\src"
py -3.12 -X utf8 -m kitchen_perception verify-models
py -3.12 -X utf8 -m kitchen_perception simulate --scenario a_to_b
py -3.12 -X utf8 -m kitchen_perception run --preview
```

## 7. 验收清单

1. `scripts/check_env.py` 返回 0 且打印“环境就绪”（无摄像头时加 `--no-camera`）。
2. `tests/run_tests.py` 打印 `Ran 103 tests` + `OK`。
3. `scripts/run_demo.ps1` 打开窗口，画面有骨架/五指颜色，右侧面板有数据。
4. 对着摄像头做几次抬手/放下动作，右侧“切菜状态”在等待/正在切菜/切菜完成之间变化。
5. 导入一段视频，确认画面比例正常，能导出 MP4（可选）。

## 8. 常见问题

- **`ModuleNotFoundError: kitchen_perception`**：没有设置源码路径。用 `scripts/run_demo.ps1`，
  或先 `$env:PYTHONPATH = "$PWD\src"` 再运行 `py -3.12 -m kitchen_perception ...`。
- **摄像头打不开**：关闭其它占用摄像头的程序（浏览器/会议软件），检查 Windows 相机隐私设置，
  或换 `--index 1`；程序会尝试 DSHOW → MSMF → ANY 三种后端。
- **Tkinter 报错**：Python 安装时没有勾选 tcl/tk，重跑 python.org 安装器修复。
- **测试正常但退出慢**：运行的是 `python -m unittest` 而不是 `tests/run_tests.py`；
  MediaPipe 遥测阻塞，换 runner 即可。
- **视频很卡/导出很慢**：4K 源会先缩到 1280 宽再检测，但解码仍是 4K；演示前可先用外部工具
  转成 720p，或直接用界面导入（程序会自动抽帧/缩放）。
- **导出的 MP4 没有声音**：这是预期行为。导出使用 OpenCV mp4v 静音输出，不依赖 ffmpeg；
  需要声音时用原视频另行合成。

## 9. 目录与文档

- `src/kitchen_perception/`：源码（引擎、几何后处理、UI、视频导出）。
- `models/`：MediaPipe task 模型 + `MANIFEST.json`（sha256），已随仓库提交。
- `tests/run_tests.py`：103 项无摄像头测试。
- `docs/migration.md`：迁移/离线部署细节。
- `docs/architecture.md`：模块、状态机、JSON schema。
- `docs/verification.md`：验证记录与已知限制。

## 10. 当前限制（配置前先知道）

- 单人、单摄像头；多人只取置信度最高的人。
- 真人切菜准确率只在 1 段侧身、固定机位、只露双手的视频上验证过，换机位/光照需彩排。
- 演示只回答“正在切菜/切菜完成”，不识别食材、菜刀或具体菜谱步骤。
- 视频导出无音轨且慢于实时；导出期间预览暂停，完成后点「继续」重放。
