# 迁移与部署（演示机）

演示机不是开发机，但代码与模型都要能原样跑起来。策略：**git 管理代码 + 模型随仓库提交 + 一台机器一个便携目录**。

## 1. 代码与模型

- 仓库包含：`src/`、`config/`、`models/`（3 个 task 文件，约 23 MB）、`assets/test/official/`（官方测试图）、`scripts/`、`docs/`、`tests/`
- 不包含：`outputs/`、`__pycache__`、虚拟环境
- `models/MANIFEST.json` 记录每个模型的 bytes/sha256/下载 URL；换机后可 `scripts/download_models.py` 重新下载并校验

## 2. 目标机准备

1. 安装 Python 3.12（勾选 Add to PATH）
2. 拷贝/克隆项目目录到任意路径（**中文路径没问题**，代码已处理）
3. 安装依赖：
   ```powershell
   cd <项目目录>
   py -3.12 -m pip install -r requirements.txt
   ```
   离线环境：先在开发机 `pip download -r requirements.txt -d wheels\`，把 wheels 目录一起拷过去，目标机 `pip install --no-index --find-links wheels -r requirements.txt`
   开发机实测版本：mediapipe 0.10.35、opencv-python 5.0.0、numpy 2.3.5、Pillow 12.3；requirements 用区间约束，
   演示机装好后可先跑 `scripts\check_env.py` 打印实际版本。
4. 校验：
   ```powershell
   $env:PYTHONPATH="$PWD\src"
   py -3.12 -X utf8 -m kitchen_perception verify-models
   py -3.12 -X utf8 scripts\check_env.py
   ```
   测试请用 `py -3.12 -X utf8 tests\run_tests.py`（52 项，约 11 秒）。不要用 `python -m unittest` 直接跑全套：
   MediaPipe 在离线网络下退出时会同步尝试上传遥测，每个 landmarker 会多等约 42 秒，虽然能跑完但很慢。
5. 启动：
   ```powershell
   .\scripts\run_demo.ps1
   ```

## 3. 摄像头检查

```powershell
py -3.12 -X utf8 -m kitchen_perception probe-cameras
```

默认摄像头 index 0；若有多个摄像头，改 `config/default.json` 的 `camera.index` 或用 `--index`（ui 子命令支持）。

## 4. git 迁移方式（推荐）

```powershell
# 开发机
git add -A; git commit -m "..."
git remote add origin <仓库地址>; git push -u origin main
# 演示机
git clone <仓库地址>; cd CSX_人体切菜检测
py -3.12 -m pip install -r requirements.txt
```

## 5. 便携包方式（无网络）

```powershell
# 开发机
.\scripts\pack_portable.ps1    # 产出 dist\CSX_kitchen_perception_portable.zip
# 演示机：解压 → 装依赖（或附带 wheels）→ 校验 → 运行
```

便携包包含代码、配置、模型、测试素材、scripts、docs，不含 outputs 与缓存。

## 6. 环境差异与已知风险

1. **CPU-only**：Windows 版 mediapipe pip wheel 不含 GPU delegate，`delegate=GPU` 会抛 NotImplementedError；性能按 CPU 规划（640×480 单帧 p95 约 66–82 ms）。
2. **离线遥测**：MediaPipe 原生库退出时会尝试上传 Clearcut 遥测；代码已用“非阻塞 close + 入口 `os._exit`”规避，检测本身不受影响。
3. **摄像头独占**：同一摄像头同一时间只能被一个进程稳定读取；演示前关掉其它用摄像头的程序（含旧窗口）。
4. **Python 版本**：只在 3.12.10 上实测；3.11 理论上可用但未实测，不建议演示机换版本。
5. **编码**：`scripts/*.ps1` 以 UTF-8 BOM 保存，Windows PowerShell 5.1 才能正确解析中文注释。

## 7. 迁移后自检清单

- [ ] `verify-models` 通过
- [ ] `check_env.py` 显示摄像头可用、依赖版本正常
- [ ] `snapshot --image assets/test/official/pose.jpg` 输出 posture 字段非空
- [ ] `tests/run_tests.py` 52 项全部通过
- [ ] `snapshot --image assets/test/official/fist.png` 输出 grip=fist
- [ ] 打开 UI 能看到画面与实时状态
- [ ] `benchmark --frames 100` 的 p95 ≤ 500 ms（若超出，程序会自动降级，必要时在 config 调整 `latency.levels`）
