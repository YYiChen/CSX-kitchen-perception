# 验证方案与记录

## 1. 验收标准（阶段一）

| 编号 | 验收项 | 通过标准 |
| --- | --- | --- |
| A1 | 摄像头调用 | 默认摄像头可打开，画面连续，掉线能重连 |
| A2 | 躯干状态识别 | 直立/前倾/后仰/侧倾/俯身 能被稳定区分，切换有滞回不抖动 |
| A3 | 五指状态识别 | 手在画面内时，5 指各自输出 伸直/半屈/弯曲/未知；张开、握拳、捏取、指向可区分 |
| A4 | 实时性 | 单帧处理 p95 ≤ 250 ms（目标），≤ 500 ms（硬上限）；不足时自动降级 |
| A5 | 工程交付 | git 仓库干净可提交；README/迁移文档完整；测试可一条命令跑通 |
| A6 | 可扩展性 | 阈值/标签可配置；custom_rules.py 可覆盖定义 |

## 2. 自动化验证

```powershell
$env:PYTHONPATH="$PWD\src"
# 模型完整性
py -3.12 -X utf8 -m kitchen_perception verify-models
# 单元/集成测试（无需摄像头）
py -3.12 -X utf8 tests/run_tests.py -v
# 静态图片端到端
py -3.12 -X utf8 -m kitchen_perception snapshot --image assets/test/official/pose.jpg --json outputs/snapshots/verify_pose.json --overlay outputs/debug/verify_pose.jpg
# 延迟实测（需要摄像头，运行时不要开其它摄像头程序）
py -3.12 -X utf8 -m kitchen_perception benchmark --frames 300 --json outputs/benchmark.json
```

> 测试请用 `tests/run_tests.py`。直接 `python -m unittest` 也能跑，但 MediaPipe 在离线网络下
> 退出时每个 landmarker 会额外阻塞约 42 秒（Clearcut 遥测）；runner 用 `os._exit` 跳过该等待。

## 3. 手工验证（演示机）

1. 站/坐在摄像头前，确认躯干标签正确：直立站立 → 前倾 → 后仰 → 侧倾 → 俯身（每次保持 2 秒）
2. 手伸到画面里，依次做：五指张开、握拳、食指指向、拇指食指捏取；确认面板与叠加颜色变化
3. 打开“标定直立基准”后再站立，确认前倾角归零
4. 快速挥动手，确认不崩溃、延迟不超 500 ms，必要时观察降级级别
5. 拔掉摄像头再插回，确认能自动重连或给出明确错误提示

## 4. 当前记录（开发机实测）

| 项目 | 结果 | 备注 |
| --- | --- | --- |
| 模型 sha256 校验 | 通过 | 3 个模型与 MANIFEST 一致 |
| unittest | **52 项全部通过**（2026-11-10，10.5 s） | 几何/配置/躯干/手指/预算/序列化/视频源/集成/CLI；用 tests/run_tests.py；含 T5 后补的二次标定/滞回/降级窗口/缺键兜底用例 |
| 单帧处理延迟（640×480，lite） | p50 ≈ 64 ms / p95 ≈ 71 ms / max ≈ 73 ms | 姿态+手部+分析，未含 UI 绘制 |
| 摄像头 benchmark（2026-11-10，120 帧） | p50 62.4 ms / p95 82.4 ms / max 90.9 ms，verdict PASS，L0 | outputs/benchmark.json；目标 p95≤250、硬上限 500 |
| T5 独立验证 benchmark（2026-11-10，120 帧，verifier 独占摄像头） | p50 47.95 ms / p95 65.86 ms / max 70.81 ms，verdict PASS，L0 | outputs/verify_t5/benchmark.json；报告 docs/verification_report.md |
| UI 摄像头自测（ui-builder，15 s） | 155 帧、FPS 16.2、p95 52 ms、L0、躯干「前倾」置信度 0.61 | 见 docs/ui_notes.md §5.2 |
| UI 关闭退出（Lead 实测） | 窗口立即关闭，进程约 4.7 s 退出 | 真实 `python -m kitchen_perception ui` + WM_CLOSE，无 40 s 卡顿 |
| 摄像头帧率 | 约 15–18 FPS | 受曝光/环境光影响 |
| 官方图片五指分类（端到端 snapshot） | fist.png → grip=fist（五指全 flexed）；right_hands.jpg → open（五指全 extended）；woman_hands.jpg → 左手 partial（拇指 partial、四指 extended）/ 右手 pinch | 产物 outputs/snapshots/{fist,right_hands,woman_hands}.json |
| MediaPipe 退出阻塞 | 单实例 close 从约 44 s 降到约 1.5 s 上限；全套测试 254 s → 10.6 s | 非阻塞 close + 入口 os._exit |

> 注：本机实测时若另一个进程占用摄像头，read 会阻塞约 1 s/帧，导致 wall-clock FPS 降低但不影响单帧处理耗时统计；跑 benchmark 前请关闭其它用摄像头的程序。

## 5. 验收对照（2026-11-10）

T5 独立验证已完成（verifier，2026-11-10）：结论“有条件通过”，详见 docs/verification_report.md；下表为综合 Lead 自测与 T5 复核后的证据。

| 编号 | 当前状态 | 证据 |
| --- | --- | --- |
| A1 摄像头调用 | 通过（Lead 实测） | probe-cameras 列出 index 0；UI 15 s 155 帧；断线重连有单测 |
| A2 躯干识别 | 通过（静态图 + UI 实测） | pose.jpg upright 0.90；UI 实时「前倾」0.61；单测覆盖 5 状态 + 滞回；风险：近景特写构图可能误判（报告 S2-3，已写入 README 已知限制） |
| A3 五指识别 | 通过（官方图端到端） | fist→fist、right_hands→open、woman_hands→partial/pinch；52 项单测 |
| A4 实时性 | 通过 | benchmark 120 帧 p95 82.4 ms（≤250）；UI p95 52 ms |
| A5 工程交付 | 通过（T5 复核） | git 已提交；verify_all.ps1 全通过；便携包 18.83 MiB；52 项测试 10.5 s |
| A6 可扩展性 | 通过 | config/default.json 阈值/标签；custom_rules.py 钩子 + 单测 |
