# 验证方案与记录

## 1. 验收标准

### 阶段一（躯干 + 五指）

| 编号 | 验收项 | 通过标准 |
| --- | --- | --- |
| A1 | 摄像头调用 | 默认摄像头可打开，画面连续，掉线能重连 |
| A2 | 躯干状态识别 | 直立/前倾/后仰/侧倾/俯身 能被稳定区分，切换有滞回不抖动 |
| A3 | 五指状态识别 | 手在画面内时，5 指各自输出 伸直/半屈/弯曲/未知；张开、握拳、捏取、指向可区分 |
| A4 | 实时性 | 单帧处理 p95 ≤ 250 ms（目标），≤ 500 ms（硬上限）；不足时自动降级 |
| A5 | 工程交付 | git 仓库干净可提交；README/迁移文档完整；测试可一条命令跑通 |
| A6 | 可扩展性 | 阈值/标签可配置；custom_rules.py 可覆盖定义 |

### 阶段二（A/B 分区 + 切菜节奏 + 活动状态）

| 编号 | 验收项 | 通过标准 |
| --- | --- | --- |
| A7 | A/B 操作区 | 一条可配置左右分割线；身体重心判区；死区/EMA/确认防抖；预览镜像方向一致；配置可保存并在重启后生效 |
| A8 | 切菜节奏与活动状态 | 只有切菜节奏才进 ACTIVE_A/B；普通移动为 MOVING；停手为 PAUSED；A↔B 有 TRANSITION；手遮挡/离开有 OCCLUDED/AWAY；状态与事件有单测/场景回放证据 |
| A9 | 事件与界面 | 事件写入 JSONL 且 UI 可查看；分区设置对话框可用；快照 schema 1.1 完整；不新增依赖 |
| A10 | 两状态视频演示 | 切菜阶段 UI 与叠加层显示“正在切菜”；停手离开后显示“切菜完成”，且在视频结束前保持；右侧状态变化只出现两状态；无头 JSON 含 demo 字段 |

## 2. 自动化验证

```powershell
$env:PYTHONPATH="$PWD\src"
# 模型完整性
py -3.12 -X utf8 -m kitchen_perception verify-models
# 单元/集成测试（无需摄像头，98 项）
py -3.12 -X utf8 tests/run_tests.py -v
# 静态图片端到端
py -3.12 -X utf8 -m kitchen_perception snapshot --image assets/test/official/pose.jpg --json outputs/snapshots/verify_pose.json --overlay outputs/debug/verify_pose.jpg
# 活动状态机场景回放（不需要摄像头）
py -3.12 -X utf8 -m kitchen_perception simulate --scenario a_to_b --json outputs/simulate_a_to_b.json
# 两状态演示：侧身切菜视频（本机分析副本；没有该文件时用自己的视频）
py -3.12 -X utf8 -m kitchen_perception run --video outputs/video_review/chopping_720p_30fps.mp4 --jsonl outputs/video_demo/run_snapshots.jsonl --save-overlay outputs/video_demo/overlays
# 延迟实测（需要摄像头，运行时不要开其它摄像头程序）
py -3.12 -X utf8 -m kitchen_perception benchmark --frames 300 --json outputs/benchmark.json
```

> 测试请用 `tests/run_tests.py`。直接 `python -m unittest` 也能跑，但 MediaPipe 在离线网络下
> 退出时每个 landmarker 会额外阻塞约 42 秒（Clearcut 遥测）；runner 用 `os._exit` 跳过该等待。

## 3. 手工验证（演示机）

### 3.1 阶段一

1. 站/坐在摄像头前，确认躯干标签正确：直立站立 → 前倾 → 后仰 → 侧倾 → 俯身（每次保持 2 秒）
2. 手伸到画面里，依次做：五指张开、握拳、食指指向、拇指食指捏取；确认面板与叠加颜色变化
3. 打开“标定直立基准”后再站立，确认前倾角归零
4. 快速挥动手，确认不崩溃、延迟不超 500 ms，必要时观察降级级别
5. 拔掉摄像头再插回，确认能自动重连或给出明确错误提示

### 3.2 阶段二

1. 打开 UI，点“分区设置…”，把分割线拖到画面中央附近，点“应用”看到 A/B 半透明底色变化；
   点“保存到配置文件”后重启 UI，确认分割线位置保持
2. 站在画面左半边做切菜动作（手臂上下往复，约 2 Hz），观察右侧“情境感知”出现“A 区切菜”
3. 换到右半边继续切，观察状态经“转移中”进入“B 区切菜”，事件日志出现区域转移事件
4. 停止切菜但手留在画面内，约 2 秒后显示“暂停”；普通移动手但不做切菜节奏，显示“移动”
5. 用身体或布遮住手约 1 秒，显示“手部遮挡”；离开画面约 5 秒，显示“用户离开”，回来后显示“继续观察”
6. 检查 `outputs/events/*.jsonl` 每行是一条带时间戳的 JSON 事件；关闭/重开 UI 不报错

### 3.3 两状态视频演示（2026-11-11）

1. 运行 py -3.12 -X utf8 -m kitchen_perception ui --video <侧身切菜视频>，确认右侧主面板出现“切菜状态”，画面底部中央出现同文案大标签
2. 切菜阶段应显示“正在切菜”；画面里手停止/离开后约 0.9 s 显示“切菜完成”
3. 右侧“状态变化”只出现“正在切菜”“切菜完成”两条，不应出现 ACTIVE_B / TRANSITION 等旧活动状态
4. 完成状态应在视频结束前保持足够时间（本机样例约 0.97 s），方便录制/截图
5. 若切菜中短暂停顿就跳到“切菜完成”，增大 demo.done_after_ms；若普通移动误报“正在切菜”，提高 demo.min_reversals / demo.min_amplitude_ratio

> 阶段二的“切菜/活动”准确率目前只经过合成观测序列与空场景 smoke，**真人切菜准确率仍需在
> 演示机上按上述步骤复核**。这不是自动化测试能替代的，报告中应如实标注。

## 4. 当前记录（开发机实测）

| 项目 | 结果 | 备注 |
| --- | --- | --- |
| 模型 sha256 校验 | 通过 | 3 个模型与 MANIFEST 一致 |
| unittest（阶段一） | 52 项通过（2026-11-10，10.5 s） | 几何/配置/躯干/手指/预算/序列化/视频源/集成/CLI |
| unittest（阶段二 + 演示） | **98 项通过**（2026-11-11，10.8 s） | 在 92 项基础上新增 test_demo 6 项；全部无摄像头 |
| 单帧处理延迟（640×480，lite） | p50 ≈ 64 ms / p95 ≈ 71 ms / max ≈ 73 ms | 姿态+手部+分析，未含 UI 绘制 |
| 摄像头 benchmark（阶段一） | p50 62.4 ms / p95 82.4 ms / max 90.9 ms，PASS，L0 | outputs/benchmark.json |
| 摄像头 benchmark（阶段二，2026-11-11） | **p50 45.52 ms / p95 69.79 ms / max 72.25 ms，PASS，L0**，FPS 11.87 | outputs/benchmark_v2.json；目标 p95≤250、硬上限 500 |
| UI smoke（阶段二，空场景，2026-11-11） | 42 帧、面板更新 21 次、latest_age 15 ms、worker 存活、分区对话框可开可关 | outputs/ui_smoke/；截图 ui_smoke.png；空场景无人属预期 |
| 活动场景回放 | 8 个场景均通过 | `simulate` 子命令；详见 `5`、outputs/simulate_*.json |
| 官方图片五指分类 | fist→fist；right_hands→open；woman_hands→左手 partial / 右手 pinch | 阶段一产物 outputs/snapshots/*.json |
| UI 关闭退出 | 窗口立即关闭，进程约 4.7 s 退出 | 非阻塞 close + 入口 os._exit，无 40 s 卡顿 |
| 空场景误检门限 | 修复前挂衣被判“后仰”（pitch −169°）；修复后 uncertain，活动 UNKNOWN | torso.py 的 plausible_pitch/roll_deg；新增单测 |
| 真实侧身切菜视频（两状态） | 2.833 s 进入“正在切菜”，17.367 s 进入“切菜完成”，视频 18.333 s 结束前保持；p50 60.9 / p95 81.3 / max 103.3 ms，L0 | 551 帧、1280x720、30 fps；状态只切换 2 次；outputs/video_demo/ |
| UI 视频演示 smoke | “正在切菜”与“切菜完成”各截图 1 张；右侧状态变化只有 2 条；面板与叠加层一致 | outputs/video_demo/ui_video_chopping.png / ui_video_done.png |
| T5 独立验证（阶段一） | “有条件通过”，S2-1/S2-2 已修复并回归 | docs/verification_report.md |

> 注：本机实测时若另一个进程占用摄像头，read 会阻塞约 1 s/帧，导致 wall-clock FPS 降低但不影响单帧处理耗时统计；跑 benchmark 前请关闭其它用摄像头的程序。

## 5. 阶段二场景回放证据

`simulate` 用 `ActivityStateMachine` 合成观测（不调用摄像头、不依赖 MediaPipe），验证状态机分支：

| 场景 | 关键期望 | 结果 |
| --- | --- | --- |
| chop_a | 进入 ACTIVE_A，发 ACTIVITY_A_STARTED / CHOPPING_STARTED | 通过 |
| chop_b | 进入 ACTIVE_B | 通过 |
| a_to_b | ACTIVE_A → TRANSITION_A_TO_B → ACTIVE_B，转移事件各 1 次 | 通过 |
| pause | ACTIVE_A 后停手 → PAUSED | 通过 |
| moving | 手可见、普通移动 → MOVING（不是 ACTIVE） | 通过 |
| occluded | 人可见、无手 → OCCLUDED，且不误报 ACTIVITY_PAUSED | 通过 |
| away_return | 离开 6 s → AWAY，回来 → USER_RETURNED，最终 PAUSED | 通过 |
| no_person | 无人 6 s → AWAY，不误报 HAND_OCCLUDED | 通过 |

### 5.1 真实侧身切菜视频（两状态演示）

用户提供的原始视频为 3840x2160、59.82 fps、1102 帧、18.42 s；为贴近摄像头输入，分析用副本为 1280x720、30 fps、551 帧（每 2 帧取 1）。视频中人物侧身、头脚不可见，只能看到双手摆动；约 16.5 s 放下菜刀并离开画面。

实际运行结果（outputs/video_demo/run_snapshots_final.jsonl）：

| 指标 | 结果 |
| --- | --- |
| 处理帧数 / 耗时 | 551 帧 / 55.0 s |
| 单帧处理 | p50 60.9 ms / p95 81.3 ms / max 103.3 ms；L0，PASS |
| 演示状态区间 | waiting 0–2.800 s；chopping 2.833–17.333 s；done 17.367–18.333 s |
| 状态切换次数 | 2 次（waiting 到 chopping、chopping 到 done），无中途抖动 |
| raw_detected | 377/551（68.4%）；chopping 状态覆盖 436 帧（79.1%） |
| 最后检测到切菜 | 16.467 s；之后 0.9 s 转 done；视频结束前 done 保持 0.967 s |

结论：在这段固定机位、侧身、只露手的视频上，两状态逻辑与显示均满足演示要求；它不代表全厨房、正面机位、多人或不同光照下的准确率。

## 6. 验收对照（2026-11-11）

| 编号 | 当前状态 | 证据 |
| --- | --- | --- |
| A1 摄像头调用 | 通过（Lead 实测） | probe-cameras 列出 index 0；UI smoke 42 帧；断线重连有单测 |
| A2 躯干识别 | 通过（静态图 + UI 实测） | pose.jpg upright；单测覆盖 5 状态 + 滞回；近景特写构图风险已写入已知限制 |
| A3 五指识别 | 通过（官方图端到端） | fist→fist、right_hands→open、woman_hands→partial/pinch；单测覆盖五指枚举 |
| A4 实时性 | 通过 | 阶段二 benchmark p95 69.79 ms（≤250）；UI smoke p95 约 71 ms |
| A5 工程交付 | 通过 | git 提交；verify_all.ps1；便携包；98 项测试 10.8 s |
| A6 可扩展性 | 通过 | config/default.json 阈值/标签；custom_rules.py 钩子 + 单测 |
| A7 A/B 分区 | 通过（单测 + UI 对话框 smoke） | test_zones 14 项；分区设置保存/重读；镜像坐标单测 |
| A8 切菜/活动状态 | 通过（1 段侧身视频 + 合成场景；全厨房待复核） | test_motion 7 项、test_activity 10 项、test_simulate 4 项；8 场景回放；真实视频见 5.1 |
| A9 事件与界面 | 通过 | JSONL 单测 4 项；UI 事件日志 smoke；快照 schema 1.1 字段齐全 |
| A10 两状态视频演示 | 通过（本机样例） | 2.833 s 正在切菜、17.367 s 切菜完成；UI 两状态截图；outputs/video_demo/ |

## 7. 已知限制与待复核项

1. **真人切菜准确率范围有限**：已用 1 段侧身、固定机位、只露双手的视频验证两状态；正面机位、全厨房、多人、不同光照/服装仍待复核，阈值可能需按演示机位微调。
2. **A/B 只按身体重心**：不做鼠标框选矩形、不做多边形；身体重心贴近分割线时依赖死区与确认时间。
3. **活动状态是粗粒度**：只回答在哪个区、是否在切、是否暂停/移动/遮挡/离开，不识别洗、盛、调味等具体步骤。
4. **单人假设**：多人入镜时取置信度最高的人，不保证区分。
5. **空场景误检**：已加角度可信范围门限，但极端挂衣/海报仍可能被斜向判为人体。
6. **摄像头独占**：同一摄像头同一时间只能被一个进程稳定读取。
7. **离线遥测**：MediaPipe 退出时会尝试上传 Clearcut，已用非阻塞 close + `os._exit` 规避。
8. **性能与分辨率**：640×480 CPU 实测 p95 约 70 ms；换更高分辨率或更慢的演示机时，程序会自动降级，必要时调整 `config/default.json` 的 `latency.levels`。
9. **UI 自适应**：窗口会按屏幕分辨率收缩，但 1280×800 以下屏幕右侧面板可能需要滚动；演示机建议 1366×768 以上。
10. **两状态演示依赖切菜节奏**：只回答“正在切/切完”，不判断切什么、切多少；切菜节奏的阈值用 demo 节单独配置，与阶段二 activity 的阈值相互独立。
