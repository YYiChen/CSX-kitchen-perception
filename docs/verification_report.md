# T5 独立验证报告（简版）

**结论：有条件通过。** 核心链路均实测可复现；发现 2 项 S2 行为缺陷 + 1 项 S2 风险 + S3 若干与 4 处文档不一致，均不阻塞阶段一交付。用户已人肉实测确认识别准确，本报告仅归档独立复核证据。

## 环境与 commit
Windows 11（26200）、Python 3.12.10、mediapipe 0.10.35、opencv-python 5.0.0、numpy 2.3.5、Pillow 12.3.0；CPU-only、摄像头 index 0（dshow）。commit **288c6c7**（`288c6c70c9a8d7802e5735d879b12ea48fe5911d`），验证开始时 `git status --short` 为空，本报告为唯一未跟踪文件（预期）。命令统一 `$env:PYTHONPATH="$PWD\src"; py -3.12 -X utf8`，产物在 `outputs/verify_t5/`。

## A1–A6
| 项 | 结论 | 证据 |
| --- | --- | --- |
| A1 摄像头调用 | **pass** | benchmark 120 帧默认摄像头 `camera:0 (dshow)` 连续读取；重连代码在 `src/kitchen_perception/sources.py:106-134` |
| A2 躯干识别 | **有条件 pass** | pose.jpg→upright 0.897；male_full_height→upright 0.698；UI 实时→前倾 0.45；**风险**：woman_hands.jpg→bent_over 1.0（S2-3） |
| A3 五指识别 | **pass** | fist.png→fist（五指 flexed）；right/left/male 官方图→open；woman_hands→左 partial/右 pinch |
| A4 实时性 | **pass** | p50 47.95 / **p95 65.86** / max 70.81 ms，verdict PASS，L0（目标 250、硬上限 500） |
| A5 工程交付 | **pass** | 48 tests 10.54 s OK；`verify_all.ps1` 全通过；便携包 19,750,305 B（18.83 MiB）54 项完整；`check_env.py --no-camera` exit 0 |
| A6 可扩展性 | **pass** | 覆盖 `upright_pitch_deg` 15→5 后 pitch=12 由 upright 变 leaning_forward；custom_rules 钩子实测被调用（torso note=`verifier hook fired`、grip=`point`） |

其他实测：3 个模型 sha256 与 MANIFEST **独立比对全部一致**；UI 自测（独立跑）：摄像头 6 s 7 帧/p95 78 ms/L0、视频 5 s 50 帧/p95 85 ms/L0，无回调错误；snapshot 冷启动首帧 65–108 ms。

## 问题清单
**S2-1 连续两次「标定直立基准」会清掉基准**：`src/kitchen_perception/torso.py:314-324` 读的 `self.last.features` 已在 `torso.py:374-381` 减过 offset。实测：标定#1 后 pitch 0.00，再标定 offsets→0，下一帧 pitch 回 25.46°、posture 变回 leaning_forward（`outputs/verify_t5/audit_probe.log` PROBE-1）。
**S2-2 降级预算用陈旧窗口，慢帧后会继续升级**：`src/kitchen_perception/budget.py:59-81` 切换级别不清 120 帧窗口、无冷却。实测 30 帧 300 ms 后转 80 ms：L1→L2→L3（t=90），t=150 才回 L2、t=180 回 L1（PROBE-3）。
**S2-3（风险，模型级）近景构图躯干误判且置信度 1.0**：`torso.py:141-146` 只看 MediaPipe 髋部 visibility、`torso.py:241-278` 无几何合理性守卫。官方图 woman_hands.jpg 输出 bent_over/1.0/pitch 82.3°/roll 75.2°；原始 landmark 髋部 y≈0.546–0.567 与肩 0.513–0.566 几乎同高（明显错姿态）但 vis≈0.997，估算惩罚不触发（`outputs/verify_t5/probe_landmarks.log`）。建议加几何守卫 + README 补限制。

**S3 建议**：`src/kitchen_perception/geometry.py:84-86` 滞回 stable_frames 恒为 1；`torso.py:335` no_person confidence=1.0 语义混乱；`torso.py:246-249` 缺 threshold 子键无兜底（实测 KeyError: 'upright_pitch_deg'）；`torso.py:219-227` roll 项把置信度封顶 0.975；`torso.py:262-271` 前倾无条件覆盖侧倾；`custom_rules.py:58-71` 加载失败缓存永久化、torso 钩子无双签名兼容；`budget.py:35,100` `infer_window` 死代码、`analysis_scale`（L3=0.75）从未生效；`types.py:121` `hands[].stale` 恒 false；`worker.py:135-139` 暂停/恢复竞态（UI 已用轮询规避）；`pipeline.py:99` 手部 busy 轨迹键不稳；`pipeline.py:227` `sleep_live` 未用、视频不按 fps 节流；人离开后 EMA/滞回不重置。

## 文档不一致（S3）
`docs/verification.md:50` 写 woman_hands 左手 open，实测 left=partial（与 Lead 自己 `outputs/snapshots/woman_hands.json` 一致）；`docs/migration.md:21` 写 opencv 4.10，实测 5.0.0；`docs/architecture.md:84` 写 L3 手部隔 2 帧，实际 `config/default.json:108` hand_every_n=3；`README.md:113` 迁移命令缺 PYTHONPATH，直接执行报 `No module named kitchen_perception`。

## 未验证项
真实切菜场景/厨房实拍、多人（num_poses=1）、遮挡与快速运动、摄像头热插拔实测、>1 分钟长跑、全新 venv 按 requirements 安装（网络受限）、演示机迁移实机演练、标定跨进程持久化。

## 附件
`outputs/verify_t5/`：benchmark.json、6 张官方图 snapshot json/jpg、tests_raw.log、audit_probe.py/.log、probe_landmarks.log、ui_camera_raw.log、ui_video_raw.log。

## Lead 修复记录（2026-11-10 23:52，验证后）
- S2-1 已修：`torso.py calibrate()` 改为 new=old+current，连续两次标定不再清基准；新增 `test_calibration_twice_keeps_baseline`。
- S2-2 已修：`budget.py` 级别切换时清空统计窗口；新增 `test_stale_window_does_not_keep_escalating`（30 帧 300ms→120 帧 80ms 不再连升到 L3）。
- S2-3 保留为已知限制：代码未加几何守卫（避免影响用户已确认准确的实时判定），已在 README「已知限制」写明近景构图风险与演示距离建议。
- S3 已修：滞回 stable_frames 累计（S3-1）、缺 threshold 子键 .get 兜底（S3-5）、custom_rules 失败缓存 3s TTL + torso 钩子单参数兼容（S3-6）、移除 budget.infer_window 死代码（S3-7 部分）、hands[].stale 按隔帧标记（S3-8）。
- 文档 D1–D5 已修：woman_hands 左手 partial、opencv 5.0.0、L3 hand_every_n=3、README 补 PYTHONPATH、migration 章节编号。
- 未修（记录为后续增强）：no_person confidence 语义（S3-2）、roll 封顶（S3-3）、前倾覆盖侧倾（S3-4）、worker 暂停竞态（S3-9）、busy 轨迹键（S3-10）、视频节流（S3-11）、人离开后 EMA/滞回重置（S3-12）、analysis_scale 未接入。
- 修复后回归：`tests/run_tests.py` **52 项全部通过**（新增 4 项：二次标定、stable_frames、降级陈旧窗口、缺 threshold 键）；T5 探针复跑：二次标定 offsets 保持 25.46°、stable_frames 1→8、缺键不再抛异常、降级序列 L1→L0 不误升。
