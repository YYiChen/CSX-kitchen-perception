# 一键验证：模型校验 + 单元/集成测试 + 静态图片端到端（可选 benchmark）
param(
    [switch]$Benchmark,
    [int]$Frames = 120
)

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $root "src"
$env:PYTHONIOENCODING = "utf-8"
Set-Location $root

$failed = 0

Write-Host ""
Write-Host "[1/5] 模型校验" -ForegroundColor Cyan
& py -3.12 -X utf8 -m kitchen_perception verify-models
if ($LASTEXITCODE -ne 0) { $failed++ }

Write-Host ""
Write-Host "[2/5] unittest" -ForegroundColor Cyan
& py -3.12 -X utf8 tests/run_tests.py
if ($LASTEXITCODE -ne 0) { $failed++ }

Write-Host ""
Write-Host "[3/5] 静态图片端到端" -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path (Join-Path $root "outputs\snapshots") | Out-Null
& py -3.12 -X utf8 -m kitchen_perception snapshot --image assets/test/official/pose.jpg --json outputs/snapshots/verify_pose.json
if ($LASTEXITCODE -ne 0) { $failed++ }

Write-Host ""
Write-Host "[4/5] 活动状态机场景回放" -ForegroundColor Cyan
& py -3.12 -X utf8 -m kitchen_perception simulate --scenario a_to_b --json outputs/simulate_verify_a_to_b.json
if ($LASTEXITCODE -ne 0) { $failed++ }

if ($Benchmark) {
    Write-Host ""
    Write-Host "[5/5] 延迟实测（关闭其它摄像头程序）" -ForegroundColor Cyan
    & py -3.12 -X utf8 -m kitchen_perception benchmark --frames $Frames
    if ($LASTEXITCODE -ne 0) { $failed++ }
} else {
    Write-Host ""
    Write-Host "[5/5] 跳过 benchmark（加 -Benchmark 开启）" -ForegroundColor DarkGray
}

Write-Host ""
Write-Host "=== 结果 ===" -ForegroundColor Cyan
if ($failed -eq 0) {
    Write-Host "全部通过" -ForegroundColor Green
    exit 0
} else {
    Write-Host "$failed 项失败" -ForegroundColor Red
    exit 1
}
