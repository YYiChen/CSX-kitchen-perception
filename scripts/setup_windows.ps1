# 另一台 Windows 电脑的一键环境配置：检查 Python/Tkinter、安装依赖、校验模型/摄像头。
param(
    [switch]$SkipInstall,
    [switch]$NoCamera
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "=== CSX 人体切菜检测 · 环境配置 ==="
Write-Host "项目根: $root"

$py = Get-Command py -ErrorAction SilentlyContinue
if ($null -eq $py) {
    throw "未找到 Python launcher。请安装 64-bit Python 3.12（python.org），安装时勾选 tcl/tk。"
}

& py -3.12 -c "import sys; print('Python', sys.version.split()[0])"
if ($LASTEXITCODE -ne 0) {
    throw "py -3.12 启动失败。请安装 64-bit Python 3.12（python.org）。"
}

$bits = & py -3.12 -c "import struct; print(struct.calcsize('P') * 8)"
if ($bits.Trim() -ne "64") {
    throw "当前 Python 是 $($bits.Trim())-bit，本项目的 MediaPipe wheel 需要 64-bit Python。"
}

& py -3.12 -c "import tkinter; print('Tkinter OK')"
if ($LASTEXITCODE -ne 0) {
    throw "缺少 Tkinter。请重新运行 python.org 安装器并勾选 tcl/tk and IDLE。"
}

if (-not $SkipInstall) {
    Write-Host ""
    Write-Host "=== 安装依赖（requirements.txt 固定版本）==="
    & py -3.12 -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "pip 升级失败" }
    & py -3.12 -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw "依赖安装失败；确认网络可访问 PyPI。" }
} else {
    Write-Host ""
    Write-Host "=== 跳过依赖安装（-SkipInstall）==="
}

Write-Host ""
Write-Host "=== 环境/模型/摄像头自检 ==="
$env:PYTHONPATH = Join-Path $root "src"
$checkArgs = @("-3.12", "-X", "utf8", "scripts/check_env.py")
if ($NoCamera) { $checkArgs += "--no-camera" }
& py @checkArgs
$code = $LASTEXITCODE

Write-Host ""
Write-Host "=== 下一步 ==="
Write-Host "测试: py -3.12 -X utf8 tests/run_tests.py"
Write-Host "界面: pwsh -File scripts/run_demo.ps1"
Write-Host "说明: AGENT_SETUP.md"
exit $code
