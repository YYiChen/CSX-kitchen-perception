# 启动演示界面（自动设置 PYTHONPATH）
param(
    [string]$Video = "",
    [int]$Index = 0,
    [ValidateSet("lite", "full")][string]$PoseModel = "lite"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $root "src"
$env:PYTHONIOENCODING = "utf-8"

$arguments = @("-3.12", "-X", "utf8", "-m", "kitchen_perception", "--pose-model", $PoseModel, "ui")
if ($Video -ne "") {
    $arguments += @("--video", $Video)
} else {
    $arguments += @("--index", "$Index")
}

Write-Host "启动: py $($arguments -join ' ')   (项目根: $root)"
Set-Location $root
& py @arguments
exit $LASTEXITCODE
