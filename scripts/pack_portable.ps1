# 打包便携目录（不含 outputs/缓存），便于拷贝到演示机
param(
    [string]$Output = "dist\CSX_kitchen_perception_portable.zip"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$stage = Join-Path $env:TEMP ("kp_pack_" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $stage | Out-Null
$items = @("src", "config", "models", "scripts", "docs", "tests", "README.md", "requirements.txt")
foreach ($item in $items) {
    $source = Join-Path $root $item
    if (Test-Path $source) {
        $dest = Join-Path $stage $item
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dest) | Out-Null
        Copy-Item -Path $source -Destination $dest -Recurse -Force
    }
}
# 官方测试图只带演示/测试需要的小图；模型卡 PDF 与抓取源属调研素材，不进便携包
$officialSrc = Join-Path $root "assets\test\official"
if (Test-Path $officialSrc) {
    $officialDest = Join-Path $stage "assets\test\official"
    New-Item -ItemType Directory -Force -Path $officialDest | Out-Null
    Get-ChildItem -Path $officialSrc -File |
        Where-Object { $_.Extension -in ".jpg", ".jpeg", ".png", ".json" } |
        Copy-Item -Destination $officialDest -Force
}
Get-ChildItem -Path $stage -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force

$outPath = Join-Path $root $Output
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $outPath) | Out-Null
if (Test-Path $outPath) { Remove-Item -LiteralPath $outPath -Force }
Compress-Archive -Path (Join-Path $stage "*") -DestinationPath $outPath -CompressionLevel Optimal
Remove-Item -Recurse -Force $stage
Write-Host "已生成 $outPath ($([math]::Round((Get-Item $outPath).Length / 1MB, 1)) MB)"
