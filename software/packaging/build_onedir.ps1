param(
    [ValidateSet('Release', 'Dev')]
    [string]$Mode = 'Release'
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$python = 'E:\CondaEnvs\rofee-bearing\python.exe'
if (!(Test-Path $python)) { throw "缺少项目专用环境：$python" }
$pyinstallerArguments = @('-m', 'PyInstaller', '--noconfirm')
if ($Mode -eq 'Release') {
    $pyinstallerArguments += '--clean'
}
$pyinstallerArguments += (Join-Path $root 'packaging\airwatch_onedir.spec')
Write-Host "AirWatch onedir 构建模式：$Mode"
& $python @pyinstallerArguments
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& (Join-Path $root 'packaging\prepare_portable_release.ps1')
& (Join-Path $root 'packaging\validate_portable_release.ps1')
if ($Mode -eq 'Dev') {
    Write-Host '开发增量构建已通过资源/哈希/禁止依赖校验；未执行耗时的包内模型 Workflow 自检。'
    Write-Host "开发构建完成：$root\dist\AirWatch\AirWatch.exe"
    exit 0
}
$smokeReport = Join-Path $root 'build\package-runtime-smoke.json'
if (Test-Path -LiteralPath $smokeReport) {
    Remove-Item -LiteralPath $smokeReport -Force
}
$env:AIRWATCH_SMOKE_REPORT = $smokeReport
$env:AIRWATCH_DATA_DIR = Join-Path $root 'build\package-smoke-data'
$env:QT_QPA_PLATFORM = 'offscreen'
$process = Start-Process `
    -FilePath (Join-Path $root 'dist\AirWatch\AirWatch.exe') `
    -ArgumentList '--runtime-smoke' `
    -WorkingDirectory (Join-Path $root 'dist\AirWatch') `
    -WindowStyle Hidden `
    -PassThru
if (!$process.WaitForExit(60000)) {
    Stop-Process -Id $process.Id -Force
    throw '发布包模型自检超过 60 秒。'
}
if ($process.ExitCode -ne 0) {
    throw "发布包模型自检失败，退出码：$($process.ExitCode)"
}
if (!(Test-Path -LiteralPath $smokeReport)) {
    throw "发布包模型自检未生成报告：$smokeReport"
}
$smoke = Get-Content -Raw -LiteralPath $smokeReport | ConvertFrom-Json
if ($smoke.status -ne 'PASS') {
    throw '发布包模型自检未通过。'
}
Write-Host "发布包模型自检通过：$smokeReport"
Write-Host "构建并验证完成：$root\dist\AirWatch\AirWatch.exe"
