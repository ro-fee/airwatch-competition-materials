# Portable release staging helper. Build with the dedicated Conda environment.
param(
    [string]$Package = (Join-Path (Split-Path $PSScriptRoot -Parent) 'dist\AirWatch'),
    [string]$Version = '0.1.0-dev'
)

$ErrorActionPreference = 'Stop'

function Get-Sha256([string]$Path) {
    $algorithm = [Security.Cryptography.SHA256]::Create()
    $stream = [IO.File]::OpenRead($Path)
    try {
        return ([BitConverter]::ToString($algorithm.ComputeHash($stream))).Replace('-', '')
    }
    finally {
        $stream.Dispose()
        $algorithm.Dispose()
    }
}

$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$exe = Join-Path $Package 'AirWatch.exe'
$internal = Join-Path $Package '_internal'
$manifestPath = Join-Path $internal 'runtime-resources.json'
if (!(Test-Path -LiteralPath $exe)) { throw "请先构建 $exe" }
if (!(Test-Path -LiteralPath $manifestPath)) {
    throw "缺少打包生成的运行时资源清单：$manifestPath"
}

$manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
$capabilities = [ordered]@{}
foreach ($slot in $manifest.model_slots) {
    $capabilities[[string]$slot.slot_id] = [ordered]@{
        display_name = [string]$slot.display_name
        status = [string]$slot.status
        release_tier = [string]$slot.release_tier
        limitation = [string]$slot.limitation
    }
}
$release = [ordered]@{
    product = '空域电波哨兵'
    package = 'AirWatch-Windows-x64-portable'
    version = $Version
    format = 'PyInstaller onedir'
    platform = 'Windows x64'
    entrypoint = 'AirWatch.exe'
    requires_python = $false
    requires_conda = $false
    requires_network = $false
    runtime_resource_manifest = '_internal/runtime-resources.json'
    runtime_resource_files = [int]$manifest.summary.resource_files
    runtime_resource_bytes = [int64]$manifest.summary.total_bytes
    capabilities = $capabilities
}

Copy-Item -LiteralPath (Join-Path $PSScriptRoot '使用说明.txt') -Destination (Join-Path $Package '使用说明.txt') -Force
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'start_airwatch.bat') -Destination (Join-Path $Package '启动空域电波哨兵.bat') -Force
$release | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 (Join-Path $Package 'release.json')

$hash = Get-Sha256 $exe
"SHA256  AirWatch.exe`r`n$hash  AirWatch.exe" |
    Set-Content -Encoding ASCII (Join-Path $Package 'SHA256SUMS.txt')
[pscustomobject]@{
    Package = (Resolve-Path $Package).Path
    RuntimeResourceFiles = [int]$manifest.summary.resource_files
    RuntimeResourceBytes = [int64]$manifest.summary.total_bytes
    AirWatchSha256 = $hash
    Status = 'PREPARED'
}
