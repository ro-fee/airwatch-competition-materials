# Validate a staged AirWatch portable package without launching the GUI.
param([string]$Package = (Join-Path (Split-Path $PSScriptRoot -Parent) 'dist\AirWatch'))

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

$exe = Join-Path $Package 'AirWatch.exe'
$internal = Join-Path $Package '_internal'
$manifestPath = Join-Path $internal 'runtime-resources.json'
$required = @(
    $exe,
    $internal,
    $manifestPath,
    (Join-Path $Package '使用说明.txt'),
    (Join-Path $Package 'release.json'),
    (Join-Path $Package 'SHA256SUMS.txt')
)
foreach ($path in $required) {
    if (!(Test-Path -LiteralPath $path)) {
        throw "缺少发布文件：$path"
    }
}

# These packages belong to development, training, or retired rendering paths.
# A successful build must prove they did not leak back into the desktop bundle.
$forbiddenTopLevel = @(
    'cv2',
    'pandas',
    'torchvision',
    'h5py',
    'IPython',
    'tensorboard'
)
foreach ($name in $forbiddenTopLevel) {
    $candidate = Join-Path $internal $name
    if (Test-Path -LiteralPath $candidate) {
        throw "发布包包含已排除的顶层依赖：$name"
    }
}

$manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
if ($manifest.schema_version -ne 1) {
    throw "不支持的运行时资源清单版本：$($manifest.schema_version)"
}
if ($null -eq $manifest.resources -or $manifest.resources.Count -eq 0) {
    throw '运行时资源清单为空'
}

$internalRoot = [IO.Path]::GetFullPath($internal).TrimEnd('\') + '\'
$resourceBytes = [int64]0
foreach ($entry in $manifest.resources) {
    $relative = [string]$entry.path
    if ([string]::IsNullOrWhiteSpace($relative) -or [IO.Path]::IsPathRooted($relative)) {
        throw "运行时资源路径无效：$relative"
    }
    $candidate = [IO.Path]::GetFullPath((Join-Path $internal $relative))
    if (!$candidate.StartsWith($internalRoot, [StringComparison]::OrdinalIgnoreCase)) {
        throw "运行时资源路径越界：$relative"
    }
    if (!(Test-Path -LiteralPath $candidate -PathType Leaf)) {
        throw "缺少运行时资源：$relative"
    }
    $file = Get-Item -LiteralPath $candidate
    if ($file.Length -ne [int64]$entry.size_bytes) {
        throw "运行时资源大小不一致：$relative"
    }
    $actualHash = Get-Sha256 $candidate
    if ($actualHash -ne ([string]$entry.sha256).ToUpperInvariant()) {
        throw "运行时资源 SHA256 不一致：$relative"
    }
    $resourceBytes += $file.Length
}

$hash = Get-Sha256 $exe
$line = Get-Content -LiteralPath (Join-Path $Package 'SHA256SUMS.txt') | Select-Object -Last 1
if ($line -notmatch $hash) {
    throw 'AirWatch.exe SHA256 与发布清单不一致'
}
$size = (Get-ChildItem -LiteralPath $Package -Recurse -File | Measure-Object Length -Sum).Sum
[pscustomobject]@{
    Package = (Resolve-Path $Package).Path
    ExecutableBytes = (Get-Item -LiteralPath $exe).Length
    RuntimeResourceFiles = $manifest.resources.Count
    RuntimeResourceBytes = $resourceBytes
    TotalBytes = $size
    Sha256 = $hash
    Status = 'PASS'
}
