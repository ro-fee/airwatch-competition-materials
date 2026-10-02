$ErrorActionPreference = 'Stop'

$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$iscc = Join-Path $env:LOCALAPPDATA 'Programs\Inno Setup 6\ISCC.exe'
$spec = Join-Path $root 'packaging\airwatch_setup.iss'
$package = Join-Path $root 'dist\AirWatch'

if (!(Test-Path -LiteralPath $iscc)) {
    throw "找不到 Inno Setup 编译器：$iscc"
}
if (!(Test-Path -LiteralPath (Join-Path $package 'AirWatch.exe'))) {
    throw "找不到已构建的便携版：$package\AirWatch.exe。请先执行 build_onedir.ps1。"
}

$output = Join-Path $root 'dist\installer'
New-Item -ItemType Directory -Force -Path $output | Out-Null

& $iscc /Qp $spec
if ($LASTEXITCODE -ne 0) {
    throw "Inno Setup 编译失败，退出码：$LASTEXITCODE"
}

$installer = Join-Path $output 'AirWatch-Setup-Windows-x64.exe'
if (!(Test-Path -LiteralPath $installer)) {
    throw "编译命令完成，但没有找到安装器：$installer"
}

$hash = (Get-FileHash -LiteralPath $installer -Algorithm SHA256).Hash
"SHA256  AirWatch-Setup-Windows-x64.exe`r`n$hash  AirWatch-Setup-Windows-x64.exe" |
    Set-Content -Encoding ASCII (Join-Path $output 'SHA256SUMS.txt')

[pscustomobject]@{
    Installer = $installer
    Bytes = (Get-Item -LiteralPath $installer).Length
    SHA256 = $hash
    Status = 'PASS'
}
