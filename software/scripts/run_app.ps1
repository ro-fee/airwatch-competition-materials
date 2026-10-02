param(
    [switch]$CheckOnly
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $ProjectRoot

Write-Host '========================================' -ForegroundColor Cyan
Write-Host ' 智能信号处理原型验证系统' -ForegroundColor Cyan
Write-Host '========================================' -ForegroundColor Cyan
Write-Host "项目目录：$ProjectRoot"

# The project has a dedicated runtime environment. Prefer it explicitly so
# double-click launching cannot silently fall back to base/system Python.
$requiredPython = 'E:\CondaEnvs\rofee-bearing\python.exe'
if (-not (Test-Path -LiteralPath $requiredPython -PathType Leaf)) {
    throw "项目专用环境不存在：$requiredPython。请恢复该环境；不回退到系统或 base Python。"
}
$python = (Resolve-Path -LiteralPath $requiredPython).Path

Write-Host "Python：$python" -ForegroundColor Green
& $python --version
if ($LASTEXITCODE -ne 0) { throw 'Python 无法正常启动。' }

$dependencyCheck = "import importlib.util; required=['PyQt5','pyqtgraph','numpy','scipy','torch','qdarkstyle','sklearn','pywt','matplotlib','PIL']; missing=[name for name in required if importlib.util.find_spec(name) is None]; print('运行依赖检查通过。' if not missing else '缺少运行依赖：'+', '.join(missing)); raise SystemExit(1 if missing else 0)"
& $python -c $dependencyCheck
if ($LASTEXITCODE -ne 0) {
    Write-Host ''
    Write-Host '可执行以下命令安装运行依赖：' -ForegroundColor Yellow
    Write-Host "  `"$python`" -m pip install -r `"$ProjectRoot\requirements-runtime.txt`"" -ForegroundColor Yellow
    exit 2
}

if ($CheckOnly) {
    Write-Host '环境检查完成，未启动界面。' -ForegroundColor Green
    exit 0
}

$env:PYQTGRAPH_QT_LIB = 'PyQt5'
$env:QT_API = 'pyqt5'
Write-Host ''
Write-Host '正在启动软件……关闭软件窗口后本脚本会自动结束。' -ForegroundColor Green
& $python (Join-Path $ProjectRoot 'main.py')
$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) {
    Write-Host "软件异常退出，退出码：$exitCode" -ForegroundColor Red
}
exit $exitCode

