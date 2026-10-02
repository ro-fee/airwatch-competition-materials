@echo off
setlocal
cd /d "%~dp0"
if not exist "AirWatch.exe" exit /b 2
start "AirWatch" /d "%~dp0" "%~dp0AirWatch.exe"
exit /b 0
