@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Khong tim thay Python .venv. Xem README de cai moi truong.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m tools.open_t07_web --port 8767 --page /walkthrough
if errorlevel 1 pause
