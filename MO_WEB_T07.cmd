@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Khong tim thay Python tai .venv\Scripts\python.exe
  echo Hay cai moi truong theo README truoc khi mo web.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m tools.open_t07_web
if errorlevel 1 (
  echo.
  echo Khong mo duoc web. Xem thong bao phia tren.
  pause
  exit /b 1
)
exit /b 0
