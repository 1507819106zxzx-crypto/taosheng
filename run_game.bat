@echo off
setlocal

cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python not found. Please install Python 3 and try again.
  pause
  exit /b 1
)

if not exist ".venv\\Scripts\\python.exe" (
  echo [INFO] Creating venv...
  python -m venv .venv
  if errorlevel 1 (
    echo [ERROR] Failed to create venv.
    pause
    exit /b 1
  )
)

echo [INFO] Installing dependencies...
".venv\\Scripts\\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
  echo [ERROR] pip install failed.
  pause
  exit /b 1
)

echo [INFO] Launching game...
".venv\\Scripts\\python.exe" game.py
if errorlevel 1 (
  echo [ERROR] Game crashed. Exit code %errorlevel%.
  pause
  exit /b %errorlevel%
)

endlocal
