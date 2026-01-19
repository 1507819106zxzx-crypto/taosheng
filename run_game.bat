@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

rem Log file (fallback if the default log is locked by another process).
set "LOG=run_game.log"
del /f /q "%LOG%" >nul 2>nul
if exist "%LOG%" (
  set "LOG=run_game_%RANDOM%.log"
)
echo ===== %DATE% %TIME% =====> "%LOG%"
echo cwd: %CD%>> "%LOG%"
echo [INFO] Log file: "%LOG%"

rem Prefer existing venv Python first (so PATH Python isn't required once venv exists).
set "VENV_PY=.venv\Scripts\python.exe"
if exist "%VENV_PY%" (
  echo [INFO] Using existing venv: %VENV_PY%>> "%LOG%"
  goto HAVE_VENV
)

echo [INFO] Creating venv...
echo [INFO] Creating venv...>> "%LOG%"

rem No venv yet: find a system Python to create it.
set "PY="
rem Prefer the Windows Python launcher if present (commonly at C:\Windows\py.exe).
if exist "%SystemRoot%\py.exe" set "PY=%SystemRoot%\py.exe -3"

if "%PY%"=="" (
  rem PATH in some environments may not include System32; call where.exe by absolute path if available.
  if exist "%SystemRoot%\System32\where.exe" (
    "%SystemRoot%\System32\where.exe" py >nul 2>nul && set "PY=py -3"
    if "%PY%"=="" "%SystemRoot%\System32\where.exe" python >nul 2>nul && set "PY=python"
    if "%PY%"=="" "%SystemRoot%\System32\where.exe" python3 >nul 2>nul && set "PY=python3"
  ) else (
    rem Fallback: best-effort probes.
    py -3 -V >nul 2>nul && set "PY=py -3"
    if "%PY%"=="" python -V >nul 2>nul && set "PY=python"
    if "%PY%"=="" python3 -V >nul 2>nul && set "PY=python3"
  )
)

if "%PY%"=="" (
  echo [ERROR] Python not found in PATH and no venv exists.>> "%LOG%"
  echo [ERROR] Python not found. Please install Python 3.10+ and try again.
  echo         (Install from https://www.python.org/downloads/ )
  if not defined CI pause
  exit /b 1
)

rem Require Python 3.10+ (codebase uses PEP604 union types like: X | None).
%PY% -c "import sys; print('python:', sys.version.replace('\n',' ')); print('executable:', sys.executable)" >> "%LOG%" 2>&1
%PY% -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 2)" >> "%LOG%" 2>&1
if errorlevel 2 (
  echo [ERROR] Python 3.10+ required.>> "%LOG%"
  echo [ERROR] Python 3.10+ is required. See "%LOG%" for details.
  if not defined CI pause
  exit /b 2
)

%PY% -m venv .venv >> "%LOG%" 2>&1
if errorlevel 1 (
  echo [ERROR] Failed to create venv.>> "%LOG%"
  echo [ERROR] Failed to create venv. See "%LOG%" for details.
  if not defined CI pause
  exit /b 1
)

if not exist "%VENV_PY%" (
  echo [ERROR] venv created but python.exe missing: %VENV_PY%>> "%LOG%"
  echo [ERROR] venv creation incomplete. See "%LOG%" for details.
  if not defined CI pause
  exit /b 1
)

:HAVE_VENV
set "PY_EXE=%VENV_PY%"

"%PY_EXE%" -c "import sys; print('python:', sys.version.replace('\n',' ')); print('executable:', sys.executable)" >> "%LOG%" 2>&1

echo [INFO] Installing dependencies...
echo [INFO] Installing dependencies...>> "%LOG%"
"%PY_EXE%" -m pip --version >> "%LOG%" 2>&1
"%PY_EXE%" -m pip install -r requirements.txt >> "%LOG%" 2>&1
if errorlevel 1 (
  echo [ERROR] pip install failed.>> "%LOG%"
  echo [ERROR] pip install failed. See "%LOG%" for details.
  if not defined CI pause
  exit /b 1
)

echo [INFO] Launching game...
echo [INFO] Launching game...>> "%LOG%"
"%PY_EXE%" game.py >> "%LOG%" 2>&1
set "EXITCODE=%ERRORLEVEL%"
if not "%EXITCODE%"=="0" (
  echo [ERROR] Game crashed. Exit code %EXITCODE%.>> "%LOG%"
  echo [ERROR] Game crashed. Exit code %EXITCODE%.
  echo         See "%LOG%" for details.
  if not defined CI pause
)

endlocal
exit /b %EXITCODE%
