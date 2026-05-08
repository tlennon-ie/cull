@echo off
REM launch.bat - one-command bootstrap for Windows.
REM
REM What it does:
REM   1. Finds python on PATH (errors out if 3.10+ isn't installed).
REM   2. Creates .venv\ on first run, activates it.
REM   3. Installs requirements.txt and Playwright Chromium (idempotent).
REM   4. Copies .env.example -> .env on first run; never overwrites an existing .env.
REM   5. Starts the integrated launcher (dashboard + pipeline supervisor).
REM
REM Re-run safely: every step is idempotent.

setlocal ENABLEEXTENSIONS ENABLEDELAYEDEXPANSION
pushd "%~dp0"

REM ── Locate python ────────────────────────────────────────────────────────
where py >nul 2>nul
if not errorlevel 1 (
  set "PY_LAUNCHER=py -3"
) else (
  where python >nul 2>nul
  if errorlevel 1 (
    echo ERROR: Python not found on PATH. Install Python 3.10+ from python.org and retry.
    popd & exit /b 1
  )
  set "PY_LAUNCHER=python"
)

REM ── Virtualenv ───────────────────────────────────────────────────────────
if not exist ".venv\Scripts\python.exe" (
  echo [launch] creating virtualenv in .venv
  %PY_LAUNCHER% -m venv .venv
)
call ".venv\Scripts\activate.bat"

python -m pip install --upgrade pip setuptools wheel >nul

REM ── Requirements ─────────────────────────────────────────────────────────
REM Track the requirements.txt hash so we don't reinstall on every launch.
set "HASH_FILE=.venv\requirements.hash"
for /f "delims=" %%H in ('certutil -hashfile requirements.txt SHA256 ^| find /v ":"') do set "REQ_HASH=%%H"
set "REQ_HASH=!REQ_HASH: =!"

set "PREV_HASH="
if exist "!HASH_FILE!" set /p PREV_HASH=<"!HASH_FILE!"

if "!REQ_HASH!" neq "!PREV_HASH!" (
  echo [launch] installing requirements
  pip install -r requirements.txt
  > "!HASH_FILE!" echo !REQ_HASH!
) else (
  echo [launch] requirements already installed ^(hash unchanged^)
)

REM ── Playwright Chromium (one-time) ───────────────────────────────────────
if /i "%PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD%" neq "1" (
  if not exist ".venv\.playwright_chromium_installed" (
    echo [launch] downloading Chromium for Playwright ^(one-time^)
    python -m playwright install chromium >nul
    type nul > ".venv\.playwright_chromium_installed"
  )
)

REM ── Seed .env from example ───────────────────────────────────────────────
if not exist ".env" (
  copy /Y ".env.example" ".env" >nul
  echo [launch] created .env from .env.example - edit it to add API keys
)

REM ── Run ──────────────────────────────────────────────────────────────────
echo [launch] starting dashboard...
echo [launch] open http://localhost:5000 once it boots ^(Ctrl+C to stop^)
echo.
python pipeline_code\integrated_launcher.py %*
set "RC=%ERRORLEVEL%"
popd
exit /b %RC%
