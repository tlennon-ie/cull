@echo off
REM launch.bat - one-command bootstrap for Windows.
REM
REM What it does:
REM   1. Finds python on PATH (py -3 launcher preferred, then python).
REM   2. Creates .venv\ on first run, activates it.
REM   3. Installs requirements.txt and Playwright Chromium (idempotent).
REM   4. Copies .env.example -> .env on first run; never overwrites an existing .env.
REM   5. Starts the integrated launcher (dashboard + pipeline supervisor).
REM
REM Env knobs:
REM   FLASK_PORT=5000                      dashboard port (default 5000)
REM   PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1   skip the one-time Chromium download
REM
REM Re-run safely: every step is idempotent.

setlocal ENABLEEXTENSIONS ENABLEDELAYEDEXPANSION
pushd "%~dp0"

REM -- Locate python ---------------------------------------------------------
where py >nul 2>nul
if not errorlevel 1 (
  set "PY_LAUNCHER=py -3"
) else (
  where python >nul 2>nul
  if errorlevel 1 (
    echo ERROR: Python not found on PATH. Install Python 3.10+ from python.org and retry.
    echo        Tick "Add python.exe to PATH" in the installer.
    popd ^& exit /b 1
  )
  set "PY_LAUNCHER=python"
)

REM -- Virtualenv ------------------------------------------------------------
if not exist ".venv\Scripts\python.exe" (
  echo [launch] creating virtualenv in .venv
  %PY_LAUNCHER% -m venv .venv
  if errorlevel 1 (
    echo ERROR: failed to create virtualenv. Is the full Python ^(not the Store stub^) installed?
    popd ^& exit /b 1
  )
)
call ".venv\Scripts\activate.bat"

python -m pip install --upgrade pip setuptools wheel >nul

REM -- Requirements ----------------------------------------------------------
REM Track the requirements.txt hash so we don't reinstall on every launch.
set "HASH_FILE=.venv\requirements.hash"
for /f "delims=" %%H in ('certutil -hashfile requirements.txt SHA256 ^| find /v ":"') do set "REQ_HASH=%%H"
set "REQ_HASH=!REQ_HASH: =!"

set "PREV_HASH="
if exist "!HASH_FILE!" set /p PREV_HASH=<"!HASH_FILE!"

if "!REQ_HASH!" neq "!PREV_HASH!" (
  echo [launch] installing requirements
  python -m pip install -r requirements.txt
  if errorlevel 1 (
    echo ERROR: pip install failed. See the output above.
    popd ^& exit /b 1
  )
  > "!HASH_FILE!" echo !REQ_HASH!
) else (
  echo [launch] requirements already installed ^(hash unchanged^)
)

REM -- Playwright Chromium (one-time) ----------------------------------------
if /i "%PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD%" neq "1" (
  if not exist ".venv\.playwright_chromium_installed" (
    echo [launch] downloading Chromium for Playwright ^(one-time^)
    python -m playwright install chromium >nul
    if not errorlevel 1 (
      type nul > ".venv\.playwright_chromium_installed"
    ) else (
      echo WARN: Playwright Chromium install failed; the X/web scrapers won't work
      echo       until you run 'python -m playwright install chromium' manually.
    )
  )
)

REM -- Seed .env from example ------------------------------------------------
if not exist ".env" (
  if exist ".env.example" (
    copy /Y ".env.example" ".env" >nul
    echo [launch] created .env from .env.example - edit it to add API keys
  ) else (
    echo WARN: no .env or .env.example found; the dashboard will use defaults.
  )
)

REM -- Run -------------------------------------------------------------------
set "PORT=%FLASK_PORT%"
if "%PORT%"=="" set "PORT=5000"
echo [launch] starting dashboard...
echo [launch] open http://localhost:%PORT% once it boots ^(Ctrl+C to stop^)
echo.
python pipeline_code\integrated_launcher.py %*
set "RC=%ERRORLEVEL%"
popd
exit /b %RC%
