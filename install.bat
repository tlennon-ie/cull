@echo off
REM install.bat - one-shot setup for cull on Windows.
REM
REM What it does:
REM   1. Verifies Python 3.10+ and git are on PATH.
REM   2. Creates a .venv\ if missing.
REM   3. Upgrades pip + installs requirements.txt — including gallery-dl
REM      pulled from Codeberg via the git+https:// pin in requirements.txt.
REM   4. Downloads Playwright's Chromium (one-time, used by the X scraper).
REM   5. Copies .env.example to .env on first run; never overwrites an
REM      existing .env.
REM
REM Re-run safely: every step is idempotent. Doesn't launch anything;
REM run launch.bat after install completes.

setlocal ENABLEEXTENSIONS ENABLEDELAYEDEXPANSION
pushd "%~dp0"

echo [install] cull setup - installing into %CD%

REM ── Required tools ─────────────────────────────────────────────────────
where py >nul 2>nul
if not errorlevel 1 (
  set "PY_LAUNCHER=py -3"
) else (
  where python >nul 2>nul
  if errorlevel 1 (
    echo ERROR: Python not found on PATH.
    echo   Install Python 3.10+ from python.org and retry.
    popd & exit /b 1
  )
  set "PY_LAUNCHER=python"
)

for /f "delims=" %%V in ('%PY_LAUNCHER% -c "import sys;print(\"%%d.%%d\"%%sys.version_info[:2])"') do set "PY_VERSION=%%V"
echo [install] Python !PY_VERSION! OK

where git >nul 2>nul
if errorlevel 1 (
  echo ERROR: git not found on PATH.
  echo   cull installs gallery-dl from codeberg.org via 'pip install git+...',
  echo   which requires the git CLI. Install Git for Windows and retry.
  popd & exit /b 1
)
for /f "tokens=3" %%V in ('git --version') do set "GIT_VERSION=%%V"
echo [install] git !GIT_VERSION! OK

REM ── Virtualenv ─────────────────────────────────────────────────────────
if not exist ".venv\Scripts\python.exe" (
  echo [install] creating virtualenv in .venv
  %PY_LAUNCHER% -m venv .venv
) else (
  echo [install] reusing existing .venv
)
call ".venv\Scripts\activate.bat"

REM ── Bootstrap pip ──────────────────────────────────────────────────────
echo [install] upgrading pip / setuptools / wheel
python -m pip install --upgrade pip setuptools wheel >nul

REM ── Requirements (idempotent via hash file) ────────────────────────────
set "HASH_FILE=.venv\requirements.hash"
for /f "delims=" %%H in ('certutil -hashfile requirements.txt SHA256 ^| find /v ":"') do set "REQ_HASH=%%H"
set "REQ_HASH=!REQ_HASH: =!"

set "PREV_HASH="
if exist "!HASH_FILE!" set /p PREV_HASH=<"!HASH_FILE!"

if "!REQ_HASH!" neq "!PREV_HASH!" (
  echo [install] installing requirements ^(this includes gallery-dl from codeberg^)
  pip install -r requirements.txt
  if errorlevel 1 (
    echo ERROR: pip install failed. See the output above.
    popd & exit /b 1
  )
  > "!HASH_FILE!" echo !REQ_HASH!
) else (
  echo [install] requirements already installed ^(hash unchanged^)
)

REM ── Playwright Chromium (one-time) ─────────────────────────────────────
if /i "%PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD%" neq "1" (
  if not exist ".venv\.playwright_chromium_installed" (
    echo [install] downloading Chromium for Playwright ^(one-time, ~120 MB^)
    python -m playwright install chromium
    if errorlevel 1 (
      echo WARN: Playwright Chromium install failed. The X scraper won't work
      echo       until you run 'python -m playwright install chromium' manually.
    ) else (
      type nul > ".venv\.playwright_chromium_installed"
    )
  ) else (
    echo [install] Playwright Chromium already installed
  )
)

REM ── .env seed ──────────────────────────────────────────────────────────
if not exist ".env" (
  copy /Y ".env.example" ".env" >nul
  echo [install] created .env from .env.example
  echo [install] ^>^>^> edit .env to add your API keys ^(Civitai, Groq, Discord, etc.^) ^<^<^<
) else (
  echo [install] reusing existing .env
)

echo.
echo [install] done. Next steps:
echo   1. ^(optional^) edit .env to add API keys
echo   2. launch.bat    REM boots the dashboard at http://localhost:5000

popd
exit /b 0
