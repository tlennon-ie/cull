@echo off
REM update.bat - pull latest cull from origin/main, reinstall deps if changed, relaunch.
REM
REM Triggered from the dashboard's "Update" toast (POST /api/update/run) or run
REM manually. Re-run safely: every step is idempotent.

setlocal ENABLEEXTENSIONS ENABLEDELAYEDEXPANSION
pushd "%~dp0"

echo [update] checking working tree
git rev-parse --is-inside-work-tree >nul 2>nul
if errorlevel 1 (
  echo ERROR: not a git checkout - cannot self-update.
  popd & exit /b 1
)

REM Refuse to clobber dirty trees.
for /f "delims=" %%S in ('git status --porcelain') do (
  echo ERROR: working tree has uncommitted changes; refusing to update.
  echo Commit or stash them first, then re-run update.bat.
  popd & exit /b 1
)

REM Capture pre/post hashes around the pull so we know whether requirements changed.
for /f "delims=" %%H in ('git rev-parse HEAD') do set "BEFORE_SHA=%%H"

echo [update] git fetch origin
git fetch origin --quiet
if errorlevel 1 (
  echo ERROR: git fetch failed.
  popd & exit /b 1
)

echo [update] git pull --ff-only origin main
git pull --ff-only origin main
if errorlevel 1 (
  echo ERROR: fast-forward pull failed - your branch may have diverged from origin/main.
  popd & exit /b 1
)

for /f "delims=" %%H in ('git rev-parse HEAD') do set "AFTER_SHA=%%H"
echo [update] %BEFORE_SHA% -> %AFTER_SHA%

REM Reinstall deps only if requirements.txt changed in the pull range.
git diff --name-only %BEFORE_SHA% %AFTER_SHA% | findstr /I /C:"requirements.txt" >nul
if not errorlevel 1 (
  if exist ".venv\Scripts\activate.bat" (
    echo [update] requirements.txt changed - reinstalling
    call ".venv\Scripts\activate.bat"
    pip install -r requirements.txt
    REM Refresh the launcher's hash file so launch.bat doesn't re-do the work.
    for /f "delims=" %%H in ('certutil -hashfile requirements.txt SHA256 ^| find /v ":"') do set "REQ_HASH=%%H"
    set "REQ_HASH=!REQ_HASH: =!"
    > ".venv\requirements.hash" echo !REQ_HASH!
  ) else (
    echo [update] no .venv found - launch.bat will create one on next launch.
  )
)

echo [update] relaunching cull
start "" "%~dp0launch.bat"
popd
exit /b 0
