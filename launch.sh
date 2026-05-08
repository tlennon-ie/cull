#!/usr/bin/env bash
# launch.sh - one-command bootstrap for Linux / macOS.
#
# What it does:
#   1. Verifies python3 is available.
#   2. Creates a .venv/ if missing, then activates it.
#   3. Installs requirements.txt + Playwright Chromium (idempotent).
#   4. Copies .env.example -> .env on first run so the dashboard has something
#      to read; existing .env is never overwritten.
#   5. Starts the integrated launcher (dashboard + pipeline supervisor).
#
# Re-run safely: every step is idempotent. Works from any cwd.
set -euo pipefail

# Resolve the script's own directory so the user can call it from anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PY_BIN="${PYTHON:-python3}"

if ! command -v "$PY_BIN" >/dev/null 2>&1; then
  echo "ERROR: python3 not found on PATH. Install Python 3.10+ and retry." >&2
  exit 1
fi

VENV="$SCRIPT_DIR/.venv"
if [ ! -d "$VENV" ]; then
  echo "[launch] creating virtualenv in $VENV"
  "$PY_BIN" -m venv "$VENV"
fi

# shellcheck disable=SC1091
. "$VENV/bin/activate"

# Pin pip so the requirements install is reproducible.
python -m pip install --upgrade pip setuptools wheel >/dev/null

REQ_HASH_FILE="$VENV/.requirements.hash"
REQ_HASH_NEW=$(sha256sum "$SCRIPT_DIR/requirements.txt" | awk '{print $1}')
REQ_HASH_OLD="$(cat "$REQ_HASH_FILE" 2>/dev/null || true)"

if [ "$REQ_HASH_NEW" != "$REQ_HASH_OLD" ]; then
  echo "[launch] installing requirements"
  pip install -r "$SCRIPT_DIR/requirements.txt"
  echo "$REQ_HASH_NEW" > "$REQ_HASH_FILE"
else
  echo "[launch] requirements already installed (hash unchanged)"
fi

# Playwright is only needed for the X/Twitter + web scrapers, but installing
# the Chromium browser is cheap and keeps things working out-of-the-box. Skip
# if PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1 is set in the environment.
if [ "${PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD:-0}" != "1" ]; then
  PLAYWRIGHT_FLAG="$VENV/.playwright_chromium_installed"
  if [ ! -f "$PLAYWRIGHT_FLAG" ]; then
    echo "[launch] downloading Chromium for Playwright (one-time)"
    python -m playwright install chromium >/dev/null
    : > "$PLAYWRIGHT_FLAG"
  fi
fi

if [ ! -f "$SCRIPT_DIR/.env" ]; then
  cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
  echo "[launch] created .env from .env.example - edit it to add API keys"
fi

echo "[launch] starting dashboard..."
echo "[launch] open http://localhost:5000 once it boots (Ctrl+C to stop)"
echo
exec python "$SCRIPT_DIR/pipeline_code/integrated_launcher.py" "$@"
