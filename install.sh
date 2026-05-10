#!/usr/bin/env bash
# install.sh - one-shot setup for cull on Linux / macOS.
#
# What it does:
#   1. Verifies python3 (>= 3.10) and git are on PATH.
#   2. Creates a .venv/ if missing.
#   3. Upgrades pip + installs requirements.txt — including gallery-dl
#      pulled from Codeberg via the git+https:// pin in requirements.txt.
#   4. Downloads Playwright's Chromium (one-time, used by the X scraper).
#   5. Copies .env.example to .env on first run; never overwrites an
#      existing .env.
#
# Re-run safely: every step is idempotent. Doesn't launch anything; run
# ./launch.sh after install completes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PY_BIN="${PYTHON:-python3}"

echo "[install] cull setup — installing into $SCRIPT_DIR"

# ── Required tools ──────────────────────────────────────────────────────────
if ! command -v "$PY_BIN" >/dev/null 2>&1; then
  echo "ERROR: python3 not found on PATH. Install Python 3.10+ and retry." >&2
  echo "  Linux:  apt/dnf/pacman install python3" >&2
  echo "  macOS:  brew install python or download from python.org" >&2
  exit 1
fi

PY_VERSION="$("$PY_BIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
PY_MAJOR="$(echo "$PY_VERSION" | cut -d. -f1)"
PY_MINOR="$(echo "$PY_VERSION" | cut -d. -f2)"
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
  echo "ERROR: Python 3.10+ required, found $PY_VERSION" >&2
  exit 1
fi
echo "[install] Python $PY_VERSION OK"

if ! command -v git >/dev/null 2>&1; then
  echo "ERROR: git not found on PATH." >&2
  echo "  cull installs gallery-dl from codeberg.org via 'pip install git+...'," >&2
  echo "  which requires the git CLI. Install git and retry." >&2
  exit 1
fi
echo "[install] git $(git --version | awk '{print $3}') OK"

# ── Virtualenv ──────────────────────────────────────────────────────────────
VENV="$SCRIPT_DIR/.venv"
if [ ! -d "$VENV" ]; then
  echo "[install] creating virtualenv in .venv"
  "$PY_BIN" -m venv "$VENV"
else
  echo "[install] reusing existing .venv"
fi

# shellcheck disable=SC1091
. "$VENV/bin/activate"

# ── Bootstrap pip ───────────────────────────────────────────────────────────
echo "[install] upgrading pip / setuptools / wheel"
python -m pip install --upgrade pip setuptools wheel >/dev/null

# ── Requirements (idempotent via hash file) ─────────────────────────────────
REQ_HASH_FILE="$VENV/.requirements.hash"
REQ_HASH_NEW=$(sha256sum "$SCRIPT_DIR/requirements.txt" | awk '{print $1}')
REQ_HASH_OLD="$(cat "$REQ_HASH_FILE" 2>/dev/null || true)"

if [ "$REQ_HASH_NEW" != "$REQ_HASH_OLD" ]; then
  echo "[install] installing requirements (this includes gallery-dl from codeberg)"
  pip install -r "$SCRIPT_DIR/requirements.txt"
  echo "$REQ_HASH_NEW" > "$REQ_HASH_FILE"
else
  echo "[install] requirements already installed (hash unchanged)"
fi

# ── Playwright Chromium (one-time) ──────────────────────────────────────────
if [ "${PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD:-0}" != "1" ]; then
  PLAYWRIGHT_FLAG="$VENV/.playwright_chromium_installed"
  if [ ! -f "$PLAYWRIGHT_FLAG" ]; then
    echo "[install] downloading Chromium for Playwright (one-time, ~120 MB)"
    python -m playwright install chromium
    : > "$PLAYWRIGHT_FLAG"
  else
    echo "[install] Playwright Chromium already installed"
  fi
fi

# ── .env seed ───────────────────────────────────────────────────────────────
if [ ! -f "$SCRIPT_DIR/.env" ]; then
  cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
  echo "[install] created .env from .env.example"
  echo "[install] >>> edit .env to add your API keys (Civitai, Groq, Discord, etc.) <<<"
else
  echo "[install] reusing existing .env"
fi

echo
echo "[install] done. Next steps:"
echo "  1. (optional) edit .env to add API keys"
echo "  2. ./launch.sh    # boots the dashboard at http://localhost:5000"
