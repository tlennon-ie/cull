#!/usr/bin/env bash
# install.sh - one-shot setup for cull on Linux / macOS / RunPod.
#
# What it does:
#   1. Finds a Python interpreter (python3, then python) and checks >= 3.10.
#   2. Verifies git is on PATH (needed for the gallery-dl git+https pin).
#   3. Creates a .venv/ if missing. If venv creation fails (bare Ubuntu /
#      RunPod without python3-venv) it prints the apt hint and falls back to
#      the system/container Python.
#   4. Upgrades pip + installs requirements.txt - including gallery-dl
#      pulled from Codeberg via the git+https:// pin in requirements.txt.
#   5. Downloads Playwright's Chromium (one-time, used by the X scraper).
#   6. Copies .env.example to .env on first run; never overwrites an
#      existing .env.
#
# Env knobs:
#   PYTHON                            override the interpreter (e.g. python3.11)
#   CULL_NO_VENV=1                    skip the venv, install into system Python
#   PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1   skip the one-time Chromium download
#
# Re-run safely: every step is idempotent. Doesn't launch anything; run
# ./launch.sh after install completes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[install] cull setup - installing into $SCRIPT_DIR"

# ── Locate Python ───────────────────────────────────────────────────────────
PY_BIN=""
for candidate in "${PYTHON:-}" python3 python; do
  [ -n "$candidate" ] || continue
  if command -v "$candidate" >/dev/null 2>&1; then
    PY_BIN="$candidate"
    break
  fi
done
if [ -z "$PY_BIN" ]; then
  echo "ERROR: no Python interpreter found on PATH (looked for python3, python)." >&2
  echo "  Install Python 3.10+ and retry:" >&2
  echo "    Debian/Ubuntu/RunPod:  apt-get update && apt-get install -y python3 python3-venv python3-pip" >&2
  echo "    Fedora:                dnf install -y python3 python3-pip" >&2
  echo "    Arch:                  pacman -S --noconfirm python python-pip" >&2
  echo "    macOS:                 brew install python (or download from python.org)" >&2
  exit 1
fi

PY_VERSION="$("$PY_BIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
PY_MAJOR="$(echo "$PY_VERSION" | cut -d. -f1)"
PY_MINOR="$(echo "$PY_VERSION" | cut -d. -f2)"
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
  echo "ERROR: Python 3.10+ required, found $PY_VERSION (interpreter: $PY_BIN)" >&2
  exit 1
fi
echo "[install] Python $PY_VERSION OK ($PY_BIN)"

# ── git (required for the gallery-dl git+https pin) ─────────────────────────
if ! command -v git >/dev/null 2>&1; then
  echo "ERROR: git not found on PATH." >&2
  echo "  cull installs gallery-dl from codeberg.org via 'pip install git+...'," >&2
  echo "  which requires the git CLI. Install git and retry:" >&2
  echo "    Debian/Ubuntu/RunPod:  apt-get update && apt-get install -y git" >&2
  echo "    macOS:                 brew install git (or xcode-select --install)" >&2
  exit 1
fi
echo "[install] git $(git --version | awk '{print $3}') OK"

# ── Virtualenv (with graceful fallback for venv-less containers) ────────────
VENV="$SCRIPT_DIR/.venv"
USE_VENV=1
if [ "${CULL_NO_VENV:-0}" = "1" ]; then
  USE_VENV=0
  echo "[install] CULL_NO_VENV=1 set - using system Python, skipping virtualenv"
fi

if [ "$USE_VENV" = "1" ]; then
  if [ ! -d "$VENV" ]; then
    echo "[install] creating virtualenv in .venv"
    if ! "$PY_BIN" -m venv "$VENV" 2>/tmp/cull_venv_err; then
      cat /tmp/cull_venv_err >&2 || true
      echo "WARN: could not create a virtualenv with '$PY_BIN -m venv'." >&2
      echo "      On Debian/Ubuntu/RunPod this usually means the venv module is missing:" >&2
      echo "          apt-get update && apt-get install -y python3-venv" >&2
      echo "      Falling back to the system/container Python (no virtualenv)." >&2
      echo "      Set CULL_NO_VENV=1 to silence this and always use system Python." >&2
      rm -rf "$VENV" 2>/dev/null || true
      USE_VENV=0
    fi
  else
    echo "[install] reusing existing .venv"
  fi
fi

if [ "$USE_VENV" = "1" ]; then
  # shellcheck disable=SC1091
  . "$VENV/bin/activate"
  PY=python
  FLAG_DIR="$VENV"
else
  PY="$PY_BIN"
  FLAG_DIR="$SCRIPT_DIR/.venv_state"
  mkdir -p "$FLAG_DIR"
fi

# ── Bootstrap pip ───────────────────────────────────────────────────────────
echo "[install] upgrading pip / setuptools / wheel"
"$PY" -m pip install --upgrade pip setuptools wheel >/dev/null

# ── Requirements (idempotent via hash file) ─────────────────────────────────
REQ_HASH_FILE="$FLAG_DIR/.requirements.hash"
REQ_HASH_NEW="$("$PY" - "$SCRIPT_DIR/requirements.txt" <<'PYHASH'
import hashlib, sys
with open(sys.argv[1], "rb") as fh:
    print(hashlib.sha256(fh.read()).hexdigest())
PYHASH
)"
REQ_HASH_OLD="$(cat "$REQ_HASH_FILE" 2>/dev/null || true)"

if [ "$REQ_HASH_NEW" != "$REQ_HASH_OLD" ]; then
  echo "[install] installing requirements (this includes gallery-dl from codeberg)"
  "$PY" -m pip install -r "$SCRIPT_DIR/requirements.txt"
  echo "$REQ_HASH_NEW" > "$REQ_HASH_FILE"
else
  echo "[install] requirements already installed (hash unchanged)"
fi

# ── Playwright Chromium (one-time) ──────────────────────────────────────────
if [ "${PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD:-0}" != "1" ]; then
  PLAYWRIGHT_FLAG="$FLAG_DIR/.playwright_chromium_installed"
  if [ ! -f "$PLAYWRIGHT_FLAG" ]; then
    echo "[install] downloading Chromium for Playwright (one-time, ~120 MB)"
    if "$PY" -m playwright install chromium; then
      : > "$PLAYWRIGHT_FLAG"
    else
      echo "WARN: Playwright Chromium install failed. The X/web scrapers won't work" >&2
      echo "      until you run '$PY -m playwright install chromium' manually." >&2
      echo "      (On headless Linux you may also need: $PY -m playwright install-deps)" >&2
    fi
  else
    echo "[install] Playwright Chromium already installed"
  fi
fi

# ── .env seed ───────────────────────────────────────────────────────────────
if [ ! -f "$SCRIPT_DIR/.env" ]; then
  if [ -f "$SCRIPT_DIR/.env.example" ]; then
    cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
    echo "[install] created .env from .env.example"
    echo "[install] >>> edit .env to add your API keys (Civitai, Groq, Discord, etc.) <<<"
  else
    echo "WARN: no .env.example found; create a .env manually before launching." >&2
  fi
else
  echo "[install] reusing existing .env"
fi

echo
echo "[install] done. Next steps:"
echo "  1. (optional) edit .env to add API keys"
echo "  2. ./launch.sh    # boots the dashboard at http://localhost:\${FLASK_PORT:-5000}"
