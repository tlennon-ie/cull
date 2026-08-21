#!/usr/bin/env bash
# launch.sh - one-command bootstrap for Linux / macOS / RunPod.
#
# What it does:
#   1. Finds a Python interpreter (python3, then python).
#   2. Creates a .venv/ if missing, then activates it. If venv creation fails
#      (common on bare Ubuntu / RunPod where python3-venv is missing) it prints
#      the apt hint and falls back to the system/container Python.
#   3. Installs requirements.txt + Playwright Chromium (idempotent).
#   4. Copies .env.example -> .env on first run so the dashboard has something
#      to read; existing .env is never overwritten.
#   5. Starts the integrated launcher (dashboard + pipeline supervisor).
#
# Env knobs:
#   PYTHON                            override the interpreter (e.g. python3.11)
#   CULL_NO_VENV=1                    skip the venv, install into system Python
#   FLASK_PORT=5000                   dashboard port (default 5000)
#   PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1   skip the one-time Chromium download
#
# Re-run safely: every step is idempotent. Works from any cwd.
set -euo pipefail

# Resolve the script's own directory so the user can call it from anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Locate Python ───────────────────────────────────────────────────────────
# Prefer an explicit override, then python3, then python.
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
  echo "  Install Python 3.10+ and retry." >&2
  echo "  Debian/Ubuntu/RunPod:  apt-get update && apt-get install -y python3 python3-venv python3-pip" >&2
  echo "  macOS:                 brew install python" >&2
  exit 1
fi

# ── Virtualenv (with graceful fallback for venv-less containers) ─────────────
VENV="$SCRIPT_DIR/.venv"
USE_VENV=1
if [ "${CULL_NO_VENV:-0}" = "1" ]; then
  USE_VENV=0
  echo "[launch] CULL_NO_VENV=1 set - using system Python, skipping virtualenv"
fi

if [ "$USE_VENV" = "1" ] && [ ! -d "$VENV" ]; then
  echo "[launch] creating virtualenv in $VENV"
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
fi

if [ "$USE_VENV" = "1" ]; then
  # shellcheck disable=SC1091
  . "$VENV/bin/activate"
  PY=python
  FLAG_DIR="$VENV"
else
  # No venv: drive the chosen interpreter directly. pip writes into the
  # system/container site-packages (fine on ephemeral RunPod containers).
  PY="$PY_BIN"
  FLAG_DIR="$SCRIPT_DIR/.venv_state"
  mkdir -p "$FLAG_DIR"
fi

# ── pip + requirements (idempotent via hash file) ───────────────────────────
# Always invoke pip via `python -m pip`, never the bare `pip` shim.
"$PY" -m pip install --upgrade pip setuptools wheel >/dev/null 2>&1 || \
  echo "WARN: could not upgrade pip/setuptools/wheel; continuing." >&2

REQ_HASH_FILE="$FLAG_DIR/.requirements.hash"
REQ_HASH_NEW="$("$PY" - "$SCRIPT_DIR/requirements.txt" <<'PYHASH'
import hashlib, sys
with open(sys.argv[1], "rb") as fh:
    print(hashlib.sha256(fh.read()).hexdigest())
PYHASH
)"
REQ_HASH_OLD="$(cat "$REQ_HASH_FILE" 2>/dev/null || true)"

if [ "$REQ_HASH_NEW" != "$REQ_HASH_OLD" ]; then
  echo "[launch] installing requirements"
  "$PY" -m pip install -r "$SCRIPT_DIR/requirements.txt"
  echo "$REQ_HASH_NEW" > "$REQ_HASH_FILE"
else
  echo "[launch] requirements already installed (hash unchanged)"
fi

# ── ML extras (torch + open_clip_torch) — enables Find similar / embeddings.
# Heavy (~800MB torch wheel) so install is one-shot + flag-guarded. Opt out
# with CULL_SKIP_ML=1 if you never need semantic image search.
ML_FLAG="$FLAG_DIR/.ml_installed"
if [ "${CULL_SKIP_ML:-0}" != "1" ] && [ ! -f "$ML_FLAG" ]; then
  echo "[launch] installing ML extras (torch, open_clip_torch — one-time, ~800MB)"
  if "$PY" -m pip install -e "$SCRIPT_DIR"[ml]; then
    : > "$ML_FLAG"
  else
    echo "WARN: ML extras install failed. 'Find similar' will show an install hint." >&2
    echo "      Retry with: $PY -m pip install -e '$SCRIPT_DIR'[ml]" >&2
    echo "      Or opt out permanently: CULL_SKIP_ML=1 ./launch.sh" >&2
  fi
fi

# ── Playwright Chromium (one-time) ──────────────────────────────────────────
# Only needed for the X/Twitter + web scrapers. Skip with
# PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1. A failure here is non-fatal.
if [ "${PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD:-0}" != "1" ]; then
  PLAYWRIGHT_FLAG="$FLAG_DIR/.playwright_chromium_installed"
  if [ ! -f "$PLAYWRIGHT_FLAG" ]; then
    echo "[launch] downloading Chromium for Playwright (one-time)"
    if "$PY" -m playwright install chromium >/dev/null 2>&1; then
      : > "$PLAYWRIGHT_FLAG"
    else
      echo "WARN: Playwright Chromium install failed. The X/web scrapers won't work" >&2
      echo "      until you run '$PY -m playwright install chromium' manually." >&2
      echo "      (On headless Linux you may also need OS libs: playwright install-deps)" >&2
    fi
  fi
fi

# ── .env seed ───────────────────────────────────────────────────────────────
if [ ! -f "$SCRIPT_DIR/.env" ]; then
  if [ -f "$SCRIPT_DIR/.env.example" ]; then
    cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
    echo "[launch] created .env from .env.example - edit it to add API keys"
  else
    echo "WARN: no .env or .env.example found; the dashboard will use defaults." >&2
  fi
fi

# ── Run ─────────────────────────────────────────────────────────────────────
PORT="${FLASK_PORT:-5000}"
echo "[launch] starting dashboard..."
echo "[launch] open http://localhost:${PORT} once it boots (Ctrl+C to stop)"
echo "[launch] (remote/RunPod: it binds 0.0.0.0 - map port ${PORT} and use the pod URL)"
echo
exec "$PY" "$SCRIPT_DIR/pipeline_code/integrated_launcher.py" "$@"
