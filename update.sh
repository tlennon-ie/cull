#!/usr/bin/env bash
# update.sh - pull latest cull from origin/main, reinstall deps if changed, relaunch.
#
# Triggered from the dashboard's "Update" toast (POST /api/update/run) or run
# manually. Re-run safely: every step is idempotent.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "ERROR: not a git checkout - cannot self-update." >&2
  exit 1
fi

if [ -n "$(git status --porcelain)" ]; then
  echo "ERROR: working tree has uncommitted changes; refusing to update." >&2
  echo "Commit or stash them first, then re-run update.sh." >&2
  exit 1
fi

BEFORE_SHA="$(git rev-parse HEAD)"

echo "[update] git fetch origin"
git fetch origin --quiet

echo "[update] git pull --ff-only origin main"
if ! git pull --ff-only origin main; then
  echo "ERROR: fast-forward pull failed - your branch may have diverged from origin/main." >&2
  exit 1
fi

AFTER_SHA="$(git rev-parse HEAD)"
echo "[update] $BEFORE_SHA -> $AFTER_SHA"

# Reinstall deps only if requirements.txt changed in the pull range.
if git diff --name-only "$BEFORE_SHA" "$AFTER_SHA" | grep -q '^requirements\.txt$'; then
  echo "[update] requirements.txt changed - reinstalling"
  if [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    . "$SCRIPT_DIR/.venv/bin/activate"
    PY=python
    FLAG_DIR="$SCRIPT_DIR/.venv"
  else
    # No venv (e.g. CULL_NO_VENV install). Reinstall into whatever Python runs.
    PY=""
    for candidate in "${PYTHON:-}" python3 python; do
      [ -n "$candidate" ] || continue
      if command -v "$candidate" >/dev/null 2>&1; then PY="$candidate"; break; fi
    done
    FLAG_DIR="$SCRIPT_DIR/.venv_state"
  fi

  if [ -n "$PY" ]; then
    "$PY" -m pip install -r "$SCRIPT_DIR/requirements.txt"
    # Refresh the launcher's hash file so launch.sh doesn't re-do the work.
    mkdir -p "$FLAG_DIR"
    "$PY" - "$SCRIPT_DIR/requirements.txt" > "$FLAG_DIR/.requirements.hash" <<'PYHASH'
import hashlib, sys
with open(sys.argv[1], "rb") as fh:
    print(hashlib.sha256(fh.read()).hexdigest())
PYHASH
  else
    echo "[update] no Python found - launch.sh will reinstall on next launch." >&2
  fi
fi

echo "[update] relaunching cull"
# Detach so the parent dashboard process can exit cleanly.
nohup "$SCRIPT_DIR/launch.sh" >/dev/null 2>&1 &
disown 2>/dev/null || true
exit 0
