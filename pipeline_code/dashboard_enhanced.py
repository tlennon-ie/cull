#!/usr/bin/env python3
"""
cull dashboard - realtime monitoring + admin controls.

Dashboard starts STANDALONE (does not auto-launch the pipeline).
The pipeline is started/stopped from the UI via /api/pipeline/start|stop.
Auto-refresh every 5 seconds.

Endpoints:
  GET  /                         - HTML UI
  GET  /api/status               - queue/sorted/error counts + pipeline state
  GET  /api/scrapers             - scrapers with enabled/disabled flag
  POST /api/scrapers/toggle      - {name, enabled}
  POST /api/pipeline/start       - spawn run_pipeline.py subprocess
  POST /api/pipeline/stop        - terminate pipeline subprocess tree
  POST /api/vision/provider      - {provider}
  POST /api/vision/throttle      - {percent: 0..100}
  POST /api/lmstudio/endpoint    - {instance, url, model?}
  GET  /api/lmstudio/models      - per-instance models
  POST /api/lmstudio/set-model   - {instance, model_id}
  GET  /api/queue/files          - newest queue items with thumbnails+prompt
  POST /api/queue/action         - {path, action}
  GET  /api/logs/history         - historical sorter log
  GET  /api/thumbnail?path=...   - JPEG thumbnail for a queue/sorted image
  GET  /api/prompt?path=...      - .txt prompt content for an image
  GET  /api/activity             - most recent vision events
"""
from __future__ import annotations

import io
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

try:
    from flask import Flask, jsonify, render_template_string, request, send_file, abort, Response
    from flask_cors import CORS
except ImportError:
    sys.stderr.write("Missing Flask. Install: pip install flask flask-cors\n")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    sys.stderr.write("Missing Pillow. Install: pip install pillow\n")
    sys.exit(1)

logger = logging.getLogger("dashboard")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

PIPELINE_CODE_DIR: Path = Path(os.environ.get("PIPELINE_CODE_DIR", Path(__file__).parent))
ENV_PATH: Path = Path(os.environ.get("WORKSPACE_ROOT", PIPELINE_CODE_DIR.parent)) / ".env"
PIPELINE_QUEUE: Path = Path(os.environ.get("PIPELINE_QUEUE", "queue"))
PIPELINE_SORTED: Path = Path(os.environ.get("PIPELINE_SORTED", "sorted"))
LOG_DIR: Path = Path(os.environ.get("LOG_DIR", "logs_test"))
FLASK_PORT: int = int(os.environ.get("FLASK_PORT", 5000))

_STATIC_SCRAPERS: list[dict[str, str]] = [
    {"name": "X.com",       "description": "X.com (Playwright + cookies, no API)"},
    {"name": "Discord-1",   "description": "Discord UD channels"},
    {"name": "Civitai-Com", "description": "Civitai (civitai.com)"},
    {"name": "Civitai-Red", "description": "Civitai (civitai.red)"},
    {"name": "Web",         "description": "Reddit / ZforFree.com / promptsref"},
    {"name": "ZFF-Local",   "description": "ZforFree local folder (legacy)"},
    {"name": "Gallery-DL",  "description": "gallery-dl (Pixiv, DeviantArt, booru, ArtStation, Tumblr, X, Reddit, Imgur, FurAffinity, e621, Flickr…). Configure URLs + cookies in Settings."},
]


def _scraper_definitions() -> list[dict[str, str]]:
    """Compute the live scraper list.

    The admin-configurable local folder runs under the label `Local-<LOCAL_IMPORT_NAME>`,
    so the toggle here must follow whatever the user set in Settings - otherwise
    toggling `Local-local` has no effect when the feeder actually ran as e.g.
    `Local-my_dataset`.
    """
    local_name = (os.environ.get("LOCAL_IMPORT_NAME") or "local").strip() or "local"
    local_entry = {
        "name": f"Local-{local_name}",
        "description": f"Admin-configured local folder ({os.environ.get('LOCAL_IMPORT_DIR') or '(LOCAL_IMPORT_DIR unset)'})",
    }
    return [*_STATIC_SCRAPERS, local_entry]


SCRAPERS = _STATIC_SCRAPERS  # kept for anything still importing this module-level constant

sys.path.insert(0, str(PIPELINE_CODE_DIR))
try:
    from lmstudio_models import get_all_models, get_recommended_models
except ImportError:
    def get_all_models() -> dict[str, Any]:
        return {"primary": {"status": "disconnected", "models": []},
                "secondary": {"status": "disconnected", "models": []}}

    def get_recommended_models() -> list[Any]:
        return []

from lmstudio_admin import unload_all as _lmstudio_unload_all
import index_store
import thumb_cache

# Configure the SQLite index + thumbnail cache against PIPELINE_BASE_DIR
# (resolved from PIPELINE_QUEUE/PIPELINE_SORTED's parent if not set explicitly).
_DATA_ROOT: Path = Path(os.environ.get(
    "PIPELINE_BASE_DIR",
    str(PIPELINE_QUEUE.parent if PIPELINE_QUEUE.exists() else Path.cwd() / "data"),
))
_DATA_ROOT.mkdir(parents=True, exist_ok=True)
index_store.configure(_DATA_ROOT / index_store.DB_FILENAME)
thumb_cache.configure(_DATA_ROOT / "thumb_cache")
try:
    _INDEXER_INTERVAL = max(5.0, float(os.environ.get("INDEX_REFRESH_SECONDS", "30")))
except ValueError:
    _INDEXER_INTERVAL = 30.0


app = Flask(__name__)
CORS(app)

_pipeline_proc: subprocess.Popen | None = None
_pipeline_lock = threading.Lock()


# ── .env helpers ───────────────────────────────────────────────────────────────

def update_env(key: str, value: str) -> None:
    print(f"[update_env] CALLED with key={key}, value={value[:50]}{'...' if len(value) > 50 else ''}")
    print(f"[update_env] ENV_PATH={ENV_PATH}, exists={ENV_PATH.exists()}, writable={os.access(ENV_PATH, os.W_OK) if ENV_PATH.exists() else 'N/A'}")
    
    if not ENV_PATH.exists():
        logger.error(".env not found at %s", ENV_PATH)
        print(f"[update_env] ERROR: ENV_PATH does not exist!")
        return
    
    pattern = re.compile(rf"^{re.escape(key)}=.*$", re.MULTILINE)
    text = ENV_PATH.read_text(encoding="utf-8")
    line = f"{key}={value}"
    
    # `re.sub` interprets backslashes in the replacement string as regex
    # escapes (\1, \A, \g<1> etc.). Windows paths like `I:\AI\openclaw` would
    # therefore raise PatternError ("bad escape \A"). Using a lambda short-
    # circuits replacement-string parsing and lets us insert the value as a
    # literal.
    found = pattern.search(text)
    print(f"[update_env] Pattern search found existing key: {found is not None}")
    
    if found:
        old_line = found.group(0)
        text = pattern.sub(lambda _match: line, text)
        print(f"[update_env] Replaced old_line: {old_line[:60]}")
    else:
        text = text.rstrip() + f"\n{line}\n"
        print(f"[update_env] Appended new line (key not found in .env)")
    
    print(f"[update_env] Writing to {ENV_PATH}...")
    try:
        ENV_PATH.write_text(text, encoding="utf-8")
        print(f"[update_env] ✓ File write successful")
    except Exception as e:
        print(f"[update_env] ✗ File write FAILED: {e}")
        logger.error("Failed to write .env: %s", e)
        raise
    
    os.environ[key] = value
    print(f"[update_env] ✓ os.environ[{key}] updated")


def get_env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


# ── stats helpers ──────────────────────────────────────────────────────────────

def get_queue_stats() -> dict[str, int]:
    """{<topic>/<source>: count} — served from the SQLite index."""
    return index_store.count_queue_by_topic_source()


def get_sorted_stats() -> dict[str, dict[str, int]]:
    """{topic: {category: count}} — served from the SQLite index."""
    return index_store.count_sorted_by_topic_category()


# Error logs are re-read incrementally: track each log file's last (size, mtime)
# and only re-parse when either changes. Old code re-read every log file in
# full on every /api/status (every 5s). Cache survives the lifetime of the
# dashboard process; on startup it does one cold read.
_ERROR_LOG_CACHE_LOCK = threading.Lock()
_ERROR_LOG_CACHE: dict[str, dict[str, Any]] = {}  # path -> {mtime, size, errors}


def _scan_log_for_errors(path: Path) -> list[dict[str, str]]:
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    mtime_iso = datetime.fromtimestamp(path.stat().st_mtime).isoformat()
    errors: list[dict[str, str]] = []
    for line in content.splitlines():
        if any(tag in line for tag in ("ERROR", "CRITICAL", "FAILED")):
            errors.append({
                "file": path.name,
                "message": line[:600],
                "timestamp": mtime_iso,
            })
    return list(reversed(errors))


def get_error_logs(limit: int = 500) -> list[dict[str, str]]:
    """Collect ERROR/CRITICAL/FAILED lines from the 5 most recent log files.

    Cached per log file by (mtime, size); a file that hasn't changed since
    the last call is served from the in-memory cache instead of re-reading
    from disk.
    """
    if not LOG_DIR.exists():
        return []
    log_files = sorted(LOG_DIR.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)[:5]
    out: list[dict[str, str]] = []
    with _ERROR_LOG_CACHE_LOCK:
        for log_file in log_files:
            key = str(log_file)
            try:
                stat = log_file.stat()
                signature = (stat.st_mtime, stat.st_size)
            except OSError:
                continue
            cached = _ERROR_LOG_CACHE.get(key)
            if cached and cached.get("signature") == signature:
                out.extend(cached["errors"])
            else:
                fresh = _scan_log_for_errors(log_file)
                _ERROR_LOG_CACHE[key] = {"signature": signature, "errors": fresh}
                out.extend(fresh)
            if len(out) >= limit:
                break
    return out[:limit]


def disabled_set() -> set[str]:
    return {s.strip() for s in os.environ.get("SCRAPER_DISABLED", "").split(",") if s.strip()}


def pipeline_running() -> bool:
    proc = _pipeline_proc
    return proc is not None and proc.poll() is None


# ── path guards ────────────────────────────────────────────────────────────────

def safe_inside(raw: str, roots: list[Path]) -> Path | None:
    try:
        path = Path(raw).resolve()
    except OSError:
        return None
    for root in roots:
        try:
            path.relative_to(root.resolve())
            return path
        except (ValueError, OSError):
            continue
    return None


# ── API: status / controls ────────────────────────────────────────────────────

@app.route("/api/status")
def api_status():
    queue = get_queue_stats()
    sorted_stats = get_sorted_stats()
    errors = get_error_logs()
    return jsonify({
        "timestamp": datetime.now().isoformat(),
        "pipeline": {
            "running": pipeline_running(),
            "pid": _pipeline_proc.pid if pipeline_running() else None,
            "throttle": int(os.environ.get("DASHBOARD_THROTTLE_PERCENT", 100)),
            "vision_worker": os.environ.get("PIPELINE_VISION_WORKER", ""),
            "vision_workers": _active_vision_workers(),
        },
        "queue": {"total": sum(queue.values()), "by_source": queue},
        "sorted": {
            "total": sum(sum(v.values()) for v in sorted_stats.values()),
            "by_topic": sorted_stats,
        },
        "errors": errors,
        "error_count": len(errors),
        "indexer": _index_status_payload(),
    })


@app.route("/api/scrapers")
def api_scrapers():
    disabled = disabled_set()
    return jsonify([{**s, "enabled": s["name"] not in disabled} for s in _scraper_definitions()])


@app.route("/api/scrapers/bulk", methods=["POST"])
def api_scrapers_bulk():
    data = request.get_json() or {}
    action = data.get("action")
    defs = _scraper_definitions()
    names = [s["name"] for s in defs]

    if action == "disable_all":
        update_env("SCRAPER_DISABLED", ",".join(sorted(names)))
        return jsonify({"success": True, "disabled": names})
    if action == "enable_all":
        update_env("SCRAPER_DISABLED", "")
        return jsonify({"success": True, "disabled": []})
    return jsonify({"error": "action must be disable_all|enable_all"}), 400


@app.route("/api/scrapers/toggle", methods=["POST"])
def api_scraper_toggle():
    data = request.get_json() or {}
    name = data.get("name")
    enabled = bool(data.get("enabled"))
    if name not in {s["name"] for s in _scraper_definitions()}:
        return jsonify({"error": "Unknown scraper"}), 400
    current = disabled_set()
    (current.discard if enabled else current.add)(name)
    update_env("SCRAPER_DISABLED", ",".join(sorted(current)))
    return jsonify({"success": True, "name": name, "enabled": enabled})


@app.route("/api/pipeline/start", methods=["POST"])
def api_pipeline_start():
    global _pipeline_proc
    with _pipeline_lock:
        if pipeline_running():
            return jsonify({"success": True, "already_running": True, "pid": _pipeline_proc.pid})
        try:
            env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1"}
            args = [sys.executable, "-u", "run_pipeline.py"]
            kwargs = {"cwd": str(PIPELINE_CODE_DIR), "env": env}
            if sys.platform == "win32":
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            _pipeline_proc = subprocess.Popen(args, **kwargs)
            update_env("DASHBOARD_PAUSED", "false")
            return jsonify({"success": True, "pid": _pipeline_proc.pid})
        except Exception as exc:
            logger.exception("failed to start pipeline")
            return jsonify({"error": str(exc)}), 500


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key, "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


def _any_lmstudio_worker_active() -> bool:
    workers = _active_vision_workers()
    return any(w.startswith("balanced-lm") or w == "lm-autodetect" or w == "lm-keepalive"
               for w in workers)


@app.route("/api/pipeline/stop", methods=["POST"])
def api_pipeline_stop():
    global _pipeline_proc
    with _pipeline_lock:
        proc = _pipeline_proc
        if proc is None or proc.poll() is not None:
            _pipeline_proc = None
            return jsonify({"success": True, "already_stopped": True})
        try:
            if sys.platform == "win32":
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                proc.terminate()
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
        _pipeline_proc = None
        update_env("DASHBOARD_PAUSED", "true")

        unload_payload: dict[str, Any] | None = None
        if _env_bool("LMSTUDIO_UNLOAD_ON_STOP", True) and _any_lmstudio_worker_active():
            result = _lmstudio_unload_all()
            unload_payload = {"ok": result.ok, "detail": result.detail, "method": result.method}
            logger.info("LM Studio unload on stop: %s", unload_payload)
        return jsonify({"success": True, "lmstudio_unload": unload_payload})


# ── API: categories (user-editable taxonomy) ──────────────────────────────────

import categories as _categories_mod

_CAT_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,31}$")
_MAX_CATEGORIES = 12


def _validate_categories_payload(payload: Any) -> tuple[dict[str, Any] | None, str]:
    """Return (clean_payload, error). On error, payload is None."""
    if not isinstance(payload, dict):
        return None, "payload must be an object"
    cats = payload.get("categories")
    if not isinstance(cats, list) or not cats:
        return None, "categories must be a non-empty list"
    if len(cats) > _MAX_CATEGORIES:
        return None, f"max {_MAX_CATEGORIES} categories"
    seen: set[str] = set()
    cleaned: list[dict[str, str]] = []
    for entry in cats:
        if not isinstance(entry, dict):
            return None, "each category must be an object with name + hint"
        name = (entry.get("name") or "").strip()
        hint = (entry.get("hint") or "").strip()
        if not _CAT_NAME_RE.match(name):
            return None, f"invalid category name {name!r} (letters/digits/_ only, must start with letter, max 32 chars)"
        if name in {"DISCARD", "CORRUPT"}:
            return None, f"{name!r} is reserved for the system"
        if name in seen:
            return None, f"duplicate category name {name!r}"
        seen.add(name)
        cleaned.append({"name": name, "hint": hint})
    rules = payload.get("global_rules") or ""
    if not isinstance(rules, str):
        return None, "global_rules must be a string"
    if len(rules) > 8000:
        return None, "global_rules too long (max 8000 chars)"
    preset = (payload.get("preset") or "custom").strip() or "custom"
    return {"preset": preset, "categories": cleaned, "global_rules": rules}, ""


@app.route("/api/categories")
def api_categories_get():
    return jsonify({
        "active": _categories_mod.get_active(),
        "presets": _categories_mod.get_presets()["presets"],
        "default_preset": _categories_mod.get_presets().get("default"),
        "system_terminal": list(_categories_mod.SYSTEM_TERMINAL),
    })


@app.route("/api/categories", methods=["POST"])
def api_categories_set():
    payload = request.get_json() or {}
    cleaned, err = _validate_categories_payload(payload)
    if err:
        return jsonify({"ok": False, "error": err}), 400

    # If user is removing a category that already has sorted images, warn
    # unless they sent ?force=1.
    force = request.args.get("force") == "1"
    new_names = {c["name"] for c in cleaned["categories"]}
    current_names = set(_categories_mod.get_categories())
    removed = current_names - new_names
    if removed and not force and PIPELINE_SORTED.exists():
        legacy: dict[str, int] = {}
        for cat in removed:
            count = 0
            for topic_dir in PIPELINE_SORTED.iterdir():
                cat_dir = topic_dir / cat
                if cat_dir.is_dir():
                    count += sum(1 for _ in cat_dir.rglob("*.jpg")) + sum(1 for _ in cat_dir.rglob("*.png"))
            if count:
                legacy[cat] = count
        if legacy:
            return jsonify({
                "ok": False,
                "error": "removing_populated_categories",
                "legacy": legacy,
                "hint": "These categories still have sorted images. Requeue them first via tools/requeue_sorted.py, or POST again with ?force=1 to remove anyway (folders stay on disk).",
            }), 409

    _categories_mod.set_active(cleaned)
    return jsonify({"ok": True, "active": cleaned})


@app.route("/api/lmstudio/unload", methods=["POST"])
def api_lmstudio_unload():
    """Manually unload every loaded LM Studio model. Always callable."""
    result = _lmstudio_unload_all()
    return jsonify({
        "ok": result.ok,
        "detail": result.detail,
        "method": result.method,
    }), (200 if result.ok else 502)


# ── API: self-update ──────────────────────────────────────────────────────────
#
# Pulls origin/main and relaunches via update.bat / update.sh. Toast in the
# dashboard polls /api/update/check; clicking [Update] hits /api/update/run.

REPO_ROOT: Path = ENV_PATH.parent
_UPDATE_CHECK_CACHE: dict[str, Any] = {"at": 0.0, "payload": None}
_UPDATE_CHECK_TTL_SECONDS = 300  # 5 minutes


def _git(*args: str, timeout: int = 15) -> tuple[int, str, str]:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


@app.route("/api/update/check")
def api_update_check():
    """Return whether origin/main is ahead of the local checkout."""
    import time as _t
    now = _t.time()
    cached = _UPDATE_CHECK_CACHE.get("payload")
    if cached and now - _UPDATE_CHECK_CACHE.get("at", 0.0) < _UPDATE_CHECK_TTL_SECONDS:
        return jsonify(cached)

    if not (REPO_ROOT / ".git").exists():
        payload = {"ok": False, "error": "not a git checkout"}
        _UPDATE_CHECK_CACHE.update({"at": now, "payload": payload})
        return jsonify(payload)

    try:
        # Fetch is the slow part; cache results so we don't hit GitHub every page-load.
        rc_fetch, _out, err_fetch = _git("fetch", "origin", "main", "--quiet")
        if rc_fetch != 0:
            payload = {"ok": False, "error": f"git fetch failed: {err_fetch[:200]}"}
            _UPDATE_CHECK_CACHE.update({"at": now, "payload": payload})
            return jsonify(payload)

        _, local_sha, _ = _git("rev-parse", "HEAD")
        _, remote_sha, _ = _git("rev-parse", "origin/main")
        if not local_sha or not remote_sha:
            return jsonify({"ok": False, "error": "rev-parse failed"})

        rc_count, behind_str, _ = _git("rev-list", "--count", f"{local_sha}..{remote_sha}")
        try:
            behind = int(behind_str) if rc_count == 0 else 0
        except ValueError:
            behind = 0

        remote_subject = ""
        if behind > 0:
            _, remote_subject, _ = _git("log", "-1", "--pretty=%s", remote_sha)

        # Check for dirty tree — we won't run an update if it's dirty, so flag it.
        _, dirty_out, _ = _git("status", "--porcelain")
        payload = {
            "ok": True,
            "behind": behind,
            "local_sha": local_sha[:12],
            "remote_sha": remote_sha[:12],
            "remote_subject": remote_subject,
            "dirty": bool(dirty_out),
        }
        _UPDATE_CHECK_CACHE.update({"at": now, "payload": payload})
        return jsonify(payload)
    except subprocess.TimeoutExpired:
        return jsonify({"ok": False, "error": "git timed out"})
    except FileNotFoundError:
        return jsonify({"ok": False, "error": "git not on PATH"})


@app.route("/api/update/run", methods=["POST"])
def api_update_run():
    """Spawn the update script detached and return immediately.

    The script pulls + reinstalls deps + relaunches. The dashboard process
    exits to release port 5000 so the relaunch can bind it; the user just
    needs to refresh once the new process boots.
    """
    if not (REPO_ROOT / ".git").exists():
        return jsonify({"ok": False, "error": "not a git checkout"}), 400

    # Block on a dirty tree so we don't clobber unstaged work.
    try:
        _, dirty_out, _ = _git("status", "--porcelain")
        if dirty_out:
            return jsonify({"ok": False, "error": "uncommitted changes present"}), 409
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return jsonify({"ok": False, "error": f"git unavailable: {exc}"}), 500

    if sys.platform == "win32":
        script = REPO_ROOT / "update.bat"
        if not script.exists():
            return jsonify({"ok": False, "error": "update.bat missing"}), 500
        # `start /B` would inherit our console; using DETACHED_PROCESS gives the
        # child its own session so it survives our exit.
        DETACHED_PROCESS = 0x00000008
        subprocess.Popen(
            ["cmd", "/c", str(script)],
            cwd=str(REPO_ROOT),
            creationflags=DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
            close_fds=True,
        )
    else:
        script = REPO_ROOT / "update.sh"
        if not script.exists():
            return jsonify({"ok": False, "error": "update.sh missing"}), 500
        # Make sure the script is executable (fresh clones may not have +x set).
        try:
            os.chmod(script, 0o755)
        except OSError:
            pass
        subprocess.Popen(
            ["/bin/bash", str(script)],
            cwd=str(REPO_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )

    # Give the script a couple of seconds to start before we exit.
    def _delayed_exit() -> None:
        import time as _t
        _t.sleep(2)
        os._exit(0)
    threading.Thread(target=_delayed_exit, daemon=True).start()

    return jsonify({"ok": True, "message": "Update started — dashboard will restart."})


ALLOWED_VISION_WORKERS = [
    "balanced-groq",
    "balanced-lm",            # targets LMSTUDIO_PRIMARY_*
    "balanced-lm-secondary",  # same worker script, forced to LMSTUDIO_SECONDARY_* via env override
    "lm-autodetect",
]

_VISION_WORKER_DESCRIPTIONS = {
    "balanced-groq":          "Groq cloud, llama-4-scout - fast",
    "balanced-lm":            "LMStudio PRIMARY endpoint",
    "balanced-lm-secondary":  "LMStudio SECONDARY endpoint (runs in parallel)",
    "lm-autodetect":          "LMStudio, auto-picks vision-capable model",
}


def _active_vision_workers() -> list[str]:
    raw = os.environ.get("PIPELINE_VISION_WORKERS", "").strip()
    if raw:
        return [w.strip() for w in raw.split(",") if w.strip()]
    single = os.environ.get("PIPELINE_VISION_WORKER", "").strip()
    return [single] if single else []


@app.route("/api/vision/workers")
def api_vision_workers():
    active = set(_active_vision_workers())
    return jsonify([
        {"name": name, "enabled": name in active} for name in ALLOWED_VISION_WORKERS
    ])


@app.route("/api/vision/workers/toggle", methods=["POST"])
def api_vision_workers_toggle():
    data = request.get_json() or {}
    name = data.get("name", "")
    enabled = bool(data.get("enabled"))
    if name not in ALLOWED_VISION_WORKERS:
        return jsonify({"error": f"name must be one of {ALLOWED_VISION_WORKERS}"}), 400
    current = _active_vision_workers()
    if enabled and name not in current:
        current.append(name)
    elif not enabled and name in current:
        current = [w for w in current if w != name]
    update_env("PIPELINE_VISION_WORKERS", ",".join(current))
    if current:
        update_env("PIPELINE_VISION_WORKER", current[0])
    return jsonify({"success": True, "active": current})


SETTINGS_KEYS: list[str] = [
    # Topic + categorisation
    "PIPELINE_TOPIC",
    "PIPELINE_SLUG",
    "TOPIC_KEYWORDS_EXTRA",
    "TOPIC_BANNED_KEYWORDS",
    "TOPIC_GENERATION_HINTS",
    "REDDIT_SUBREDDITS",
    "MIN_PROMPT_LENGTH",
    "REQUIRE_PROMPT",
    "X_ACCOUNTS",
    # Vision quality + UX
    "VISION_OVR_MIN_SCORE",
    "VISION_REL_MIN_SCORE",
    "BLUR_NSFW_THUMBS",
    "VISION_SCORE_NOTES",
    "PIPELINE_RECONCILE_SECONDS",
    # Auto-captioning
    "AUTO_CAPTION_ENABLED",
    "AUTO_CAPTION_STYLE",
    "AUTO_CAPTION_OVERWRITE",
    # Storage paths
    "PIPELINE_BASE_DIR",
    "PIPELINE_QUEUE",
    "PIPELINE_SORTED",
    "LOG_DIR",
    "ZFORFREE_LOCAL_SRC",
    "LOCAL_IMPORT_DIR",
    "LOCAL_IMPORT_NAME",
    "LOCAL_IMPORT_ENABLED",
    "LOCAL_IMPORT_MIGRATE_FROM",
    # Vision provider credentials
    "GROQ_API_KEY",
    "GROQ_API_KEYS",
    "GROQ_MODEL",
    "LMSTUDIO_PRIMARY_TIMEOUT",
    "LMSTUDIO_SECONDARY_TIMEOUT",
    "LMSTUDIO_UNLOAD_ON_STOP",
    "LMSTUDIO_IDLE_UNLOAD_MINUTES",
    # Scraper credentials
    "CIVITAI_API_KEY",
    "CIVITAI_API_RED_KEY",
    "CIVITAI_DOMAINS",
    "TWITTER_COOKIES",
    "DISCORD_BOT_TOKEN",
    "DISCORD_AUTH_MODE",
    "DISCORD_CHANNELS_JSON",
    "REDDIT_CLIENT_ID",
    "REDDIT_CLIENT_SECRET",
    "REDDIT_USER_AGENT",
    # ZForFree feeders
    "ZFORFREE_LOCAL_ENABLED",
    "ZFORFREE_WEB_ENABLED",
    # gallery-dl scraper
    "GALLERY_DL_ENABLED",
    "GALLERY_DL_URLS",
    "GALLERY_DL_LIMIT_PER_URL",
    "GALLERY_DL_COOKIES_FILE",
    "GALLERY_DL_CONFIG_PATH",
]
SECRET_KEYS: set[str] = {
    "GROQ_API_KEY", "GROQ_API_KEYS",
    "CIVITAI_API_KEY", "CIVITAI_API_RED_KEY",
    "TWITTER_COOKIES", "DISCORD_BOT_TOKEN", "DISCORD_CHANNELS_JSON",
    "REDDIT_CLIENT_SECRET",
}
PATH_KEYS: set[str] = {"PIPELINE_BASE_DIR", "PIPELINE_QUEUE", "PIPELINE_SORTED",
                       "LOG_DIR", "ZFORFREE_LOCAL_SRC", "LOCAL_IMPORT_DIR"}


def _slugify(topic: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", topic.lower()).strip("_")
    return slug or "default"


@app.route("/api/settings")
def api_settings_get():
    return jsonify({key: os.environ.get(key, "") for key in SETTINGS_KEYS})


@app.route("/api/settings", methods=["POST"])
def api_settings_post():
    print(f"[api_settings_post] REQUEST RECEIVED")
    print(f"[api_settings_post] ENV_PATH={ENV_PATH}, exists={ENV_PATH.exists()}")
    
    data = request.get_json() or {}
    errors: dict[str, str] = {}
    changes: dict[str, str] = {}

    for key, value in data.items():
        if key not in SETTINGS_KEYS:
            errors[key] = "unknown setting"
            continue
        value = ("" if value is None else str(value)).strip()
        if key == "MIN_PROMPT_LENGTH" and value:
            try:
                parsed = int(value)
                if parsed < 0 or parsed > 10000:
                    raise ValueError
                value = str(parsed)
            except ValueError:
                errors[key] = "must be a non-negative integer"
                continue
        if key in {"VISION_OVR_MIN_SCORE", "VISION_REL_MIN_SCORE"} and value:
            try:
                parsed = int(value)
                if parsed < 0 or parsed > 100:
                    raise ValueError
                value = str(parsed)
            except ValueError:
                errors[key] = "must be an integer 0-100"
                continue
        if key == "PIPELINE_RECONCILE_SECONDS" and value:
            try:
                parsed = int(value)
                if parsed < 1 or parsed > 3600:
                    raise ValueError
                value = str(parsed)
            except ValueError:
                errors[key] = "must be an integer 1-3600"
                continue
        if key == "DISCORD_AUTH_MODE" and value:
            if value.lower() not in {"bot", "user", "auto"}:
                errors[key] = "must be 'bot', 'user', or 'auto'"
                continue
            value = value.lower()
        if key in PATH_KEYS and value:
            try:
                p = Path(value)
                if not p.is_absolute():
                    errors[key] = "must be an absolute path"
                    continue
                # Refuse obviously unsafe roots.
                if str(p).lower() in {r"c:\\windows", r"c:\\"}:
                    errors[key] = "path not allowed"
                    continue
            except Exception as exc:
                errors[key] = f"invalid path: {exc}"
                continue
        changes[key] = value

    if errors:
        return jsonify({"success": False, "errors": errors, "applied": {}}), 400

    # Auto-derive slug if topic changed and slug not explicitly provided.
    if "PIPELINE_TOPIC" in changes and "PIPELINE_SLUG" not in changes:
        changes["PIPELINE_SLUG"] = _slugify(changes["PIPELINE_TOPIC"])

    for key, value in changes.items():
        update_env(key, value)

    return jsonify({"success": True, "applied": changes, "restart_required": True})


@app.route("/api/vision/test", methods=["POST"])
def api_vision_test():
    """Verify the user's stored credentials hit a real backend.

    Caller passes ``{"provider": "groq"|"lmstudio"}``. We do the
    cheapest possible probe per provider and return ``{ok, message, latency_ms}``
    so the Settings UI can surface a working/broken indicator next to each
    credential field without forcing the user to start the whole pipeline.
    """
    import time as _t
    data = request.get_json() or {}
    provider = (data.get("provider") or "").strip().lower()
    started = _t.time()

    def _done(ok: bool, message: str, status: int = 200) -> Any:
        return jsonify({
            "ok": ok, "message": message,
            "latency_ms": int((_t.time() - started) * 1000),
            "provider": provider,
        }), status

    try:
        import requests
        if provider == "lmstudio":
            url = (data.get("url") or os.environ.get("LMSTUDIO_PRIMARY_URL", "")).rstrip("/")
            if not url:
                return _done(False, "no LMSTUDIO_PRIMARY_URL configured", 400)
            r = requests.get(f"{url}/v1/models", timeout=5)
            if r.status_code != 200:
                return _done(False, f"HTTP {r.status_code}: {r.text[:200]}")
            payload = r.json()
            n = len(payload.get("data") or [])
            return _done(True, f"connected, {n} model(s) loaded")
        if provider == "groq":
            key = os.environ.get("GROQ_API_KEY", "") or (
                os.environ.get("GROQ_API_KEYS", "").split(",")[0].strip()
            )
            if not key:
                return _done(False, "no GROQ_API_KEY configured", 400)
            r = requests.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {key}"},
                timeout=10,
            )
            if r.status_code == 401:
                return _done(False, "401 Unauthorized - key invalid")
            if r.status_code != 200:
                return _done(False, f"HTTP {r.status_code}: {r.text[:200]}")
            return _done(True, "Groq key accepted")
        return _done(False, f"unknown provider: {provider!r}", 400)
    except requests.RequestException as exc:
        return _done(False, f"connection error: {exc}")


@app.route("/api/vision/provider", methods=["POST"])
def api_vision_provider():
    """Legacy single-provider endpoint (kept so existing clients keep working)."""
    data = request.get_json() or {}
    provider = data.get("provider", "")
    if provider not in ALLOWED_VISION_WORKERS:
        return jsonify({"error": f"provider must be one of {ALLOWED_VISION_WORKERS}"}), 400
    update_env("PIPELINE_VISION_WORKER", provider)
    update_env("PIPELINE_VISION_WORKERS", provider)
    return jsonify({"success": True, "provider": provider})


@app.route("/api/vision/throttle", methods=["POST"])
def api_vision_throttle():
    data = request.get_json() or {}
    try:
        percent = int(data.get("percent", 100))
    except (TypeError, ValueError):
        return jsonify({"error": "percent must be int"}), 400
    percent = max(0, min(100, percent))
    update_env("DASHBOARD_THROTTLE_PERCENT", str(percent))
    return jsonify({"success": True, "throttle": percent})


@app.route("/api/lmstudio/endpoint", methods=["POST"])
def api_lmstudio_endpoint():
    data = request.get_json() or {}
    instance = data.get("instance")
    url = data.get("url", "").strip()
    model = data.get("model", "").strip()
    if instance not in {"primary", "secondary"}:
        return jsonify({"error": "instance must be primary|secondary"}), 400
    if not url:
        return jsonify({"error": "url required"}), 400
    prefix = "LMSTUDIO_PRIMARY" if instance == "primary" else "LMSTUDIO_SECONDARY"
    update_env(f"{prefix}_URL", url)
    if model:
        update_env(f"{prefix}_MODEL", model)
    return jsonify({"success": True, "instance": instance, "url": url, "model": model})


@app.route("/api/lmstudio/models")
def api_lmstudio_models():
    return jsonify({
        "instances": get_all_models(),
        "recommended": get_recommended_models(),
        "current": {
            "primary": {"url": get_env("LMSTUDIO_PRIMARY_URL"), "model": get_env("LMSTUDIO_PRIMARY_MODEL")},
            "secondary": {"url": get_env("LMSTUDIO_SECONDARY_URL"), "model": get_env("LMSTUDIO_SECONDARY_MODEL")},
        },
    })


@app.route("/api/lmstudio/set-model", methods=["POST"])
def api_set_model():
    data = request.get_json() or {}
    instance = data.get("instance")
    model_id = data.get("model_id")
    if instance not in {"primary", "secondary"} or not model_id:
        return jsonify({"error": "instance and model_id required"}), 400
    key = "LMSTUDIO_PRIMARY_MODEL" if instance == "primary" else "LMSTUDIO_SECONDARY_MODEL"
    update_env(key, model_id)
    return jsonify({"success": True, "instance": instance, "model_id": model_id})


# ── API: queue / thumbnails / prompts ─────────────────────────────────────────


@app.route("/api/queue/files")
def api_queue_files():
    """Newest queued images, served from the SQLite index.

    The old version globbed the entire queue tree on every request. Index-
    backed listing is sub-millisecond regardless of queue depth.
    """
    limit = int(request.args.get("limit", 60))
    results: list[dict[str, Any]] = []
    for item in index_store.list_recent_queue(limit=limit):
        path = Path(item.path)
        if not path.exists():
            continue  # indexer hasn't caught up to a worker that just moved it
        prompt = (item.prompt or "")[:300]
        results.append({
            "path": item.path,
            "name": path.name,
            "source": item.source,
            "size": item.size,
            "corrupt": item.size < 5000,
            "modified": datetime.fromtimestamp(item.mtime).isoformat(),
            "thumbnail": f"/api/thumbnail?path={item.path}",
            "prompt": prompt,
        })
    return jsonify(results)


@app.route("/api/queue/action", methods=["POST"])
def api_queue_action():
    data = request.get_json() or {}
    path = safe_inside(data.get("path", ""), [PIPELINE_QUEUE])
    action = data.get("action")
    target = data.get("target")
    if path is None or not path.exists():
        return jsonify({"error": "path outside queue or missing"}), 400

    if action == "delete":
        for sibling in path.parent.glob(f"{path.stem}.*"):
            sibling.unlink(missing_ok=True)
        return jsonify({"success": True, "action": "delete"})
    if action == "move":
        if not target:
            return jsonify({"error": "target required"}), 400
        dest = PIPELINE_QUEUE / target
        dest.mkdir(parents=True, exist_ok=True)
        for sibling in path.parent.glob(f"{path.stem}.*"):
            shutil.move(str(sibling), str(dest / sibling.name))
        return jsonify({"success": True, "action": "move", "target": str(dest)})
    return jsonify({"error": "unknown action"}), 400


# ── Brand assets ──────────────────────────────────────────────────────────────

# Whitelist - only these filenames are served from /brand/<name>. Keeps the
# endpoint from devolving into an arbitrary static-file server. Add new
# allowed assets here when extending the brand pack.
_BRAND_ASSETS: frozenset[str] = frozenset({
    "logo.png",
    "logo-transparent.png",
    "logo-transparent-dark.png",
})


@app.route("/brand/<filename>")
def brand_asset(filename: str):
    """Serve cull's brand assets (logos / wordmarks) from the repo's
    ``assets/`` folder. Used by the dashboard nav, About tab, welcome card,
    favicon, and empty states. Whitelist-gated; no path traversal possible.
    """
    if filename not in _BRAND_ASSETS:
        abort(404)
    from paths import REPO_ROOT
    asset_path = REPO_ROOT / "assets" / filename
    if not asset_path.exists():
        abort(404)
    return send_file(asset_path, mimetype="image/png", max_age=86400)


@app.route("/api/thumbnail")
def api_thumbnail():
    """Serve a JPEG thumbnail from the disk cache.

    First request for a (path, size) pair generates with PIL and writes
    to data/thumb_cache/<sha>/<sha>_<size>.jpg. Subsequent requests serve
    the file directly. send_file sets Last-Modified + ETag headers so
    browsers 304 on F5.
    """
    raw = request.args.get("path", "")
    try:
        size = max(64, min(2400, int(request.args.get("size", 240))))
    except ValueError:
        size = 240
    path = safe_inside(raw, [PIPELINE_QUEUE, PIPELINE_SORTED])
    if path is None or not path.exists():
        abort(404)
    try:
        cache_file = thumb_cache.get_or_create(path, size)
    except FileNotFoundError:
        abort(404)
    except Exception as exc:
        logger.debug("thumbnail fail: %s", exc)
        abort(404)
    return send_file(
        cache_file,
        mimetype="image/jpeg",
        max_age=86400,           # 1 day; the file path is content-hashed so it never goes stale.
        conditional=True,        # respect If-Modified-Since / If-None-Match for 304s.
    )


@app.route("/api/prompt")
def api_prompt():
    raw = request.args.get("path", "")
    path = safe_inside(raw, [PIPELINE_QUEUE, PIPELINE_SORTED])
    if path is None:
        abort(404)
    txt = path.with_suffix(".txt")
    if not txt.exists():
        return Response("", mimetype="text/plain")
    try:
        return Response(txt.read_text(encoding="utf-8", errors="replace"), mimetype="text/plain")
    except OSError:
        abort(404)


@app.route("/api/logs/history")
def api_logs_history():
    """Historical log of every classification, newest first.

    Served from the SQLite index, filterable by source/category. Pagination
    is via ?limit=. Pre-index versions globbed PIPELINE_SORTED on every call.
    """
    source = request.args.get("source")
    category = request.args.get("category")
    limit = int(request.args.get("limit", 200))

    items, _total = index_store.list_sorted(
        sources=[source] if source else None,
        categories=[category] if category else None,
        sort="newest",
        limit=limit,
        offset=0,
    )

    out: list[dict[str, Any]] = []
    for item in items:
        path = Path(item.path)
        if not path.exists():
            continue
        out.append({
            "timestamp": datetime.fromtimestamp(item.mtime).isoformat(),
            "image": path.name,
            "source": item.source,
            "classification": item.category,
            "quality": item.quality,
            "summary": (item.vision_json or {}).get("reason", "") if item.vision_json else "",
            "thumbnail": f"/api/thumbnail?path={item.path}",
            "prompt_url": f"/api/prompt?path={item.path}",
            "path": item.path,
        })
    return jsonify(out)


_IMAGE_EXTS_FOR_ACTIVITY: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in _IMAGE_EXTS_FOR_ACTIVITY


def _is_in_archive(path: Path) -> bool:
    """Skip files under the bookkeeping folders the requeue tool maintains."""
    return any(part.startswith(".") for part in path.relative_to(PIPELINE_SORTED).parts)


@app.route("/api/activity")
def api_activity():
    """Newest classified items, served from the SQLite index."""
    limit = int(request.args.get("limit", 12))
    results: list[dict[str, Any]] = []
    for item in index_store.list_recent_sorted(limit=limit):
        if not Path(item.path).exists():
            continue  # indexer hasn't caught up to a deletion / move
        results.append({
            "name": Path(item.path).name,
            "path": item.path,
            "category": item.category,
            "source": item.source,
            "modified": datetime.fromtimestamp(item.mtime).isoformat(),
            "thumbnail": f"/api/thumbnail?path={item.path}",
            "prompt_url": f"/api/prompt?path={item.path}",
            "summary": (item.vision_json or {}).get("reason", "") if item.vision_json else "",
            "quality": item.quality,
        })
    return jsonify(results)


# ── Sorted-folder scan + cache (powers stats + gallery) ───────────────────────

import time as _time
import zipfile as _zipfile
from collections import Counter as _Counter
from dataclasses import dataclass as _dataclass, field as _field

_STOPWORDS: frozenset[str] = frozenset({
    "the", "and", "with", "for", "from", "this", "that", "are", "was", "were",
    "have", "has", "had", "but", "not", "you", "your", "they", "their", "them",
    "she", "her", "his", "him", "its", "our", "their", "all", "any", "some",
    "one", "two", "more", "very", "much", "many", "most", "into", "over",
    "under", "than", "then", "there", "where", "when", "what", "which", "who",
    "how", "why", "out", "off", "out", "down", "back", "also", "just", "only",
    "even", "still", "yet", "such", "like", "can", "could", "would", "should",
    "may", "might", "will", "shall", "must", "about", "above", "below",
    "between", "through", "during", "before", "after", "while", "though",
    "because", "though", "since", "until", "without", "within", "across",
    "around", "behind", "beyond", "instead", "rather", "really", "quite",
    "hand", "front", "side", "scene", "view", "shot", "image", "photo",
})

_GALLERY_PAGE_LIMIT: int = 60
_ZIP_HARD_LIMIT: int = 5000
_STATS_CACHE_TTL: float = 60.0
_RESOLUTION_BUCKETS: tuple[tuple[str, int], ...] = (
    ("low", 768),
    ("medium", 1280),
    ("high", 2048),
)


@_dataclass
class _SortedItem:
    image_path: Path
    meta_path: Path
    txt_path: Path | None
    category: str
    source: str
    mtime: float
    payload: dict[str, Any]
    prompt_text: str
    width: int = 0
    height: int = 0


_sorted_cache: dict[str, Any] = {"items": [], "ts": 0.0, "signature": None}
_sorted_cache_lock = threading.Lock()
_dim_cache: dict[str, tuple[int, int]] = {}


def _resolve_image_dims(path: Path) -> tuple[int, int]:
    """Fallback dim resolver. Most rows have width/height stored in the index;
    this only fires for legacy data missing them."""
    key = str(path)
    cached = _dim_cache.get(key)
    if cached is not None:
        return cached
    try:
        with Image.open(path) as img:
            dims = img.size
    except (OSError, ValueError):
        dims = (0, 0)
    _dim_cache[key] = dims
    return dims


def _index_row_to_item(row: index_store.IndexedImage) -> _SortedItem:
    """Convert an IndexedImage from the SQLite index into the legacy
    _SortedItem shape every downstream helper still consumes (filters,
    keyword analytics, item_to_card, ZIP downloads)."""
    image_path = Path(row.path)
    return _SortedItem(
        image_path=image_path,
        meta_path=image_path.with_name(image_path.stem + ".vision.json"),
        txt_path=image_path.with_suffix(".txt") if (row.prompt or "") else None,
        category=row.category or "",
        source=row.source,
        mtime=row.mtime,
        payload=row.vision_json or {},
        prompt_text=row.prompt or "",
        width=row.width or 0,
        height=row.height or 0,
    )


def _get_sorted_items(force: bool = False) -> list[_SortedItem]:
    """Return every sorted image as a legacy _SortedItem.

    Backed by the SQLite index — the filesystem scan that this used to do
    has been replaced by the background indexer. We still cache the
    materialised _SortedItem list for `_STATS_CACHE_TTL` seconds because
    constructing 100k Python objects per /api/stats call would dominate
    the request time, and stats is the one place we actually need them all.
    Filter / paginated callers (gallery, history, activity) hit the index
    directly via index_store.list_sorted instead.
    """
    with _sorted_cache_lock:
        now = _time.time()
        if not force and _sorted_cache["items"] and (now - _sorted_cache["ts"]) < _STATS_CACHE_TTL:
            return _sorted_cache["items"]
        rows, _total = index_store.list_sorted(
            sort="newest", limit=1_000_000, offset=0,
        )
        items = [_index_row_to_item(r) for r in rows]
        _sorted_cache["items"] = items
        _sorted_cache["ts"] = now
        _sorted_cache["signature"] = (len(items), now)
        return items


_TOKEN_RX = re.compile(r"[A-Za-z][A-Za-z'\-]{2,}")


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _TOKEN_RX.findall(text or "")]


def _is_content_word(word: str) -> bool:
    return word not in _STOPWORDS and len(word) >= 3


def _resolution_bucket(width: int, height: int) -> str:
    if width <= 0 or height <= 0:
        return "unknown"
    longest = max(width, height)
    for label, ceiling in _RESOLUTION_BUCKETS:
        if longest <= ceiling:
            return label
    return "ultra"


def _item_to_card(item: _SortedItem) -> dict[str, Any]:
    """Compact JSON representation used by gallery + stats payloads."""
    payload = item.payload
    width, height = item.width, item.height
    if width <= 0 or height <= 0:
        width, height = _resolve_image_dims(item.image_path)
        item.width, item.height = width, height
    return {
        "name": item.image_path.name,
        "path": str(item.image_path),
        "category": item.category,
        "source": item.source,
        "modified": datetime.fromtimestamp(item.mtime).isoformat() if item.mtime else None,
        "thumbnail": f"/api/thumbnail?path={item.image_path}",
        "prompt_url": f"/api/prompt?path={item.image_path}",
        "ovr": payload.get("OVR_Quality_Score"),
        "rel": payload.get("REL_Quality_Score"),
        "quality": payload.get("quality_score"),
        "nsfw": bool(payload.get("nsfw")),
        "summary": payload.get("reason", ""),
        "score_reason": payload.get("score_reason", ""),
        "width": width,
        "height": height,
        "resolution_bucket": _resolution_bucket(width, height),
        "prompt_excerpt": (item.prompt_text or "").strip()[:240],
    }


# ── API: stats ────────────────────────────────────────────────────────────────

@app.route("/api/stats")
def api_stats():
    """Aggregates over every sorted image: keyword frequency, top thumbnails,
    per-source platform analytics. Cached for 60s via _get_sorted_items.
    """
    items = _get_sorted_items()
    non_discard = [it for it in items if it.category.upper() != "DISCARD"]

    # Keyword frequency over non-DISCARD prompts.
    keyword_counter: _Counter[str] = _Counter()
    for it in non_discard:
        keyword_counter.update(w for w in _tokenize(it.prompt_text) if _is_content_word(w))
    top_keywords = [
        {"term": term, "count": count}
        for term, count in keyword_counter.most_common(50)
    ]

    # Top-10 thumbnails by three lenses. Only consider non-DISCARD so DISCARD
    # entries with stale high scores can't pollute the leaderboards.
    def _top_by(metric_keys: tuple[str, ...]) -> list[dict[str, Any]]:
        scored: list[tuple[float, _SortedItem]] = []
        for it in non_discard:
            vals = [it.payload.get(k) for k in metric_keys]
            try:
                numeric = [float(v) for v in vals if v is not None]
            except (TypeError, ValueError):
                continue
            if not numeric:
                continue
            score = sum(numeric) / len(numeric)
            scored.append((score, it))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            {**_item_to_card(it), "metric_value": round(score, 1)}
            for score, it in scored[:10]
        ]

    top_overall = _top_by(("OVR_Quality_Score", "REL_Quality_Score"))
    top_quality = _top_by(("OVR_Quality_Score",))
    top_relative = _top_by(("REL_Quality_Score",))

    # Per-source platform analytics. NSFW% counts category=='NSFW'; discard%
    # counts category=='DISCARD'. Quality/REL averages exclude DISCARD so a
    # noisy reject pile doesn't drag the bar down.
    by_source: dict[str, dict[str, Any]] = {}
    for it in items:
        bucket = by_source.setdefault(
            it.source,
            {"total": 0, "discard": 0, "nsfw": 0, "ovr_sum": 0.0, "ovr_n": 0,
             "rel_sum": 0.0, "rel_n": 0},
        )
        bucket["total"] += 1
        cat_upper = it.category.upper()
        if cat_upper == "DISCARD":
            bucket["discard"] += 1
        if cat_upper == "NSFW":
            bucket["nsfw"] += 1
        if cat_upper != "DISCARD":
            ovr = it.payload.get("OVR_Quality_Score")
            rel = it.payload.get("REL_Quality_Score")
            try:
                if ovr is not None:
                    bucket["ovr_sum"] += float(ovr)
                    bucket["ovr_n"] += 1
                if rel is not None:
                    bucket["rel_sum"] += float(rel)
                    bucket["rel_n"] += 1
            except (TypeError, ValueError):
                pass

    sources = []
    for name, b in by_source.items():
        total = max(1, b["total"])
        sources.append({
            "source": name,
            "total": b["total"],
            "discard_pct": round(100.0 * b["discard"] / total, 1),
            "nsfw_pct": round(100.0 * b["nsfw"] / total, 1),
            "avg_ovr": round(b["ovr_sum"] / b["ovr_n"], 1) if b["ovr_n"] else None,
            "avg_rel": round(b["rel_sum"] / b["rel_n"], 1) if b["rel_n"] else None,
        })
    sources.sort(key=lambda r: r["total"], reverse=True)

    return jsonify({
        "totals": {
            "all": len(items),
            "non_discard": len(non_discard),
            "discard": len(items) - len(non_discard),
        },
        "keywords": top_keywords,
        "top_overall": top_overall,
        "top_quality": top_quality,
        "top_relative": top_relative,
        "sources": sources,
        "cached_age": round(_time.time() - _sorted_cache["ts"], 1),
    })


# ── API: gallery (paginated + filtered) ───────────────────────────────────────

def _parse_int(value: str | None, default: int = 0) -> int:
    try:
        return int(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _filter_items(items: list[_SortedItem], args: Any) -> list[_SortedItem]:
    """Apply gallery filters from query-string arguments."""
    sources_filter = {s.strip() for s in (args.get("sources") or "").split(",") if s.strip()}
    categories_filter = {c.strip() for c in (args.get("categories") or "").split(",") if c.strip()}
    resolutions_filter = {r.strip() for r in (args.get("resolutions") or "").split(",") if r.strip()}

    keyword = (args.get("q") or "").strip().lower()
    nsfw_mode = (args.get("nsfw") or "any").lower()  # any | only | exclude

    min_ovr = _parse_int(args.get("min_ovr"))
    min_rel = _parse_int(args.get("min_rel"))
    min_quality = _parse_int(args.get("min_quality"))

    date_from = (args.get("date_from") or "").strip()
    date_to = (args.get("date_to") or "").strip()

    def _date_to_ts(value: str, end_of_day: bool) -> float | None:
        if not value:
            return None
        try:
            base = datetime.fromisoformat(value)
        except ValueError:
            return None
        if end_of_day:
            base = base.replace(hour=23, minute=59, second=59)
        return base.timestamp()

    from_ts = _date_to_ts(date_from, end_of_day=False)
    to_ts = _date_to_ts(date_to, end_of_day=True)

    out: list[_SortedItem] = []
    for it in items:
        if sources_filter and it.source not in sources_filter:
            continue
        if categories_filter and it.category not in categories_filter:
            continue
        is_nsfw = it.category.upper() == "NSFW" or bool(it.payload.get("nsfw"))
        if nsfw_mode == "only" and not is_nsfw:
            continue
        if nsfw_mode == "exclude" and is_nsfw:
            continue
        if from_ts is not None and it.mtime < from_ts:
            continue
        if to_ts is not None and it.mtime > to_ts:
            continue
        if min_ovr:
            try:
                if float(it.payload.get("OVR_Quality_Score") or 0) < min_ovr:
                    continue
            except (TypeError, ValueError):
                continue
        if min_rel:
            try:
                if float(it.payload.get("REL_Quality_Score") or 0) < min_rel:
                    continue
            except (TypeError, ValueError):
                continue
        if min_quality:
            try:
                if float(it.payload.get("quality_score") or 0) < min_quality:
                    continue
            except (TypeError, ValueError):
                continue
        if resolutions_filter:
            if it.width <= 0 or it.height <= 0:
                w, h = _resolve_image_dims(it.image_path)
                it.width, it.height = w, h
            if _resolution_bucket(it.width, it.height) not in resolutions_filter:
                continue
        if keyword:
            haystack = (it.prompt_text + "\n" + it.payload.get("description", "")
                        + "\n" + it.payload.get("primary_subject", "")).lower()
            if keyword not in haystack:
                continue
        out.append(it)
    return out


@app.route("/api/gallery")
def api_gallery():
    items = _get_sorted_items()
    filtered = _filter_items(items, request.args)
    sort_key = (request.args.get("sort") or "newest").lower()
    if sort_key == "ovr":
        filtered.sort(key=lambda it: float(it.payload.get("OVR_Quality_Score") or 0), reverse=True)
    elif sort_key == "rel":
        filtered.sort(key=lambda it: float(it.payload.get("REL_Quality_Score") or 0), reverse=True)
    elif sort_key == "quality":
        filtered.sort(key=lambda it: float(it.payload.get("quality_score") or 0), reverse=True)
    else:
        filtered.sort(key=lambda it: it.mtime, reverse=True)
    page = max(1, _parse_int(request.args.get("page"), default=1))
    page_size = max(1, min(240, _parse_int(request.args.get("page_size"), default=_GALLERY_PAGE_LIMIT)))
    start = (page - 1) * page_size
    page_items = filtered[start:start + page_size]
    return jsonify({
        "page": page,
        "page_size": page_size,
        "total": len(filtered),
        "total_unfiltered": len(items),
        "items": [_item_to_card(it) for it in page_items],
        "available": {
            "sources": sorted({it.source for it in items}),
            "categories": sorted({it.category for it in items}),
            "resolutions": [b[0] for b in _RESOLUTION_BUCKETS] + ["ultra", "unknown"],
        },
    })


@app.route("/api/gallery/insights")
def api_gallery_insights():
    """N-gram insights computed only over the FILTERED set so they reflect
    whatever the user is actively browsing."""
    items = _get_sorted_items()
    filtered = _filter_items(items, request.args)
    non_discard = [it for it in filtered if it.category.upper() != "DISCARD"]

    ngram_min = max(2, _parse_int(request.args.get("ngram_min"), default=3))
    ngram_max = max(ngram_min, _parse_int(request.args.get("ngram_max"), default=5))

    ngrams: _Counter[str] = _Counter()
    high_quality_keywords: _Counter[str] = _Counter()
    if non_discard:
        ovr_values = [
            float(it.payload.get("OVR_Quality_Score") or 0) for it in non_discard
        ]
        ovr_threshold = sorted(ovr_values, reverse=True)[max(0, len(ovr_values) // 4 - 1)] if ovr_values else 0
    else:
        ovr_threshold = 0

    for it in non_discard:
        tokens = _tokenize(it.prompt_text)
        # N-grams: keep raw token sequences (preserve stopwords for fluency).
        for n in range(ngram_min, ngram_max + 1):
            if len(tokens) < n:
                continue
            for i in range(len(tokens) - n + 1):
                window = tokens[i:i + n]
                if sum(1 for w in window if _is_content_word(w)) < max(1, n // 2):
                    continue  # drop windows that are mostly stopwords
                ngrams[" ".join(window)] += 1
        try:
            ovr = float(it.payload.get("OVR_Quality_Score") or 0)
        except (TypeError, ValueError):
            ovr = 0.0
        if ovr >= ovr_threshold:
            high_quality_keywords.update(
                w for w in tokens if _is_content_word(w)
            )

    return jsonify({
        "ngrams": [
            {"phrase": phrase, "count": count}
            for phrase, count in ngrams.most_common(40)
        ],
        "top_quality_keywords": [
            {"term": term, "count": count}
            for term, count in high_quality_keywords.most_common(40)
        ],
        "ovr_quality_threshold": round(ovr_threshold, 1),
        "considered": len(non_discard),
    })


# ── API: prompt save (overwrite, no backup per user spec) ─────────────────────

@app.route("/api/prompt", methods=["POST"])
def api_prompt_save():
    data = request.get_json() or {}
    raw = data.get("path") or ""
    text = data.get("text")
    if text is None:
        return jsonify({"error": "missing 'text' field"}), 400
    image_path = safe_inside(raw, [PIPELINE_QUEUE, PIPELINE_SORTED])
    if image_path is None:
        return jsonify({"error": "path outside pipeline roots"}), 400
    txt_path = image_path.with_suffix(".txt")
    try:
        txt_path.write_text(text, encoding="utf-8")
    except OSError as exc:
        return jsonify({"error": f"write failed: {exc}"}), 500
    # Invalidate the scan cache so the next /api/stats sees the new prompt.
    with _sorted_cache_lock:
        _sorted_cache["ts"] = 0.0
        _sorted_cache["signature"] = None
    return jsonify({"success": True, "path": str(txt_path), "bytes": len(text)})


# ── API: zip download of filtered gallery ─────────────────────────────────────

@app.route("/api/gallery/download.zip")
def api_gallery_download():
    items = _get_sorted_items()
    filtered = _filter_items(items, request.args)
    if not filtered:
        return Response("no items match the current filters", status=404, mimetype="text/plain")
    if len(filtered) > _ZIP_HARD_LIMIT:
        return Response(
            f"filter selects {len(filtered)} items; cap is {_ZIP_HARD_LIMIT}. "
            "Tighten filters and retry.",
            status=413, mimetype="text/plain",
        )

    buffer = io.BytesIO()
    seen_names: set[str] = set()
    with _zipfile.ZipFile(buffer, mode="w", compression=_zipfile.ZIP_DEFLATED, compresslevel=4) as zf:
        for it in filtered:
            base = f"{it.category}/{it.source}/{it.image_path.stem}"
            # Avoid name collisions across folders by suffixing with the
            # parent path hash if the same stem already landed in the archive.
            suffix = ""
            attempt = 0
            while f"{base}{suffix}{it.image_path.suffix}" in seen_names:
                attempt += 1
                suffix = f"_{attempt}"
            stem = f"{base}{suffix}"
            try:
                zf.writestr(f"{stem}{it.image_path.suffix}", it.image_path.read_bytes())
                seen_names.add(f"{stem}{it.image_path.suffix}")
            except OSError:
                continue
            try:
                zf.writestr(f"{stem}.vision.json",
                            it.meta_path.read_text(encoding="utf-8"))
            except OSError:
                pass
            if it.txt_path is not None:
                try:
                    zf.writestr(f"{stem}.txt", it.txt_path.read_text(encoding="utf-8", errors="replace"))
                except OSError:
                    pass
    buffer.seek(0)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return send_file(
        buffer,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"pipeline_gallery_{stamp}.zip",
        max_age=0,
    )


# ── UI ─────────────────────────────────────────────────────────────────────────

HTML_TEMPLATE = r"""<!doctype html>
<html lang="en" class="dark">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title x-text="'cull · ' + (tabs.find(t => t.id === active)?.label || 'overview')">cull</title>
<link rel="icon" type="image/png" href="/brand/logo-transparent-dark.png">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://unpkg.com/alpinejs@3.x.x/dist/cdn.min.js" defer></script>
<style>
  /* Brand palette
     ink     #0F1115   surface  #F5F2EC   keep-accent  #E8B73A
     discard #C8553D   subtle   #7A8088
     The dashboard runs dark, so ink/surface flip vs print. Yellow remains
     the "kept" signal everywhere; rust red is reserved for DISCARD. */
  body { font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
  .font-brand { font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace; }
  .brand-ink { color: #0F1115; }
  .brand-keep { color: #E8B73A; }
  .brand-discard { color: #C8553D; }
  .bg-brand-keep { background-color: #E8B73A; }
  .bg-brand-discard { background-color: #C8553D; }
  .card { background: rgba(15,23,42,0.78); border:1px solid rgba(51,65,85,0.5); backdrop-filter: blur(8px); }
  .pill { font-size:.65rem; letter-spacing:.05em; text-transform:uppercase; }
  table { width:100%; font-size:.85rem; }
  th { text-align:left; font-weight:600; color:#94a3b8; padding:.4rem .6rem; position:sticky; top:0; background:#0f172a; }
  td { padding:.4rem .6rem; border-top:1px solid rgba(51,65,85,.4); vertical-align:middle; }
  .thumb { width:56px; height:56px; object-fit:cover; border-radius:6px; background:#0f172a; border:1px solid #334155; }
  .thumb-lg { width:88px; height:88px; object-fit:cover; border-radius:8px; background:#0f172a; border:1px solid #334155; }
  .scroll-box { max-height: 70vh; overflow-y:auto; }
  details > summary { cursor: pointer; }
  details > summary::-webkit-details-marker { display:none; }
  .thumb, .thumb-lg { cursor: zoom-in; transition: transform .15s; }
  .thumb:hover, .thumb-lg:hover { transform: scale(1.04); border-color:#6366f1; }
  .prompt-snippet { display:-webkit-box; -webkit-line-clamp:3; -webkit-box-orient:vertical; overflow:hidden; }
  .link-btn { color:#818cf8; cursor:pointer; font-size:.75rem; }
  .link-btn:hover { color:#a5b4fc; text-decoration:underline; }
  /* NSFW blur container - the eye button overlays so the user can reveal a single thumb. */
  .nsfw-wrap { position:relative; display:inline-block; line-height:0; }
  .nsfw-blur { filter: blur(18px) saturate(0.7); transition: filter .2s; }
  .nsfw-blur:hover { filter: blur(14px) saturate(0.7); }
  .nsfw-eye {
    position:absolute; inset:0; display:flex; align-items:center; justify-content:center;
    background: rgba(15,23,42,0.45); border-radius:6px; cursor:pointer; user-select:none;
    color:#fbbf24; font-size:1rem; line-height:1;
  }
  .nsfw-eye:hover { background: rgba(15,23,42,0.65); color:#fde68a; }
  .nsfw-eye svg { width:60%; height:60%; max-width:34px; max-height:34px; opacity:.95; }
</style>
</head>
<body class="min-h-screen bg-slate-950 text-slate-100">
<div x-data="dashboard()" x-init="start()" class="flex min-h-screen">

  <!-- Indexer cold-scan toast — bottom-left, fires once on first run when
       the SQLite index is being populated for the first time. -->
  <div x-show="indexerToast.show" x-cloak x-transition.opacity
       class="fixed bottom-4 left-4 z-50 max-w-sm bg-slate-900 border border-amber-700 rounded-lg shadow-2xl p-4 text-sm">
    <div class="flex items-start gap-3">
      <svg class="w-5 h-5 mt-0.5 text-amber-400 animate-spin shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" stroke-linecap="round"/>
      </svg>
      <div class="flex-1">
        <div class="font-semibold mb-1">First-time indexing</div>
        <div class="text-xs text-slate-300 mb-2">
          cull is building its SQLite index from the existing
          <span x-show="indexer.queue_total + indexer.sorted_total === 0">queue + sorted folders</span>
          dataset for the first time. This is a one-time cost — subsequent
          launches read from <code class="text-amber-300">data/cull_index.sqlite3</code>
          and start instantly.
        </div>
        <div class="text-xs text-slate-400 font-mono mb-3"
             x-text="(indexer.files_seen || 0).toLocaleString() + ' files scanned · ' + (indexer.files_added || 0).toLocaleString() + ' rows committed'">
        </div>
        <button @click="indexerToast.show = false; indexerToast.dismissed = true"
                class="px-3 py-1.5 bg-slate-700 hover:bg-slate-600 rounded text-xs">Dismiss</button>
      </div>
    </div>
  </div>

  <!-- Update toast — bottom-right, rendered only when a newer commit exists on origin/main.
       Dismissal is keyed on remote SHA so a new release re-toasts. -->
  <div x-show="update.available" x-cloak x-transition.opacity
       class="fixed bottom-4 right-4 z-50 max-w-sm bg-slate-900 border border-slate-700 rounded-lg shadow-2xl p-4 text-sm">
    <div class="flex items-start gap-3">
      <div class="text-amber-400 mt-0.5">⬆</div>
      <div class="flex-1">
        <div class="font-semibold mb-1">
          Update available
          <span class="text-xs font-normal text-slate-400" x-text="update.behind + ' commit' + (update.behind === 1 ? '' : 's') + ' behind origin/main'"></span>
        </div>
        <div class="text-xs text-slate-300 mb-2 truncate" :title="update.remote_subject" x-text="update.remote_subject || ''"></div>
        <div class="text-xs text-slate-500 mb-3 font-mono" x-text="'→ ' + update.remote_sha"></div>
        <div x-show="update.error" class="text-xs text-rose-300 mb-2" x-text="update.error"></div>
        <div class="flex gap-2">
          <button @click="runUpdate()" :disabled="update.running"
            class="px-3 py-1.5 bg-amber-500 hover:bg-amber-400 text-slate-900 font-semibold rounded text-xs disabled:opacity-60">
            <span x-text="update.running ? 'Updating…' : 'Update'"></span>
          </button>
          <button @click="dismissUpdate()" :disabled="update.running"
            class="px-3 py-1.5 bg-slate-700 hover:bg-slate-600 rounded text-xs">Later</button>
        </div>
        <div x-show="update.running" class="text-xs text-slate-400 mt-2">
          Pulling, reinstalling deps if needed, and relaunching. Dashboard will restart — refresh in a moment.
        </div>
      </div>
    </div>
  </div>

  <!-- Mobile hamburger - hidden on lg+ where the sidebar is always visible. -->
  <button @click="sidebarOpen = !sidebarOpen" aria-label="Toggle navigation"
    class="lg:hidden fixed top-3 left-3 z-40 bg-slate-800 hover:bg-slate-700 rounded p-2">
    <svg class="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M3 6h18M3 12h18M3 18h18" stroke-linecap="round"/>
    </svg>
  </button>

  <aside :class="sidebarOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'"
    class="fixed lg:static inset-y-0 left-0 z-30 w-56 shrink-0 border-r border-slate-800 bg-slate-900/95 lg:bg-slate-900/60 p-4 space-y-1 transform transition-transform">
    <div class="mb-4">
      <!-- Brand lockup: inline SVG funnel mark + JetBrains Mono lowercase wordmark.
           One bead falling through, rendered in brand mustard. Keeps the dashboard
           branded before the user has dropped real logo SVGs into docs/brand/. -->
      <a href="#" @click.prevent="active = 'about'"
         class="flex items-center gap-2 hover:opacity-80 transition" aria-label="About cull">
        <img src="/brand/logo-transparent-dark.png" alt="" width="32" height="32"
             class="shrink-0"/>
        <span class="font-brand text-2xl font-medium tracking-tight">cull</span>
      </a>
      <p class="text-xs text-slate-400 mt-1" x-text="'Worker: ' + (status.pipeline?.vision_worker || '...')"></p>
      <div class="mt-2">
        <span class="pill px-2 py-0.5 rounded"
          :class="status.pipeline?.running ? 'bg-emerald-900/60 text-emerald-300' : 'bg-slate-800 text-slate-400'"
          x-text="status.pipeline?.running ? 'running' : 'stopped'"></span>
      </div>
    </div>
    <template x-for="tab in tabs" :key="tab.id">
      <button @click="active = tab.id; sidebarOpen = false"
        class="w-full text-left px-3 py-2 rounded text-sm transition"
        :class="active === tab.id ? 'bg-indigo-600 text-white' : 'text-slate-300 hover:bg-slate-800'"
        x-text="tab.label"></button>
    </template>
    <div class="pt-6 text-xs text-slate-500" x-text="'Refreshed: ' + lastRefresh"></div>
    <!-- Indexer status: shown only while a scan is running OR while there are
         indexed rows but no successful scan timestamp yet (initial bootstrap).
         The dashboard polls /api/status every 5 s; the indexer payload is
         baked into that response so we don't add a third poll. -->
    <div x-show="indexer.in_progress" x-cloak
         class="mt-2 flex items-center gap-2 text-[11px] text-amber-300">
      <svg class="w-3 h-3 animate-spin" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" stroke-linecap="round"/>
      </svg>
      <span x-text="'Indexing ' + (indexer.files_seen || 0).toLocaleString() + ' files'"></span>
    </div>
    <div x-show="!indexer.in_progress && indexer.last_scan_at" x-cloak
         class="mt-2 text-[11px] text-slate-500"
         x-text="'Indexed ' + ((indexer.queue_total || 0) + (indexer.sorted_total || 0)).toLocaleString() + ' images'">
    </div>
  </aside>

  <!-- Backdrop when sidebar is open on mobile. -->
  <div x-show="sidebarOpen" x-cloak class="lg:hidden fixed inset-0 z-20 bg-black/50"
    @click="sidebarOpen = false"></div>

  <main class="flex-1 p-6 space-y-6 overflow-y-auto">
    <header class="flex items-center justify-between">
      <div>
        <h2 class="text-2xl font-bold" x-text="tabs.find(t => t.id === active)?.label"></h2>
        <p class="text-sm text-slate-400">
          <span x-show="active !== 'gallery'">Realtime operations console - auto-refresh every 5 s</span>
          <span x-show="active === 'gallery'">Filter, browse, edit, and export the sorted library</span>
        </p>
      </div>
      <div class="flex gap-2">
        <template x-if="!status.pipeline?.running">
          <button @click="startPipeline()" class="px-4 py-2 rounded text-sm font-medium bg-emerald-600 hover:bg-emerald-500">Start pipeline</button>
        </template>
        <template x-if="status.pipeline?.running">
          <button @click="stopPipeline()" class="px-4 py-2 rounded text-sm font-medium bg-rose-600 hover:bg-rose-500">Stop pipeline</button>
        </template>
        <button @click="refresh()" class="px-4 py-2 rounded bg-slate-700 hover:bg-slate-600 text-sm">Refresh</button>
      </div>
    </header>

    <!-- OVERVIEW -->
    <section x-show="active === 'overview'" class="space-y-4">
      <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div class="card rounded-xl p-5">
          <div class="pill text-indigo-300">Queue</div>
          <div class="text-4xl font-bold mt-1" x-text="status.queue?.total ?? 0"></div>
          <div class="text-xs text-slate-400">images waiting for vision</div>
        </div>
        <div class="card rounded-xl p-5">
          <div class="pill text-emerald-300">Sorted</div>
          <div class="text-4xl font-bold mt-1" x-text="status.sorted?.total ?? 0"></div>
          <div class="text-xs text-slate-400">classified into categories</div>
        </div>
        <div class="card rounded-xl p-5">
          <div class="pill" :class="status.error_count ? 'text-rose-300' : 'text-slate-400'">Errors</div>
          <div class="text-4xl font-bold mt-1" :class="status.error_count ? 'text-rose-300' : ''" x-text="status.error_count ?? 0"></div>
          <div class="text-xs text-slate-400">in last 5 logs</div>
        </div>
      </div>

      <div class="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div class="card rounded-xl p-5">
          <h3 class="font-semibold mb-3">Queue by source</h3>
          <div class="scroll-box"><table>
            <thead><tr><th>Source</th><th class="text-right">Count</th></tr></thead>
            <tbody>
              <template x-for="[name, count] in Object.entries(status.queue?.by_source ?? {})" :key="name">
                <tr><td x-text="name"></td><td class="text-right font-mono" x-text="count"></td></tr>
              </template>
            </tbody>
          </table></div>
        </div>
        <div class="card rounded-xl p-5 lg:col-span-2">
          <h3 class="font-semibold mb-3">Recent vision activity</h3>
          <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
            <template x-for="a in activity" :key="a.path">
              <div class="bg-slate-900/60 border border-slate-800 rounded-lg p-2 flex gap-3 items-start">
                <span class="nsfw-wrap">
                  <img :src="a.thumbnail" :alt="a.name + ' - ' + a.category" class="thumb-lg" :class="{ 'nsfw-blur': shouldBlurNsfw(a) }"
                       loading="lazy" referrerpolicy="no-referrer"
                       @click="shouldBlurNsfw(a) ? revealNsfw(a) : openModalFromActivity(a)"/>
                  <span class="nsfw-eye" role="button" tabindex="0" aria-label="Reveal NSFW image" x-show="shouldBlurNsfw(a)" @click.stop="revealNsfw(a)" title="Reveal NSFW">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                      <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/>
                    </svg>
                  </span>
                </span>
                <div class="min-w-0 text-xs">
                  <div class="pill px-1.5 py-0.5 rounded bg-slate-800 text-slate-200" x-text="a.category"></div>
                  <div class="mt-1 font-mono truncate" x-text="a.name"></div>
                  <div class="text-slate-400 mt-0.5" x-text="a.source + ' - Q:' + (a.quality ?? '?')"></div>
                  <span class="link-btn" @click="openModalFromActivity(a)">See more</span>
                </div>
              </div>
            </template>
            <template x-if="activity.length === 0">
              <div class="text-sm text-slate-500 col-span-full">Nothing classified yet - start the pipeline.</div>
            </template>
          </div>
        </div>
      </div>

      <div class="card rounded-xl p-5">
        <h3 class="font-semibold mb-3">Sorted categories</h3>
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
          <template x-for="[topic, cats] in Object.entries(status.sorted?.by_topic ?? {})" :key="topic">
            <div>
              <div class="text-xs text-slate-400 mb-1" x-text="topic"></div>
              <template x-for="[cat, n] in Object.entries(cats)" :key="cat">
                <div class="flex justify-between py-1 text-sm">
                  <span x-text="cat"></span><span class="font-mono text-slate-300" x-text="n"></span>
                </div>
              </template>
            </div>
          </template>
        </div>
      </div>
    </section>

    <!-- STATS -->
    <section x-show="active === 'stats'" class="space-y-4">
      <div class="card rounded-xl p-5">
        <div class="flex items-center justify-between mb-2">
          <div>
            <h3 class="font-semibold">Pipeline analytics</h3>
            <p class="text-xs text-slate-400">Aggregates over every sorted image (DISCARD excluded for keyword + leaderboards). Cached for 60s.</p>
          </div>
          <button @click="loadStats()" :disabled="statsLoading"
            class="px-3 py-1.5 text-xs bg-slate-700 hover:bg-slate-600 rounded disabled:opacity-60">
            <span x-show="!statsLoading">Refresh stats</span>
            <span x-show="statsLoading">Scanning...</span>
          </button>
        </div>
        <div x-show="statsLoading && (stats.totals?.all ?? 0) === 0" class="mt-3 text-sm text-slate-400">
          Scanning sorted folder for the first time. This can take 30-60 seconds for large libraries; subsequent loads are cached for 60s.
        </div>
        <div class="grid grid-cols-2 md:grid-cols-4 gap-3 mt-2">
          <div class="bg-slate-900/60 border border-slate-800 rounded p-3"><div class="pill text-slate-400">Total classified</div><div class="text-2xl font-mono mt-1" x-text="stats.totals?.all ?? 0"></div></div>
          <div class="bg-slate-900/60 border border-slate-800 rounded p-3"><div class="pill text-emerald-300">Kept (non-DISCARD)</div><div class="text-2xl font-mono mt-1 text-emerald-200" x-text="stats.totals?.non_discard ?? 0"></div></div>
          <div class="bg-slate-900/60 border border-slate-800 rounded p-3"><div class="pill text-rose-300">Discarded</div><div class="text-2xl font-mono mt-1 text-rose-200" x-text="stats.totals?.discard ?? 0"></div></div>
          <div class="bg-slate-900/60 border border-slate-800 rounded p-3"><div class="pill text-slate-400">Cache age (s)</div><div class="text-2xl font-mono mt-1" x-text="stats.cached_age ?? 0"></div></div>
        </div>
      </div>

      <div class="card rounded-xl p-5">
        <h3 class="font-semibold mb-3">Top keywords across kept prompts</h3>
        <div class="flex flex-wrap gap-2">
          <template x-for="kw in stats.keywords" :key="kw.term">
            <button @click="active='gallery'; insertGalleryQuery(kw.term)"
              class="px-2 py-1 text-xs rounded bg-slate-800 hover:bg-indigo-600 hover:text-white border border-slate-700 transition"
              :title="'Show in Gallery: ' + kw.term">
              <span class="font-mono" x-text="kw.term"></span>
              <span class="ml-1 text-slate-400" x-text="'(' + kw.count + ')'"></span>
            </button>
          </template>
          <template x-if="(stats.keywords?.length ?? 0) === 0">
            <span class="text-xs text-slate-500">No keywords yet - process images first.</span>
          </template>
        </div>
      </div>

      <template x-for="bucket in [
          {key:'top_overall',  label:'Top 10 by overall (avg of OVR + REL)', metric:'metric_value'},
          {key:'top_quality',  label:'Top 10 by craft quality (OVR)',          metric:'metric_value'},
          {key:'top_relative', label:'Top 10 by topic relevance (REL)',        metric:'metric_value'},
        ]" :key="bucket.key">
        <div class="card rounded-xl p-5">
          <h3 class="font-semibold mb-3" x-text="bucket.label"></h3>
          <div class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 lg:grid-cols-10 gap-3">
            <template x-for="(c, idx) in (stats[bucket.key] || [])" :key="bucket.key + '_' + c.path">
              <div class="bg-slate-900/60 border border-slate-800 rounded p-2 text-center">
                <span class="nsfw-wrap block">
                  <img :src="c.thumbnail" :alt="c.name" class="thumb-lg mx-auto" :class="{ 'nsfw-blur': shouldBlurNsfw(c) }"
                       loading="lazy"
                       @click="shouldBlurNsfw(c) ? revealNsfw(c) : openModalFromCard(c)"/>
                  <span class="nsfw-eye" role="button" tabindex="0" aria-label="Reveal NSFW image" x-show="shouldBlurNsfw(c)" @click.stop="revealNsfw(c)" title="Reveal NSFW">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                      <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/>
                    </svg>
                  </span>
                </span>
                <div class="mt-1 text-xs text-slate-400" x-text="'#' + (idx+1) + ' · ' + c.metric_value"></div>
                <div class="text-xs font-mono truncate" x-text="c.source"></div>
              </div>
            </template>
            <template x-if="(stats[bucket.key]?.length ?? 0) === 0">
              <div class="col-span-full text-xs text-slate-500">No data yet.</div>
            </template>
          </div>
        </div>
      </template>

      <div class="card rounded-xl p-5">
        <h3 class="font-semibold mb-3">Per-source platform analytics</h3>
        <div class="scroll-box"><table>
          <thead><tr>
            <th>Source</th><th class="text-right">Total</th>
            <th class="text-right">DISCARD %</th><th class="text-right">NSFW %</th>
            <th class="text-right">Avg OVR</th><th class="text-right">Avg REL</th>
          </tr></thead>
          <tbody>
            <template x-for="s in stats.sources" :key="s.source">
              <tr>
                <td class="font-mono text-xs" x-text="s.source"></td>
                <td class="text-right font-mono" x-text="s.total"></td>
                <td class="text-right font-mono" :class="s.discard_pct >= 50 ? 'text-rose-300' : ''" x-text="s.discard_pct + '%'"></td>
                <td class="text-right font-mono" x-text="s.nsfw_pct + '%'"></td>
                <td class="text-right font-mono" x-text="s.avg_ovr ?? '-'"></td>
                <td class="text-right font-mono" x-text="s.avg_rel ?? '-'"></td>
              </tr>
            </template>
            <template x-if="(stats.sources?.length ?? 0) === 0">
              <tr><td colspan="6" class="text-xs text-slate-500">No sources yet.</td></tr>
            </template>
          </tbody>
        </table></div>
      </div>
    </section>

    <!-- GALLERY -->
    <section x-show="active === 'gallery'" class="space-y-4">
      <div class="card rounded-xl p-4">
        <div class="grid grid-cols-1 lg:grid-cols-12 gap-3">
          <div class="lg:col-span-4">
            <label class="text-xs text-slate-400">Search prompts / descriptions</label>
            <input x-model="galleryFilters.q" @keydown.enter="galleryReload()"
              class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 text-sm"
              placeholder="e.g. blonde hair, beach, neon"/>
          </div>
          <div class="lg:col-span-2">
            <label class="text-xs text-slate-400">Sort by</label>
            <select x-model="galleryFilters.sort" @change.debounce.300ms="galleryReload()"
              class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 text-sm">
              <option value="newest">Newest first</option>
              <option value="ovr">OVR (craft)</option>
              <option value="rel">REL (relevance)</option>
              <option value="quality">quality_score</option>
            </select>
          </div>
          <div class="lg:col-span-2">
            <label class="text-xs text-slate-400">NSFW</label>
            <select x-model="galleryFilters.nsfw" @change.debounce.300ms="galleryReload()"
              class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 text-sm">
              <option value="any">Show all</option>
              <option value="exclude">Hide NSFW</option>
              <option value="only">Only NSFW</option>
            </select>
          </div>
          <div class="lg:col-span-2">
            <label class="text-xs text-slate-400">Date from</label>
            <input type="date" x-model="galleryFilters.dateFrom" @change.debounce.300ms="galleryReload()"
              class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 text-sm"/>
          </div>
          <div class="lg:col-span-2">
            <label class="text-xs text-slate-400">Date to</label>
            <input type="date" x-model="galleryFilters.dateTo" @change.debounce.300ms="galleryReload()"
              class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 text-sm"/>
          </div>

          <div class="lg:col-span-2">
            <label class="text-xs text-slate-400">Min OVR</label>
            <input type="number" min="0" max="100" x-model.number="galleryFilters.minOvr" @change.debounce.300ms="galleryReload()"
              class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 text-sm"/>
          </div>
          <div class="lg:col-span-2">
            <label class="text-xs text-slate-400">Min REL</label>
            <input type="number" min="0" max="100" x-model.number="galleryFilters.minRel" @change.debounce.300ms="galleryReload()"
              class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 text-sm"/>
          </div>
          <div class="lg:col-span-2">
            <label class="text-xs text-slate-400">Min quality (1-10)</label>
            <input type="number" min="0" max="10" x-model.number="galleryFilters.minQuality" @change.debounce.300ms="galleryReload()"
              class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 text-sm"/>
          </div>
          <div class="lg:col-span-6 flex items-end gap-2">
            <button @click="galleryReload()" class="px-3 py-2 bg-indigo-600 hover:bg-indigo-500 rounded text-sm">Apply</button>
            <button @click="galleryClearFilters()" class="px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded text-sm">Clear</button>
            <button @click="galleryDownload()" class="px-3 py-2 bg-emerald-600 hover:bg-emerald-500 rounded text-sm"
              title="Stream a zip of the current filtered view (image + .txt + .vision.json)">Download view as ZIP</button>
            <span class="text-xs text-slate-400 ml-auto">
              <span x-text="gallery.total"></span> / <span x-text="gallery.total_unfiltered"></span> match
            </span>
          </div>

          <div class="lg:col-span-4">
            <label class="text-xs text-slate-400">Sources</label>
            <div class="flex flex-wrap gap-1 mt-1">
              <template x-for="src in gallery.available?.sources ?? []" :key="src">
                <button @click="toggleGalleryFilter('sources', src)"
                  class="px-2 py-0.5 text-xs rounded border"
                  :class="galleryFilters.sources.includes(src) ? 'bg-indigo-600 border-indigo-400 text-white' : 'bg-slate-800 border-slate-700 text-slate-300'"
                  x-text="src"></button>
              </template>
            </div>
          </div>
          <div class="lg:col-span-4">
            <label class="text-xs text-slate-400">Categories</label>
            <div class="flex flex-wrap gap-1 mt-1">
              <template x-for="cat in gallery.available?.categories ?? []" :key="cat">
                <button @click="toggleGalleryFilter('categories', cat)"
                  class="px-2 py-0.5 text-xs rounded border"
                  :class="galleryFilters.categories.includes(cat) ? 'bg-indigo-600 border-indigo-400 text-white' : 'bg-slate-800 border-slate-700 text-slate-300'"
                  x-text="cat"></button>
              </template>
            </div>
          </div>
          <div class="lg:col-span-4">
            <label class="text-xs text-slate-400">Resolution</label>
            <div class="flex flex-wrap gap-1 mt-1">
              <template x-for="res in gallery.available?.resolutions ?? []" :key="res">
                <button @click="toggleGalleryFilter('resolutions', res)"
                  class="px-2 py-0.5 text-xs rounded border"
                  :class="galleryFilters.resolutions.includes(res) ? 'bg-indigo-600 border-indigo-400 text-white' : 'bg-slate-800 border-slate-700 text-slate-300'"
                  x-text="res"></button>
              </template>
            </div>
          </div>
        </div>
      </div>

      <div class="card rounded-xl p-4">
        <div class="flex items-center justify-between mb-3">
          <h3 class="font-semibold">Results</h3>
          <div class="flex items-center gap-2 text-xs">
            <button @click="galleryPrev()" class="px-2 py-1 bg-slate-800 hover:bg-slate-700 rounded disabled:opacity-50" :disabled="gallery.page <= 1">Prev</button>
            <span>Page <span x-text="gallery.page"></span> / <span x-text="Math.max(1, Math.ceil(gallery.total / gallery.pageSize))"></span></span>
            <button @click="galleryNext()" class="px-2 py-1 bg-slate-800 hover:bg-slate-700 rounded disabled:opacity-50"
              :disabled="gallery.page >= Math.max(1, Math.ceil(gallery.total / gallery.pageSize))">Next</button>
          </div>
        </div>
        <div x-show="galleryLoading" class="text-xs text-slate-400">Loading...</div>
        <div class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3">
          <template x-for="c in gallery.items" :key="c.path">
            <div class="bg-slate-900/60 border border-slate-800 rounded p-2 text-xs flex flex-col">
              <span class="nsfw-wrap block">
                <img :src="c.thumbnail" :alt="c.name" class="w-full aspect-square object-cover rounded"
                     :class="{ 'nsfw-blur': shouldBlurNsfw(c) }"
                     loading="lazy"
                     @click="shouldBlurNsfw(c) ? revealNsfw(c) : openModalFromCard(c)"/>
                <span class="nsfw-eye" role="button" tabindex="0" aria-label="Reveal NSFW image" x-show="shouldBlurNsfw(c)" @click.stop="revealNsfw(c)" title="Reveal NSFW">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/>
                  </svg>
                </span>
              </span>
              <div class="mt-1 flex items-center justify-between gap-1">
                <span class="pill px-1.5 py-0.5 rounded bg-slate-800" x-text="c.category"></span>
                <span class="text-slate-400" x-text="c.source"></span>
              </div>
              <div class="mt-1 grid grid-cols-3 gap-1 text-center">
                <span class="bg-slate-800/60 rounded px-1" :title="'OVR ' + (c.ovr ?? '-')">O <span x-text="c.ovr ?? '-'"></span></span>
                <span class="bg-slate-800/60 rounded px-1" :title="'REL ' + (c.rel ?? '-')">R <span x-text="c.rel ?? '-'"></span></span>
                <span class="bg-slate-800/60 rounded px-1" :title="(c.width||'?') + 'x' + (c.height||'?')" x-text="c.resolution_bucket"></span>
              </div>
              <div class="mt-1 text-slate-400 text-[11px] truncate" x-text="c.prompt_excerpt || '(no prompt)'"></div>
              <span class="link-btn mt-1" @click="openModalFromCard(c)">Open</span>
            </div>
          </template>
          <template x-if="!galleryLoading && (gallery.items?.length ?? 0) === 0">
            <div class="col-span-full text-center py-10">
              <img src="/brand/logo-transparent-dark.png" alt="" width="56" height="56"
                   class="mx-auto opacity-60"/>
              <div class="mt-3 text-sm text-slate-300">Nothing kept yet for this filter.</div>
              <div class="text-xs text-slate-500 mt-1">
                Start a scraper, or run
                <code class="font-brand bg-slate-800 px-1.5 py-0.5 rounded text-slate-300">python tools/seed_demo_data.py</code>
                from the repo root to populate a synthetic preview.
              </div>
            </div>
          </template>
        </div>
      </div>

      <div class="grid grid-cols-1 lg:grid-cols-2 gap-3">
        <div class="card rounded-xl p-4">
          <h3 class="font-semibold mb-2">Frequent phrases (3-5 word n-grams) in current view</h3>
          <p class="text-xs text-slate-400 mb-2">Computed over the filtered, non-DISCARD subset.</p>
          <div class="flex flex-wrap gap-1">
            <template x-for="ng in galleryInsights.ngrams" :key="ng.phrase">
              <button @click="insertGalleryQuery(ng.phrase)"
                class="px-2 py-0.5 text-xs rounded bg-slate-800 hover:bg-indigo-600 hover:text-white border border-slate-700">
                <span class="font-mono" x-text="ng.phrase"></span>
                <span class="ml-1 text-slate-400" x-text="'(' + ng.count + ')'"></span>
              </button>
            </template>
            <template x-if="(galleryInsights.ngrams?.length ?? 0) === 0">
              <span class="text-xs text-slate-500">No phrases - try widening filters.</span>
            </template>
          </div>
        </div>
        <div class="card rounded-xl p-4">
          <h3 class="font-semibold mb-2">Top keywords in highest-scoring quartile</h3>
          <p class="text-xs text-slate-400 mb-2">
            Threshold: OVR >= <span x-text="galleryInsights.ovr_quality_threshold ?? '-'"></span>
            (over <span x-text="galleryInsights.considered ?? 0"></span> kept items in current filter).
          </p>
          <div class="flex flex-wrap gap-1">
            <template x-for="kw in galleryInsights.top_quality_keywords" :key="'hq_' + kw.term">
              <button @click="insertGalleryQuery(kw.term)"
                class="px-2 py-0.5 text-xs rounded bg-slate-800 hover:bg-emerald-600 hover:text-white border border-slate-700">
                <span class="font-mono" x-text="kw.term"></span>
                <span class="ml-1 text-slate-400" x-text="'(' + kw.count + ')'"></span>
              </button>
            </template>
          </div>
        </div>
      </div>
    </section>

    <!-- SCRAPERS -->
    <section x-show="active === 'scrapers'" class="card rounded-xl p-5">
      <div class="flex items-start justify-between gap-4 mb-3">
        <div>
          <h3 class="font-semibold">Scraper controls</h3>
          <p class="text-xs text-slate-400 mt-1">Toggles persist to <code>.env</code> (SCRAPER_DISABLED). Restart the pipeline to pick up changes.</p>
        </div>
        <div class="flex gap-2">
          <button @click="scrapersBulk('disable_all')" class="px-3 py-1.5 text-xs bg-rose-600/80 hover:bg-rose-500 rounded font-medium">All off (vision only)</button>
          <button @click="scrapersBulk('enable_all')" class="px-3 py-1.5 text-xs bg-emerald-600/80 hover:bg-emerald-500 rounded font-medium">All on</button>
        </div>
      </div>
      <div class="grid gap-2">
        <template x-for="s in scrapers" :key="s.name">
          <div class="flex items-center justify-between bg-slate-900/60 border border-slate-800 rounded px-4 py-3">
            <div>
              <div class="font-medium" x-text="s.name"></div>
              <div class="text-xs text-slate-400" x-text="s.description"></div>
            </div>
            <label class="inline-flex items-center cursor-pointer gap-2">
              <span class="text-xs" x-text="s.enabled ? 'On' : 'Off'"></span>
              <input type="checkbox" :checked="s.enabled" :aria-label="'Toggle scraper ' + s.name" @change="toggleScraper(s.name, $event.target.checked)"
                class="w-10 h-5 appearance-none bg-slate-700 rounded-full relative transition
                  checked:bg-indigo-500 before:content-[''] before:absolute before:top-0.5 before:left-0.5
                  before:w-4 before:h-4 before:bg-white before:rounded-full before:transition
                  checked:before:translate-x-5"/>
            </label>
          </div>
        </template>
      </div>
    </section>

    <!-- VISION -->
    <section x-show="active === 'vision'" class="space-y-4">
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div class="card rounded-xl p-5">
          <h3 class="font-semibold mb-3">Vision workers</h3>
          <p class="text-xs text-slate-400 mb-3">
            Toggle multiple providers on to run them in parallel. Each enabled worker runs as its own subprocess and
            pulls from the shared queue via round-robin. Writes <code>PIPELINE_VISION_WORKERS</code>. Restart pipeline
            to apply changes.
          </p>
          <div class="grid gap-2">
            <template x-for="w in visionWorkers" :key="w.name">
              <div class="flex items-center justify-between bg-slate-900/60 border border-slate-800 rounded px-3 py-2">
                <div>
                  <div class="font-medium" x-text="w.name"></div>
                  <div class="text-xs text-slate-400" x-text="workerDescriptions[w.name] || ''"></div>
                </div>
                <label class="inline-flex items-center cursor-pointer gap-2">
                  <span class="text-xs" x-text="w.enabled ? 'On' : 'Off'"></span>
                  <input type="checkbox" :checked="w.enabled" :aria-label="'Toggle worker ' + w.name" @change="toggleVisionWorker(w.name, $event.target.checked)"
                    class="w-10 h-5 appearance-none bg-slate-700 rounded-full relative transition
                      checked:bg-indigo-500 before:content-[''] before:absolute before:top-0.5 before:left-0.5
                      before:w-4 before:h-4 before:bg-white before:rounded-full before:transition
                      checked:before:translate-x-5"/>
                </label>
              </div>
            </template>
          </div>
          <div class="text-xs text-slate-500 mt-3" x-text="'Active: ' + ((status.pipeline?.vision_workers || []).join(', ') || '(none)')"></div>
        </div>
        <div class="card rounded-xl p-5">
          <h3 class="font-semibold mb-3">Throttle (<span x-text="throttle + '%'"></span>)</h3>
          <input type="range" min="0" max="100" step="5" x-model="throttle" @change="setThrottle()" class="w-full accent-indigo-500"/>
        </div>
      </div>

      <!-- Auto-captioning + prompt-required toggles. Writes AUTO_CAPTION_*
           and REQUIRE_PROMPT to .env via /api/settings; the supervisor
           soft-restarts the vision worker pool the next reconcile tick. -->
      <div class="card rounded-xl p-5">
        <h3 class="font-semibold mb-3">Auto-captioning</h3>
        <p class="text-xs text-slate-400 mb-3">
          When enabled, the vision worker writes a training-ready caption to
          <code class="font-brand bg-slate-800 px-1.5 py-0.5 rounded">&lt;image&gt;.txt</code>
          alongside the existing <code class="font-brand bg-slate-800 px-1.5 py-0.5 rounded">.vision.json</code>.
          The same LLM call that classifies the image also returns the caption,
          so there's no extra request. Captions never overwrite an existing
          source-side prompt unless the overwrite toggle is also on.
        </p>
        <div class="grid md:grid-cols-2 gap-4">
          <div>
            <label class="flex items-center gap-3 cursor-pointer">
              <input type="checkbox" :checked="settings.AUTO_CAPTION_ENABLED === 'true'"
                     @change="settings.AUTO_CAPTION_ENABLED = $event.target.checked ? 'true' : 'false'; saveSettings()"
                     class="w-10 h-5 appearance-none bg-slate-700 rounded-full relative transition
                       checked:bg-indigo-500 before:content-[''] before:absolute before:top-0.5 before:left-0.5
                       before:w-4 before:h-4 before:bg-white before:rounded-full before:transition
                       checked:before:translate-x-5"/>
              <span class="text-sm">Enable auto-captioning</span>
            </label>
            <label class="flex items-center gap-3 cursor-pointer mt-3">
              <input type="checkbox" :checked="settings.AUTO_CAPTION_OVERWRITE === 'true'"
                     :disabled="settings.AUTO_CAPTION_ENABLED !== 'true'"
                     @change="settings.AUTO_CAPTION_OVERWRITE = $event.target.checked ? 'true' : 'false'; saveSettings()"
                     class="w-10 h-5 appearance-none bg-slate-700 rounded-full relative transition
                       checked:bg-indigo-500 disabled:opacity-40 before:content-[''] before:absolute before:top-0.5 before:left-0.5
                       before:w-4 before:h-4 before:bg-white before:rounded-full before:transition
                       checked:before:translate-x-5"/>
              <span class="text-sm">Overwrite any existing captions found</span>
            </label>
            <label class="flex items-center gap-3 cursor-pointer mt-3">
              <input type="checkbox" :checked="settings.REQUIRE_PROMPT === 'false'"
                     @change="settings.REQUIRE_PROMPT = $event.target.checked ? 'false' : 'true'; saveSettings()"
                     class="w-10 h-5 appearance-none bg-slate-700 rounded-full relative transition
                       checked:bg-indigo-500 before:content-[''] before:absolute before:top-0.5 before:left-0.5
                       before:w-4 before:h-4 before:bg-white before:rounded-full before:transition
                       checked:before:translate-x-5"/>
              <span class="text-sm">Allow scrapers to ingest images without a prompt
                <span class="block text-xs text-slate-500">(off = current behaviour: require MIN_PROMPT_LENGTH chars)</span>
              </span>
            </label>
          </div>
          <div>
            <label class="text-xs text-slate-400 block mb-1">Caption style</label>
            <select x-model="settings.AUTO_CAPTION_STYLE" @change="saveSettings()"
                    :disabled="settings.AUTO_CAPTION_ENABLED !== 'true'"
                    class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 disabled:opacity-50">
              <option value="sd_prompt">SD / Flux prompt (comma-separated descriptive phrases)</option>
              <option value="booru_tags">Booru tags (lowercase_underscored, comma-separated)</option>
              <option value="natural_language">Natural-language description (1-3 sentences)</option>
            </select>
            <p class="text-xs text-slate-500 mt-2">
              Pick the style your downstream trainer expects. SD/Flux for general
              photo LoRAs, booru tags for anime LoRAs, natural-language for
              CLIP/BLIP-style image-text models.
            </p>
          </div>
        </div>
      </div>

      <div class="card rounded-xl p-5">
        <h3 class="font-semibold mb-3">LMStudio endpoints</h3>
        <div class="grid md:grid-cols-2 gap-6">
          <template x-for="inst in ['primary', 'secondary']" :key="inst">
            <div>
              <div class="pill mb-2 text-indigo-300" x-text="inst + ' - ' + (models.instances?.[inst]?.status || 'unknown')"></div>
              <label class="block text-xs text-slate-400 mb-1">URL</label>
              <input :value="models.current?.[inst]?.url" @change="setEndpoint(inst, $event.target.value)"
                class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mb-2"/>
              <label class="block text-xs text-slate-400 mb-1">Model</label>
              <select @change="setModel(inst, $event.target.value)" class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2">
                <option :value="models.current?.[inst]?.model || ''" x-text="models.current?.[inst]?.model || '-'"></option>
                <template x-for="m in (models.instances?.[inst]?.models ?? [])" :key="m.id">
                  <option :value="m.id" x-text="m.name"></option>
                </template>
              </select>
            </div>
          </template>
        </div>
      </div>

      <div class="card rounded-xl p-5">
        <h3 class="font-semibold mb-3">Recent classifications</h3>
        <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
          <template x-for="a in activity" :key="a.path">
            <div class="bg-slate-900/60 border border-slate-800 rounded-lg p-2 flex gap-2 items-start">
              <span class="nsfw-wrap">
                <img :src="a.thumbnail" :alt="a.name + ' - ' + a.category" class="thumb-lg" :class="{ 'nsfw-blur': shouldBlurNsfw(a) }"
                     loading="lazy"
                     @click="shouldBlurNsfw(a) ? revealNsfw(a) : openModalFromActivity(a)"/>
                <span class="nsfw-eye" role="button" tabindex="0" aria-label="Reveal NSFW image" x-show="shouldBlurNsfw(a)" @click.stop="revealNsfw(a)" title="Reveal NSFW">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/>
                  </svg>
                </span>
              </span>
              <div class="min-w-0 text-xs">
                <div class="pill px-1.5 py-0.5 rounded bg-slate-800" x-text="a.category"></div>
                <div class="mt-1 font-mono truncate" x-text="a.name"></div>
                <span class="link-btn mt-1" @click="openModalFromActivity(a)">See more</span>
              </div>
            </div>
          </template>
        </div>
      </div>
    </section>

    <!-- QUEUE -->
    <section x-show="active === 'queue'" class="space-y-4">
      <!-- First-run welcome card: shown only when nothing has been queued OR
           classified yet. Keeps the empty state from looking like a bug. -->
      <template x-if="(status.queue?.total ?? 0) === 0 && (status.sorted?.total ?? 0) === 0">
        <div class="card rounded-xl p-6">
          <div class="flex items-start gap-4">
            <img src="/brand/logo-transparent-dark.png" alt="" width="64" height="64"
                 class="shrink-0"/>
            <div class="flex-1">
              <h3 class="font-brand text-2xl font-medium tracking-tight">Welcome to cull</h3>
              <p class="text-sm text-slate-300 mt-1 mb-4">
                cull is a curation engine for AI image datasets. Configure a scraper and a vision worker to begin,
                or seed a synthetic dataset to see what the dashboard looks like with data in it.
              </p>
              <div class="flex flex-wrap gap-2">
                <button @click="active = 'scrapers'" class="px-3 py-2 bg-indigo-600 hover:bg-indigo-500 rounded text-sm">Configure scrapers</button>
                <button @click="active = 'vision'" class="px-3 py-2 bg-indigo-600 hover:bg-indigo-500 rounded text-sm">Configure vision worker</button>
                <button @click="active = 'settings'" class="px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded text-sm">Open settings</button>
                <button @click="active = 'about'" class="px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded text-sm">About cull</button>
              </div>
              <p class="text-xs text-slate-500 mt-4">
                Want a demo first? Run <code class="font-brand bg-slate-800 px-1.5 py-0.5 rounded">python tools/seed_demo_data.py</code> from the repo root.
              </p>
            </div>
          </div>
        </div>
      </template>

      <div class="card rounded-xl p-5">
      <h3 class="font-semibold mb-3">Queue (newest 60)</h3>
      <div class="scroll-box"><table>
        <thead><tr><th></th><th>Name</th><th>Source</th><th>Size</th><th>Prompt</th><th></th></tr></thead>
        <tbody>
          <template x-for="f in queueFiles" :key="f.path">
            <tr :class="f.corrupt ? 'bg-rose-900/25' : ''">
              <td><img :src="f.thumbnail" :alt="f.name" class="thumb" loading="lazy" @click="openModalFromFile(f)"/></td>
              <td class="font-mono text-xs" x-text="f.name"></td>
              <td x-text="f.source"></td>
              <td class="font-mono text-xs" x-text="(f.size/1024).toFixed(1) + ' KB'"></td>
              <td class="max-w-md">
                <div class="prompt-snippet text-xs text-slate-300" x-text="f.prompt || '-'"></div>
                <template x-if="(f.prompt || '').length > 140">
                  <span class="link-btn" @click="openModalFromFile(f)">See more</span>
                </template>
              </td>
              <td class="text-right space-x-1 whitespace-nowrap">
                <button @click="queueAction(f.path, 'delete')" class="px-2 py-1 bg-rose-600/70 hover:bg-rose-500 rounded text-xs">Delete</button>
              </td>
            </tr>
          </template>
          <template x-if="queueFiles.length === 0">
            <tr><td colspan="6" class="text-center text-slate-500 py-6">Queue is empty.</td></tr>
          </template>
        </tbody>
      </table></div>
      </div>
    </section>

    <!-- LOGS -->
    <section x-show="active === 'logs'" class="card rounded-xl p-5">
      <h3 class="font-semibold mb-3">Historical sorter log</h3>
      <div class="scroll-box"><table>
        <thead><tr><th></th><th>Time</th><th>Image</th><th>Source</th><th>Classification</th></tr></thead>
        <tbody>
          <template x-for="h in history" :key="(h.image||'') + (h.timestamp||'')">
            <tr>
              <td>
                <span class="nsfw-wrap">
                  <img :src="h.thumbnail || ''" :alt="h.image" class="thumb" :class="{ 'nsfw-blur': shouldBlurNsfw(h) }"
                       loading="lazy"
                       onerror="this.style.visibility='hidden'"
                       @click="h.thumbnail && (shouldBlurNsfw(h) ? revealNsfw(h) : openModalFromHistory(h))"/>
                  <span class="nsfw-eye" role="button" tabindex="0" aria-label="Reveal NSFW image" x-show="shouldBlurNsfw(h)" @click.stop="revealNsfw(h)" title="Reveal NSFW">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                      <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/>
                    </svg>
                  </span>
                </span>
              </td>
              <td class="text-xs text-slate-400 font-mono" x-text="h.timestamp"></td>
              <td class="font-mono text-xs" x-text="h.image"></td>
              <td x-text="h.source"></td>
              <td>
                <span class="pill bg-slate-800 rounded px-2 py-0.5" x-text="h.classification"></span>
                <template x-if="h.thumbnail">
                  <span class="link-btn ml-2" @click="openModalFromHistory(h)">See more</span>
                </template>
              </td>
            </tr>
          </template>
          <template x-if="history.length === 0">
            <tr><td colspan="5" class="text-center text-slate-500 py-6">No historical entries yet.</td></tr>
          </template>
        </tbody>
      </table></div>
    </section>

    <!-- SETTINGS -->
    <section x-show="active === 'settings'" class="space-y-4"
      @input="markSettingsDirty()" @change="markSettingsDirty()">
      <div class="card rounded-xl p-5">
        <div class="flex items-start justify-between gap-4 mb-3">
          <div>
            <h3 class="font-semibold">Topic &amp; scraper filters</h3>
            <p class="text-xs text-slate-400">Changes write to <code>.env</code>. Stop + Start the pipeline to apply.</p>
          </div>
          <div class="flex gap-2">
            <button @click="reloadSettings()" class="px-3 py-1.5 text-xs bg-slate-800 hover:bg-slate-700 rounded">Reload</button>
            <button @click="saveSettings()" class="px-3 py-1.5 text-xs bg-indigo-600 hover:bg-indigo-500 rounded font-medium">Save</button>
          </div>
        </div>
        <template x-if="settingsBanner">
          <div class="border text-xs px-3 py-2 rounded mb-3"
            :class="settingsBannerOk ? 'bg-indigo-950/60 border-indigo-700 text-indigo-200'
                                     : 'bg-rose-950/60 border-rose-700 text-rose-200'"
            x-text="settingsBanner"></div>
        </template>
        <div class="grid md:grid-cols-2 gap-4">
          <label class="block">
            <span class="text-xs text-slate-400">Topic</span>
            <input x-model="settings.PIPELINE_TOPIC" class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1"/>
            <span class="text-[10px] text-slate-500">Slug auto-derives from topic unless you override below.</span>
          </label>
          <label class="block">
            <span class="text-xs text-slate-400">Slug (folder name)</span>
            <input x-model="settings.PIPELINE_SLUG" class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1"/>
          </label>
          <label class="block md:col-span-2">
            <span class="text-xs text-slate-400">Required keywords (comma-sep — post must contain at least one)</span>
            <input x-model="settings.TOPIC_KEYWORDS_EXTRA" placeholder="leave empty = auto-derive from topic"
                   class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1"/>
          </label>
          <label class="block md:col-span-2">
            <span class="text-xs text-slate-400">Banned keywords (comma-sep — any match rejects)</span>
            <input x-model="settings.TOPIC_BANNED_KEYWORDS" placeholder="link in bio, dm me, patreon, onlyfans..."
                   class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1"/>
          </label>
          <label class="block md:col-span-2">
            <span class="text-xs text-slate-400">Generation hints (prompt must contain at least one)</span>
            <input x-model="settings.TOPIC_GENERATION_HINTS" placeholder="photorealistic, cinematic, cfg, lora..."
                   class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1"/>
          </label>
          <label class="block md:col-span-2">
            <span class="text-xs text-slate-400">Reddit subreddit allowlist (comma-sep)</span>
            <input x-model="settings.REDDIT_SUBREDDITS"
                   class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1"/>
          </label>
          <label class="block">
            <span class="text-xs text-slate-400">Minimum prompt length (chars)</span>
            <input x-model="settings.MIN_PROMPT_LENGTH" type="number" min="0"
                   class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1" :class="settingsErrors.MIN_PROMPT_LENGTH ? 'border-rose-600' : ''"/>
          </label>
          <label class="block md:col-span-2">
            <span class="text-xs text-slate-400">X.com accounts (comma-sep, no @). Empty = search-only.</span>
            <input x-model="settings.X_ACCOUNTS" placeholder="account1,account2,account3"
                   class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1"/>
          </label>
        </div>
      </div>

      <!-- Categories: user-editable taxonomy. Lives in data/cull_categories.json,
           not .env, so the supervisor watches its mtime separately. -->
      <div class="card rounded-xl p-5">
        <div class="flex items-start justify-between gap-4 mb-3">
          <div>
            <h3 class="font-semibold">Categories</h3>
            <p class="text-xs text-slate-400">
              The classification taxonomy: keep-buckets, per-bucket prompt hints, and the global judgement rules
              that get injected into every vision call. Saved to <code>data/cull_categories.json</code>;
              workers soft-restart automatically on save. <code>DISCARD</code> and <code>CORRUPT</code> are reserved.
            </p>
          </div>
          <div class="flex gap-2 shrink-0">
            <button @click="loadCategories()" class="px-3 py-1.5 text-xs bg-slate-800 hover:bg-slate-700 rounded">Reload</button>
            <button @click="saveCategories()" :disabled="cats.saving"
                    class="px-3 py-1.5 text-xs bg-indigo-600 hover:bg-indigo-500 rounded font-medium disabled:opacity-50">
              <span x-text="cats.saving ? 'Saving...' : 'Save categories'"></span>
            </button>
          </div>
        </div>
        <div x-show="cats.banner" x-cloak
             class="border text-xs px-3 py-2 rounded mb-3"
             :class="cats.bannerOk ? 'bg-indigo-950/60 border-indigo-700 text-indigo-200'
                                   : 'bg-rose-950/60 border-rose-700 text-rose-200'"
             x-text="cats.banner"></div>

        <div class="flex flex-wrap items-center gap-2 mb-4">
          <label class="text-xs text-slate-400">Load preset:</label>
          <select x-model="cats.presetSelect"
                  class="bg-slate-800 border border-slate-700 rounded px-2 py-1 text-xs">
            <template x-for="(preset, key) in cats.presets" :key="key">
              <option :value="key" x-text="preset.label || key"></option>
            </template>
          </select>
          <button @click="applyPreset()"
                  class="px-2 py-1 text-xs bg-slate-700 hover:bg-slate-600 rounded">
            Apply preset (replaces current edits)
          </button>
          <span class="text-[11px] text-slate-500 ml-auto" x-text="'Active preset: ' + (cats.draft.preset || 'custom')"></span>
        </div>

        <div class="space-y-2 mb-4">
          <template x-for="(cat, idx) in cats.draft.categories" :key="idx">
            <div class="flex items-start gap-2 bg-slate-900/40 border border-slate-800 rounded p-2">
              <div class="flex flex-col gap-1 shrink-0">
                <button @click="moveCategory(idx, -1)" :disabled="idx === 0"
                        class="px-1.5 py-0.5 text-[10px] bg-slate-800 hover:bg-slate-700 rounded disabled:opacity-30"
                        title="Move up">↑</button>
                <button @click="moveCategory(idx, 1)" :disabled="idx === cats.draft.categories.length - 1"
                        class="px-1.5 py-0.5 text-[10px] bg-slate-800 hover:bg-slate-700 rounded disabled:opacity-30"
                        title="Move down">↓</button>
              </div>
              <div class="flex-1 grid md:grid-cols-3 gap-2">
                <label class="block">
                  <span class="text-[10px] text-slate-500 uppercase tracking-wider">Name</span>
                  <input x-model="cat.name" placeholder="CategoryName"
                         class="w-full bg-slate-800 border border-slate-700 rounded px-2 py-1 mt-0.5 font-mono text-xs"/>
                </label>
                <label class="block md:col-span-2">
                  <span class="text-[10px] text-slate-500 uppercase tracking-wider">Hint (injected into the prompt)</span>
                  <textarea x-model="cat.hint" rows="2"
                            placeholder="when should the model pick this category?"
                            class="w-full bg-slate-800 border border-slate-700 rounded px-2 py-1 mt-0.5 text-xs leading-snug"></textarea>
                </label>
              </div>
              <button @click="removeCategory(idx)"
                      class="px-2 py-1 text-xs bg-rose-900/60 hover:bg-rose-800 text-rose-100 rounded shrink-0"
                      title="Remove this category">✕</button>
            </div>
          </template>
          <button @click="addCategory()"
                  :disabled="cats.draft.categories.length >= 12"
                  class="text-xs px-3 py-1.5 bg-slate-800 hover:bg-slate-700 rounded disabled:opacity-50">
            + Add category <span class="text-slate-500" x-text="'(' + cats.draft.categories.length + '/12)'"></span>
          </button>
        </div>

        <label class="block">
          <span class="text-xs text-slate-400">Global judgement rules (prepended to CATEGORY ASSIGNMENT in the prompt)</span>
          <textarea x-model="cats.draft.global_rules" rows="6"
                    class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1 font-mono text-[11px] leading-snug"></textarea>
          <span class="text-[10px] text-slate-500">Free-text. Used verbatim by every vision worker. Keep portrait-specific gates here when using the default taxonomy; relax them when sorting by art-style or quality only.</span>
        </label>
      </div>

      <div class="card rounded-xl p-5">
        <h3 class="font-semibold mb-3">Discord access</h3>
        <p class="text-xs text-slate-400 mb-3">
          The Discord scraper supports two auth styles:
          <code>bot</code> sends <code>Authorization: Bot &lt;token&gt;</code> (use a token from
          the Discord Developer Portal, the bot must be a member of each server).
          <code>user</code> sends the token raw and lets your <em>personal</em> Discord account
          read every server you're a member of - including private ones where you can't add a bot.
          Grab your account token from devtools: discord.com → F12 → Network tab → any
          <code>/api/v9/&hellip;</code> request → Headers → copy the <code>authorization</code> value.
          <span class="text-amber-300">User mode is against Discord's TOS; your account, your risk.</span>
        </p>
        <div class="grid md:grid-cols-2 gap-4">
          <label class="block md:col-span-2">
            <span class="text-xs text-slate-400">Discord token (bot or user account)</span>
            <input x-model="settings.DISCORD_BOT_TOKEN" type="password"
                   placeholder="paste token, no quotes"
                   class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1 font-mono text-xs"/>
          </label>
          <label class="block">
            <span class="text-xs text-slate-400">Auth mode</span>
            <select x-model="settings.DISCORD_AUTH_MODE"
                    class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1">
              <option value="auto">auto (try bot → fall back to user on 401)</option>
              <option value="bot">bot (Developer-Portal token)</option>
              <option value="user">user (personal account token)</option>
            </select>
          </label>
        </div>
      </div>

      <div class="card rounded-xl p-5">
        <h3 class="font-semibold mb-3">Vision provider credentials</h3>
        <p class="text-xs text-slate-400 mb-3">
          You only need keys for the providers you select on the <strong>Vision</strong> tab.
          Click <em>Test</em> after saving to confirm the credential works without starting the pipeline.
        </p>
        <div class="grid md:grid-cols-2 gap-4">
          <label class="block">
            <span class="text-xs text-slate-400">GROQ_API_KEY (single key)</span>
            <input x-model="settings.GROQ_API_KEY" type="password" placeholder="gsk_..."
              class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1 font-mono text-xs"
              :class="settingsErrors.GROQ_API_KEY ? 'border-rose-600' : ''"/>
            <span x-show="settingsErrors.GROQ_API_KEY" class="text-xs text-rose-300 mt-1 block" x-text="settingsErrors.GROQ_API_KEY"></span>
          </label>
          <label class="block">
            <span class="text-xs text-slate-400">GROQ_API_KEYS (comma-sep, rotated round-robin)</span>
            <input x-model="settings.GROQ_API_KEYS" type="password" placeholder="gsk_one,gsk_two,gsk_three"
              class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1 font-mono text-xs"/>
          </label>
          <label class="block">
            <span class="text-xs text-slate-400">GROQ_MODEL</span>
            <input x-model="settings.GROQ_MODEL" placeholder="meta-llama/llama-4-scout-17b-16e-instruct"
              class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1 font-mono text-xs"/>
          </label>
          <div class="flex items-end">
            <button @click="testProvider('groq')" :disabled="providerTest.groq?.testing"
              class="px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded text-sm">
              <span x-text="providerTest.groq?.testing ? 'Testing...' : 'Test Groq'"></span>
            </button>
            <span class="ml-3 text-xs" x-show="providerTest.groq?.message"
              :class="providerTest.groq?.ok ? 'text-emerald-300' : 'text-rose-300'"
              x-text="providerTest.groq?.message"></span>
          </div>

          <div class="md:col-span-2 flex items-center gap-3">
            <button @click="testProvider('lmstudio')" :disabled="providerTest.lmstudio?.testing"
              class="px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded text-sm">
              <span x-text="providerTest.lmstudio?.testing ? 'Testing...' : 'Test LM Studio'"></span>
            </button>
            <span class="text-xs" x-show="providerTest.lmstudio?.message"
              :class="providerTest.lmstudio?.ok ? 'text-emerald-300' : 'text-rose-300'"
              x-text="providerTest.lmstudio?.message"></span>
            <button @click="unloadLmStudio()" :disabled="lmstudioUnload.busy"
              class="px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded text-sm"
              title="Free VRAM by unloading every model in LM Studio. Safe to call any time — JIT load fires on the next image.">
              <span x-text="lmstudioUnload.busy ? 'Unloading...' : 'Unload LM Studio'"></span>
            </button>
            <span class="text-xs" x-show="lmstudioUnload.message"
              :class="lmstudioUnload.ok ? 'text-emerald-300' : 'text-rose-300'"
              x-text="lmstudioUnload.message"></span>
          </div>

          <label class="block md:col-span-2">
            <span class="text-xs text-slate-400">LMSTUDIO_UNLOAD_ON_STOP — unload models when the pipeline stops (true/false)</span>
            <input x-model="settings.LMSTUDIO_UNLOAD_ON_STOP" placeholder="true"
              class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1 font-mono text-xs"/>
          </label>
          <label class="block md:col-span-2">
            <span class="text-xs text-slate-400">LMSTUDIO_IDLE_UNLOAD_MINUTES — auto-unload after this many idle minutes (0 = off, ignored when lm-keepalive is active)</span>
            <input x-model="settings.LMSTUDIO_IDLE_UNLOAD_MINUTES" type="number" min="0" placeholder="10"
              class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1 font-mono text-xs"/>
          </label>
        </div>
      </div>

      <div class="card rounded-xl p-5">
        <h3 class="font-semibold mb-3">Scraper credentials</h3>
        <p class="text-xs text-slate-400 mb-3">Required only for the scrapers you've enabled on the <strong>Scrapers</strong> tab.</p>
        <div class="grid md:grid-cols-2 gap-4">
          <label class="block">
            <span class="text-xs text-slate-400">CIVITAI_API_KEY (civitai.com)</span>
            <input x-model="settings.CIVITAI_API_KEY" type="password"
              class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1 font-mono text-xs"/>
          </label>
          <label class="block">
            <span class="text-xs text-slate-400">CIVITAI_API_RED_KEY (civitai.red)</span>
            <input x-model="settings.CIVITAI_API_RED_KEY" type="password"
              class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1 font-mono text-xs"/>
          </label>
          <label class="block md:col-span-2">
            <span class="text-xs text-slate-400">CIVITAI_DOMAINS (comma-sep)</span>
            <input x-model="settings.CIVITAI_DOMAINS" placeholder="civitai.com,civitai.red"
              class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1"/>
          </label>
          <label class="block md:col-span-2">
            <span class="text-xs text-slate-400">TWITTER_COOKIES (full cookie string from a logged-in browser)</span>
            <textarea x-model="settings.TWITTER_COOKIES" rows="2"
              placeholder="auth_token=...; ct0=...; twid=..."
              class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1 font-mono text-xs"></textarea>
          </label>
          <label class="block md:col-span-2">
            <span class="text-xs text-slate-400">DISCORD_CHANNELS_JSON</span>
            <textarea x-model="settings.DISCORD_CHANNELS_JSON" rows="3"
              placeholder='{"channels":[{"id":"...","name":"...","guild":"...","kind":"png_embed"}]}'
              class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1 font-mono text-xs"></textarea>
          </label>
          <label class="block">
            <span class="text-xs text-slate-400">REDDIT_CLIENT_ID (optional)</span>
            <input x-model="settings.REDDIT_CLIENT_ID"
              class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1 font-mono text-xs"/>
          </label>
          <label class="block">
            <span class="text-xs text-slate-400">REDDIT_CLIENT_SECRET (optional)</span>
            <input x-model="settings.REDDIT_CLIENT_SECRET" type="password"
              class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1 font-mono text-xs"/>
          </label>
          <label class="block md:col-span-2">
            <span class="text-xs text-slate-400">REDDIT_USER_AGENT</span>
            <input x-model="settings.REDDIT_USER_AGENT" placeholder="cull/0.1"
              class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1 font-mono text-xs"/>
          </label>
        </div>
      </div>

      <div class="card rounded-xl p-5">
        <h3 class="font-semibold mb-3">Vision quality scoring</h3>
        <p class="text-xs text-slate-400 mb-3">
          Every classification now emits two 0-100 scores. <code>OVR_Quality_Score</code> judges absolute
          craft (composition, lighting, colour, emotion, …). <code>REL_Quality_Score</code> judges how
          closely the image matches the configured topic at its platonic best. Reserve 90+ for the rarest
          images. Either threshold forces a DISCARD when not met; set 0 to disable.
        </p>
        <div class="grid md:grid-cols-2 gap-4">
          <label class="block">
            <span class="text-xs text-slate-400">Minimum OVR score (0-100)</span>
            <input x-model="settings.VISION_OVR_MIN_SCORE" type="number" min="0" max="100"
                   class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1" :class="settingsErrors.VISION_OVR_MIN_SCORE ? 'border-rose-600' : ''"/>
          </label>
          <label class="block">
            <span class="text-xs text-slate-400">Minimum REL score (0-100)</span>
            <input x-model="settings.VISION_REL_MIN_SCORE" type="number" min="0" max="100"
                   class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1" :class="settingsErrors.VISION_REL_MIN_SCORE ? 'border-rose-600' : ''"/>
          </label>
          <label class="block md:col-span-2">
            <span class="text-xs text-slate-400">Scoring notes (appended to rubric - nudge the model toward your taste)</span>
            <textarea x-model="settings.VISION_SCORE_NOTES" rows="3"
                      placeholder="e.g. prefer golden-hour natural light, penalise heavy over-smoothing"
                      class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1 font-mono text-xs"></textarea>
          </label>
          <label class="block">
            <span class="text-xs text-slate-400">Supervisor reconcile interval (seconds)</span>
            <input x-model="settings.PIPELINE_RECONCILE_SECONDS" type="number" min="1" max="3600"
                   class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1" :class="settingsErrors.PIPELINE_RECONCILE_SECONDS ? 'border-rose-600' : ''"/>
            <span class="text-[10px] text-slate-500">How quickly toggles take effect without a restart. Lower = snappier.</span>
          </label>
          <label class="flex items-start gap-2 md:col-span-2">
            <input type="checkbox" :checked="settings.BLUR_NSFW_THUMBS === 'true'"
                   @change="settings.BLUR_NSFW_THUMBS = $event.target.checked ? 'true' : 'false'"
                   class="mt-1"/>
            <div>
              <div class="text-xs text-slate-300">Blur NSFW thumbnails by default</div>
              <div class="text-[10px] text-slate-500">A small eye icon on each NSFW image lets you reveal it temporarily; the setting persists per browser session for revealed items only.</div>
            </div>
          </label>
        </div>
      </div>

      <div class="card rounded-xl p-5">
        <h3 class="font-semibold mb-3">Paths</h3>
        <p class="text-xs text-slate-400 mb-3">Absolute paths only. Pipeline must be stopped before changing these.</p>
        <div class="grid md:grid-cols-2 gap-4">
          <label class="block md:col-span-2">
            <span class="text-xs text-slate-400">Pipeline base dir (parent of queue/, sorted/)</span>
            <input x-model="settings.PIPELINE_BASE_DIR"
                   class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1 font-mono text-xs"/>
          </label>
          <label class="block">
            <span class="text-xs text-slate-400">Queue dir</span>
            <input x-model="settings.PIPELINE_QUEUE"
                   class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1 font-mono text-xs"/>
          </label>
          <label class="block">
            <span class="text-xs text-slate-400">Sorted dir</span>
            <input x-model="settings.PIPELINE_SORTED"
                   class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1 font-mono text-xs"/>
          </label>
          <label class="block">
            <span class="text-xs text-slate-400">Log dir</span>
            <input x-model="settings.LOG_DIR"
                   class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1 font-mono text-xs"/>
          </label>
          <label class="block">
            <span class="text-xs text-slate-400">ZforFree local source</span>
            <input x-model="settings.ZFORFREE_LOCAL_SRC"
                   class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1 font-mono text-xs"/>
          </label>
        </div>
      </div>

      <div class="card rounded-xl p-5">
        <h3 class="font-semibold mb-3">Local folder import</h3>
        <p class="text-xs text-slate-400 mb-3">
          Mirror any folder on disk into the queue. Expected layout: <code>&lt;n&gt;.jpg|png|webp</code>
          paired with <code>&lt;n&gt;.txt</code> (prompt). Items already under
          <code>&lt;sorted&gt;/&lt;slug&gt;/*/&lt;label&gt;/</code> are skipped automatically.
        </p>
        <div class="grid md:grid-cols-2 gap-4">
          <label class="block md:col-span-2">
            <span class="text-xs text-slate-400">Local import folder (absolute path)</span>
            <input x-model="settings.LOCAL_IMPORT_DIR" placeholder="e.g. D:\\my-dataset"
                   class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1 font-mono text-xs"/>
          </label>
          <label class="block">
            <span class="text-xs text-slate-400">Source label (queue subfolder)</span>
            <input x-model="settings.LOCAL_IMPORT_NAME" placeholder="local"
                   class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1"/>
            <span class="text-[10px] text-slate-500">Lowercase, letters/digits/underscore. Used as the queue + sorted subfolder name.</span>
          </label>
          <label class="block">
            <span class="text-xs text-slate-400">Enabled</span>
            <select x-model="settings.LOCAL_IMPORT_ENABLED"
                    class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1">
              <option value="true">true</option>
              <option value="false">false</option>
            </select>
          </label>
          <label class="block md:col-span-2">
            <span class="text-xs text-slate-400">Migrate dedup from (optional, comma-sep)</span>
            <input x-model="settings.LOCAL_IMPORT_MIGRATE_FROM" placeholder="zff_local:zff:zforfree"
                   class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1 font-mono text-xs"/>
            <span class="text-[10px] text-slate-500">Each spec <code>&lt;seen_prefix&gt;:&lt;stem_prefix&gt;:&lt;sorted_src&gt;</code> re-uses a legacy feeder's dedup history.</span>
          </label>
        </div>
      </div>

      <!-- gallery-dl scraper config. URLs is a textarea; everything else is
           an env var the supervisor picks up live. -->
      <div class="card rounded-xl p-5">
        <h3 class="font-semibold mb-3">gallery-dl scraper</h3>
        <p class="text-xs text-slate-400 mb-3">
          gallery-dl handles 340+ sites - <strong>Pixiv, DeviantArt, Danbooru/Gelbooru/e621, ArtStation, Tumblr, Newgrounds, FurAffinity, X, Reddit, Imgur, Flickr</strong>, and many more. Paste one or more URLs (newline or comma separated), enable the toggle, and the supervisor spawns a Gallery-DL agent that downloads, dedupes, and routes each image through the same vision pipeline as the rest. Cookies file is required for Pixiv / X / login-walled sites - export <code class="font-brand bg-slate-800 px-1.5 py-0.5 rounded">cookies.txt</code> from your browser. Captions are mined from each site's metadata (description / caption / selftext / tags) and written as the image's <code class="font-brand bg-slate-800 px-1.5 py-0.5 rounded">.txt</code> automatically.
        </p>
        <div class="grid md:grid-cols-2 gap-4">
          <label class="block">
            <span class="text-xs text-slate-400">Enabled</span>
            <select x-model="settings.GALLERY_DL_ENABLED"
                    class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1">
              <option value="true">true</option>
              <option value="false">false</option>
            </select>
          </label>
          <label class="block">
            <span class="text-xs text-slate-400">Images per URL (limit)</span>
            <input x-model="settings.GALLERY_DL_LIMIT_PER_URL" type="number" min="1" max="5000" placeholder="50"
                   class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1"/>
            <span class="text-[10px] text-slate-500">Capped via gallery-dl <code>image-range</code>; default 50.</span>
          </label>
          <label class="block md:col-span-2">
            <span class="text-xs text-slate-400">URLs (one per line, # comments OK)</span>
            <textarea x-model="settings.GALLERY_DL_URLS" rows="5"
                      placeholder="https://www.pixiv.net/users/123456&#10;https://danbooru.donmai.us/posts?tags=portrait&#10;https://www.deviantart.com/SOMEONE/gallery"
                      class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1 font-mono text-xs"></textarea>
          </label>
          <label class="block md:col-span-2">
            <span class="text-xs text-slate-400">Cookies file (Netscape cookies.txt)</span>
            <input x-model="settings.GALLERY_DL_COOKIES_FILE" placeholder="e.g. C:\\Users\\you\\cookies.txt"
                   class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1 font-mono text-xs"/>
            <span class="text-[10px] text-slate-500">Optional. Required for Pixiv, X, and any private gallery. Use the &quot;Get cookies.txt LOCALLY&quot; browser extension.</span>
          </label>
          <label class="block md:col-span-2">
            <span class="text-xs text-slate-400">Custom gallery-dl config path (advanced)</span>
            <input x-model="settings.GALLERY_DL_CONFIG_PATH" placeholder="e.g. C:\\Users\\you\\gallery-dl\\config.json"
                   class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1 font-mono text-xs"/>
            <span class="text-[10px] text-slate-500">Optional. Loaded on top of cull's defaults so power users can tune per-extractor settings.</span>
          </label>
        </div>
      </div>
    </section>

    <!-- ERRORS -->
    <section x-show="active === 'errors'" class="card rounded-xl p-5">
      <h3 class="font-semibold mb-3 text-rose-300">Recent errors</h3>
      <div class="space-y-2">
        <template x-for="e in status.errors ?? []" :key="e.timestamp + e.message">
          <div class="bg-rose-950/40 border border-rose-800/50 rounded px-3 py-2 text-sm">
            <div class="text-xs text-rose-300 font-mono" x-text="e.file + ' - ' + e.timestamp"></div>
            <div class="font-mono text-xs whitespace-pre-wrap" x-text="e.message"></div>
          </div>
        </template>
        <template x-if="(status.errors ?? []).length === 0">
          <div class="text-sm text-slate-500">No errors logged recently.</div>
        </template>
      </div>
    </section>

    <!-- FAQ ───────────────────────────────────────────────────────────── -->
    <section x-show="active === 'faq'" class="space-y-4">
      <div class="card rounded-xl p-6">
        <h3 class="font-semibold text-lg mb-1">Frequently asked questions</h3>
        <p class="text-xs text-slate-400 mb-5">Pre-empts the GitHub issues. Browse the source if you want the long answer.</p>

        <div class="space-y-5 text-sm leading-relaxed">
          <div>
            <div class="font-semibold text-slate-100 mb-1">Why no Redis?</div>
            <p class="text-slate-300">Because the filesystem is already a queue. <code class="font-brand text-xs bg-slate-800 px-1.5 py-0.5 rounded">image.jpg.processing</code> is the lock; <code class="font-brand text-xs bg-slate-800 px-1.5 py-0.5 rounded">os.rename</code> is atomic on every platform that matters; the supervisor's stale-processing sweep recovers from crashes on restart. cull runs on a Raspberry Pi if you want it to.</p>
          </div>
          <div>
            <div class="font-semibold text-slate-100 mb-1">Why force a JSON schema on every backend?</div>
            <p class="text-slate-300">Because vision models love to reply with <code class="font-brand text-xs bg-slate-800 px-1.5 py-0.5 rounded">&lt;think&gt;...&lt;/think&gt;</code> blocks, markdown fences, or "I'd be happy to help!" prefixes that break regex parsers. The schema constraint moves the problem one layer down — the model literally cannot emit invalid output. Adding a new backend reduces to the API call shape.</p>
          </div>
          <div>
            <div class="font-semibold text-slate-100 mb-1">What is the <span class="brand-keep">Watermarked</span> bucket?</div>
            <p class="text-slate-300">A photo that passes every other gate (photoreal, real human, scores above threshold, not NSFW) but the model flagged a watermark. The shot is salvageable if you remove the overlay; the bucket exists so you don't lose those to <span class="brand-discard">DISCARD</span>.</p>
          </div>
          <div>
            <div class="font-semibold text-slate-100 mb-1">How do I add a new scraper?</div>
            <p class="text-slate-300">Copy <code class="font-brand text-xs bg-slate-800 px-1.5 py-0.5 rounded">pipeline_code/scraper_civitai.py</code>, swap the API specifics, register in <code class="font-brand text-xs bg-slate-800 px-1.5 py-0.5 rounded">run_pipeline.compute_desired_agents</code>, add a row to <code class="font-brand text-xs bg-slate-800 px-1.5 py-0.5 rounded">_STATIC_SCRAPERS</code> in the dashboard so it shows up as a toggle. <code class="font-brand text-xs bg-slate-800 px-1.5 py-0.5 rounded">SeenStore</code> and <code class="font-brand text-xs bg-slate-800 px-1.5 py-0.5 rounded">credentials.get_required</code> handle dedup and key resolution.</p>
          </div>
          <div>
            <div class="font-semibold text-slate-100 mb-1">How do I switch LM Studio endpoints without restarting?</div>
            <p class="text-slate-300">You can't fully — endpoint config is read at worker spawn. But you can hot-swap the loaded <em>model</em> via the Vision tab without touching the supervisor.</p>
          </div>
          <div>
            <div class="font-semibold text-slate-100 mb-1">Where does my data live?</div>
            <p class="text-slate-300"><code class="font-brand text-xs bg-slate-800 px-1.5 py-0.5 rounded">data/</code> next to the repo by default (<code class="font-brand text-xs bg-slate-800 px-1.5 py-0.5 rounded">data/queue/&lt;slug&gt;/&lt;source&gt;/</code>, <code class="font-brand text-xs bg-slate-800 px-1.5 py-0.5 rounded">data/sorted/&lt;slug&gt;/&lt;category&gt;/&lt;source&gt;/</code>). Set <code class="font-brand text-xs bg-slate-800 px-1.5 py-0.5 rounded">PIPELINE_BASE_DIR</code> in <code class="font-brand text-xs bg-slate-800 px-1.5 py-0.5 rounded">.env</code> to put it on a different disk. The path lives in one module, <code class="font-brand text-xs bg-slate-800 px-1.5 py-0.5 rounded">paths.py</code>.</p>
          </div>
          <div>
            <div class="font-semibold text-slate-100 mb-1">Why "cull"?</div>
            <p class="text-slate-300">Because that's the verb. Photographers cull. Editors cull. ML engineers cull. The product automates a workflow that already had a name.</p>
          </div>
        </div>
      </div>
    </section>

    <!-- ABOUT ─────────────────────────────────────────────────────────── -->
    <section x-show="active === 'about'" class="space-y-4">
      <div class="card rounded-xl p-6">
        <div class="flex items-center gap-4 mb-4">
          <img src="/brand/logo-transparent-dark.png" alt="cull" width="96" height="96"
               class="shrink-0"/>
          <div>
            <div class="font-brand text-3xl font-medium tracking-tight">cull</div>
            <p class="text-sm text-slate-300 mt-1">The curation engine for AI image datasets.</p>
          </div>
        </div>

        <div class="grid md:grid-cols-2 gap-4 text-sm">
          <div>
            <div class="text-xs uppercase tracking-wider text-slate-400 mb-2">What it is</div>
            <p class="text-slate-300">A single-machine pipeline that pulls AI-generated images from seven sources, classifies each one under a strict 16-field JSON schema, and drops the keepers into category folders next to the prompt that made them. No Redis. No database. No Docker required.</p>
          </div>
          <div>
            <div class="text-xs uppercase tracking-wider text-slate-400 mb-2">License + repo</div>
            <p class="text-slate-300">MIT. <a href="https://github.com/tlennon-ie/cull" class="text-indigo-300 hover:underline">github.com/tlennon-ie/cull</a></p>
            <div class="text-xs uppercase tracking-wider text-slate-400 mt-3 mb-2">Architecture brief</div>
            <p class="text-slate-300">For AI agents working on the codebase: <a href="https://github.com/tlennon-ie/cull/blob/main/CLAUDE.md" class="text-indigo-300 hover:underline">CLAUDE.md</a> + <a href="https://github.com/tlennon-ie/cull/blob/main/.claude/skills/cull-helper/SKILL.md" class="text-indigo-300 hover:underline">.claude/skills/cull-helper/</a>.</p>
          </div>
        </div>

        <div class="mt-5 pt-5 border-t border-slate-800 grid grid-cols-3 gap-3 text-center">
          <div>
            <div class="text-xs text-slate-400">Total classified</div>
            <div class="text-xl font-mono mt-1" x-text="status.sorted?.total ?? 0"></div>
          </div>
          <div>
            <div class="text-xs text-slate-400">In queue</div>
            <div class="text-xl font-mono mt-1" x-text="status.queue?.total ?? 0"></div>
          </div>
          <div>
            <div class="text-xs text-slate-400">Active vision worker</div>
            <div class="text-xs font-brand mt-2" x-text="status.pipeline?.vision_worker || '-'"></div>
          </div>
        </div>
      </div>

      <div class="card rounded-xl p-6">
        <h3 class="font-semibold mb-3">Brand</h3>
        <p class="text-xs text-slate-400 mb-3">Logo variants live in <code class="font-brand bg-slate-800 px-1.5 py-0.5 rounded">assets/</code> and are served from <code class="font-brand bg-slate-800 px-1.5 py-0.5 rounded">/brand/&lt;filename&gt;</code>: <a href="/brand/logo.png" class="text-indigo-300 hover:underline">logo.png</a> for light surfaces, <a href="/brand/logo-transparent.png" class="text-indigo-300 hover:underline">logo-transparent.png</a> for flexible drops, <a href="/brand/logo-transparent-dark.png" class="text-indigo-300 hover:underline">logo-transparent-dark.png</a> for the dark dashboard. Wordmark is JetBrains Mono, all-lowercase.</p>
        <div class="grid grid-cols-5 gap-2 text-xs">
          <div><div class="h-12 rounded" style="background:#0F1115; border:1px solid #1f2937;"></div><div class="mt-1 text-slate-400">ink #0F1115</div></div>
          <div><div class="h-12 rounded" style="background:#F5F2EC;"></div><div class="mt-1 text-slate-400">surface #F5F2EC</div></div>
          <div><div class="h-12 rounded" style="background:#E8B73A;"></div><div class="mt-1 text-slate-400">keep #E8B73A</div></div>
          <div><div class="h-12 rounded" style="background:#C8553D;"></div><div class="mt-1 text-slate-400">discard #C8553D</div></div>
          <div><div class="h-12 rounded" style="background:#7A8088;"></div><div class="mt-1 text-slate-400">subtle #7A8088</div></div>
        </div>
      </div>
    </section>

    <!-- Brand footer: cheap reinforcement, no chrome. -->
    <footer class="mt-8 pt-6 border-t border-slate-800/60 text-center text-xs text-slate-500">
      <span class="font-brand">cull</span>
      <span class="mx-2">·</span>
      MIT
      <span class="mx-2">·</span>
      <a href="https://github.com/tlennon-ie/cull" class="hover:text-slate-300">github.com/tlennon-ie/cull</a>
    </footer>

  </main>

  <!-- DETAIL MODAL -->
  <div x-show="modal.open" x-cloak role="dialog" aria-modal="true" aria-labelledby="modalName"
       @keydown.escape.window="closeModal()"
       class="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-6"
       @click.self="closeModal()">
    <div class="card rounded-xl p-5 max-w-5xl w-full max-h-[90vh] overflow-hidden flex flex-col">
      <div class="flex items-start justify-between gap-4 mb-3">
        <div class="min-w-0">
          <div class="text-xs text-slate-400" x-text="modal.source + (modal.category ? ' · ' + modal.category : '')"></div>
          <div id="modalName" class="font-mono text-sm truncate" x-text="modal.name"></div>
        </div>
        <button @click="closeModal()" x-ref="modalClose"
          class="shrink-0 px-3 py-1 text-sm bg-slate-800 hover:bg-slate-700 rounded">Close (Esc)</button>
      </div>
      <div class="grid lg:grid-cols-2 gap-4 overflow-hidden">
        <div class="bg-slate-950 rounded flex items-center justify-center min-h-[300px] overflow-auto relative">
          <img :src="modal.imageUrl" :alt="modal.name"
               class="max-w-full max-h-[75vh] object-contain"
               :class="{ 'nsfw-blur': shouldBlurNsfw({ category: modal.category, path: modal.imageUrl }) }"/>
          <span class="nsfw-eye" style="background:rgba(15,23,42,0.55);"
                x-show="shouldBlurNsfw({ category: modal.category, path: modal.imageUrl })"
                @click="revealNsfw({ path: modal.imageUrl })" title="Reveal NSFW">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/>
            </svg>
          </span>
        </div>
        <div class="overflow-y-auto">
          <div class="flex items-center justify-between mb-2">
            <h4 class="text-xs uppercase tracking-wider text-slate-400">Full prompt</h4>
            <div class="flex items-center gap-2 text-xs">
              <span class="text-emerald-300" x-text="modal.savedFlash" x-show="modal.savedFlash"></span>
              <template x-if="!modal.editing && modal.path">
                <button @click="modal.editing = true" class="px-2 py-1 bg-slate-800 hover:bg-slate-700 rounded">Edit</button>
              </template>
              <template x-if="modal.editing">
                <button @click="savePrompt()" :disabled="modal.saving"
                  class="px-2 py-1 bg-amber-600 hover:bg-amber-500 rounded disabled:opacity-50"
                  title="Overwrites the .txt file with no backup">
                  <span x-text="modal.saving ? 'Saving...' : 'Save (overwrite)'"></span>
                </button>
              </template>
              <template x-if="modal.editing">
                <button @click="cancelPromptEdit()" class="px-2 py-1 bg-slate-700 hover:bg-slate-600 rounded">Cancel</button>
              </template>
            </div>
          </div>
          <template x-if="!modal.editing">
            <pre class="whitespace-pre-wrap text-sm font-mono"
              :class="modal.prompt ? 'text-slate-200' : 'text-slate-500 italic'"
              x-text="modal.prompt || '(no prompt saved for this image)'"></pre>
          </template>
          <template x-if="modal.editing">
            <textarea x-model="modal.prompt" rows="14"
              class="w-full bg-slate-950 border border-slate-700 rounded p-3 text-sm font-mono text-slate-200"></textarea>
          </template>
          <template x-if="modal.summary">
            <div class="mt-4">
              <h4 class="text-xs uppercase tracking-wider text-slate-400 mb-1">Vision summary</h4>
              <p class="text-sm" x-text="modal.summary"></p>
            </div>
          </template>
          <template x-if="modal.meta">
            <div class="mt-4 grid grid-cols-3 gap-2 text-xs">
              <div class="bg-slate-900/60 border border-slate-800 rounded p-2"><div class="text-slate-400">OVR</div><div class="font-mono" x-text="modal.meta.ovr ?? '-'"></div></div>
              <div class="bg-slate-900/60 border border-slate-800 rounded p-2"><div class="text-slate-400">REL</div><div class="font-mono" x-text="modal.meta.rel ?? '-'"></div></div>
              <div class="bg-slate-900/60 border border-slate-800 rounded p-2"><div class="text-slate-400">quality</div><div class="font-mono" x-text="modal.meta.quality ?? '-'"></div></div>
              <div class="bg-slate-900/60 border border-slate-800 rounded p-2"><div class="text-slate-400">Resolution</div><div class="font-mono" x-text="(modal.meta.width || '?') + 'x' + (modal.meta.height || '?')"></div></div>
              <div class="bg-slate-900/60 border border-slate-800 rounded p-2"><div class="text-slate-400">NSFW</div><div class="font-mono" x-text="modal.meta.nsfw ? 'true' : 'false'"></div></div>
              <div class="bg-slate-900/60 border border-slate-800 rounded p-2"><div class="text-slate-400">Score reason</div><div class="font-mono truncate" :title="modal.meta.score_reason" x-text="modal.meta.score_reason || '-'"></div></div>
            </div>
          </template>
        </div>
      </div>
    </div>
  </div>

</div>

<script>
function dashboard() {
  return {
    active: 'overview',
    sidebarOpen: false,
    tabs: [
      {id:'overview', label:'Overview'},
      {id:'stats',    label:'Stats'},
      {id:'gallery',  label:'Gallery'},
      {id:'scrapers', label:'Scrapers'},
      {id:'vision',   label:'Vision'},
      {id:'queue',    label:'Queue'},
      {id:'logs',     label:'Historical'},
      {id:'errors',   label:'Errors'},
      {id:'settings', label:'Settings'},
      {id:'faq',      label:'FAQ'},
      {id:'about',    label:'About'},
    ],
    providers: ['balanced-groq','balanced-lm','balanced-lm-secondary','lm-autodetect'],
    provider: 'balanced-groq',
    throttle: 100,
    status: {}, scrapers: [], models: {}, visionWorkers: [],
    settings: {}, settingsBanner: '', settingsBannerOk: true,
    settingsDirty: false, settingsErrors: {},
    providerTest: {},
    cats: {
      draft: { preset: 'portrait_curation', categories: [], global_rules: '' },
      presets: {},
      presetSelect: 'portrait_curation',
      banner: '', bannerOk: true, saving: false, loaded: false,
    },
    lmstudioUnload: { busy: false, ok: null, message: '' },
    update: { available: false, behind: 0, remote_sha: '', remote_subject: '', dismissed_sha: '', running: false, error: '' },
    indexer: { in_progress: false, files_seen: 0, files_added: 0, queue_total: 0, sorted_total: 0, last_scan_at: null, scan_started_at: null },
    indexerToast: { show: false, dismissed: false },
    workerDescriptions: {
      'balanced-groq':          'Groq cloud, llama-4-scout - fast, handles NSFW',
      'balanced-lm':            'LMStudio PRIMARY endpoint',
      'balanced-lm-secondary':  'LMStudio SECONDARY endpoint (parallel with -primary)',
      'lm-autodetect':          'LMStudio, auto-picks the loaded vision model (VL)',
    },
    queueFiles: [], history: [], activity: [], promptCache: {},
    revealedNsfw: {},  // path -> true once the user clicks the eye
    lastRefresh: '...',
    modal: { open:false, imageUrl:'', prompt:'', promptOriginal:'', editing:false,
             saving:false, savedFlash:'', name:'', source:'', category:'', summary:'',
             path:'', meta:null },
    // Stats tab state
    stats: { totals:{}, keywords:[], top_overall:[], top_quality:[], top_relative:[], sources:[] },
    statsLoading: false,
    // Gallery tab state
    gallery: {
      page: 1, pageSize: 60, total: 0, totalUnfiltered: 0, items: [],
      available: { sources:[], categories:[], resolutions:[] },
    },
    galleryLoading: false,
    galleryFilters: {
      q: '', sources: [], categories: [], resolutions: [],
      minOvr: 0, minRel: 0, minQuality: 0,
      nsfw: 'any', sort: 'newest', dateFrom: '', dateTo: '',
    },
    galleryInsights: { ngrams:[], top_quality_keywords:[], ovr_quality_threshold:0, considered:0 },
    modalReturnFocus: null,

    async refresh() {
      // Per-tab gating: only fetch what the active tab needs. Always pull
      // /api/status (it powers the sidebar pill) and /api/scrapers + workers
      // (cheap, used in multiple tabs). Skip the heavy 200-row history poll
      // unless the user is on the Historical or Overview tab.
      const tab = this.active;
      const need = (id) => tab === id || tab === 'overview';
      const j = (url) => fetch(url).then(r => r.ok ? r.json() : Promise.reject(r.status));
      const tasks = {
        status:    j('/api/status'),
        scrapers:  j('/api/scrapers'),
        workers:   j('/api/vision/workers'),
        models:    (need('vision') ? j('/api/lmstudio/models') : null),
        queue:     (need('queue')  ? j('/api/queue/files?limit=60') : null),
        history:   (need('logs')   ? j('/api/logs/history?limit=200') : null),
        activity:  (need('overview') || tab === 'vision' ? j('/api/activity?limit=12') : null),
        // Vision tab also needs settings for the auto-caption toggles.
        settings:  (need('settings') || tab === 'vision' ? j('/api/settings') : null),
      };
      const keys = Object.keys(tasks);
      const results = await Promise.allSettled(Object.values(tasks).map(p => p ?? Promise.resolve(null)));
      const out = {};
      results.forEach((r, i) => { out[keys[i]] = r.status === 'fulfilled' ? r.value : null; });
      if (out.status)    this.status = out.status;
      if (out.scrapers)  this.scrapers = out.scrapers;
      if (out.workers)   this.visionWorkers = out.workers;
      if (out.models)    this.models = out.models;
      if (out.queue)     this.queueFiles = out.queue;
      if (out.history)   this.history = out.history;
      if (out.activity)  this.activity = out.activity;
      // Indexer state rides on /api/status so we don't add a third poll.
      if (out.status && out.status.indexer) {
        this.indexer = out.status.indexer;
        // Show the cold-scan toast once: when in_progress flips on AND no
        // prior successful scan timestamp exists. Stays dismissed for the
        // session so it doesn't pop again on refresh.
        if (this.indexer.in_progress && !this.indexer.last_scan_at && !this.indexerToast.dismissed) {
          this.indexerToast.show = true;
        } else if (!this.indexer.in_progress) {
          this.indexerToast.show = false;
        }
      }
      // Only seed settings on first load OR when the user explicitly hits Reload.
      if (out.settings && !this.settingsDirty && Object.keys(this.settings).length === 0) {
        this.settings = out.settings;
      }
      this.provider = this.status.pipeline?.vision_worker || this.provider;
      this.throttle = this.status.pipeline?.throttle ?? this.throttle;
      this.lastRefresh = new Date().toLocaleTimeString();
    },
    async saveSettings() {
      this.settingsBanner = 'Saving...';
      this.settingsBannerOk = true;
      this.settingsErrors = {};
      try {
        const r = await fetch('/api/settings', {method:'POST', headers:{'Content-Type':'application/json'},
          body: JSON.stringify(this.settings)});
        const j = await r.json();
        if (j.success) {
          this.settingsBanner = 'Saved. Stop + Start the pipeline to pick up changes.';
          this.settingsBannerOk = true;
          this.settingsDirty = false;
          this.settings = {};  // Force reload from server on next refresh
          setTimeout(() => this.settingsBanner = '', 6000);
        } else {
          // Render per-field errors next to inputs; banner just summarises.
          this.settingsErrors = j.errors || {};
          const fields = Object.keys(this.settingsErrors);
          this.settingsBanner = fields.length
            ? `Validation failed: ${fields.join(', ')}`
            : 'Save failed.';
          this.settingsBannerOk = false;
        }
      } catch (e) {
        this.settingsBanner = 'Network error - see console.';
        this.settingsBannerOk = false;
      }
      this.refresh();
    },
    markSettingsDirty() { this.settingsDirty = true; },
    async testProvider(name) {
      this.providerTest[name] = { testing: true, message: 'Connecting...', ok: null };
      try {
        const r = await fetch('/api/vision/test', {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({provider: name}),
        });
        const j = await r.json();
        this.providerTest[name] = {
          testing: false,
          ok: j.ok,
          message: (j.ok ? '✓ ' : '✗ ') + (j.message || 'unknown') + ` (${j.latency_ms}ms)`,
        };
      } catch (e) {
        this.providerTest[name] = { testing: false, ok: false, message: '✗ network error' };
      }
    },
    async loadCategories() {
      try {
        const r = await fetch('/api/categories');
        const j = await r.json();
        this.cats.presets = j.presets || {};
        this.cats.draft = JSON.parse(JSON.stringify(j.active || { preset: 'custom', categories: [], global_rules: '' }));
        this.cats.presetSelect = this.cats.draft.preset in this.cats.presets ? this.cats.draft.preset : (j.default_preset || Object.keys(this.cats.presets)[0] || 'portrait_curation');
        this.cats.loaded = true;
        this.cats.banner = '';
      } catch (e) {
        this.cats.banner = 'Failed to load categories: ' + e.message;
        this.cats.bannerOk = false;
      }
    },
    applyPreset() {
      const preset = this.cats.presets[this.cats.presetSelect];
      if (!preset) return;
      if (this.cats.draft.categories.length && !confirm('Replace current categories + rules with the "' + (preset.label || this.cats.presetSelect) + '" preset?')) return;
      this.cats.draft = {
        preset: this.cats.presetSelect,
        categories: JSON.parse(JSON.stringify(preset.categories || [])),
        global_rules: preset.global_rules || '',
      };
    },
    addCategory() {
      if (this.cats.draft.categories.length >= 12) return;
      this.cats.draft.categories.push({ name: '', hint: '' });
    },
    removeCategory(idx) {
      this.cats.draft.categories.splice(idx, 1);
    },
    moveCategory(idx, delta) {
      const target = idx + delta;
      if (target < 0 || target >= this.cats.draft.categories.length) return;
      const list = this.cats.draft.categories;
      [list[idx], list[target]] = [list[target], list[idx]];
    },
    async saveCategories(force) {
      this.cats.saving = true;
      this.cats.banner = '';
      const url = '/api/categories' + (force ? '?force=1' : '');
      try {
        const r = await fetch(url, {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(this.cats.draft),
        });
        const j = await r.json();
        if (r.ok && j.ok) {
          this.cats.banner = 'Saved. Workers will soft-restart on the next reconcile tick (~1s).';
          this.cats.bannerOk = true;
          this.cats.draft = JSON.parse(JSON.stringify(j.active));
          setTimeout(() => this.cats.banner = '', 6000);
        } else if (r.status === 409 && j.error === 'removing_populated_categories') {
          const list = Object.entries(j.legacy).map(([k, v]) => `${k} (${v})`).join(', ');
          if (confirm(`These categories still have sorted images: ${list}.\n\nRequeue them via tools/requeue_sorted.py first if you want them reclassified.\n\nProceed anyway? Folders stay on disk.`)) {
            await this.saveCategories(true);
            return;
          }
          this.cats.banner = 'Save cancelled — populated categories not removed.';
          this.cats.bannerOk = false;
        } else {
          this.cats.banner = 'Save failed: ' + (j.error || 'unknown error');
          this.cats.bannerOk = false;
        }
      } catch (e) {
        this.cats.banner = 'Network error: ' + e.message;
        this.cats.bannerOk = false;
      }
      this.cats.saving = false;
    },
    async unloadLmStudio() {
      this.lmstudioUnload = { busy: true, ok: null, message: 'Unloading...' };
      try {
        const r = await fetch('/api/lmstudio/unload', { method: 'POST' });
        const j = await r.json();
        this.lmstudioUnload = {
          busy: false, ok: j.ok,
          message: (j.ok ? '✓ ' : '✗ ') + (j.detail || 'unknown'),
        };
      } catch (e) {
        this.lmstudioUnload = { busy: false, ok: false, message: '✗ network error' };
      }
    },
    async checkUpdate() {
      try {
        const r = await fetch('/api/update/check');
        const j = await r.json();
        if (!j.ok) { this.update.error = j.error || ''; return; }
        const dismissed = localStorage.getItem('cull_update_dismissed_sha') || '';
        this.update = {
          available: j.behind > 0 && j.remote_sha !== dismissed,
          behind: j.behind || 0,
          remote_sha: j.remote_sha || '',
          remote_subject: j.remote_subject || '',
          dismissed_sha: dismissed,
          running: false,
          error: '',
        };
      } catch (e) {
        // Silent — updater is best-effort.
      }
    },
    dismissUpdate() {
      if (this.update.remote_sha) {
        localStorage.setItem('cull_update_dismissed_sha', this.update.remote_sha);
      }
      this.update.available = false;
    },
    async runUpdate() {
      if (!confirm('cull will pull the latest version, reinstall dependencies if needed, and restart. Continue?')) return;
      this.update.running = true;
      try {
        const r = await fetch('/api/update/run', { method: 'POST' });
        const j = await r.json();
        if (!j.ok) {
          this.update.error = j.error || 'Update failed.';
          this.update.running = false;
        }
        // On success the dashboard process will exit imminently; the update
        // script restarts it. The user just needs to refresh.
      } catch (e) {
        // Connection drop is expected once the dashboard restarts.
      }
    },
    reloadSettings() {
      if (this.settingsDirty && !window.confirm('Discard your unsaved settings changes?')) return;
      this.settings = {};
      this.settingsDirty = false;
      this.settingsErrors = {};
      this.refresh();
    },
    async toggleVisionWorker(name, enabled) {
      await fetch('/api/vision/workers/toggle', {method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({name, enabled})}); this.refresh();
    },
    async loadPrompt(item) {
      if (!item || !item.path || this.promptCache[item.path]) return;
      try {
        const txt = await fetch(item.prompt_url).then(r=>r.text());
        this.promptCache[item.path] = txt;
      } catch (e) { this.promptCache[item.path] = ''; }
    },

    // Pipeline lifecycle: POST returns the new state; we apply it optimistically
    // so the UI flips immediately, then poll quickly for ~3s in case the backend
    // takes a moment for the subprocess to settle (Windows spawn + supervisor
    // print loop). Eliminates the "I clicked Start but the chip still says
    // stopped" feeling.
    async _settlePipelineState(showAs) {
      if (!this.status.pipeline) this.status.pipeline = {};
      this.status.pipeline.running = showAs;
      // Quick burst: 6 polls at 500ms intervals so the UI reflects the
      // real subprocess state within ~3s without waiting for the 5s tick.
      for (let i = 0; i < 6; i++) {
        await new Promise(r => setTimeout(r, 500));
        await this.refresh();
        if (Boolean(this.status.pipeline?.running) === showAs) break;
      }
    },
    async startPipeline() {
      await fetch('/api/pipeline/start', {method:'POST'});
      this._settlePipelineState(true);
    },
    async stopPipeline() {
      await fetch('/api/pipeline/stop', {method:'POST'});
      this._settlePipelineState(false);
    },

    async toggleScraper(name, enabled) {
      await fetch('/api/scrapers/toggle', {method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({name, enabled})}); this.refresh();
    },
    async scrapersBulk(action) {
      await fetch('/api/scrapers/bulk', {method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({action})}); this.refresh();
    },
    async setProvider() {
      await fetch('/api/vision/provider', {method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({provider: this.provider})});
    },
    async setThrottle() {
      await fetch('/api/vision/throttle', {method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({percent: Number(this.throttle)})});
    },
    async setEndpoint(instance, url) {
      await fetch('/api/lmstudio/endpoint', {method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({instance, url})}); this.refresh();
    },
    async setModel(instance, model_id) {
      if (!model_id) return;
      await fetch('/api/lmstudio/set-model', {method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({instance, model_id})}); this.refresh();
    },
    async queueAction(path, action, target) {
      await fetch('/api/queue/action', {method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({path, action, target})});
      this.refresh();
    },

    async openModal({imageUrl, promptUrl, name, source, category, summary, path, meta}) {
      // Remember the trigger so we can restore focus on close (a11y).
      this.modalReturnFocus = document.activeElement;
      this.modal = {
        open: true, imageUrl,
        prompt: 'Loading...', promptOriginal: '',
        editing: false, saving: false, savedFlash: '',
        name: name || '', source: source || '', category: category || '',
        summary: summary || '', path: path || '', meta: meta || null,
      };
      try {
        const txt = promptUrl ? await fetch(promptUrl).then(r=>r.text()) : '';
        const value = txt || '';
        this.modal.prompt = value || '(empty)';
        this.modal.promptOriginal = value;
      } catch (e) { this.modal.prompt = '(failed to load prompt)'; }
    },
    upsize(url) { if (!url) return url; return url + (url.includes('?') ? '&' : '?') + 'size=1600'; },
    openModalFromFile(f) {
      const encoded = encodeURIComponent(f.path);
      this.openModal({
        imageUrl: '/api/thumbnail?size=1600&path=' + encoded,
        promptUrl: '/api/prompt?path=' + encoded,
        name: f.name, source: f.source, path: f.path,
      });
    },
    openModalFromActivity(a) {
      this.openModal({
        imageUrl: this.upsize(a.thumbnail),
        promptUrl: a.prompt_url,
        name: a.name, source: a.source, category: a.category, summary: a.summary,
        path: a.path,
      });
    },
    openModalFromHistory(h) {
      this.openModal({
        imageUrl: this.upsize(h.thumbnail),
        promptUrl: h.prompt_url,
        name: h.image, source: h.source, category: h.classification,
        path: h.path,
      });
    },
    openModalFromCard(c) {
      // Used by the new Stats/Gallery sections - card has full vision payload.
      this.openModal({
        imageUrl: this.upsize(c.thumbnail),
        promptUrl: c.prompt_url,
        name: c.name, source: c.source, category: c.category, summary: c.summary,
        path: c.path, meta: c,
      });
    },
    closeModal() {
      // Warn on unsaved prompt edits before discarding the modal.
      if (this.modal.editing && this.modal.prompt !== this.modal.promptOriginal) {
        if (!window.confirm('You have unsaved prompt edits. Discard them?')) return;
      }
      this.modal.open = false;
      this.modal.editing = false;
      this.modal.savedFlash = '';
      // Restore focus to whatever opened the modal (a11y).
      if (this.modalReturnFocus && this.modalReturnFocus.focus) {
        try { this.modalReturnFocus.focus(); } catch (_) { /* element gone */ }
        this.modalReturnFocus = null;
      }
    },

    // Edit-and-save the prompt that lives next to an image. Per spec we
    // overwrite without backup, and we invalidate the stats cache server-side
    // so keyword counts reflect the edit on the next refresh.
    async savePrompt() {
      if (!this.modal.path || this.modal.saving) return;
      this.modal.saving = true;
      try {
        const r = await fetch('/api/prompt', {
          method: 'POST', headers: {'Content-Type':'application/json'},
          body: JSON.stringify({path: this.modal.path, text: this.modal.prompt || ''})
        });
        if (!r.ok) {
          // Surface non-200s explicitly - r.json() on a 5xx HTML page throws.
          let detail = '';
          try { detail = (await r.json())?.error || ''; } catch (_) { detail = await r.text(); }
          this.modal.savedFlash = `Error ${r.status}: ${detail || 'save failed'}`;
          return;
        }
        const j = await r.json();
        if (j.success) {
          this.modal.promptOriginal = this.modal.prompt;
          this.modal.editing = false;
          this.modal.savedFlash = 'Saved.';
          setTimeout(() => this.modal.savedFlash = '', 2500);
        } else {
          this.modal.savedFlash = 'Error: ' + (j.error || 'unknown');
        }
      } catch (e) {
        this.modal.savedFlash = 'Network error - check console';
        console.error('savePrompt failed', e);
      } finally {
        this.modal.saving = false;
      }
    },
    cancelPromptEdit() {
      this.modal.prompt = this.modal.promptOriginal;
      this.modal.editing = false;
    },

    // ── Stats tab ────────────────────────────────────────────────────────
    async loadStats() {
      this.statsLoading = true;
      try {
        const r = await fetch('/api/stats').then(r=>r.json());
        this.stats = r;
      } catch (e) { /* fall through silently */ }
      this.statsLoading = false;
    },

    // ── Gallery tab ──────────────────────────────────────────────────────
    galleryQueryString(extra) {
      const f = this.galleryFilters;
      const params = new URLSearchParams();
      if (f.q) params.set('q', f.q);
      if (f.sources?.length) params.set('sources', f.sources.join(','));
      if (f.categories?.length) params.set('categories', f.categories.join(','));
      if (f.resolutions?.length) params.set('resolutions', f.resolutions.join(','));
      if (f.minOvr) params.set('min_ovr', f.minOvr);
      if (f.minRel) params.set('min_rel', f.minRel);
      if (f.minQuality) params.set('min_quality', f.minQuality);
      if (f.nsfw && f.nsfw !== 'any') params.set('nsfw', f.nsfw);
      if (f.sort) params.set('sort', f.sort);
      if (f.dateFrom) params.set('date_from', f.dateFrom);
      if (f.dateTo) params.set('date_to', f.dateTo);
      Object.entries(extra || {}).forEach(([k, v]) => {
        if (v !== null && v !== undefined && v !== '') params.set(k, v);
      });
      return params.toString();
    },
    async loadGallery(page) {
      if (page) this.gallery.page = page;
      this.galleryLoading = true;
      const qs = this.galleryQueryString({page: this.gallery.page, page_size: this.gallery.pageSize});
      try {
        const j = await fetch('/api/gallery?' + qs).then(r=>r.json());
        this.gallery = { ...this.gallery, ...j };
      } catch (e) { /* swallow */ }
      this.galleryLoading = false;
    },
    async loadGalleryInsights() {
      try {
        const qs = this.galleryQueryString();
        this.galleryInsights = await fetch('/api/gallery/insights?' + qs).then(r=>r.json());
      } catch (e) { /* swallow */ }
    },
    galleryReload() {
      this.gallery.page = 1;
      this.loadGallery();
      this.loadGalleryInsights();
    },
    toggleGalleryFilter(field, value) {
      const list = this.galleryFilters[field];
      const idx = list.indexOf(value);
      if (idx === -1) list.push(value); else list.splice(idx, 1);
      this.galleryReload();
    },
    galleryClearFilters() {
      this.galleryFilters = {
        q: '', sources: [], categories: [], resolutions: [],
        minOvr: 0, minRel: 0, minQuality: 0,
        nsfw: 'any', sort: 'newest', dateFrom: '', dateTo: '',
      };
      this.galleryReload();
    },
    galleryDownload() {
      const qs = this.galleryQueryString();
      window.open('/api/gallery/download.zip?' + qs, '_blank');
    },
    galleryPrev() { if (this.gallery.page > 1) this.loadGallery(this.gallery.page - 1); },
    galleryNext() {
      const totalPages = Math.max(1, Math.ceil(this.gallery.total / this.gallery.pageSize));
      if (this.gallery.page < totalPages) this.loadGallery(this.gallery.page + 1);
    },
    insertGalleryQuery(term) {
      this.galleryFilters.q = term;
      this.galleryReload();
    },

    // NSFW blur helpers. shouldBlurNsfw(item) is the source of truth - callers
    // pass {category, path}, we decide based on the BLUR_NSFW_THUMBS setting
    // and whether the user has clicked the eye for this path. revealNsfw()
    // flips the per-path reveal so a single image is unblurred without
    // disabling the global setting.
    shouldBlurNsfw(item) {
      if (!item) return false;
      const cat = String(item.category || item.classification || '').toUpperCase();
      if (cat !== 'NSFW') return false;
      if ((this.settings.BLUR_NSFW_THUMBS || 'true') !== 'true') return false;
      const key = item.path || item.thumbnail || item.image || '';
      return !this.revealedNsfw[key];
    },
    revealNsfw(item) {
      const key = item?.path || item?.thumbnail || item?.image || '';
      if (key) this.revealedNsfw[key] = true;
    },

    start() {
      this.refresh();
      setInterval(() => this.refresh(), 5000);
      // Lazy-load stats / gallery only when their tab is opened. Stats then
      // auto-refreshes alongside the rest of the dashboard. Gallery stays
      // user-driven because filter state shouldn't be clobbered on poll.
      this.$watch('active', (tab) => {
        if (tab === 'stats') this.loadStats();
        if (tab === 'gallery' && this.gallery.items.length === 0 && !this.galleryLoading) {
          this.loadGallery();
          this.loadGalleryInsights();
        }
        if (tab === 'settings' && !this.cats.loaded) this.loadCategories();
      });
      setInterval(() => { if (this.active === 'stats') this.loadStats(); }, 30000);
      // Self-update: check on load, then every 30 minutes. The endpoint is
      // server-side cached for 5 minutes so polls are cheap.
      this.checkUpdate();
      setInterval(() => this.checkUpdate(), 30 * 60 * 1000);
    },
  };
}
</script>
</body>
</html>
"""


@app.route("/")
def dashboard():
    return render_template_string(HTML_TEMPLATE)


# ── Idle LM Studio unloader ───────────────────────────────────────────────────
#
# When the queue stays empty for LMSTUDIO_IDLE_UNLOAD_MINUTES and an LM Studio
# worker is active, unload the model to free VRAM. The next image triggers
# JIT-load on LM Studio's side automatically. Disabled when:
#   * minutes <= 0
#   * pipeline not running
#   * lm-keepalive worker is in PIPELINE_VISION_WORKERS (user explicitly wants
#     the model resident — pinging every 15s would defeat us anyway)

import time as _time

_IDLE_POLL_SECONDS = 60


def _idle_unload_loop() -> None:
    last_idle_at: float | None = None
    already_unloaded = False
    while True:
        try:
            _time.sleep(_IDLE_POLL_SECONDS)
            try:
                idle_minutes = int(os.environ.get("LMSTUDIO_IDLE_UNLOAD_MINUTES", "0").strip() or "0")
            except ValueError:
                idle_minutes = 0
            if idle_minutes <= 0:
                last_idle_at = None
                already_unloaded = False
                continue

            queue_total = sum(get_queue_stats().values())
            if queue_total > 0:
                last_idle_at = None
                already_unloaded = False
                continue

            if not pipeline_running():
                # Pipeline stopped — stop hook handles unload; don't double-fire.
                last_idle_at = None
                continue

            workers = _active_vision_workers()
            uses_lmstudio = any(
                w.startswith("balanced-lm") or w == "lm-autodetect" for w in workers
            )
            if not uses_lmstudio:
                continue
            if "lm-keepalive" in workers:
                continue  # explicit keep-loaded mode

            now = _time.monotonic()
            if last_idle_at is None:
                last_idle_at = now
                continue
            if already_unloaded:
                continue
            if now - last_idle_at < idle_minutes * 60:
                continue

            result = _lmstudio_unload_all()
            already_unloaded = True
            logger.info(
                "idle unload after %d min: ok=%s method=%s detail=%s",
                idle_minutes, result.ok, result.method, result.detail,
            )
        except Exception as exc:  # never let the watcher die
            logger.warning("idle-unload watcher hiccup: %s", exc)


def _start_idle_unload_thread() -> None:
    thread = threading.Thread(target=_idle_unload_loop, name="lmstudio-idle-unload", daemon=True)
    thread.start()


def _start_indexer_thread() -> None:
    """Spawn the background SQLite indexer. First scan can take a while
    on a 100k-image dataset; subsequent ticks are incremental."""
    print(f"[indexer] starting (refresh every {int(_INDEXER_INTERVAL)}s)", flush=True)
    index_store.start_background_indexer(
        queue_root=PIPELINE_QUEUE,
        sorted_root=PIPELINE_SORTED,
        interval_seconds=_INDEXER_INTERVAL,
    )


def _index_status_payload() -> dict[str, Any]:
    """Snapshot of the indexer's state for the UI.

    Reads scan_meta keys that scan() updates as it runs. ``in_progress``
    flips to true when a scan starts, false when it ends (including on
    failure). ``files_seen`` / ``files_added`` climb live during a cold
    backfill so the UI can display 'scanning 12,345 files...' instead of
    appearing frozen.
    """
    in_progress_raw = index_store.get_meta("scan_in_progress")
    last_report_raw = index_store.get_meta("last_scan_report")
    try:
        last_report = json.loads(last_report_raw) if last_report_raw else None
    except json.JSONDecodeError:
        last_report = None
    started_at = index_store.get_meta("scan_started_at")
    return {
        "queue_total": index_store.total("queue"),
        "sorted_total": index_store.total("sorted"),
        "in_progress": in_progress_raw == "1",
        "files_seen": int(index_store.get_meta("scan_files_seen") or 0),
        "files_added": int(index_store.get_meta("scan_files_added") or 0),
        "scan_started_at": float(started_at) if started_at else None,
        "last_scan_at": float(index_store.get_meta("last_scan_at") or 0) or None,
        "last_scan_report": last_report,
    }


@app.route("/api/index/status")
def api_index_status():
    return jsonify(_index_status_payload())


if __name__ == "__main__":
    print(f"Dashboard: http://localhost:{FLASK_PORT}", flush=True)
    print(f"Settings: {ENV_PATH}", flush=True)
    print("Pipeline is NOT auto-started. Use the Start button in the UI.", flush=True)
    _start_idle_unload_thread()
    _start_indexer_thread()
    app.run(host="0.0.0.0", port=FLASK_PORT, debug=False, use_reloader=False)
