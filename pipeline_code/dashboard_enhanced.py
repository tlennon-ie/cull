#!/usr/bin/env python3
"""
Pipeline Dashboard - realtime monitoring + admin controls.

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


app = Flask(__name__)
CORS(app)

_pipeline_proc: subprocess.Popen | None = None
_pipeline_lock = threading.Lock()


# ── .env helpers ───────────────────────────────────────────────────────────────

def update_env(key: str, value: str) -> None:
    if not ENV_PATH.exists():
        logger.error(".env not found at %s", ENV_PATH)
        return
    pattern = re.compile(rf"^{re.escape(key)}=.*$", re.MULTILINE)
    text = ENV_PATH.read_text(encoding="utf-8")
    line = f"{key}={value}"
    # `re.sub` interprets backslashes in the replacement string as regex
    # escapes (\1, \A, \g<1> etc.). Windows paths like `I:\AI\openclaw` would
    # therefore raise PatternError ("bad escape \A"). Using a lambda short-
    # circuits replacement-string parsing and lets us insert the value as a
    # literal.
    if pattern.search(text):
        text = pattern.sub(lambda _match: line, text)
    else:
        text = text.rstrip() + f"\n{line}\n"
    ENV_PATH.write_text(text, encoding="utf-8")
    os.environ[key] = value


def get_env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


# ── stats helpers ──────────────────────────────────────────────────────────────

def get_queue_stats() -> dict[str, int]:
    stats: dict[str, int] = {}
    if not PIPELINE_QUEUE.exists():
        return stats
    for source_dir in PIPELINE_QUEUE.glob("*/*"):
        if source_dir.is_dir():
            count = len(list(source_dir.glob("*.jpg"))) + len(list(source_dir.glob("*.png")))
            if count:
                stats[f"{source_dir.parent.name}/{source_dir.name}"] = count
    return stats


def get_sorted_stats() -> dict[str, dict[str, int]]:
    stats: dict[str, dict[str, int]] = {}
    if not PIPELINE_SORTED.exists():
        return stats
    for topic_dir in PIPELINE_SORTED.iterdir():
        if not topic_dir.is_dir():
            continue
        topic_stats: dict[str, int] = {}
        for category_dir in topic_dir.iterdir():
            if category_dir.is_dir():
                count = len(list(category_dir.glob("**/*.jpg"))) + len(list(category_dir.glob("**/*.png")))
                if count:
                    topic_stats[category_dir.name] = count
        if topic_stats:
            stats[topic_dir.name] = topic_stats
    return stats


def get_error_logs(limit: int = 500) -> list[dict[str, str]]:
    """Collect ERROR/CRITICAL/FAILED lines from the 5 most recent log files.

    Returns newest-first (by log file mtime, then reversed line order within each
    file) so the dashboard's Errors tab always shows the latest failures at top.
    """
    errors: list[dict[str, str]] = []
    if not LOG_DIR.exists():
        return errors
    for log_file in sorted(LOG_DIR.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)[:5]:
        try:
            content = log_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        mtime_iso = datetime.fromtimestamp(log_file.stat().st_mtime).isoformat()
        file_errors: list[dict[str, str]] = []
        for line in content.splitlines():
            if any(tag in line for tag in ("ERROR", "CRITICAL", "FAILED")):
                file_errors.append({
                    "file": log_file.name,
                    "message": line[:600],
                    "timestamp": mtime_iso,
                })
        errors.extend(reversed(file_errors))
        if len(errors) >= limit:
            break
    return errors[:limit]


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
        return jsonify({"success": True})


ALLOWED_VISION_WORKERS = [
    "balanced-groq",
    "balanced-lm",            # targets LMSTUDIO_PRIMARY_*
    "balanced-lm-secondary",  # same worker script, forced to LMSTUDIO_SECONDARY_* via env override
    "lm-autodetect",
    "gemini",
    "groq",
]

_VISION_WORKER_DESCRIPTIONS = {
    "balanced-groq":          "Groq cloud, llama-4-scout - fast",
    "balanced-lm":            "LMStudio PRIMARY endpoint",
    "balanced-lm-secondary":  "LMStudio SECONDARY endpoint (runs in parallel)",
    "lm-autodetect":          "LMStudio, auto-picks vision-capable model",
    "gemini":                 "Gemini 2.5 Flash (cloud)",
    "groq":                   "Legacy single-threaded Groq",
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
    "PIPELINE_TOPIC",
    "PIPELINE_SLUG",
    "TOPIC_KEYWORDS_EXTRA",
    "TOPIC_BANNED_KEYWORDS",
    "TOPIC_GENERATION_HINTS",
    "REDDIT_SUBREDDITS",
    "MIN_PROMPT_LENGTH",
    "X_ACCOUNTS",
    "DISCORD_BOT_TOKEN",
    "DISCORD_AUTH_MODE",
    "VISION_OVR_MIN_SCORE",
    "VISION_REL_MIN_SCORE",
    "BLUR_NSFW_THUMBS",
    "VISION_SCORE_NOTES",
    "PIPELINE_RECONCILE_SECONDS",
    "PIPELINE_BASE_DIR",
    "PIPELINE_QUEUE",
    "PIPELINE_SORTED",
    "LOG_DIR",
    "ZFORFREE_LOCAL_SRC",
    "LOCAL_IMPORT_DIR",
    "LOCAL_IMPORT_NAME",
    "LOCAL_IMPORT_ENABLED",
    "LOCAL_IMPORT_MIGRATE_FROM",
]
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

def _recent_queue_files(limit: int = 60) -> list[Path]:
    """List newest queue images. Race-tolerant: vision workers can move files
    out from under us between glob and stat - we just drop those entries."""
    if not PIPELINE_QUEUE.exists():
        return []
    files: list[Path] = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        files.extend(PIPELINE_QUEUE.glob(f"**/{pattern}"))

    def _safe_mtime(p: Path) -> float:
        try:
            return p.stat().st_mtime
        except OSError:
            return -1.0  # bubble missing files to the bottom

    files.sort(key=_safe_mtime, reverse=True)
    return files[:limit]


@app.route("/api/queue/files")
def api_queue_files():
    limit = int(request.args.get("limit", 60))
    results: list[dict[str, Any]] = []
    for path in _recent_queue_files(limit):
        try:
            stat = path.stat()
        except OSError:
            continue  # vision worker grabbed it mid-listing - skip
        prompt_path = path.with_suffix(".txt")
        prompt = ""
        if prompt_path.exists():
            try:
                prompt = prompt_path.read_text(encoding="utf-8", errors="replace")[:300]
            except OSError:
                prompt = ""
        results.append({
            "path": str(path),
            "name": path.name,
            "source": path.parent.name,
            "size": stat.st_size,
            "corrupt": stat.st_size < 5000,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "thumbnail": f"/api/thumbnail?path={path}",
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
    if action == "requeue":
        return jsonify({"success": True, "action": "requeue"})
    if action == "move":
        if not target:
            return jsonify({"error": "target required"}), 400
        dest = PIPELINE_QUEUE / target
        dest.mkdir(parents=True, exist_ok=True)
        for sibling in path.parent.glob(f"{path.stem}.*"):
            shutil.move(str(sibling), str(dest / sibling.name))
        return jsonify({"success": True, "action": "move", "target": str(dest)})
    return jsonify({"error": "unknown action"}), 400


@app.route("/api/thumbnail")
def api_thumbnail():
    raw = request.args.get("path", "")
    try:
        size = max(64, min(2400, int(request.args.get("size", 240))))
    except ValueError:
        size = 240
    path = safe_inside(raw, [PIPELINE_QUEUE, PIPELINE_SORTED])
    if path is None or not path.exists():
        abort(404)
    try:
        img = Image.open(path)
        img.thumbnail((size, size))
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=82)
        buf.seek(0)
        return send_file(buf, mimetype="image/jpeg", max_age=60)
    except Exception as exc:
        logger.debug("thumbnail fail: %s", exc)
        abort(404)


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

    Source of truth is <PIPELINE_SORTED>/**/<stem>.vision.json files (one per
    sorted image). The old sorter.jsonl path is honored too if it exists, but
    the vision-json scan is always authoritative - it never goes stale.
    """
    source = request.args.get("source")
    category = request.args.get("category")
    limit = int(request.args.get("limit", 200))

    if not PIPELINE_SORTED.exists():
        return jsonify([])

    meta_files = [
        p for p in PIPELINE_SORTED.glob("**/*.vision.json")
        if not _is_in_archive(p)
    ]
    meta_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    out: list[dict[str, Any]] = []
    for meta in meta_files[:limit * 2]:  # over-fetch; we may filter
        try:
            payload = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        stem = meta.name.replace(".vision.json", "")
        img_candidate = next(
            (p for p in meta.parent.glob(f"{stem}.*") if _is_image(p)),
            None,
        )
        if img_candidate is None or not img_candidate.exists():
            continue
        # meta.parent = .../sorted/<slug>/<Category>/<source>
        category_name = meta.parent.parent.name
        source_name = meta.parent.name
        if source and source_name != source:
            continue
        if category and category_name != category:
            continue
        out.append({
            "timestamp": datetime.fromtimestamp(meta.stat().st_mtime).isoformat(),
            "image": img_candidate.name,
            "source": source_name,
            "classification": category_name,
            "quality": payload.get("quality_score"),
            "summary": payload.get("reason", ""),
            "thumbnail": f"/api/thumbnail?path={img_candidate}",
            "prompt_url": f"/api/prompt?path={img_candidate}",
            "path": str(img_candidate),
        })
        if len(out) >= limit:
            break
    return jsonify(out)


_IMAGE_EXTS_FOR_ACTIVITY: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in _IMAGE_EXTS_FOR_ACTIVITY


def _is_in_archive(path: Path) -> bool:
    """Skip files under the bookkeeping folders the requeue tool maintains."""
    return any(part.startswith(".") for part in path.relative_to(PIPELINE_SORTED).parts)


@app.route("/api/activity")
def api_activity():
    """Newest classified items from <PIPELINE_SORTED> so the UI can show what just got processed."""
    limit = int(request.args.get("limit", 12))
    if not PIPELINE_SORTED.exists():
        return jsonify([])
    meta_files: list[Path] = [
        p for p in PIPELINE_SORTED.glob("**/*.vision.json")
        if not _is_in_archive(p)
    ]
    meta_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    results: list[dict[str, Any]] = []
    for meta in meta_files:
        if len(results) >= limit:
            break
        try:
            payload = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        stem = meta.name.replace(".vision.json", "")
        # Restrict the alongside-the-meta lookup to actual image files - the
        # naive `glob(stem.*)` matched the .vision.json itself when the
        # image had been archived elsewhere, then the dashboard tried to
        # render JSON as a thumbnail.
        img_candidate = next(
            (p for p in meta.parent.glob(f"{stem}.*") if _is_image(p)),
            None,
        )
        if img_candidate is None or not img_candidate.exists():
            continue
        results.append({
            "name": img_candidate.name,
            "path": str(img_candidate),
            "category": meta.parent.parent.name,
            "source": meta.parent.name,
            "modified": datetime.fromtimestamp(meta.stat().st_mtime).isoformat(),
            "thumbnail": f"/api/thumbnail?path={img_candidate}",
            "prompt_url": f"/api/prompt?path={img_candidate}",
            "summary": payload.get("reason", ""),
            "quality": payload.get("quality_score"),
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


def _sorted_signature() -> tuple[int, float]:
    if not PIPELINE_SORTED.exists():
        return (0, 0.0)
    paths = [p for p in PIPELINE_SORTED.glob("**/*.vision.json") if not _is_in_archive(p)]
    if not paths:
        return (0, 0.0)
    return (len(paths), max(p.stat().st_mtime for p in paths))


def _resolve_image_dims(path: Path) -> tuple[int, int]:
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


def _scan_sorted_items() -> list[_SortedItem]:
    """One-shot scan of PIPELINE_SORTED returning every classified image with
    its vision-json + .txt prompt body. Skips archived/dot-prefixed paths.
    """
    if not PIPELINE_SORTED.exists():
        return []
    items: list[_SortedItem] = []
    for meta in PIPELINE_SORTED.glob("**/*.vision.json"):
        if _is_in_archive(meta):
            continue
        try:
            payload = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        stem = meta.name.replace(".vision.json", "")
        img = next(
            (p for p in meta.parent.glob(f"{stem}.*") if _is_image(p)),
            None,
        )
        if img is None or not img.exists():
            continue
        txt = meta.parent / f"{stem}.txt"
        prompt_text = ""
        if txt.exists():
            try:
                prompt_text = txt.read_text(encoding="utf-8", errors="replace")
            except OSError:
                prompt_text = ""
        try:
            mtime = meta.stat().st_mtime
        except OSError:
            mtime = 0.0
        items.append(
            _SortedItem(
                image_path=img,
                meta_path=meta,
                txt_path=txt if txt.exists() else None,
                category=meta.parent.parent.name,
                source=meta.parent.name,
                mtime=mtime,
                payload=payload,
                prompt_text=prompt_text,
            )
        )
    return items


def _get_sorted_items(force: bool = False) -> list[_SortedItem]:
    """Cached scan; refreshes when the underlying folder signature changes
    or when the TTL expires. Thread-safe."""
    with _sorted_cache_lock:
        now = _time.time()
        sig = _sorted_signature()
        if (
            not force
            and _sorted_cache["signature"] == sig
            and (now - _sorted_cache["ts"]) < _STATS_CACHE_TTL
        ):
            return _sorted_cache["items"]
        items = _scan_sorted_items()
        _sorted_cache["items"] = items
        _sorted_cache["ts"] = now
        _sorted_cache["signature"] = sig
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
<title>Pipeline Dashboard</title>
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://unpkg.com/alpinejs@3.x.x/dist/cdn.min.js" defer></script>
<style>
  body { font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
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

  <aside class="w-56 shrink-0 border-r border-slate-800 bg-slate-900/60 p-4 space-y-1">
    <div class="mb-4">
      <h1 class="text-lg font-bold">Pipeline</h1>
      <p class="text-xs text-slate-400" x-text="'Worker: ' + (status.pipeline?.vision_worker || '...')"></p>
      <div class="mt-2">
        <span class="pill px-2 py-0.5 rounded"
          :class="status.pipeline?.running ? 'bg-emerald-900/60 text-emerald-300' : 'bg-slate-800 text-slate-400'"
          x-text="status.pipeline?.running ? 'running' : 'stopped'"></span>
      </div>
    </div>
    <template x-for="tab in tabs" :key="tab.id">
      <button @click="active = tab.id"
        class="w-full text-left px-3 py-2 rounded text-sm transition"
        :class="active === tab.id ? 'bg-indigo-600 text-white' : 'text-slate-300 hover:bg-slate-800'"
        x-text="tab.label"></button>
    </template>
    <div class="pt-6 text-xs text-slate-500" x-text="'Refreshed: ' + lastRefresh"></div>
  </aside>

  <main class="flex-1 p-6 space-y-6 overflow-y-auto">
    <header class="flex items-center justify-between">
      <div>
        <h2 class="text-2xl font-bold capitalize" x-text="tabs.find(t => t.id === active)?.label"></h2>
        <p class="text-sm text-slate-400">Realtime operations console - auto-refresh every 5 s</p>
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
                  <img :src="a.thumbnail" class="thumb-lg" :class="{ 'nsfw-blur': shouldBlurNsfw(a) }"
                       loading="lazy" referrerpolicy="no-referrer"
                       @click="shouldBlurNsfw(a) ? revealNsfw(a) : openModalFromActivity(a)"/>
                  <span class="nsfw-eye" x-show="shouldBlurNsfw(a)" @click.stop="revealNsfw(a)" title="Reveal NSFW">
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
          <button @click="loadStats()" class="px-3 py-1.5 text-xs bg-slate-700 hover:bg-slate-600 rounded">Refresh stats</button>
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
                  <img :src="c.thumbnail" class="thumb-lg mx-auto" :class="{ 'nsfw-blur': shouldBlurNsfw(c) }"
                       loading="lazy"
                       @click="shouldBlurNsfw(c) ? revealNsfw(c) : openModalFromCard(c)"/>
                  <span class="nsfw-eye" x-show="shouldBlurNsfw(c)" @click.stop="revealNsfw(c)" title="Reveal NSFW">
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
            <select x-model="galleryFilters.sort" @change="galleryReload()"
              class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 text-sm">
              <option value="newest">Newest first</option>
              <option value="ovr">OVR (craft)</option>
              <option value="rel">REL (relevance)</option>
              <option value="quality">quality_score</option>
            </select>
          </div>
          <div class="lg:col-span-2">
            <label class="text-xs text-slate-400">NSFW</label>
            <select x-model="galleryFilters.nsfw" @change="galleryReload()"
              class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 text-sm">
              <option value="any">Show all</option>
              <option value="exclude">Hide NSFW</option>
              <option value="only">Only NSFW</option>
            </select>
          </div>
          <div class="lg:col-span-2">
            <label class="text-xs text-slate-400">Date from</label>
            <input type="date" x-model="galleryFilters.dateFrom" @change="galleryReload()"
              class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 text-sm"/>
          </div>
          <div class="lg:col-span-2">
            <label class="text-xs text-slate-400">Date to</label>
            <input type="date" x-model="galleryFilters.dateTo" @change="galleryReload()"
              class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 text-sm"/>
          </div>

          <div class="lg:col-span-2">
            <label class="text-xs text-slate-400">Min OVR</label>
            <input type="number" min="0" max="100" x-model.number="galleryFilters.minOvr" @change="galleryReload()"
              class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 text-sm"/>
          </div>
          <div class="lg:col-span-2">
            <label class="text-xs text-slate-400">Min REL</label>
            <input type="number" min="0" max="100" x-model.number="galleryFilters.minRel" @change="galleryReload()"
              class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 text-sm"/>
          </div>
          <div class="lg:col-span-2">
            <label class="text-xs text-slate-400">Min quality (1-10)</label>
            <input type="number" min="0" max="10" x-model.number="galleryFilters.minQuality" @change="galleryReload()"
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
            <button @click="galleryPrev()" class="px-2 py-1 bg-slate-800 hover:bg-slate-700 rounded" :disabled="gallery.page <= 1">Prev</button>
            <span>Page <span x-text="gallery.page"></span> / <span x-text="Math.max(1, Math.ceil(gallery.total / gallery.pageSize))"></span></span>
            <button @click="galleryNext()" class="px-2 py-1 bg-slate-800 hover:bg-slate-700 rounded">Next</button>
          </div>
        </div>
        <div x-show="galleryLoading" class="text-xs text-slate-400">Loading...</div>
        <div class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3">
          <template x-for="c in gallery.items" :key="c.path">
            <div class="bg-slate-900/60 border border-slate-800 rounded p-2 text-xs flex flex-col">
              <span class="nsfw-wrap block">
                <img :src="c.thumbnail" class="w-full aspect-square object-cover rounded"
                     :class="{ 'nsfw-blur': shouldBlurNsfw(c) }"
                     loading="lazy"
                     @click="shouldBlurNsfw(c) ? revealNsfw(c) : openModalFromCard(c)"/>
                <span class="nsfw-eye" x-show="shouldBlurNsfw(c)" @click.stop="revealNsfw(c)" title="Reveal NSFW">
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
            <div class="col-span-full text-sm text-slate-500">No images match these filters.</div>
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
              <input type="checkbox" :checked="s.enabled" @change="toggleScraper(s.name, $event.target.checked)"
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
                  <input type="checkbox" :checked="w.enabled" @change="toggleVisionWorker(w.name, $event.target.checked)"
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
                <img :src="a.thumbnail" class="thumb-lg" :class="{ 'nsfw-blur': shouldBlurNsfw(a) }"
                     loading="lazy"
                     @click="shouldBlurNsfw(a) ? revealNsfw(a) : openModalFromActivity(a)"/>
                <span class="nsfw-eye" x-show="shouldBlurNsfw(a)" @click.stop="revealNsfw(a)" title="Reveal NSFW">
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
    <section x-show="active === 'queue'" class="card rounded-xl p-5">
      <h3 class="font-semibold mb-3">Queue (newest 60)</h3>
      <div class="scroll-box"><table>
        <thead><tr><th></th><th>Name</th><th>Source</th><th>Size</th><th>Prompt</th><th></th></tr></thead>
        <tbody>
          <template x-for="f in queueFiles" :key="f.path">
            <tr :class="f.corrupt ? 'bg-rose-900/25' : ''">
              <td><img :src="f.thumbnail" class="thumb" loading="lazy" @click="openModalFromFile(f)"/></td>
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
                <button @click="queueAction(f.path, 'requeue')" class="px-2 py-1 bg-indigo-600/70 hover:bg-indigo-500 rounded text-xs">Requeue</button>
                <button @click="queueAction(f.path, 'delete')" class="px-2 py-1 bg-rose-600/70 hover:bg-rose-500 rounded text-xs">Delete</button>
              </td>
            </tr>
          </template>
          <template x-if="queueFiles.length === 0">
            <tr><td colspan="6" class="text-center text-slate-500 py-6">Queue is empty.</td></tr>
          </template>
        </tbody>
      </table></div>
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
                  <img :src="h.thumbnail || ''" class="thumb" :class="{ 'nsfw-blur': shouldBlurNsfw(h) }"
                       loading="lazy"
                       onerror="this.style.visibility='hidden'"
                       @click="h.thumbnail && (shouldBlurNsfw(h) ? revealNsfw(h) : openModalFromHistory(h))"/>
                  <span class="nsfw-eye" x-show="shouldBlurNsfw(h)" @click.stop="revealNsfw(h)" title="Reveal NSFW">
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
    <section x-show="active === 'settings'" class="space-y-4">
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
          <div class="bg-indigo-950/60 border border-indigo-700 text-indigo-200 text-xs px-3 py-2 rounded mb-3" x-text="settingsBanner"></div>
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
                   class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1"/>
          </label>
          <label class="block md:col-span-2">
            <span class="text-xs text-slate-400">X.com accounts (comma-sep, no @). Empty = search-only.</span>
            <input x-model="settings.X_ACCOUNTS" placeholder="account1,account2,account3"
                   class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1"/>
          </label>
        </div>
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
                   class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1"/>
          </label>
          <label class="block">
            <span class="text-xs text-slate-400">Minimum REL score (0-100)</span>
            <input x-model="settings.VISION_REL_MIN_SCORE" type="number" min="0" max="100"
                   class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1"/>
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
                   class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1"/>
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

  </main>

  <!-- DETAIL MODAL -->
  <div x-show="modal.open" x-cloak @keydown.escape.window="closeModal()"
       class="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-6"
       @click.self="closeModal()">
    <div class="card rounded-xl p-5 max-w-5xl w-full max-h-[90vh] overflow-hidden flex flex-col">
      <div class="flex items-start justify-between gap-4 mb-3">
        <div class="min-w-0">
          <div class="text-xs text-slate-400" x-text="modal.source + (modal.category ? ' · ' + modal.category : '')"></div>
          <div class="font-mono text-sm truncate" x-text="modal.name"></div>
        </div>
        <button @click="closeModal()" class="shrink-0 px-3 py-1 text-sm bg-slate-800 hover:bg-slate-700 rounded">Close (Esc)</button>
      </div>
      <div class="grid lg:grid-cols-2 gap-4 overflow-hidden">
        <div class="bg-slate-950 rounded flex items-center justify-center min-h-[300px] overflow-auto relative">
          <img :src="modal.imageUrl"
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
                <button @click="savePrompt()" :disabled="modal.saving" class="px-2 py-1 bg-emerald-600 hover:bg-emerald-500 rounded disabled:opacity-50">
                  <span x-text="modal.saving ? 'Saving...' : 'Save (overwrite)'"></span>
                </button>
              </template>
              <template x-if="modal.editing">
                <button @click="cancelPromptEdit()" class="px-2 py-1 bg-slate-700 hover:bg-slate-600 rounded">Cancel</button>
              </template>
            </div>
          </div>
          <template x-if="!modal.editing">
            <pre class="whitespace-pre-wrap text-sm font-mono text-slate-200" x-text="modal.prompt || '(empty)'"></pre>
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
    ],
    providers: ['balanced-groq','balanced-lm','balanced-lm-secondary','lm-autodetect','groq','gemini'],
    provider: 'balanced-groq',
    throttle: 100,
    status: {}, scrapers: [], models: {}, visionWorkers: [],
    settings: {}, settingsBanner: '',
    workerDescriptions: {
      'balanced-groq':          'Groq cloud, llama-4-scout - fast, handles NSFW',
      'balanced-lm':            'LMStudio PRIMARY endpoint',
      'balanced-lm-secondary':  'LMStudio SECONDARY endpoint (parallel with -primary)',
      'lm-autodetect':          'LMStudio, auto-picks the loaded vision model (VL)',
      'gemini':                 'Google Gemini 2.5 Flash - paid cloud',
      'groq':                   'Legacy single-threaded Groq worker',
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

    async refresh() {
      const [s, scr, mods, q, h, a, vw, settings] = await Promise.all([
        fetch('/api/status').then(r=>r.json()),
        fetch('/api/scrapers').then(r=>r.json()),
        fetch('/api/lmstudio/models').then(r=>r.json()),
        fetch('/api/queue/files?limit=60').then(r=>r.json()),
        fetch('/api/logs/history?limit=200').then(r=>r.json()),
        fetch('/api/activity?limit=12').then(r=>r.json()),
        fetch('/api/vision/workers').then(r=>r.json()),
        fetch('/api/settings').then(r=>r.json()),
      ]);
      this.status = s;
      this.scrapers = scr;
      this.models = mods;
      this.queueFiles = q;
      this.history = h;
      this.activity = a;
      this.visionWorkers = vw;
      // Only overwrite settings if the user isn't mid-edit (empty settings object).
      if (Object.keys(this.settings).length === 0) this.settings = settings;
      this.provider = s.pipeline?.vision_worker || this.provider;
      this.throttle = s.pipeline?.throttle ?? this.throttle;
      this.lastRefresh = new Date().toLocaleTimeString();
    },
    async saveSettings() {
      this.settingsBanner = 'Saving...';
      const r = await fetch('/api/settings', {method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify(this.settings)});
      const j = await r.json();
      if (j.success) {
        this.settingsBanner = 'Saved. Stop + Start the pipeline to pick up changes.';
        setTimeout(() => this.settingsBanner = '', 6000);
      } else {
        this.settingsBanner = 'Errors: ' + JSON.stringify(j.errors);
      }
      this.refresh();
    },
    reloadSettings() { this.settings = {}; this.refresh(); },
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
    closeModal() { this.modal.open = false; this.modal.editing = false; },

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
        this.modal.savedFlash = 'Network error';
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
      });
      setInterval(() => { if (this.active === 'stats') this.loadStats(); }, 30000);
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


if __name__ == "__main__":
    print(f"Dashboard: http://localhost:{FLASK_PORT}", flush=True)
    print(f"Settings: {ENV_PATH}", flush=True)
    print("Pipeline is NOT auto-started. Use the Start button in the UI.", flush=True)
    app.run(host="0.0.0.0", port=FLASK_PORT, debug=False, use_reloader=False)
