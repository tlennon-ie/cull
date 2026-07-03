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
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote as _urlquote

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

# Always resolve to absolute path. If WORKSPACE_ROOT is set, use it; otherwise infer from this file's location.
if "WORKSPACE_ROOT" in os.environ and os.environ["WORKSPACE_ROOT"]:
    WORKSPACE_ROOT = Path(os.environ["WORKSPACE_ROOT"]).resolve()
else:
    # Assume this script is in <workspace_root>/pipeline_code/dashboard_enhanced.py
    WORKSPACE_ROOT = Path(__file__).parent.parent.resolve()

PIPELINE_CODE_DIR: Path = WORKSPACE_ROOT / "pipeline_code"
ENV_PATH: Path = WORKSPACE_ROOT / ".env"
PIPELINE_QUEUE: Path = Path(os.environ.get("PIPELINE_QUEUE", "queue"))
PIPELINE_SORTED: Path = Path(os.environ.get("PIPELINE_SORTED", "sorted"))
LOG_DIR: Path = Path(os.environ.get("LOG_DIR", "logs_test"))
FLASK_PORT: int = int(os.environ.get("FLASK_PORT", 5000))

# Scraper display descriptions, keyed by the canonical name. The NAME list
# itself is owned by job_config.SCRAPER_NAMES (single source of truth shared
# with the jobs model); we only annotate them here for the UI. A name without
# an entry here falls back to an empty description.
_SCRAPER_DESCRIPTIONS: dict[str, str] = {
    "X.com":       "X.com (Playwright + cookies, no API)",
    "Discord-1":   "Discord UD channels",
    "Civitai-Com": "Civitai (civitai.com)",
    "Civitai-Red": "Civitai (civitai.red)",
    "Web":         "Reddit",
    "Gallery-DL":  "gallery-dl (Pixiv, DeviantArt, booru, ArtStation, Tumblr, X, Reddit, Imgur, FurAffinity, e621, Flickr…). Configure URLs + cookies in the job's Scraper targets.",
}

# _STATIC_SCRAPERS is derived from job_config.SCRAPER_NAMES (built just below,
# once job_config is importable) so the dashboard toggle list and the job
# config's scrapers.enabled map can never drift apart. Local-import folders are
# NOT here — they're a per-job list (scrapers.local_imports) surfaced as
# read-only Local-<name> rows by _scraper_enabled_response.


_STATIC_SCRAPERS: list[dict[str, str]] = []


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
import job_config
import config_io
import scraper_test
import paths as _paths
import vision_model_catalog
import phash_dedup
import export_profiles
import hf_export
import scheduler
import fleet_health
import vision_prompt
import credentials

# Now that job_config is importable, derive the canonical scraper toggle list
# from its SCRAPER_NAMES (single source of truth) annotated with UI descriptions.
_STATIC_SCRAPERS[:] = [
    {"name": name, "description": _SCRAPER_DESCRIPTIONS.get(name, "")}
    for name in job_config.SCRAPER_NAMES
]
SCRAPERS = _STATIC_SCRAPERS  # kept for anything still importing this module-level constant

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

# Jobs model: ensure existing installs get a `default` job built from the
# current .env so the pipeline has something to activate. Idempotent — a no-op
# once any job file exists. Done at import so both `python dashboard_enhanced.py`
# and an embedding test client see a migrated store.
try:
    job_config.migrate_env_to_default_job()
    # Fold legacy single-endpoint vision env (LMSTUDIO_*/OLLAMA_*/OPENAI_COMPAT_*)
    # into the default preset's worker fleet so an upgrade keeps working endpoints.
    job_config.migrate_legacy_vision_to_fleet()
except Exception as _exc:  # pragma: no cover - never block dashboard boot
    logger.warning("job migration skipped: %s", _exc)


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
    """{<topic>/<source>: count} — served from the SQLite index. Still used by
    the idle-unload watcher to detect an empty queue."""
    return index_store.count_queue_by_topic_source()


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


# ── error responses ─────────────────────────────────────────────────────────────

def _err(message: str, exc: BaseException | None = None, code: int = 500):
    """Build a generic error JSON response without leaking exception detail.

    The client only ever sees ``message`` (a fixed, non-sensitive string); the
    underlying exception text is logged server-side so operators keep the detail
    for debugging while a stack trace / internal path never reaches the browser.
    Returns the standard ``{success, data, error}`` envelope with ``code``.
    """
    if exc is not None:
        logger.warning("%s: %s", message, exc)
    return jsonify({"success": False, "data": None, "error": message}), code


# ── path guards ────────────────────────────────────────────────────────────────

def safe_inside(raw: str, roots: list[Path]) -> Path | None:
    """Return the resolved ``raw`` path iff it is contained in one of ``roots``.

    Uses ``os.path.realpath`` to fully normalise the candidate (collapsing
    ``..`` / symlinks) and ``os.path.commonpath`` for the containment check — a
    normalize-then-contain pattern static analysers recognise as a path-traversal
    sanitiser. Returns the resolved :class:`~pathlib.Path` when it lives under a
    root, else ``None``. Behaviour is identical to the previous
    ``Path.relative_to`` loop (in-roots → resolved Path, out-of-roots / traversal
    → None); ``commonpath`` raising :class:`ValueError` on a different Windows
    drive is treated as "not contained".
    """
    try:
        real = os.path.realpath(str(raw))
    except (OSError, ValueError):
        return None
    for root in roots:
        try:
            root_real = os.path.realpath(str(root))
            if os.path.commonpath([real, root_real]) == root_real:
                return Path(real)
        except (OSError, ValueError):
            continue
    return None


# ── Jobs: slug resolution + per-slug index scoping ────────────────────────────
#
# Every job-aware endpoint resolves an effective slug from an optional
# ``?job=<slug>`` query param, defaulting to the active job. The index_store
# layer keys every row by ``topic_slug`` already, so scoping is just adding a
# ``topic_slug = ?`` predicate. We query the shared connection directly because
# index_store's public listing helpers don't take a slug filter.

# Slugs that would collide with literal route segments under /api/jobs/. A job
# with one of these slugs would shadow (or be shadowed by) the static-path
# handlers below, so we forbid creating them. Defence in depth — job_config
# could add the same guard, but the dashboard owns these route names.
RESERVED_JOB_SLUGS: frozenset[str] = frozenset({"queue", "advance"})


def _resolve_job_slug(default: str | None = "__active__") -> str | None:
    """Effective slug for a request.

    ``?job=<slug>`` wins when present and well-formed; otherwise fall back to
    ``default`` (sentinel ``"__active__"`` means "use the active job"). A
    malformed ``?job=`` falls back too (never silently widens scope to *all*
    jobs). Returns None only when there is no active job and none was supplied —
    callers treat that as "unscoped / behave as today" for pre-jobs installs.
    """
    raw = (request.args.get("job") or "").strip()
    if raw and job_config.JOB_SLUG_RE.match(raw):
        return raw
    if default == "__active__":
        return job_config.get_active_slug()
    return default


def _scoped_queue_by_source(slug: str | None) -> dict[str, int]:
    """{'<slug>/<source>': count} for one slug (or all when slug is None)."""
    with index_store.with_conn() as conn:
        if slug is None:
            cur = conn.execute(
                "SELECT topic_slug, source, COUNT(*) FROM images "
                "WHERE status = 'queue' GROUP BY topic_slug, source"
            )
        else:
            cur = conn.execute(
                "SELECT topic_slug, source, COUNT(*) FROM images "
                "WHERE status = 'queue' AND topic_slug = ? GROUP BY topic_slug, source",
                (slug,),
            )
        return {f"{row[0] or 'default'}/{row[1]}": int(row[2]) for row in cur.fetchall()}


def _scoped_sorted_by_category(slug: str | None) -> dict[str, dict[str, int]]:
    """{topic: {category: count}} for one slug (or all when slug is None)."""
    with index_store.with_conn() as conn:
        if slug is None:
            cur = conn.execute(
                "SELECT topic_slug, category, COUNT(*) FROM images "
                "WHERE status = 'sorted' AND category IS NOT NULL "
                "GROUP BY topic_slug, category"
            )
        else:
            cur = conn.execute(
                "SELECT topic_slug, category, COUNT(*) FROM images "
                "WHERE status = 'sorted' AND category IS NOT NULL AND topic_slug = ? "
                "GROUP BY topic_slug, category",
                (slug,),
            )
        out: dict[str, dict[str, int]] = {}
        for topic, cat, count in cur.fetchall():
            out.setdefault(topic or "default", {})[cat] = int(count)
        return out


def _scoped_counts(slug: str | None) -> dict[str, int]:
    """{'queued': n, 'sorted': n} for a single slug. Used by the jobs grid."""
    with index_store.with_conn() as conn:
        if slug is None:
            q = conn.execute("SELECT COUNT(*) FROM images WHERE status = 'queue'").fetchone()[0]
            s = conn.execute("SELECT COUNT(*) FROM images WHERE status = 'sorted'").fetchone()[0]
        else:
            q = conn.execute(
                "SELECT COUNT(*) FROM images WHERE status = 'queue' AND topic_slug = ?",
                (slug,),
            ).fetchone()[0]
            s = conn.execute(
                "SELECT COUNT(*) FROM images WHERE status = 'sorted' AND topic_slug = ?",
                (slug,),
            ).fetchone()[0]
        return {"queued": int(q), "sorted": int(s)}


def _job_status_label(slug: str, job: job_config.Job, *, active: str | None,
                      queue: list[str]) -> str:
    """Derive a display status for a job from the index + runtime state.

    The job file carries a ``status`` field but the live truth is: the active
    job is 'running' iff the supervisor process is up, else 'active'; queued
    jobs are 'queued'; everything else is 'idle'. We don't mutate the job file
    here — this is a read-only projection for the UI.
    """
    if slug == active:
        return "running" if pipeline_running() else "active"
    if slug in queue:
        return "queued"
    return job.status or "idle"


def _job_card(job: job_config.Job, *, active: str | None, queue: list[str]) -> dict[str, Any]:
    """Compact per-job payload for the jobs grid (list view)."""
    counts = _scoped_counts(job.slug)
    if job.slug == active:
        position = 0
    elif job.slug in queue:
        position = 1 + queue.index(job.slug)
    else:
        position = None
    return {
        "slug": job.slug,
        "name": job.name,
        "status": _job_status_label(job.slug, job, active=active, queue=queue),
        "is_active": job.slug == active,
        "queue_position": position,
        "queued": counts["queued"],
        "sorted": counts["sorted"],
        "updated_at": job.updated_at,
    }


# ── API: status / controls ────────────────────────────────────────────────────

@app.route("/api/status")
def api_status():
    # Scope queue/sorted counts to ?job=<slug> (default active). When no jobs
    # exist yet (pre-migration installs) the slug resolves to None and we show
    # the global totals exactly as before.
    slug = _resolve_job_slug()
    queue = _scoped_queue_by_source(slug)
    sorted_stats = _scoped_sorted_by_category(slug)
    errors = get_error_logs()
    return jsonify({
        "timestamp": datetime.now().isoformat(),
        "job": slug,
        "active_job": job_config.get_active_slug(),
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


def _job_for_scope(slug: str | None) -> job_config.Job | None:
    """Load the job a scrapers/categories request targets, or None when no jobs
    exist yet (pre-migration). Raises nothing — callers decide on None."""
    if slug is None:
        return None
    return job_config.get_job(slug)


def _job_effective_scrapers(job: job_config.Job) -> dict[str, Any]:
    eff = job_config.effective_config(job)
    s = eff.get("scrapers")
    return s if isinstance(s, dict) else {}


def _save_scraper_enabled(job: job_config.Job, enabled_map: dict[str, bool]) -> job_config.Job:
    """Persist the job's ``scrapers.enabled`` as an OVERRIDE (v2) and return the
    saved Job. We do NOT re-project here — consistent with the auto-save model,
    the active job is re-projected once on apply/leave (POST .../activate)."""
    updated = job_config.set_override(job, "scrapers.enabled", enabled_map)
    return job_config.save_job(updated)


def _existing_enabled_override(job: job_config.Job) -> dict[str, bool]:
    """The job's SPARSE ``scrapers.enabled`` override map (only keys the user has
    actually overridden), or ``{}`` when the scrapers are fully inherited. Reads
    the override map — NOT the effective map — so toggling one scraper doesn't
    drag the other five into the override and lose their inherit 'global' chip."""
    cur = (job.overrides or {}).get("scrapers", {})
    enabled = cur.get("enabled") if isinstance(cur, dict) else None
    if not isinstance(enabled, dict):
        return {}
    return {n: bool(v) for n, v in enabled.items()
            if n in job_config.SCRAPER_NAMES and isinstance(v, bool)}


def _scraper_enabled_response(job: job_config.Job | None) -> list[dict[str, Any]]:
    """[{name, description, enabled}] for the Scrapers tab. v2: the canonical
    scraper rows come from the job's EFFECTIVE enabled map, plus one read-only
    ``Local-<name>`` row per enabled local-import folder. Pre-migration falls
    back to the legacy env."""
    if job is not None:
        scr = _job_effective_scrapers(job)
        enabled = scr.get("enabled", {}) if isinstance(scr.get("enabled"), dict) else {}
        rows = [
            {"name": s["name"], "description": s["description"],
             "enabled": bool(enabled.get(s["name"], True)), "kind": "scraper"}
            for s in _STATIC_SCRAPERS
        ]
        local_list = scr.get("local_imports") if isinstance(scr.get("local_imports"), list) else []
        for folder in local_list:
            if not isinstance(folder, dict):
                continue
            nm = str(folder.get("name", "local") or "local")
            rows.append({
                "name": f"Local-{nm}",
                "description": f"Local folder ({folder.get('dir') or '(no dir set)'})",
                "enabled": bool(folder.get("enabled", False)),
                "kind": "local",          # read-only; managed in Job Settings
            })
        return rows
    disabled = disabled_set()
    return [{**s, "enabled": s["name"] not in disabled, "kind": "scraper"}
            for s in _STATIC_SCRAPERS]


@app.route("/api/scrapers")
def api_scrapers():
    job = _job_for_scope(_resolve_job_slug())
    return jsonify(_scraper_enabled_response(job))


@app.route("/api/scrapers/bulk", methods=["POST"])
def api_scrapers_bulk():
    data = request.get_json() or {}
    action = data.get("action")
    if action not in {"disable_all", "enable_all"}:
        return jsonify({"error": "action must be disable_all|enable_all"}), 400
    want_enabled = action == "enable_all"

    job = _job_for_scope(_resolve_job_slug())
    if job is not None:
        # "All on/off" is a deliberate full override of the 6 canonical
        # scrapers; the dynamic Local-<name> rows are read-only and untouched.
        enabled_map = {n: want_enabled for n in job_config.SCRAPER_NAMES}
        saved = _save_scraper_enabled(job, enabled_map)
        disabled = sorted(n for n, on in enabled_map.items() if not on)
        # Return fresh overrides so the JS can resync je.overrides (clobber-safe).
        return jsonify({"success": True, "job": job.slug, "disabled": disabled,
                        "overrides": saved.overrides or {}})

    # No job in scope (pre-migration): fall back to the legacy global env.
    names = list(job_config.SCRAPER_NAMES)
    if want_enabled:
        update_env("SCRAPER_DISABLED", "")
        return jsonify({"success": True, "disabled": []})
    update_env("SCRAPER_DISABLED", ",".join(sorted(names)))
    return jsonify({"success": True, "disabled": names})


@app.route("/api/scrapers/toggle", methods=["POST"])
def api_scraper_toggle():
    data = request.get_json() or {}
    name = data.get("name")
    enabled = bool(data.get("enabled"))

    job = _job_for_scope(_resolve_job_slug())
    if job is not None:
        # Only the 6 canonical scrapers are togglable; Local-<name> rows are
        # read-only status (configure folders in Job Settings → Local folders).
        if name not in job_config.SCRAPER_NAMES:
            return jsonify({
                "error": "local folders are configured in Job Settings → Local folders",
                "name": name,
            }), 400
        # SPARSE override: merge only the toggled key into the EXISTING override
        # map (not the effective map), so untouched scrapers keep inheriting from
        # the preset (their 'global' chip stays). The first toggle stores exactly
        # one key under overrides.scrapers.enabled.
        enabled_map = {**_existing_enabled_override(job), name: enabled}
        saved = _save_scraper_enabled(job, enabled_map)
        # Return fresh overrides so the JS resyncs je.overrides and a later
        # debounced job-editor PUT can't overwrite this toggle with a stale map.
        return jsonify({"success": True, "name": name, "enabled": enabled,
                        "job": job.slug, "overrides": saved.overrides or {}})

    # Pre-migration (no jobs yet): fall back to the legacy global env.
    if name not in job_config.SCRAPER_NAMES:
        return jsonify({"error": "Unknown scraper"}), 400
    current = disabled_set()
    (current.discard if enabled else current.add)(name)
    update_env("SCRAPER_DISABLED", ",".join(sorted(current)))
    return jsonify({"success": True, "name": name, "enabled": enabled})


@app.route("/api/pipeline/start", methods=["POST"])
def api_pipeline_start():
    global _pipeline_proc
    # The jobs model requires an active job: the supervisor projects the active
    # job's config (env + categories) into the spawn environment. Without one
    # there is nothing to run.
    if job_config.get_active_slug() is None:
        return jsonify({"error": "Activate a job before starting the pipeline"}), 409
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
            return jsonify({"error": "failed to start pipeline"}), 500


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

# Mirror of config_io._CAT_NAME_RE — allow spaces/hyphens for readable labels
# ("Irish cars"); start with a letter, max 40 chars. _safe_component() guards the
# filesystem downstream. Keep this in lockstep with config_io.py.
_CAT_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9 _-]{0,39}$")
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
            return None, f"invalid category name {name!r} (letters, digits, spaces, _ or - only, must start with a letter, max 40 chars)"
        if name in {"DISCARD", "CORRUPT"}:
            return None, f"{name!r} is reserved for the system"
        if len(hint) > 2000:
            return None, f"hint for {name!r} too long (max 2000 chars)"
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


def _job_active_taxonomy(job: job_config.Job | None) -> dict[str, Any]:
    """Shape the ``active`` taxonomy block for the categories editor (v2).

    Per-job: the EFFECTIVE categories + rules (preset ⊕ overrides). Pre-migration
    (no job in scope): the global active taxonomy file as before."""
    if job is not None:
        eff = job_config.effective_config(job)
        cats = eff.get("categories")
        return {
            "preset": "custom",
            "categories": [dict(c) for c in (cats or []) if isinstance(c, dict)],
            "global_rules": str(eff.get("category_rules", "") or ""),
        }
    return _categories_mod.get_active()


@app.route("/api/categories")
def api_categories_get():
    job = _job_for_scope(_resolve_job_slug())
    overridden = {
        "categories": job_config.is_overridden(job, "categories") if job else False,
        "category_rules": job_config.is_overridden(job, "category_rules") if job else False,
    }
    return jsonify({
        "active": _job_active_taxonomy(job),
        "presets": _categories_mod.get_presets()["presets"],
        "default_preset": _categories_mod.get_presets().get("default"),
        "system_terminal": list(_categories_mod.SYSTEM_TERMINAL),
        "job": job.slug if job is not None else None,
        "overridden": overridden,
    })


def _count_sorted_in_categories(slug: str | None, cats: set[str]) -> dict[str, int]:
    """{category: count} of sorted rows in the given categories, scoped to slug.
    Index-backed so we don't walk the filesystem."""
    if not cats:
        return {}
    placeholders = ",".join("?" * len(cats))
    params: list[Any] = list(cats)
    where = f"status = 'sorted' AND category IN ({placeholders})"
    if slug is not None:
        where += " AND topic_slug = ?"
        params.append(slug)
    with index_store.with_conn() as conn:
        cur = conn.execute(
            f"SELECT category, COUNT(*) FROM images WHERE {where} GROUP BY category",
            params,
        )
        return {row[0]: int(row[1]) for row in cur.fetchall() if row[1]}


@app.route("/api/categories", methods=["POST"])
def api_categories_set():
    payload = request.get_json() or {}
    cleaned, err = _validate_categories_payload(payload)
    if err:
        return jsonify({"ok": False, "error": err}), 400

    job = _job_for_scope(_resolve_job_slug())
    force = request.args.get("force") == "1"
    new_names = {c["name"] for c in cleaned["categories"]}

    # Warn before dropping a category that still has sorted images. Scope the
    # count + current-name comparison to the EFFECTIVE categories of the job we
    # actually loaded (NOT the active-default), so editing an INACTIVE job
    # reflects that job's data. Index-backed; honours ?force=1 to proceed.
    count_slug = job.slug if job is not None else None
    if job is not None:
        eff_cats = job_config.effective_config(job).get("categories") or []
        current_names = {c.get("name") for c in eff_cats if isinstance(c, dict)}
    else:
        current_names = set(_categories_mod.get_categories())
    removed = {n for n in (current_names - new_names) if n}
    if removed and not force:
        legacy = _count_sorted_in_categories(count_slug, removed)
        if legacy:
            return jsonify({
                "ok": False,
                "error": "removing_populated_categories",
                "legacy": legacy,
                "hint": "These categories still have sorted images. Requeue them first via tools/requeue_sorted.py, or POST again with ?force=1 to remove anyway (folders stay on disk).",
            }), 409

    if job is not None:
        # v2: categories + rules become job OVERRIDES. Consistent with the
        # auto-save model, we do NOT re-project here — the active job applies on
        # leave (POST /api/jobs/<slug>/activate).
        updated = job_config.set_override(job, "categories", cleaned["categories"])
        updated = job_config.set_override(updated, "category_rules", cleaned["global_rules"])
        saved = job_config.save_job(updated)
        return jsonify({"ok": True, "active": _job_active_taxonomy(saved), "job": saved.slug})

    # Pre-migration fallback: write the global active taxonomy file directly.
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
        logger.warning("git unavailable during update: %s", exc)
        return jsonify({"ok": False, "error": "git unavailable"}), 500

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
    "openai",                 # OpenAI cloud (GPT vision)
    "anthropic",              # Anthropic Claude vision
    "gemini",                 # Google Gemini vision (google-genai SDK)
    "openrouter",             # OpenRouter OpenAI-compatible gateway
    "balanced-lm",            # targets LMSTUDIO_PRIMARY_*
    "balanced-lm-secondary",  # same worker script, forced to LMSTUDIO_SECONDARY_* via env override
    "lm-autodetect",
    "balanced-openai-compat", # llama.cpp / vLLM / koboldcpp / LocalAI / text-generation-webui
    "balanced-ollama",        # Ollama native /api/chat
]

_VISION_WORKER_DESCRIPTIONS = {
    "balanced-groq":           "Groq cloud, llama-4-scout - fast, paid",
    "openai":                  "OpenAI cloud (GPT vision) - structured JSON schema. Reads OPENAI_API_KEY + OPENAI_VISION_MODEL",
    "anthropic":               "Anthropic Claude vision - structured output via forced tool-use. Reads ANTHROPIC_API_KEY + ANTHROPIC_VISION_MODEL",
    "gemini":                  "Google Gemini vision (new google-genai SDK) - response_schema output. Reads GEMINI_API_KEY/GOOGLE_API_KEY + GEMINI_VISION_MODEL",
    "openrouter":              "OpenRouter - OpenAI-compatible gateway to many vision models. Reads OPENROUTER_API_KEY + OPENROUTER_VISION_MODEL",
    "balanced-lm":             "LM Studio PRIMARY endpoint",
    "balanced-lm-secondary":   "LM Studio SECONDARY endpoint (runs in parallel to primary)",
    "lm-autodetect":           "LM Studio, auto-picks vision-capable model",
    "balanced-openai-compat":  "Local OpenAI-compatible server (llama.cpp, vLLM, koboldcpp, LocalAI, text-gen-webui)",
    "balanced-ollama":         "Ollama via native /api/chat (best structured-output path for Ollama)",
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
    # Keep the legacy singular in lockstep — including CLEARING it when the list
    # empties. Otherwise _active_vision_workers() falls back to a stale
    # PIPELINE_VISION_WORKER and the just-disabled worker snaps back to enabled.
    update_env("PIPELINE_VISION_WORKER", current[0] if current else "")
    return jsonify({"success": True, "active": current})


# GLOBAL settings only. Under the jobs model, everything that defines *what to
# scrape and how to judge it* (topic, keywords, scraper targets, scoring,
# captioning, gallery-dl URLs, local-import) lives in the per-job config and is
# edited via the Job Settings tab — NOT here. This list is credentials, model
# endpoints, vision-worker selection, throttle, storage roots, and global UX.
SETTINGS_KEYS: list[str] = [
    # Global UX / runtime
    "BLUR_NSFW_THUMBS",
    "PIPELINE_RECONCILE_SECONDS",
    # Wave-2 global feature toggles (scheduler/notifications/embeddings/prefilter/video)
    "PREFILTER_ENABLED",
    "PREFILTER_MIN_SCORE",
    "WEBHOOK_URL",
    "DESKTOP_NOTIFICATIONS",
    "EMBEDDINGS_ENABLED",
    "SCHEDULER_ENABLED",
    "VIDEO_CLASSIFY_ENABLED",
    # yt-dlp scraper (global toggle + targets; credentials/cookies are global)
    "YT_DLP_ENABLED",
    "YT_DLP_URLS",
    "YT_DLP_LIMIT",
    "YT_DLP_COOKIES",
    # Storage paths (global roots; per-slug subdirs derive from the job slug)
    "PIPELINE_BASE_DIR",
    "PIPELINE_QUEUE",
    "PIPELINE_SORTED",
    "LOG_DIR",
    # Vision provider credentials + endpoints
    "GROQ_API_KEY",
    "GROQ_API_KEYS",
    "GROQ_MODEL",
    "LMSTUDIO_PRIMARY_TIMEOUT",
    "LMSTUDIO_SECONDARY_TIMEOUT",
    "LMSTUDIO_UNLOAD_ON_STOP",
    "LMSTUDIO_IDLE_UNLOAD_MINUTES",
    # Generic OpenAI-compatible local server (llama.cpp / vLLM / koboldcpp / LocalAI / text-gen-webui)
    "OPENAI_COMPAT_URL",
    "OPENAI_COMPAT_MODEL",
    "OPENAI_COMPAT_API_KEY",
    "OPENAI_COMPAT_TIMEOUT",
    # Ollama (native /api/chat)
    "OLLAMA_URL",
    "OLLAMA_MODEL",
    "OLLAMA_TIMEOUT",
    # Cloud vision providers (key + model). Test Connection fetches the live
    # model catalogue into the dropdown; the worker reads <PROVIDER>_VISION_MODEL.
    "OPENAI_API_KEY",
    "OPENAI_VISION_MODEL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_VISION_MODEL",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_VISION_MODEL",
    "OPENROUTER_API_KEY",
    "OPENROUTER_VISION_MODEL",
    # Scraper credentials (the keys are global; per-job targets live in the job)
    "CIVITAI_API_KEY",
    "CIVITAI_API_RED_KEY",
    "TWITTER_COOKIES",
    "DISCORD_BOT_TOKEN",
    "DISCORD_AUTH_MODE",
    "REDDIT_CLIENT_ID",
    "REDDIT_CLIENT_SECRET",
    "REDDIT_USER_AGENT",
    "REDDIT_COOKIES",
]
SECRET_KEYS: set[str] = {
    "GROQ_API_KEY", "GROQ_API_KEYS",
    "CIVITAI_API_KEY", "CIVITAI_API_RED_KEY",
    "TWITTER_COOKIES", "DISCORD_BOT_TOKEN",
    "REDDIT_CLIENT_SECRET", "REDDIT_COOKIES",
    "OPENAI_COMPAT_API_KEY",
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY",
}
# Placeholder shown in the UI for a credential that IS set. The real value never
# leaves the server: GET returns this for any non-empty secret, and POST treats
# an incoming mask as "unchanged" so saving the form can't wipe a stored secret.
SECRET_MASK = "********"
PATH_KEYS: set[str] = {"PIPELINE_BASE_DIR", "PIPELINE_QUEUE", "PIPELINE_SORTED",
                       "LOG_DIR"}


@app.route("/api/settings")
def api_settings_get():
    # Never ship credentials to the browser in plaintext: a set secret returns
    # the mask; an unset secret returns "" (so the UI shows an empty field).
    out: dict[str, str] = {}
    for key in SETTINGS_KEYS:
        val = os.environ.get(key, "")
        out[key] = SECRET_MASK if (key in SECRET_KEYS and val) else val
    return jsonify(out)


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
        # A secret echoed back as the mask means "unchanged" — skip it so the
        # stored credential is preserved (a real typed value still updates it,
        # and an explicit empty string still clears it).
        if key in SECRET_KEYS and value == SECRET_MASK:
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

    for key, value in changes.items():
        update_env(key, value)

    return jsonify({"success": True, "applied": changes, "restart_required": True})


# Cloud vision providers whose model catalogue /api/vision/test fetches via
# vision_model_catalog. Value = the env var names a key is resolved from, in
# order (first non-empty wins). Mirrors the cloud workers' own env lookup.
_CLOUD_VISION_PROVIDERS: dict[str, tuple[str, ...]] = {
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "openrouter": ("OPENROUTER_API_KEY",),
}


def _cloud_provider_key(provider: str) -> str:
    """Resolve a cloud provider's API key from its env var(s), '' when unset."""
    for env_name in _CLOUD_VISION_PROVIDERS.get(provider, ()):  # first set wins
        val = (os.environ.get(env_name) or "").strip()
        if val:
            return val
    return ""


@app.route("/api/vision/test", methods=["POST"])
def api_vision_test():
    """Probe one vision endpoint and (for local providers) return its model list.

    Body: ``{provider, url?, api_key?}``. ``provider`` is one of groq, lmstudio,
    llamacpp, ollama, openai-compat, OR a cloud provider (openai / anthropic /
    gemini / openrouter). For local providers the caller-supplied ``url``/
    ``api_key`` are probed directly so a worker can be verified BEFORE it is
    saved; for cloud providers the key resolves from the body or the provider's
    env var and the live model catalogue is fetched via vision_model_catalog.
    Either way the discovered models are returned so the UI can fill the model
    dropdown. Returns ``{ok, message, models, latency_ms, provider}``.

    Security: this is a localhost single-user admin tool whose job is to test
    user-entered GPU endpoints — which live on private / LAN / Tailscale / link-
    local addresses. We therefore do NOT block private IPs (that would defeat the
    feature); the only guard is an http(s) scheme check.
    """
    import time as _t
    data = request.get_json() or {}
    provider = (data.get("provider") or "").strip().lower()
    body_url = (data.get("url") or "").strip()
    body_key = (data.get("api_key") or "").strip()
    # A masked sentinel means "the field still shows the stored secret, I didn't
    # type a new key" — treat it as empty so the key falls back to the env var
    # rather than literally sending "********" to the provider (mirrors the
    # scraper-test guard against SECRET_MASK).
    if body_key == SECRET_MASK:
        body_key = ""
    started = _t.time()

    def _done(ok: bool, message: str, status: int = 200, models: list | None = None) -> Any:
        return jsonify({
            "ok": ok, "message": message, "models": models or [],
            "latency_ms": int((_t.time() - started) * 1000), "provider": provider,
        }), status

    def _http(u: str) -> bool:
        return bool(re.match(r"^https?://", u, re.I))

    try:
        import requests
        if provider == "groq":
            key = body_key or os.environ.get("GROQ_API_KEY", "") or (
                os.environ.get("GROQ_API_KEYS", "").split(",")[0].strip())
            if not key:
                return _done(False, "no GROQ_API_KEY configured", 400)
            r = requests.get("https://api.groq.com/openai/v1/models",
                             headers={"Authorization": f"Bearer {key}"}, timeout=10,
                             allow_redirects=False)  # no 302-bounce to metadata
            if r.status_code == 401:
                return _done(False, "401 Unauthorized - key invalid")
            if r.status_code != 200:
                return _done(False, f"HTTP {r.status_code}: {r.text[:200]}")
            # Groq's /openai/v1/models is OpenAI-shaped — return the catalogue so
            # the Settings dropdown mirrors the other cloud providers.
            models = [m.get("id", "") for m in (r.json().get("data") or [])
                      if isinstance(m, dict) and m.get("id")]
            return _done(True, f"connected, {len(models)} model(s) available", models=models)

        if provider == "ollama":
            url = (body_url or os.environ.get("OLLAMA_URL", "") or "http://127.0.0.1:11434").rstrip("/")
            if not _http(url):
                return _done(False, "url must be an http(s) address", 400)
            headers = {"Authorization": f"Bearer {body_key}"} if body_key else {}
            # allow_redirects=False so a hostile endpoint can't 302-bounce the
            # probe to a metadata URL (169.254.169.254) the user didn't type.
            r = requests.get(f"{url}/api/tags", headers=headers, timeout=6,
                             allow_redirects=False)
            if r.status_code in (401, 403):
                return _done(False, f"auth rejected (HTTP {r.status_code}) - check the API key")
            if r.status_code != 200:
                return _done(False, f"HTTP {r.status_code}: {r.text[:200]}")
            models = [m.get("name", "") for m in (r.json().get("models") or [])
                      if isinstance(m, dict) and m.get("name")]
            return _done(True, f"connected, {len(models)} model(s) available", models=models)

        if provider in ("lmstudio", "llamacpp", "openai-compat", "openai_compat"):
            default_env = "LMSTUDIO_PRIMARY_URL" if provider == "lmstudio" else "OPENAI_COMPAT_URL"
            url = (body_url or os.environ.get(default_env, "")).rstrip("/")
            if not url:
                return _done(False, "no endpoint URL provided", 400)
            if not _http(url):
                return _done(False, "url must be an http(s) address", 400)
            key = body_key or os.environ.get("OPENAI_COMPAT_API_KEY", "")
            headers = {"Authorization": f"Bearer {key}"} if key else {}
            r = requests.get(f"{url}/v1/models", headers=headers, timeout=6,
                             allow_redirects=False)   # no 302-bounce to metadata
            if r.status_code in (401, 403):
                return _done(False, f"auth rejected (HTTP {r.status_code}) - check the API key")
            if r.status_code != 200:
                return _done(False, f"HTTP {r.status_code}: {r.text[:200]}")
            models = [m.get("id", "") for m in (r.json().get("data") or [])
                      if isinstance(m, dict) and m.get("id")]
            return _done(True, f"connected, {len(models)} model(s) loaded", models=models)

        if provider in _CLOUD_VISION_PROVIDERS:
            # Cloud providers (openai/anthropic/gemini/openrouter): fetch the live
            # model catalogue so the UI dropdown mirrors the local-fleet picker.
            # The key comes from the request body first (lets you test an unsaved
            # key) then the provider's env var(s). list_models is best-effort and
            # never raises; an empty list with a key set means auth/endpoint fail.
            key = body_key or _cloud_provider_key(provider)
            if not key:
                env_hint = " / ".join(_CLOUD_VISION_PROVIDERS[provider])
                return _done(False, f"no API key — set {env_hint} in Settings or type one above", 400)
            models = vision_model_catalog.list_models(
                provider, base_url=body_url or None, api_key=key)
            if models:
                return _done(True, f"connected, {len(models)} model(s) available", models=models)
            return _done(False, "no models returned — check the API key"
                                + (" / base URL" if body_url else ""))

        return _done(False, f"unknown provider: {provider!r}", 400)
    except (requests.RequestException, ValueError) as exc:
        # ValueError covers JSONDecodeError on a 200-but-non-JSON response.
        logger.warning("vision provider test connection error (%s): %s", provider, exc)
        return _done(False, "connection error — check the base URL and that the endpoint is reachable")


@app.route("/api/scrapers/test", methods=["POST"])
def api_scrapers_test():
    """Live connectivity + auth probe for one scraper.

    Body: ``{"scraper": "<name>", "settings": {<cred form values>}?, "config": {<targets>}?}``
    Credentials resolve from the live process env, overlaid with any NON-MASKED
    values the user has typed in the Settings form — so an unsaved key can be
    tested immediately, while a masked secret falls back to the stored value.
    Returns scraper_test.test_scraper's ``{ok, message, latency_ms, detail}``.
    """
    data = request.get_json() or {}
    name = (data.get("scraper") or "").strip()
    if name not in scraper_test.SUPPORTED:
        return jsonify({"ok": False, "message": f"unsupported scraper: {name}",
                        "latency_ms": None, "detail": ""}), 400
    env = dict(os.environ)
    form = data.get("settings")
    if isinstance(form, dict):
        for key, val in form.items():
            if key in SETTINGS_KEYS and isinstance(val, str) and val != SECRET_MASK:
                env[key] = val
    config = data.get("config") if isinstance(data.get("config"), dict) else None
    return jsonify(scraper_test.test_scraper(name, config=config, env=env))


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


# ── API: jobs (the jobs model) ────────────────────────────────────────────────
#
# job_config.py is the single source of truth for job state. These routes are a
# thin HTTP surface over it; they never write job JSON directly. Every <slug>
# path param is validated against JOB_SLUG_RE as defence in depth (job_config
# also guards on its own).

def _valid_slug_or_400(slug: str) -> bool:
    return bool(job_config.JOB_SLUG_RE.match(slug or ""))


def _preset_exists(name: str) -> bool:
    return name in job_config.list_presets().get("presets", {})


def _validate_inheritable_cfg(cfg: Any, *, partial: bool) -> tuple[dict | None, str]:
    """Validate a preset cfg (``partial=False``) or a job override map
    (``partial=True``). Returns (clean_cfg, error).

    Delegates to ``config_io.validate_inheritable_cfg`` — the single source of
    truth for the inheritable-config schema (topic_filters / scrapers /
    categories / category_rules / scoring / captioning / media / vision) — so the
    dashboard's job & preset save paths can never drift from config_io's
    importer/exporter validation. Adapts its raise-based API to the (clean, err)
    tuple the dashboard callers expect.
    """
    try:
        return config_io.validate_inheritable_cfg(cfg, partial=partial), ""
    except config_io.ValidationError as exc:
        return None, str(exc)


@app.route("/api/jobs")
def api_jobs_list():
    """List jobs with status, queue position, and queued/sorted counts, plus the
    active slug and queue order from the index."""
    idx = job_config.get_index()
    active = idx.get("active")
    queue = list(idx.get("queue") or [])
    jobs = job_config.list_jobs()
    return jsonify({
        "active": active,
        "queue": queue,
        "pipeline_running": pipeline_running(),
        "jobs": [_job_card(j, active=active, queue=queue) for j in jobs],
    })


@app.route("/api/jobs", methods=["POST"])
def api_jobs_create():
    data = request.get_json() or {}
    name = (data.get("name") or "").strip()
    base_on = (data.get("base_on") or "").strip() or None
    subject = data.get("subject")
    preset = (data.get("preset") or "").strip() or None
    if not name:
        return jsonify({"error": "name is required"}), 400
    if base_on and not _valid_slug_or_400(base_on):
        return jsonify({"error": "invalid base_on slug"}), 400
    if job_config.slugify(name) in RESERVED_JOB_SLUGS:
        return jsonify({"error": f"'{job_config.slugify(name)}' is a reserved slug"}), 400
    if preset is not None and not _preset_exists(preset):
        return jsonify({"error": f"unknown preset: {preset}"}), 400
    try:
        job = job_config.create_job(
            name,
            subject=(str(subject).strip() if subject is not None else None),
            preset=preset,
            base_on=base_on,
        )
    except ValueError as exc:
        # Duplicate slug / empty slug / unknown base_on all surface here.
        logger.warning("create_job rejected: %s", exc)
        return jsonify({"error": "could not create job (duplicate name or invalid base_on)"}), 400
    return jsonify({"success": True, "job": _job_detail(job)}), 201


def _job_detail(job: job_config.Job) -> dict[str, Any]:
    """v2 GET shape: identity + effective (resolved) config + the sparse override
    map + the preset library names. The UI needs effective values to display and
    the overrides map to decide chip-vs-reset per field."""
    lib = job_config.list_presets()
    return {
        "job": {
            "slug": job.slug, "name": job.name, "status": job.status,
            "subject": job.subject, "preset": job.preset,
        },
        "effective": job_config.effective_config(job),
        "overrides": job.overrides or {},
        "presets": sorted(lib.get("presets", {}).keys()),
        "default_preset": lib.get("default", "default"),
    }


@app.route("/api/jobs/<slug>")
def api_jobs_get(slug: str):
    if not _valid_slug_or_400(slug):
        return jsonify({"error": "invalid slug"}), 400
    job = job_config.get_job(slug)
    if job is None:
        return jsonify({"error": "job not found"}), 404
    return jsonify(_job_detail(job))


@app.route("/api/jobs/<slug>", methods=["PUT"])
def api_jobs_update(slug: str):
    """Persist {subject?, preset?, overrides?}. Validates the preset exists and
    runs the inheritable-config validators (categories, scoring) over any
    overrides present. Does NOT auto-activate — the UI applies on leave via
    POST /api/jobs/<slug>/activate, so the active job isn't re-projected and
    restarted on every keystroke."""
    if not _valid_slug_or_400(slug):
        return jsonify({"error": "invalid slug"}), 400
    existing = job_config.get_job(slug)
    if existing is None:
        return jsonify({"error": "job not found"}), 404
    data = request.get_json() or {}
    if not isinstance(data, dict):
        return jsonify({"error": "body must be an object"}), 400

    merged = existing.to_dict()
    if "subject" in data:
        merged["subject"] = str(data["subject"] or "").strip()
    if "preset" in data:
        preset = (str(data["preset"]) or "").strip()
        if not _preset_exists(preset):
            return jsonify({"error": f"unknown preset: {preset}"}), 400
        merged["preset"] = preset
    if "overrides" in data:
        ov = data["overrides"]
        if not isinstance(ov, dict):
            return jsonify({"error": "overrides must be an object"}), 400
        clean_ov, err = _validate_inheritable_cfg(ov, partial=True)
        if err:
            return jsonify({"error": err}), 400
        merged["overrides"] = clean_ov

    try:
        saved = job_config.save_job(job_config.Job.from_dict(merged))
    except (ValueError, TypeError) as exc:
        logger.warning("save_job rejected invalid config for %s: %s", slug, exc)
        return jsonify({"error": "invalid job config"}), 400

    return jsonify({"success": True, "job": _job_detail(saved)})


@app.route("/api/jobs/<slug>", methods=["DELETE"])
def api_jobs_delete(slug: str):
    if not _valid_slug_or_400(slug):
        return jsonify({"error": "invalid slug"}), 400
    if job_config.get_job(slug) is None:
        return jsonify({"error": "job not found"}), 404
    if slug == job_config.get_active_slug():
        return jsonify({"error": "cannot delete the active job"}), 409

    force = request.args.get("force") == "1"
    removed_dirs: list[str] = []
    if force:
        # Optionally drop the job's on-disk data — but ONLY paths that pass the
        # safe_inside() guard against the queue/sorted roots. A path that didn't
        # come back from the guard is never unlinked.
        for candidate in (_paths.queue_dir(slug), _paths.sorted_dir(slug)):
            roots = [_paths.queue_root(), _paths.sorted_root()]
            safe = safe_inside(str(candidate), roots)
            if safe is not None and safe.exists() and safe.is_dir():
                shutil.rmtree(safe, ignore_errors=True)
                removed_dirs.append(str(safe))

    job_config.delete_job(slug)
    return jsonify({"success": True, "slug": slug, "removed_dirs": removed_dirs})


@app.route("/api/jobs/<slug>/clone", methods=["POST"])
def api_jobs_clone(slug: str):
    if not _valid_slug_or_400(slug):
        return jsonify({"error": "invalid slug"}), 400
    if job_config.get_job(slug) is None:
        return jsonify({"error": "job not found"}), 404
    data = request.get_json() or {}
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "name is required"}), 400
    if job_config.slugify(name) in RESERVED_JOB_SLUGS:
        return jsonify({"error": f"'{job_config.slugify(name)}' is a reserved slug"}), 400
    try:
        job = job_config.create_job(name, base_on=slug)
    except ValueError as exc:
        logger.warning("clone job rejected (base_on=%s): %s", slug, exc)
        return jsonify({"error": "could not clone job (duplicate name)"}), 400
    return jsonify({"success": True, "job": job.to_dict()}), 201


@app.route("/api/jobs/<slug>/activate", methods=["POST"])
def api_jobs_activate(slug: str):
    if not _valid_slug_or_400(slug):
        return jsonify({"error": "invalid slug"}), 400
    if job_config.get_job(slug) is None:
        return jsonify({"error": "job not found"}), 404
    job_config.activate(slug)
    return jsonify({"success": True, "active": slug})


@app.route("/api/jobs/queue", methods=["POST"])
def api_jobs_set_queue():
    data = request.get_json() or {}
    order = data.get("order")
    if not isinstance(order, list) or any(not isinstance(s, str) for s in order):
        return jsonify({"error": "order must be a list of slugs"}), 400
    if any(not _valid_slug_or_400(s) for s in order):
        return jsonify({"error": "order contains an invalid slug"}), 400
    try:
        job_config.set_queue(order)
    except ValueError as exc:
        logger.warning("set_queue rejected: %s", exc)
        return jsonify({"error": "invalid queue order"}), 400
    return jsonify({"success": True, "queue": job_config.get_index().get("queue", [])})


@app.route("/api/jobs/<slug>/enqueue", methods=["POST"])
def api_jobs_enqueue(slug: str):
    if not _valid_slug_or_400(slug):
        return jsonify({"error": "invalid slug"}), 400
    if job_config.get_job(slug) is None:
        return jsonify({"error": "job not found"}), 404
    try:
        job_config.enqueue(slug)
    except ValueError as exc:
        logger.warning("enqueue rejected (%s): %s", slug, exc)
        return jsonify({"error": "could not enqueue job"}), 400
    return jsonify({"success": True, "queue": job_config.get_index().get("queue", [])})


@app.route("/api/jobs/<slug>/dequeue", methods=["POST"])
def api_jobs_dequeue(slug: str):
    if not _valid_slug_or_400(slug):
        return jsonify({"error": "invalid slug"}), 400
    job_config.dequeue(slug)
    return jsonify({"success": True, "queue": job_config.get_index().get("queue", [])})


@app.route("/api/jobs/advance", methods=["POST"])
@app.route("/api/jobs/<slug>/advance", methods=["POST"])
def api_jobs_advance(slug: str | None = None):
    """Promote the next queued job to active. The optional <slug> is accepted
    for symmetry but advance() always pops the queue head (sequential model)."""
    if slug is not None and not _valid_slug_or_400(slug):
        return jsonify({"error": "invalid slug"}), 400
    new_active = job_config.advance()
    return jsonify({"success": True, "active": new_active})


# ── API: presets (global inheritable-config library) ──────────────────────────
#
# Presets are the named defaults a job inherits from. job_config owns the
# library (data/jobs/_presets.json); these routes are a thin HTTP surface. Every
# <name> is validated against PRESET_NAME_RE at the boundary.

def _valid_preset_name(name: str) -> bool:
    return bool(job_config.PRESET_NAME_RE.match(name or ""))


@app.route("/api/presets")
def api_presets_list():
    lib = job_config.list_presets()
    return jsonify({
        "default": lib.get("default", "default"),
        "presets": sorted(lib.get("presets", {}).keys()),
        "builtins": sorted(job_config.builtin_preset_names()),
    })


@app.route("/api/presets", methods=["POST"])
def api_presets_create():
    data = request.get_json() or {}
    name = (data.get("name") or "").strip()
    base_on = (data.get("base_on") or "").strip() or None
    if not _valid_preset_name(name):
        return jsonify({"error": "invalid preset name (letters/digits/space/_/- , max 40)"}), 400
    if _preset_exists(name):
        return jsonify({"error": f"preset already exists: {name}"}), 400
    if base_on is not None and not _preset_exists(base_on):
        return jsonify({"error": f"unknown base_on preset: {base_on}"}), 400
    # Seed from base_on (or the default preset) so a new preset is a full cfg.
    seed = job_config.get_preset(base_on) if base_on else job_config.get_preset(
        job_config.default_preset_name())
    try:
        job_config.save_preset(name, seed)
    except ValueError as exc:
        logger.warning("create preset rejected (%s): %s", name, exc)
        return jsonify({"error": "could not save preset"}), 400
    return jsonify({"success": True, "name": name, "cfg": job_config.get_preset(name)}), 201


@app.route("/api/presets/<name>")
def api_presets_get(name: str):
    if not _valid_preset_name(name):
        return jsonify({"error": "invalid preset name"}), 400
    if not _preset_exists(name):
        return jsonify({"error": "preset not found"}), 404
    lib = job_config.list_presets()
    return jsonify({
        "name": name,
        "cfg": job_config.get_preset(name),
        "is_default": name == lib.get("default"),
        "referenced_by": sorted(j.slug for j in job_config.list_jobs() if j.preset == name),
    })


@app.route("/api/presets/<name>", methods=["PUT"])
def api_presets_update(name: str):
    if not _valid_preset_name(name):
        return jsonify({"error": "invalid preset name"}), 400
    if not _preset_exists(name):
        return jsonify({"error": "preset not found"}), 404
    data = request.get_json() or {}
    cfg = data.get("cfg", data)          # accept {cfg:{...}} or a bare cfg body
    clean, err = _validate_inheritable_cfg(cfg, partial=False)
    if err:
        return jsonify({"error": err}), 400
    # Deep-merge onto the existing TOTAL preset so a partial PUT keeps untouched
    # nested leaves (a shallow update would wipe e.g. scrapers.local_imports when
    # the body includes only scrapers.enabled). Lists still replace wholesale.
    merged = job_config._deep_merge(job_config.get_preset(name), clean)
    try:
        job_config.save_preset(name, merged)
    except ValueError as exc:
        logger.warning("update preset rejected (%s): %s", name, exc)
        return jsonify({"error": "could not save preset"}), 400
    return jsonify({"success": True, "name": name, "cfg": job_config.get_preset(name)})


@app.route("/api/presets/<name>", methods=["DELETE"])
def api_presets_delete(name: str):
    if not _valid_preset_name(name):
        return jsonify({"error": "invalid preset name"}), 400
    if not _preset_exists(name):
        return jsonify({"error": "preset not found"}), 404
    try:
        job_config.delete_preset(name)
    except ValueError as exc:
        # default / referenced-by-a-job → 409 conflict.
        logger.warning("delete preset rejected (%s): %s", name, exc)
        return jsonify({"error": "cannot delete this preset (it is the default or in use by a job)"}), 409
    return jsonify({"success": True, "name": name})


@app.route("/api/presets/<name>/clone", methods=["POST"])
def api_presets_clone(name: str):
    if not _valid_preset_name(name):
        return jsonify({"error": "invalid preset name"}), 400
    if not _preset_exists(name):
        return jsonify({"error": "preset not found"}), 404
    data = request.get_json() or {}
    new_name = (data.get("name") or "").strip()
    if not _valid_preset_name(new_name):
        return jsonify({"error": "invalid new preset name"}), 400
    if _preset_exists(new_name):
        return jsonify({"error": f"preset already exists: {new_name}"}), 400
    try:
        job_config.save_preset(new_name, job_config.get_preset(name))
    except ValueError as exc:
        logger.warning("clone preset rejected (%s -> %s): %s", name, new_name, exc)
        return jsonify({"error": "could not clone preset"}), 400
    return jsonify({"success": True, "name": new_name, "cfg": job_config.get_preset(new_name)}), 201


@app.route("/api/presets/default", methods=["POST"])
def api_presets_set_default():
    data = request.get_json() or {}
    name = (data.get("name") or "").strip()
    if not _valid_preset_name(name):
        return jsonify({"error": "invalid preset name"}), 400
    try:
        job_config.set_default_preset(name)
    except ValueError as exc:
        logger.warning("set_default_preset rejected (%s): %s", name, exc)
        return jsonify({"error": "could not set default preset"}), 400
    return jsonify({"success": True, "default": name})


@app.route("/api/presets/<name>/reset", methods=["POST"])
def api_presets_reset(name: str):
    """Restore a shipped (builtin) preset to its library definition."""
    if not _valid_preset_name(name):
        return jsonify({"error": "invalid preset name"}), 400
    try:
        cfg = job_config.reset_preset_to_builtin(name)
    except ValueError as exc:
        logger.warning("reset_preset_to_builtin rejected (%s): %s", name, exc)
        return jsonify({"error": "could not reset preset (not a shipped preset)"}), 400
    return jsonify({"success": True, "name": name, "cfg": cfg})


# ── API: export / import (presets + full config backup) ──────────────────────

EXPORT_VERSION = 1


@app.route("/api/presets/<name>/export")
def api_presets_export(name: str):
    """Download one preset as a portable, versioned JSON envelope."""
    if not _valid_preset_name(name):
        return jsonify({"error": "invalid preset name"}), 400
    if not _preset_exists(name):
        return jsonify({"error": "preset not found"}), 404
    payload = {"kind": "cull.preset", "version": EXPORT_VERSION,
               "name": name, "cfg": job_config.get_preset(name)}
    resp = jsonify(payload)
    safe_name = re.sub(r"[^A-Za-z0-9_-]", "_", name)  # header-injection defence
    resp.headers["Content-Disposition"] = f'attachment; filename="{safe_name}.preset.json"'
    return resp


def _import_one_preset(name: str, cfg: Any, *, overwrite: bool) -> tuple[bool, str]:
    """Validate + save a single imported preset. Returns (ok, reason)."""
    if not _valid_preset_name(name):
        return False, "bad name"
    clean, err = _validate_inheritable_cfg(cfg, partial=False)
    if err:
        return False, err
    if _preset_exists(name) and not overwrite:
        return False, "exists"
    merged = job_config._deep_merge(job_config._default_preset_cfg(), clean)
    job_config.save_preset(name, merged)
    return True, "ok"


@app.route("/api/presets/import", methods=["POST"])
def api_presets_import():
    """Import one preset from {name, cfg} (or a cull.preset envelope)."""
    data = request.get_json() or {}
    name = (data.get("name") or "").strip()
    overwrite = bool(data.get("overwrite"))
    if not _valid_preset_name(name):
        return jsonify({"error": "invalid preset name (letters/digits/space/_/- , max 40)"}), 400
    if _preset_exists(name) and not overwrite:
        return jsonify({"error": f"preset already exists: {name}", "exists": True}), 409
    ok, reason = _import_one_preset(name, data.get("cfg"), overwrite=overwrite)
    if not ok:
        return jsonify({"error": f"invalid preset: {reason}"}), 400
    return jsonify({"success": True, "name": name})


@app.route("/api/config/export")
def api_config_export():
    """Download a full backup: every job, the preset library, and the queue index."""
    payload = {
        "kind": "cull.config", "version": EXPORT_VERSION,
        "jobs": [j.to_dict() for j in job_config.list_jobs()],
        "presets": job_config.list_presets(),
        "index": job_config.get_index(),
    }
    resp = jsonify(payload)
    resp.headers["Content-Disposition"] = 'attachment; filename="cull-config.json"'
    return resp


@app.route("/api/config/import", methods=["POST"])
def api_config_import():
    """Restore a cull.config bundle or a single cull.job export.

    Additive by default (existing presets/jobs are skipped); pass overwrite=true
    to replace. Everything is validated; nothing can escape the jobs dir."""
    data = request.get_json() or {}
    kind = data.get("kind")
    if kind not in ("cull.config", "cull.job"):
        return jsonify({"error": "not a cull config/job export (bad 'kind')"}), 400
    overwrite = bool(data.get("overwrite"))
    summary: dict[str, Any] = {"presets_added": 0, "jobs_added": 0, "skipped": []}

    def _do_preset(pname: Any, pcfg: Any) -> None:
        ok, reason = _import_one_preset(str(pname or ""), pcfg, overwrite=overwrite)
        if ok:
            summary["presets_added"] += 1
        else:
            summary["skipped"].append(f"preset {pname!r} ({reason})")

    def _do_job(jd: Any) -> None:
        if not isinstance(jd, dict):
            summary["skipped"].append("job (not an object)")
            return
        try:
            job = job_config.Job.from_dict(jd)
        except Exception:
            summary["skipped"].append("job (unreadable)")
            return
        if not job_config.JOB_SLUG_RE.match(job.slug or ""):
            summary["skipped"].append(f"job {job.slug!r} (bad slug)")
            return
        if job_config.get_job(job.slug) is not None and not overwrite:
            summary["skipped"].append(f"job {job.slug!r} (exists)")
            return
        # Validate overrides like the PUT path does, and reset runtime state.
        if job.overrides:
            _, ov_err = _validate_inheritable_cfg(job.overrides, partial=True)
            if ov_err:
                summary["skipped"].append(f"job {job.slug!r} (bad overrides: {ov_err})")
                return
        job_config.save_job(job.with_updates(status="idle"))
        summary["jobs_added"] += 1

    if kind == "cull.job":
        pre = data.get("preset")
        if isinstance(pre, dict) and pre.get("name"):
            _do_preset(pre.get("name"), pre.get("cfg"))
        if isinstance(data.get("job"), dict):
            _do_job(data.get("job"))
    else:  # cull.config
        presets = ((data.get("presets") or {}).get("presets")) or {}
        if isinstance(presets, dict):
            for pname, pcfg in presets.items():
                _do_preset(pname, pcfg)
        for jd in (data.get("jobs") or []):
            _do_job(jd)

    return jsonify({"success": True, **summary})


# ── API: queue / thumbnails / prompts ─────────────────────────────────────────


def _list_recent_queue_scoped(slug: str | None, limit: int) -> list[index_store.IndexedImage]:
    """Newest queued images, optionally scoped to one slug. When slug is None
    (pre-migration), behaves like index_store.list_recent_queue."""
    if slug is None:
        return index_store.list_recent_queue(limit=limit)
    with index_store.with_conn() as conn:
        cur = conn.execute(
            "SELECT * FROM images WHERE status = 'queue' AND topic_slug = ? "
            "ORDER BY mtime DESC LIMIT ?",
            (slug, int(limit)),
        )
        return [index_store.IndexedImage.from_row(row) for row in cur.fetchall()]


def _list_recent_sorted_scoped(slug: str | None, limit: int) -> list[index_store.IndexedImage]:
    """Newest classified images, optionally scoped to one slug."""
    if slug is None:
        return index_store.list_recent_sorted(limit=limit)
    with index_store.with_conn() as conn:
        cur = conn.execute(
            "SELECT * FROM images WHERE status = 'sorted' AND topic_slug = ? "
            "ORDER BY mtime DESC LIMIT ?",
            (slug, int(limit)),
        )
        return [index_store.IndexedImage.from_row(row) for row in cur.fetchall()]


def _list_sorted_scoped(
    slug: str | None,
    *,
    sources: list[str] | None = None,
    categories: list[str] | None = None,
    sort: str = "newest",
    limit: int = 200,
) -> list[index_store.IndexedImage]:
    """Sorted listing with the same filters as index_store.list_sorted but with
    an optional topic_slug scope. Falls back to the library helper when slug is
    None so pre-migration installs behave exactly as before."""
    if slug is None:
        items, _total = index_store.list_sorted(
            sources=sources, categories=categories, sort=sort, limit=limit, offset=0,
        )
        return items
    sort_clause = {
        "newest": "mtime DESC", "oldest": "mtime ASC",
        "ovr": "ovr DESC NULLS LAST, mtime DESC",
        "rel": "rel DESC NULLS LAST, mtime DESC",
        "quality": "quality DESC NULLS LAST, mtime DESC",
    }.get(sort, "mtime DESC")
    where = ["status = 'sorted'", "topic_slug = ?"]
    params: list[Any] = [slug]
    if sources:
        where.append("source IN (" + ",".join("?" * len(sources)) + ")")
        params.extend(sources)
    if categories:
        where.append("category IN (" + ",".join("?" * len(categories)) + ")")
        params.extend(categories)
    with index_store.with_conn() as conn:
        cur = conn.execute(
            f"SELECT * FROM images WHERE {' AND '.join(where)} "
            f"ORDER BY {sort_clause} LIMIT ?",
            [*params, int(limit)],
        )
        return [index_store.IndexedImage.from_row(row) for row in cur.fetchall()]


@app.route("/api/queue/files")
def api_queue_files():
    """Newest queued images, served from the SQLite index.

    The old version globbed the entire queue tree on every request. Index-
    backed listing is sub-millisecond regardless of queue depth. Scoped to
    ?job=<slug> (default active).
    """
    limit = int(request.args.get("limit", 60))
    slug = _resolve_job_slug()
    results: list[dict[str, Any]] = []
    for item in _list_recent_queue_scoped(slug, limit):
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


def _read_vision_prediction(image_path: Path) -> tuple[str, dict[str, Any]]:
    """Best-effort ``(predicted_category, scores)`` for an image being re-labelled.

    Reads the sibling ``<stem>.vision.json`` audit record (the full classifier
    result, which carries both ``category`` and the score keys). Falls back to
    the image's current parent-folder name for the predicted category when no
    audit record exists or it is unparseable. Never raises — a missing/garbage
    record just yields ``("", {})`` for the scores so the caller can still record
    the corrected target.
    """
    predicted = image_path.parent.name
    scores: dict[str, Any] = {}
    meta_path = image_path.with_name(image_path.stem + ".vision.json")
    try:
        if meta_path.exists():
            record = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(record, dict):
                scores = record
                cat = str(record.get("category") or "").strip()
                if cat:
                    predicted = cat
    except (OSError, json.JSONDecodeError, ValueError):
        pass
    return predicted, scores


def _record_relabel_correction(
    stem: str, predicted: str, target_category: str, scores: dict[str, Any],
) -> None:
    """Feed a manual queue re-label into the active-learning correction log.

    ``predicted`` / ``scores`` are snapshotted from the image's ORIGINAL location
    (before the move relocated it). Lazily imports ``active_learning`` and records
    ``(predicted → target)`` keyed by the image stem so corrections accumulate for
    the prompt's few-shot block. Wrapped so it can NEVER break the move that
    already succeeded — and ``active_learning`` is best-effort, including against
    a ``SystemExit``-subclass error (so we catch ``BaseException``).
    """
    if not predicted or predicted == target_category:
        return  # not a correction — same bucket (or unknown source bucket)
    try:
        import active_learning
        slug = _resolve_job_slug() or "default"
        active_learning.record_correction(
            slug, stem, predicted, target_category, scores,
        )
    except BaseException as exc:  # noqa: BLE001 - learning is best-effort, never fatal
        logger.debug("record_correction skipped for %s: %s", stem, exc)


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
        # Re-validate the destination through safe_inside so a traversal target
        # (e.g. "../../etc") can never resolve outside the queue root. We resolve
        # BEFORE any mkdir/move so a bad target never creates dirs or relocates a
        # file outside the queue; all subsequent file ops use the returned Path.
        dest = safe_inside(str(PIPELINE_QUEUE / target), [PIPELINE_QUEUE])
        if dest is None:
            return jsonify({"error": "target outside queue"}), 400
        dest.mkdir(parents=True, exist_ok=True)
        # Snapshot the original prediction + scores BEFORE the move relocates the
        # sidecar (the predicted bucket is the source folder / the .vision.json's
        # category — both gone once the image lands under <target>).
        predicted, scores = _read_vision_prediction(path)
        for sibling in path.parent.glob(f"{path.stem}.*"):
            shutil.move(str(sibling), str(dest / sibling.name))
        # After a SUCCESSFUL move, log the re-label for active learning.
        # Best-effort — never breaks the move.
        _record_relabel_correction(path.stem, predicted, target, scores)
        return jsonify({"success": True, "action": "move", "target": str(dest)})
    return jsonify({"error": "unknown action"}), 400


# ── API: gallery culling (recategorise a SORTED image) ────────────────────────
#
# The gallery shows images that have already been *sorted* into category folders
# under PIPELINE_SORTED (layout: <sorted>/[<slug>/]<category>/<source>/<file>).
# api_queue_action only moves within the *queue*, so the keyboard-driven cull
# needs its own move path for the sorted tree. It mirrors the queue mover's
# sibling-sweep but recomputes the destination by swapping the category folder,
# and returns the resolved {from_category, to_category, path} so the client can
# build a reversible undo entry. safe_inside guards both the source and the
# (recomputed) destination — neither may escape PIPELINE_SORTED.

def _recategorise_dest_dir(image_path: Path, target_category: str) -> Path | None:
    """Destination ``<category>/<source>`` folder for recategorising a sorted image.

    The category folder is the image's grandparent (``…/<category>/<source>/<file>``)
    and the source folder is its parent. We swap only the category segment, keeping
    the same source sub-folder, so a Civitai image stays under ``civitai/``. Returns
    None when the image isn't nested under a ``<category>/<source>`` pair (so we never
    fabricate a bogus destination from a too-shallow path).
    """
    source_folder = image_path.parent
    category_folder = source_folder.parent
    # category_folder must be a real directory below the sorted root, not the
    # root itself or a drive anchor — otherwise the layout is unexpected.
    if category_folder == source_folder or category_folder.parent == category_folder:
        return None
    return category_folder.parent / target_category / source_folder.name


@app.route("/api/gallery/recategorize", methods=["POST"])
def api_gallery_recategorize():
    """Move one sorted image (and its sidecars) into a different category folder.

    Body: ``{path, target, job?}``. ``path`` must resolve inside PIPELINE_SORTED
    via ``safe_inside``; ``target`` must be a known category (keep bucket or a
    system-terminal bucket such as DISCARD) so a typo can't spray arbitrary
    folders across the sorted tree. The recomputed destination is re-checked with
    ``safe_inside`` before any file is touched. Returns the envelope plus
    ``{from_category, to_category, path}`` for the client's undo stack.
    """
    data = request.get_json() or {}
    path = safe_inside(data.get("path", ""), [PIPELINE_SORTED])
    target = (data.get("target") or "").strip()
    if path is None or not path.exists():
        return jsonify({"success": False, "error": "path outside sorted roots or missing"}), 400
    if not target:
        return jsonify({"success": False, "error": "target category required"}), 400
    # Validate against the active taxonomy (keep buckets + DISCARD/CORRUPT). This
    # is the single source of truth — never inline a category list here.
    allowed = set(_categories_mod.get_all_categories())
    if target not in allowed:
        return jsonify({"success": False, "error": f"unknown target category '{target}'"}), 400

    from_category = path.parent.parent.name
    if from_category == target:
        # No-op move — report success without churning the filesystem so the
        # client can skip pushing an undo entry that reverses nothing.
        return jsonify({"success": True, "data": {
            "path": str(path), "from_category": from_category, "to_category": target,
            "moved": False,
        }, "error": None})

    raw_dest_dir = _recategorise_dest_dir(path, target)
    # safe_inside both validates AND returns the resolved destination; every file
    # op below uses that returned Path (never the raw recomputed one) so a
    # destination escaping the sorted root can never be created or written to.
    dest_dir = safe_inside(str(raw_dest_dir), [PIPELINE_SORTED]) if raw_dest_dir is not None else None
    if dest_dir is None:
        return jsonify({"success": False, "error": "could not resolve a destination inside sorted"}), 400

    predicted, scores = _read_vision_prediction(path)
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        new_image_path = dest_dir / path.name
        for sibling in path.parent.glob(f"{path.stem}.*"):
            shutil.move(str(sibling), str(dest_dir / sibling.name))
    except OSError as exc:
        logger.warning("recategorize move failed: %s", exc)
        return jsonify({"success": False, "error": "move failed"}), 500

    # Best-effort active-learning correction (same contract as the queue mover).
    _record_relabel_correction(path.stem, predicted, target, scores)
    # Invalidate the sorted cache so the next gallery/stats read reflects the move.
    with _sorted_cache_lock:
        _sorted_cache["ts"] = 0.0
        _sorted_cache["signature"] = None
    return jsonify({"success": True, "data": {
        "path": str(new_image_path), "from_category": from_category,
        "to_category": target, "moved": True,
    }, "error": None})


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
    # max_age=0 forces a conditional revalidation on every request: the disk
    # cache file is content-hashed (path + mtime + size — see thumb_cache.py),
    # but the URL the browser sees is keyed only on the source image path, so a
    # long max_age would serve stale thumbnails when an image at the same path
    # gets replaced. With conditional=True, send_file emits Last-Modified +
    # ETag and a matching client gets a small 304 instead of a fresh JPEG.
    return send_file(
        cache_file,
        mimetype="image/jpeg",
        max_age=0,
        conditional=True,
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
    slug = _resolve_job_slug()

    items = _list_sorted_scoped(
        slug,
        sources=[source] if source else None,
        categories=[category] if category else None,
        sort="newest",
        limit=limit,
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
    """Newest classified items, served from the SQLite index. Scoped to
    ?job=<slug> (default active)."""
    limit = int(request.args.get("limit", 12))
    slug = _resolve_job_slug()
    results: list[dict[str, Any]] = []
    for item in _list_recent_sorted_scoped(slug, limit):
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
    topic_slug: str | None = None


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
        topic_slug=row.topic_slug,
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


def _get_sorted_items_scoped(slug: str | None, force: bool = False) -> list[_SortedItem]:
    """Sorted items filtered to one job slug. When slug is None (pre-migration)
    returns the full set so behaviour is unchanged. Reuses the materialised
    cache from _get_sorted_items so stats/gallery stay fast."""
    items = _get_sorted_items(force=force)
    if slug is None:
        return items
    return [it for it in items if (it.topic_slug or "default") == slug]


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
    Scoped to ?job=<slug> (default active).
    """
    items = _get_sorted_items_scoped(_resolve_job_slug())
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
    items = _get_sorted_items_scoped(_resolve_job_slug())
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
    whatever the user is actively browsing. Scoped to ?job=<slug>."""
    items = _get_sorted_items_scoped(_resolve_job_slug())
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
        logger.warning("prompt write failed: %s", exc)
        return jsonify({"error": "write failed"}), 500
    # Invalidate the scan cache so the next /api/stats sees the new prompt.
    with _sorted_cache_lock:
        _sorted_cache["ts"] = 0.0
        _sorted_cache["signature"] = None
    return jsonify({"success": True, "path": str(txt_path), "bytes": len(text)})


# ── API: inline caption editor (training .txt next to an image) ───────────────
#
# A focused, envelope-returning surface for the gallery's inline caption editor.
# It is intentionally separate from /api/prompt (which is the modal's free-form
# prompt editor and predates the envelope convention): the caption editor wants a
# small, predictable {success,data,error} shape, a hard length cap, and a GET that
# returns the current sibling .txt. Both verbs are safe_inside-guarded against the
# queue + sorted roots — the .txt is always written next to the image stem, never
# at an attacker-chosen location.

_MAX_CAPTION_CHARS: int = 20_000


@app.route("/api/caption", methods=["GET"])
def api_caption_get():
    """Return the current ``<stem>.txt`` caption for an image (or empty).

    Query: ``?path=<image>``. The path must resolve inside the queue/sorted
    roots; a traversal or out-of-roots path is a clean 400 (never a file read
    outside the pipeline). Envelope: ``{success, data:{path,text}, error}``.
    """
    raw = request.args.get("path", "")
    image_path = safe_inside(raw, [PIPELINE_QUEUE, PIPELINE_SORTED])
    if image_path is None:
        return jsonify({"success": False, "data": None,
                        "error": "path outside pipeline roots"}), 400
    txt_path = image_path.with_suffix(".txt")
    text = ""
    if txt_path.exists():
        try:
            text = txt_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return _err("read failed", exc, 500)
    return jsonify({"success": True,
                    "data": {"path": str(txt_path), "text": text}, "error": None})


@app.route("/api/caption", methods=["POST"])
def api_caption_save():
    """Write ``<stem>.txt`` next to an image.

    Body: ``{path, text, job?}``. ``path`` must resolve inside the queue/sorted
    roots via ``safe_inside`` (traversal / out-of-roots → 400). ``text`` is
    required and length-capped at ``_MAX_CAPTION_CHARS`` so a runaway client can't
    write an unbounded file. Envelope: ``{success, data:{path,bytes}, error}``.
    """
    data = request.get_json() or {}
    raw = data.get("path") or ""
    text = data.get("text")
    if not isinstance(text, str):
        return jsonify({"success": False, "data": None,
                        "error": "missing or non-string 'text' field"}), 400
    if len(text) > _MAX_CAPTION_CHARS:
        return jsonify({"success": False, "data": None,
                        "error": f"caption exceeds {_MAX_CAPTION_CHARS} chars"}), 400
    image_path = safe_inside(raw, [PIPELINE_QUEUE, PIPELINE_SORTED])
    if image_path is None:
        return jsonify({"success": False, "data": None,
                        "error": "path outside pipeline roots"}), 400
    txt_path = image_path.with_suffix(".txt")
    try:
        txt_path.write_text(text, encoding="utf-8")
    except OSError as exc:
        return _err("write failed", exc, 500)
    # Invalidate the scan cache so the next /api/stats sees the new caption.
    with _sorted_cache_lock:
        _sorted_cache["ts"] = 0.0
        _sorted_cache["signature"] = None
    return jsonify({"success": True,
                    "data": {"path": str(txt_path), "bytes": len(text)}, "error": None})


# ── API: zip download of filtered gallery ─────────────────────────────────────

@app.route("/api/gallery/download.zip")
def api_gallery_download():
    items = _get_sorted_items_scoped(_resolve_job_slug())
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


# ── API: near-duplicate detection (perceptual hash) ───────────────────────────

@app.route("/api/duplicates")
def api_duplicates():
    """Group near-duplicate sorted images by perceptual hash, scoped to ?job=.

    Query: ``threshold`` (Hamming distance, default 10) and ``job``. Returns the
    envelope ``{success, data: {groups, threshold}, error}`` where each group is
    a list of ``{path, thumb_url, category}`` for images that are within
    ``threshold`` bits of one another. On-demand only — never run on the status
    poll. Every path is re-validated with ``safe_inside`` so a stale index row
    pointing outside the pipeline roots can never leak a thumbnail URL.
    """
    slug = _resolve_job_slug()
    threshold = _parse_int(request.args.get("threshold"), default=phash_dedup.DEFAULT_THRESHOLD)
    threshold = max(0, min(64, threshold))
    try:
        raw_groups = phash_dedup.find_near_duplicates(threshold=threshold, slug=slug)
    except Exception as exc:  # noqa: BLE001 - a scan failure must not 500 the UI
        return _err("duplicate scan failed", exc, 500)

    # Map index keys (image paths) to display cards via the cached sorted items,
    # so we reuse the category/thumb info without a second filesystem walk.
    by_path = {str(it.image_path): it for it in _get_sorted_items_scoped(slug)}
    groups: list[list[dict[str, Any]]] = []
    for keys in raw_groups:
        members: list[dict[str, Any]] = []
        for key in keys:
            safe = safe_inside(key, [PIPELINE_QUEUE, PIPELINE_SORTED])
            if safe is None:
                continue
            item = by_path.get(key)
            members.append({
                "path": key,
                # URL-encode so a path with spaces / ? / # / + can't break the
                # query string (api_thumbnail re-validates with safe_inside).
                "thumb_url": f"/api/thumbnail?path={_urlquote(key, safe='')}",
                "category": item.category if item is not None else "",
            })
        if len(members) > 1:
            groups.append(members)
    return jsonify({"success": True,
                    "data": {"groups": groups, "threshold": threshold},
                    "error": None})


@app.route("/api/duplicates/delete", methods=["POST"])
def api_duplicates_delete():
    """Delete one image (and its sidecars) from the duplicates view.

    Body: ``{path}``. The path MUST resolve inside the queue/sorted roots via
    ``safe_inside`` — the queue-only delete endpoint can't touch sorted images,
    so this mirrors its sibling-sweep for the sorted tree behind the same guard.
    Returns ``{success, error}``.
    """
    data = request.get_json() or {}
    path = safe_inside(data.get("path", ""), [PIPELINE_QUEUE, PIPELINE_SORTED])
    if path is None or not path.exists():
        return jsonify({"success": False, "error": "path outside pipeline roots or missing"}), 400
    try:
        for sibling in path.parent.glob(f"{path.stem}.*"):
            sibling.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("duplicate delete failed: %s", exc)
        return jsonify({"success": False, "error": "delete failed"}), 500
    # Invalidate the sorted cache so the next scan/gallery doesn't show the dupe.
    with _sorted_cache_lock:
        _sorted_cache["ts"] = 0.0
        _sorted_cache["signature"] = None
    return jsonify({"success": True, "error": None})


# ── API: dataset export (training-ready layouts) ──────────────────────────────

@app.route("/api/export/dataset", methods=["POST"])
def api_export_dataset():
    """Export the active (or ?job=) sorted library into a training-ready layout.

    Body: ``{profile, trigger_word?, train_val_split?, resolution_bucketing?,
    repeats?, video_bucketing?, categories?}``. Validates ``profile`` against
    ``export_profiles.PROFILES``, writes under
    ``<base_dir>/exports/<slug>/<profile>_<stamp>`` (never touching the source),
    and returns ``{success, data: <summary>, error}``.
    """
    slug = _resolve_job_slug(default=None) or job_config.get_active_slug()
    if not slug:
        return jsonify({"success": False, "data": None,
                        "error": "no active job — activate one first"}), 409
    data = request.get_json() or {}
    profile = (data.get("profile") or "").strip()
    if profile not in export_profiles.PROFILES:
        return jsonify({"success": False, "data": None,
                        "error": f"profile must be one of {list(export_profiles.PROFILES)}"}), 400

    # Collect the allowed options, validating the scalar shapes here so a
    # malformed value (e.g. a dict where an int is expected) returns a clean 400
    # rather than a TypeError that would fall through to the 500 branch.
    opts: dict[str, Any] = {}
    for key in ("trigger_word", "train_val_split", "resolution_bucketing",
                "repeats", "video_bucketing", "categories"):
        if key not in data or data[key] in (None, ""):
            continue
        val = data[key]
        if key in ("train_val_split", "repeats") and not isinstance(val, (int, float)):
            return jsonify({"success": False, "data": None,
                            "error": f"{key} must be a number"}), 400
        if key in ("trigger_word", "video_bucketing") and not isinstance(val, str):
            return jsonify({"success": False, "data": None,
                            "error": f"{key} must be a string"}), 400
        # bool is a subclass of int, so check it explicitly (and before any int
        # coercion) so a stray dict/str for the bucketing flag is a clean 400.
        if key == "resolution_bucketing" and not isinstance(val, bool):
            return jsonify({"success": False, "data": None,
                            "error": "resolution_bucketing must be a boolean"}), 400
        if key == "categories" and not isinstance(val, list):
            return jsonify({"success": False, "data": None,
                            "error": "categories must be a list"}), 400
        opts[key] = val

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = _paths.base_dir() / "exports" / slug / f"{profile}_{stamp}"
    try:
        summary = export_profiles.export_dataset(slug, profile, out_dir, **opts)
    except ValueError as exc:  # bad option / unknown profile
        return _err("invalid export option", exc, 400)
    except Exception as exc:  # noqa: BLE001 - export I/O failure shouldn't 500-crash
        return _err("dataset export failed", exc, 500)
    return jsonify({"success": True, "data": summary, "error": None})


# ── API: push a curated dataset to HuggingFace ────────────────────────────────

_HF_REPO_RE = re.compile(r"^[\w.-]+/[\w.-]+$")

# Background HF-export jobs, keyed by a generated id. Each value is
# ``{state: running|done|error, message, data, repo_id, slug}``. The upload runs
# off the request thread so a large dataset push doesn't block the dashboard; the
# UI polls the status endpoint below. Guarded by a lock for cross-thread access.
_HF_JOBS: dict[str, dict[str, Any]] = {}
_HF_JOBS_LOCK = threading.Lock()
# How long the POST waits for the worker before handing back a 202 + id. A
# mocked/instant push (tests) finishes inside this window and returns its terminal
# result synchronously; a real upload exceeds it and switches to async polling.
_HF_SYNC_WAIT_SECONDS = 0.5
# Cap on retained finished jobs so the dict can't grow without bound.
_HF_JOBS_MAX = 64


def _hf_error_status(exc: Exception) -> tuple[str, int]:
    """Map a ``push_to_hf`` exception to ``(message, http_status)``.

    Mirrors the original synchronous mapping so the terminal envelope (and the
    status endpoint) report the same actionable messages.
    """
    if isinstance(exc, credentials.MissingCredentialError):
        return "set HF_TOKEN in the environment to push to HuggingFace", 400
    if isinstance(exc, ValueError):            # nothing to export
        return str(exc), 400
    if isinstance(exc, RuntimeError):          # huggingface_hub not installed
        return str(exc), 400
    logger.warning("HF push failed: %s", exc)  # upload / network failure
    return str(exc), 500


def _hf_set(job_id: str, **fields: Any) -> None:
    with _HF_JOBS_LOCK:
        entry = _HF_JOBS.setdefault(job_id, {})
        entry.update(fields)
        # Evict the oldest finished jobs if we're over the cap (insertion order).
        if len(_HF_JOBS) > _HF_JOBS_MAX:
            for stale in [k for k, v in _HF_JOBS.items()
                          if v.get("state") in ("done", "error")][: len(_HF_JOBS) - _HF_JOBS_MAX]:
                _HF_JOBS.pop(stale, None)


def _hf_get(job_id: str) -> dict[str, Any] | None:
    with _HF_JOBS_LOCK:
        entry = _HF_JOBS.get(job_id)
        return dict(entry) if entry is not None else None


def _run_hf_push(job_id: str, *, slug: str, repo_id: str, private: bool,
                 include_video: bool, categories_arg: list | None) -> None:
    """Thread body: run the push, recording terminal state into ``_HF_JOBS``.

    Catches ``BaseException`` because ``credentials.MissingCredentialError`` is a
    ``SystemExit`` subclass (so it would otherwise escape a plain ``except
    Exception`` and kill the worker thread silently).
    """
    try:
        result = hf_export.push_to_hf(
            slug=slug, repo_id=repo_id, private=private,
            include_video=include_video, categories=categories_arg)
    except BaseException as exc:  # noqa: BLE001 - mapped to a friendly status below
        message, status = _hf_error_status(exc)
        _hf_set(job_id, state="error", message=message, status=status, data=None)
        return
    _hf_set(job_id, state="done", message="upload complete", status=200, data=result)


@app.route("/api/jobs/<slug>/export/huggingface", methods=["POST"])
def api_export_huggingface(slug: str):
    """Kick off a push of ``data/sorted/<slug>`` to a (private by default) HF repo.

    Body: ``{repo_id, private?, include_video?, categories?}``. ``repo_id`` must
    match ``namespace/name`` (validated synchronously). The upload then runs on a
    background thread keyed by a generated id; poll
    ``GET /api/jobs/<slug>/export/huggingface/status?id=`` for progress. If the
    work finishes within ``_HF_SYNC_WAIT_SECONDS`` (e.g. a fast failure) the
    terminal ``{success, data, error}`` envelope is returned immediately with the
    mapped status; otherwise a ``202`` with ``{data: {id, state: "running"}}``.
    """
    if not job_config.JOB_SLUG_RE.match(slug) or job_config.get_job(slug) is None:
        return jsonify({"success": False, "data": None, "error": "unknown job"}), 404
    data = request.get_json() or {}
    repo_id = (data.get("repo_id") or "").strip()
    if not _HF_REPO_RE.match(repo_id):
        return jsonify({"success": False, "data": None,
                        "error": "repo_id must look like 'namespace/name'"}), 400
    private = bool(data.get("private", True))
    include_video = bool(data.get("include_video", False))
    categories_arg = data.get("categories") if isinstance(data.get("categories"), list) else None

    job_id = uuid.uuid4().hex
    _hf_set(job_id, state="running", message="upload in progress", status=202,
            data=None, slug=slug, repo_id=repo_id)
    thread = threading.Thread(
        target=_run_hf_push, name=f"hf-push-{job_id[:8]}", daemon=True,
        kwargs={"job_id": job_id, "slug": slug, "repo_id": repo_id,
                "private": private, "include_video": include_video,
                "categories_arg": categories_arg})
    thread.start()
    # Give a fast/mocked push a chance to finish so its terminal result (and the
    # original status-code mapping) is returned inline; real uploads time out
    # here and hand back a 202 for polling.
    thread.join(_HF_SYNC_WAIT_SECONDS)
    entry = _hf_get(job_id) or {}
    state = entry.get("state")
    if state == "done":
        return jsonify({"success": True, "data": entry.get("data"), "error": None})
    if state == "error":
        return jsonify({"success": False, "data": None,
                        "error": entry.get("message")}), int(entry.get("status", 500))
    # Still running: hand back the id for status polling.
    return jsonify({"success": True, "error": None,
                    "data": {"id": job_id, "state": "running", "repo_id": repo_id}}), 202


@app.route("/api/jobs/<slug>/export/huggingface/status", methods=["GET"])
def api_export_huggingface_status(slug: str):
    """Report the state of a background HF-export job started for ``slug``.

    Query: ``?id=<job id>``. Returns ``{success, data: {id, state, repo_id,
    result?}, error}`` where ``state`` is ``running|done|error``; ``error`` carries
    the friendly message when the job failed. Unknown ids are a clean 404.
    """
    if not job_config.JOB_SLUG_RE.match(slug):
        return jsonify({"success": False, "data": None, "error": "invalid job slug"}), 400
    job_id = (request.args.get("id") or "").strip()
    entry = _hf_get(job_id) if job_id else None
    if entry is None:
        return jsonify({"success": False, "data": None, "error": "unknown export job"}), 404
    state = entry.get("state")
    payload = {"id": job_id, "state": state, "repo_id": entry.get("repo_id")}
    if state == "done":
        payload["result"] = entry.get("data")
    return jsonify({
        "success": state != "error",
        "data": payload,
        "error": entry.get("message") if state == "error" else None,
    })


# ── API: CLIP embeddings (status + find-similar) ──────────────────────────────
#
# Embeddings are an OPTIONAL feature gated on the ML extra (open_clip + torch).
# Both endpoints degrade gracefully: when the extra is absent OR the per-slug
# index is empty they return a friendly, actionable envelope rather than a 500.
# `embeddings` / `embeddings_index` import cleanly without the ML stack (the
# heavy deps are lazy inside embed_image), so a plain `import` is safe here, but
# we keep it lazy to mirror the active_learning pattern and avoid loading them
# until an embeddings feature is actually used.

_EMBED_HINT = ("Embeddings need the ML extra. Install it with `pip install .[ml]`, "
               "then build the index with `python tools/embed_sorted.py`.")
_EMBED_EMPTY_HINT = ("No embeddings indexed for this job yet. Build the index with "
                     "`python tools/embed_sorted.py`.")


@app.route("/api/embeddings/status", methods=["GET"])
def api_embeddings_status():
    """Report whether semantic features are usable for ``?job=<slug>``.

    Returns ``{success, data: {available, indexed, slug}, error}`` where
    ``available`` is ``embeddings.open_clip_available()`` and ``indexed`` is the
    number of vectors stored for the slug. Never 500s — a missing ML extra just
    reports ``available: false`` with a hint.
    """
    slug = _resolve_job_slug() or "default"
    try:
        import embeddings
        import embeddings_index
        available = bool(embeddings.open_clip_available())
        indexed = len(embeddings_index.EmbeddingIndex(slug=slug))
    except Exception as exc:  # noqa: BLE001 - optional feature, never fatal
        logger.debug("embeddings status unavailable: %s", exc)
        return jsonify({"success": True, "error": None,
                        "data": {"available": False, "indexed": 0, "slug": slug,
                                 "hint": _EMBED_HINT}})
    data: dict[str, Any] = {"available": available, "indexed": indexed, "slug": slug}
    if not available:
        data["hint"] = _EMBED_HINT
    elif indexed == 0:
        data["hint"] = _EMBED_EMPTY_HINT
    return jsonify({"success": True, "data": data, "error": None})


@app.route("/api/embeddings/similar", methods=["GET"])
def api_embeddings_similar():
    """Nearest neighbours of an indexed image for ``?job=<slug>``.

    Query: ``?key=<image key>&top_k=&job=``. Looks up the key's stored vector in
    ``EmbeddingIndex(slug)`` and ranks the rest by cosine similarity. Returns
    ``{success, data: {results: [{path, thumb_url, score}], ...}, error}`` —
    paths are ``safe_inside``-guarded against the queue/sorted roots. Degrades
    gracefully (never 500) when the ML extra is missing, the index is empty, or
    the key isn't indexed.
    """
    slug = _resolve_job_slug() or "default"
    key = (request.args.get("key") or "").strip()
    try:
        top_k = max(1, min(60, int(request.args.get("top_k", 12))))
    except (TypeError, ValueError):
        top_k = 12
    if not key:
        return jsonify({"success": False, "data": None, "error": "key is required"}), 400

    try:
        import embeddings
        import embeddings_index
        available = bool(embeddings.open_clip_available())
        index = embeddings_index.EmbeddingIndex(slug=slug)
    except Exception as exc:  # noqa: BLE001 - optional feature, never fatal
        logger.debug("embeddings similar unavailable: %s", exc)
        return jsonify({"success": True, "error": None,
                        "data": {"results": [], "available": False, "indexed": 0,
                                 "message": _EMBED_HINT}})

    indexed = len(index)
    if not available or indexed == 0:
        return jsonify({"success": True, "error": None,
                        "data": {"results": [], "available": available, "indexed": indexed,
                                 "message": _EMBED_HINT if not available else _EMBED_EMPTY_HINT}})

    query_vec = index.get(key)
    if query_vec is None:
        return jsonify({"success": True, "error": None,
                        "data": {"results": [], "available": True, "indexed": indexed,
                                 "message": "That image isn't in the embeddings index yet."}})

    # Ask for one extra so we can drop the query image itself from the results.
    neighbours = index.search(query_vec, top_k=top_k + 1)
    results: list[dict[str, Any]] = []
    for neighbour_key, score in neighbours:
        if neighbour_key == key:
            continue
        safe = safe_inside(neighbour_key, [PIPELINE_QUEUE, PIPELINE_SORTED])
        if safe is None:
            continue  # traversal-guarded out (stale index row pointing elsewhere)
        results.append({
            "path": neighbour_key,
            "thumb_url": f"/api/thumbnail?path={_urlquote(neighbour_key)}",
            "score": round(float(score), 4),
        })
        if len(results) >= top_k:
            break
    return jsonify({"success": True, "error": None,
                    "data": {"results": results, "available": True, "indexed": indexed}})


# ── API: per-job schedules ────────────────────────────────────────────────────

@app.route("/api/schedules", methods=["GET"])
def api_schedules_list():
    """All persisted schedules as ``{success, data: {schedules, actions}, error}``."""
    rows = [s.to_dict() for s in scheduler.list_schedules()]
    return jsonify({"success": True,
                    "data": {"schedules": rows, "actions": list(scheduler.ACTIONS)},
                    "error": None})


@app.route("/api/schedules", methods=["POST"])
def api_schedules_add():
    """Upsert a schedule. Body: ``{slug, cadence, action, enabled?}``."""
    data = request.get_json() or {}
    slug = (data.get("slug") or "").strip()
    if not job_config.JOB_SLUG_RE.match(slug) or job_config.get_job(slug) is None:
        return jsonify({"success": False, "data": None, "error": "unknown job slug"}), 400
    try:
        sched = scheduler.Schedule(
            slug=slug,
            cadence=str(data.get("cadence", "@daily")),
            action=str(data.get("action", "scrape")),
            enabled=bool(data.get("enabled", True)),
        )
    except ValueError as exc:  # bad cadence / action
        return _err("invalid cadence or action", exc, 400)
    try:
        scheduler.add_schedule(sched)
    except OSError as exc:  # disk full / perms — clean error, not a 500 traceback
        return _err("could not persist", exc, 500)
    return jsonify({"success": True, "data": sched.to_dict(), "error": None})


@app.route("/api/schedules", methods=["DELETE"])
def api_schedules_delete():
    """Delete the schedule identified by ``?slug=&action=``."""
    slug = (request.args.get("slug") or "").strip()
    action = (request.args.get("action") or "").strip()
    if not slug or not action:
        return jsonify({"success": False, "data": None,
                        "error": "slug and action are required"}), 400
    # Validate the slug shape (mirrors the POST handler) so a malformed slug can
    # never reach the persistence layer. We deliberately don't require the job to
    # still exist, so an orphaned schedule remains deletable.
    if not job_config.JOB_SLUG_RE.match(slug):
        return jsonify({"success": False, "data": None,
                        "error": "invalid job slug"}), 400
    try:
        removed = scheduler.remove_schedule(slug, action)
    except OSError as exc:  # disk write failure on the rewrite
        return _err("could not persist", exc, 500)
    return jsonify({"success": True, "data": {"removed": removed}, "error": None})


# ── API: vision fleet health telemetry ────────────────────────────────────────

def _scrub_fleet(fleet: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip the raw ``api_key`` from each fleet worker before it leaves the box.

    The health panel echoes the fleet so the UI can label probes, but the worker
    dicts carry the per-instance API key. Replace it with a ``has_key`` bool so
    the browser learns whether a key is set without ever receiving the secret.
    """
    scrubbed: list[dict[str, Any]] = []
    for worker in fleet:
        if not isinstance(worker, dict):
            continue
        safe = {k: v for k, v in worker.items() if k != "api_key"}
        safe["has_key"] = bool(str(worker.get("api_key", "") or "").strip())
        scrubbed.append(safe)
    return scrubbed


@app.route("/api/vision/health")
def api_vision_health():
    """Probe the active (or ?job=) job's vision fleet — a live telemetry panel.

    Returns ``{success, data: {probes, fleet}, error}`` where ``probes`` is
    ``{id: {ok, latency_ms, error}}`` from ``fleet_health.probe_fleet`` over the
    cleaned worker list. On-demand only.
    """
    slug = _resolve_job_slug()
    job = _job_for_scope(slug)
    fleet: list[dict[str, Any]] = []
    if job is not None:
        eff = job_config.effective_config(job)
        vision = eff.get("vision") if isinstance(eff.get("vision"), dict) else {}
        fleet = job_config.clean_vision_fleet(vision.get("workers"))
    try:
        probes = fleet_health.probe_fleet(fleet)
    except Exception as exc:  # noqa: BLE001 - probe must never 500 the panel
        return _err("fleet health probe failed", exc, 500)
    # Never ship raw per-worker api_keys to the browser — replace with has_key.
    return jsonify({"success": True,
                    "data": {"probes": probes, "fleet": _scrub_fleet(fleet)},
                    "error": None})


# ── UI ─────────────────────────────────────────────────────────────────────────

HTML_TEMPLATE = r"""{% macro tip(body, example='') -%}
<span class="tip" tabindex="0" aria-label="Help"><span class="tip-i" aria-hidden="true">i</span><span class="tip-pop" role="tooltip">{{ body|safe }}{% if example %}<span class="ex">{{ example|safe }}</span>{% endif %}</span></span>
{%- endmacro -%}
{% macro testbtn(scraper, label='Test connection') -%}
<div class="mt-1.5 flex items-center gap-2 flex-wrap">
  <button type="button" @click="testScraper('{{ scraper }}')" :disabled="!!scraperTest['{{ scraper }}']?.pending"
    class="px-2 py-0.5 text-xs bg-slate-700 hover:bg-slate-600 rounded disabled:opacity-50">
    <span x-text="scraperTest['{{ scraper }}']?.pending ? 'Testing…' : '{{ label }}'"></span>
  </button>
  <span x-show="scraperTest['{{ scraper }}'] && !scraperTest['{{ scraper }}'].pending" class="text-xs"
    :class="scraperTest['{{ scraper }}']?.ok ? 'text-emerald-400' : 'text-rose-400'"
    x-text="(scraperTest['{{ scraper }}']?.ok ? '✓ ' : '✕ ') + (scraperTest['{{ scraper }}']?.message || '') + (scraperTest['{{ scraper }}']?.latency_ms != null ? ' (' + scraperTest['{{ scraper }}'].latency_ms + 'ms)' : '')"></span>
</div>
{%- endmacro -%}
{# toggle(key, label, help) — a real switch bound to a settings value stored as
   the string "true"/"false". A transparent native checkbox handles interaction +
   keyboard focus and its @change bubbles so the Settings form dirty-tracking
   fires (same contract as the legacy BLUR_NSFW_THUMBS checkbox); Alpine :class
   drives the visual state. #}
{% macro toggle(key, label, help='') -%}
<label class="flex items-center justify-between gap-3 py-1.5 cursor-pointer select-none">
  <span class="text-xs text-slate-300">{{ label }}{% if help %}{{ tip(help) }}{% endif %}</span>
  <span class="relative inline-flex h-5 w-9 shrink-0 items-center rounded-full transition-colors"
        :class="settings.{{ key }} === 'true' ? 'bg-indigo-500' : 'bg-slate-600'">
    <input type="checkbox" class="absolute inset-0 w-full h-full opacity-0 cursor-pointer m-0"
      :checked="settings.{{ key }} === 'true'"
      @change="settings.{{ key }} = $event.target.checked ? 'true' : 'false'"/>
    <span class="inline-block h-4 w-4 transform rounded-full bg-white shadow transition-transform pointer-events-none"
      :class="settings.{{ key }} === 'true' ? 'translate-x-4' : 'translate-x-1'"></span>
  </span>
</label>
{%- endmacro -%}
{# slider(key, label, min, max, step, help, suffix) — a min..max range bound to a
   numeric settings value, with a live value bubble. x-model fires input events
   that bubble for dirty-tracking; accent colours the native track. #}
{% macro slider(key, label, min=0, max=100, step=1, help='', suffix='') -%}
<label class="block py-1.5">
  <span class="flex items-center justify-between text-xs text-slate-300">
    <span>{{ label }}{% if help %}{{ tip(help) }}{% endif %}</span>
    <span class="font-mono text-indigo-300" x-text="(settings.{{ key }} || '{{ min }}') + '{{ suffix }}'"></span>
  </span>
  <input type="range" min="{{ min }}" max="{{ max }}" step="{{ step }}"
    :value="settings.{{ key }} || '{{ min }}'" @input="settings.{{ key }} = $event.target.value"
    class="w-full mt-2 accent-indigo-500 cursor-pointer"/>
</label>
{%- endmacro -%}
<!doctype html>
<html lang="en" class="dark">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title x-text="'cull · ' + (currentTabLabel() || 'jobs')">cull</title>
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
  /* Keyboard-cull legend key caps. */
  .kbd { display:inline-block; padding:0 .35em; min-width:1.2em; text-align:center;
         font-family:ui-monospace,monospace; font-size:.7rem; line-height:1.4;
         color:#e2e8f0; background:#1e293b; border:1px solid #334155;
         border-bottom-width:2px; border-radius:.25rem; }
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
  /* Field help tooltips — an "i" trigger; hover/focus reveals a positioned
     popover. CSS-only so it needs no per-instance Alpine state. See tip() macro. */
  .tip { position:relative; display:inline-flex; align-items:center; vertical-align:middle; margin-left:.3rem; }
  .tip-i { display:inline-flex; align-items:center; justify-content:center; width:15px; height:15px;
           border-radius:9999px; border:1px solid #64748b; color:#94a3b8; font-size:10px; line-height:1;
           font-style:normal; font-weight:700; cursor:help; user-select:none; }
  .tip:hover .tip-i, .tip:focus-within .tip-i { border-color:#38bdf8; color:#7dd3fc; }
  .tip-pop { visibility:hidden; opacity:0; transition:opacity .12s ease; position:absolute; z-index:80;
             left:0; top:1.5em; width:18rem; max-width:min(18rem,70vw); padding:.55rem .65rem;
             border-radius:.4rem; background:#1e293b; border:1px solid #475569; color:#e2e8f0;
             font-size:.72rem; line-height:1.4; box-shadow:0 10px 25px rgba(0,0,0,.45);
             text-transform:none; letter-spacing:normal; font-weight:400; white-space:normal; }
  .tip:hover .tip-pop, .tip:focus-within .tip-pop { visibility:visible; opacity:1; }
  .tip.tip-r .tip-pop { left:auto; right:0; }
  .tip-pop b { color:#f1f5f9; font-weight:600; }
  .tip-pop code { background:#0f172a; border:1px solid #334155; border-radius:3px; padding:0 .25rem; color:#fcd34d; }
  .tip-pop .ex { color:#9aa6b6; display:block; margin-top:.35rem; }
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

  <!-- Generic toast stack — top-right. Replaces window.alert(): success/error/
       warn/info messages, auto-dismissing. Driven by notify(message, type). -->
  <div class="fixed top-4 right-4 z-[60] flex flex-col gap-2 w-[22rem] max-w-[calc(100vw-2rem)] pointer-events-none">
    <template x-for="t in toasts" :key="t.id">
      <div x-transition.opacity.duration.200ms
           class="pointer-events-auto flex items-start gap-2 rounded-lg shadow-2xl p-3 text-sm border bg-slate-900"
           :class="{
             'border-emerald-600 text-emerald-50': t.type === 'success',
             'border-rose-600 text-rose-50': t.type === 'error',
             'border-amber-600 text-amber-50': t.type === 'warn',
             'border-sky-700 text-slate-100': t.type === 'info' || !t.type,
           }">
        <span class="mt-0.5 shrink-0 font-bold" x-text="({success:'✓', error:'✕', warn:'⚠', info:'ℹ'})[t.type] || 'ℹ'"></span>
        <div class="flex-1 whitespace-pre-line break-words" x-text="t.message"></div>
        <button @click="dismissToast(t.id)" class="shrink-0 text-slate-400 hover:text-slate-100 leading-none" aria-label="Dismiss">✕</button>
      </div>
    </template>
  </div>

  <!-- Styled confirm dialog — replaces window.confirm() for destructive actions.
       askConfirm(message, {title, confirmLabel, danger}) resolves to a boolean. -->
  <div x-show="confirmDialog.open" x-cloak x-transition.opacity
       class="fixed inset-0 z-[70] flex items-center justify-center bg-black/60 p-4"
       @click.self="_closeConfirm(false)"
       @keydown.escape.window="confirmDialog.open && _closeConfirm(false)">
    <div class="bg-slate-900 border border-slate-700 rounded-xl shadow-2xl max-w-md w-full p-5">
      <div class="font-semibold text-base mb-2" x-text="confirmDialog.title"></div>
      <div class="text-sm text-slate-300 mb-4 whitespace-pre-line" x-text="confirmDialog.message"></div>
      <input x-show="confirmDialog.input" x-ref="dialogInput" type="text"
             x-model="confirmDialog.inputValue" :placeholder="confirmDialog.inputPlaceholder"
             @keydown.enter.prevent="_closeConfirm(true)"
             class="w-full mb-4 px-3 py-2 bg-slate-800 border border-slate-600 rounded text-sm focus:outline-none focus:border-sky-500">
      <div class="flex justify-end gap-2">
        <button @click="_closeConfirm(false)"
                class="px-3 py-1.5 bg-slate-700 hover:bg-slate-600 rounded text-sm"
                x-text="confirmDialog.cancelLabel"></button>
        <button @click="_closeConfirm(true)" x-ref="confirmBtn"
                class="px-3 py-1.5 rounded text-sm font-semibold text-white"
                :class="confirmDialog.danger ? 'bg-rose-600 hover:bg-rose-500' : 'bg-sky-600 hover:bg-sky-500'"
                x-text="confirmDialog.confirmLabel"></button>
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
      <a href="#" @click.prevent="backToJobs()"
         class="flex items-center gap-2 hover:opacity-80 transition" aria-label="cull — back to Jobs">
        <img src="/brand/logo-transparent-dark.png" alt="" width="32" height="32"
             class="shrink-0"/>
        <span class="font-brand text-2xl font-medium tracking-tight">cull</span>
      </a>
      <p class="text-xs text-slate-400 mt-1" x-text="'Worker: ' + (status.pipeline?.vision_worker || '...')"></p>
      <div class="mt-2">
        <span class="pill px-2 py-0.5 rounded"
          :class="status.pipeline?.running ? 'bg-emerald-900/60 text-emerald-300' : 'bg-slate-800 text-slate-400'"
          x-text="status.pipeline?.running ? 'running' : 'stopped'"></span>
        <span class="pill px-2 py-0.5 rounded bg-slate-800 text-slate-300 ml-1"
          x-show="jobsActive" x-text="'active: ' + (jobsActive || '')"></span>
      </div>
    </div>

    <!-- GLOBAL nav (Jobs landing view) -->
    <template x-if="view === 'jobs'">
      <div class="space-y-1">
        <template x-for="tab in globalTabs" :key="tab.id">
          <button @click="active = tab.id; sidebarOpen = false"
            class="w-full text-left px-3 py-2 rounded text-sm transition"
            :class="active === tab.id ? 'bg-indigo-600 text-white' : 'text-slate-300 hover:bg-slate-800'"
            x-text="tab.label"></button>
        </template>
      </div>
    </template>

    <!-- JOB-SCOPED nav (single job view) -->
    <template x-if="view === 'job'">
      <div class="space-y-1">
        <button @click="backToJobs()"
          class="w-full text-left px-3 py-2 rounded text-sm text-slate-300 hover:bg-slate-800 transition flex items-center gap-2">
          <span aria-hidden="true">←</span> Jobs
        </button>
        <div class="px-3 py-1 text-xs text-slate-500 truncate" x-text="currentJob"></div>
        <template x-for="tab in jobTabs" :key="tab.id">
          <button @click="active = tab.id; sidebarOpen = false"
            class="w-full text-left px-3 py-2 rounded text-sm transition"
            :class="active === tab.id ? 'bg-indigo-600 text-white' : 'text-slate-300 hover:bg-slate-800'"
            x-text="tab.label"></button>
        </template>
      </div>
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
        <h2 class="text-2xl font-bold" x-text="currentTabLabel()"></h2>
        <p class="text-sm text-slate-400">
          <span x-show="view === 'jobs' && active === 'jobs'">Each job is a curation target with its own scrapers, taxonomy, and scoring. Activate one to run it.</span>
          <span x-show="view === 'jobs' && active === 'settings'">Global credentials, model endpoints, and storage roots. Per-job settings live inside each job.</span>
          <span x-show="view === 'jobs' && active === 'gstats'">Aggregate analytics across every job — filter to one job to drill in.</span>
          <span x-show="view === 'job' && active !== 'gallery'" x-text="'Job: ' + currentJob + ' — auto-refresh every 5 s'"></span>
          <span x-show="view === 'job' && active === 'gallery'" x-text="'Filter, browse, edit, and export ' + currentJob + '\'s sorted library'"></span>
        </p>
      </div>
      <div class="flex gap-2" x-show="view === 'job'">
        <template x-if="!status.pipeline?.running">
          <button @click="startPipeline()" class="px-4 py-2 rounded text-sm font-medium bg-emerald-600 hover:bg-emerald-500">Start pipeline</button>
        </template>
        <template x-if="status.pipeline?.running">
          <button @click="stopPipeline()" class="px-4 py-2 rounded text-sm font-medium bg-rose-600 hover:bg-rose-500">Stop pipeline</button>
        </template>
        <button @click="refresh()" class="px-4 py-2 rounded bg-slate-700 hover:bg-slate-600 text-sm">Refresh</button>
      </div>
    </header>

    <!-- JOBS LANDING ──────────────────────────────────────────────────────
         The first thing the user sees: a grid of job cards + a New Job card.
         Each job is its own curation target; activating one makes the
         supervisor run it. -->
    <section x-show="view === 'jobs' && active === 'jobs'" class="space-y-4">
      <!-- Job queue strip: active job + next queued, with advance control. -->
      <div class="card rounded-xl p-5" x-show="jobsActive || jobsQueue.length">
        <div class="flex items-center justify-between mb-3">
          <h3 class="font-semibold">Run queue</h3>
          <button @click="advanceQueue()" :disabled="!jobsQueue.length"
            class="px-3 py-1.5 text-xs bg-slate-700 hover:bg-slate-600 rounded disabled:opacity-50"
            title="Promote the next queued job to active">Advance →</button>
        </div>
        <div class="flex flex-wrap items-center gap-2 text-sm">
          <span class="text-xs text-slate-400">Active:</span>
          <span class="px-2 py-1 rounded bg-indigo-900/60 text-indigo-200 font-mono text-xs"
                x-text="jobsActive || '(none — activate a job)'"></span>
          <template x-if="jobsQueue.length">
            <span class="text-xs text-slate-400 ml-2">then:</span>
          </template>
          <template x-for="(slug, i) in jobsQueue" :key="slug">
            <span class="px-2 py-1 rounded bg-amber-900/40 text-amber-200 font-mono text-xs"
                  x-text="(i+1) + '. ' + slug"></span>
          </template>
        </div>
      </div>

      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <!-- Existing jobs -->
        <template x-for="jb in jobsList" :key="jb.slug">
          <div class="card rounded-xl p-5 flex flex-col">
            <div class="flex items-start justify-between gap-2 mb-2">
              <div class="min-w-0">
                <div class="font-semibold truncate" x-text="jb.name"></div>
                <div class="text-xs text-slate-500 font-mono truncate" x-text="jb.slug"></div>
              </div>
              <span class="pill px-2 py-0.5 rounded shrink-0" :class="jobStatusClass(jb.status)" x-text="jb.status"></span>
            </div>
            <div class="grid grid-cols-2 gap-2 text-center my-2">
              <div class="bg-slate-900/60 border border-slate-800 rounded p-2">
                <div class="pill text-indigo-300">Queued</div>
                <div class="text-xl font-mono mt-0.5" x-text="jb.queued"></div>
              </div>
              <div class="bg-slate-900/60 border border-slate-800 rounded p-2">
                <div class="pill text-emerald-300">Sorted</div>
                <div class="text-xl font-mono mt-0.5" x-text="jb.sorted"></div>
              </div>
            </div>
            <div class="text-[11px] text-slate-500 mb-3" x-show="jb.queue_position !== null"
                 x-text="jb.queue_position === 0 ? 'Active' : ('Queue position #' + jb.queue_position)"></div>
            <div class="mt-auto flex flex-wrap gap-1.5">
              <button @click="openJob(jb.slug)" class="px-2.5 py-1 text-xs bg-indigo-600 hover:bg-indigo-500 rounded font-medium">Open</button>
              <button @click="activateJob(jb.slug)" :disabled="jb.is_active"
                class="px-2.5 py-1 text-xs bg-slate-700 hover:bg-slate-600 rounded disabled:opacity-40"
                x-text="jb.is_active ? 'Active' : 'Activate'"></button>
              <button @click="openJob(jb.slug); active='jobSettings'" class="px-2.5 py-1 text-xs bg-slate-700 hover:bg-slate-600 rounded">Edit</button>
              <button @click="cloneJob(jb.slug)" class="px-2.5 py-1 text-xs bg-slate-700 hover:bg-slate-600 rounded">Clone</button>
              <template x-if="!jb.is_active && jb.queue_position === null">
                <button @click="enqueueJob(jb.slug)" class="px-2.5 py-1 text-xs bg-slate-700 hover:bg-slate-600 rounded">Enqueue</button>
              </template>
              <template x-if="jb.queue_position !== null && jb.queue_position > 0">
                <button @click="dequeueJob(jb.slug)" class="px-2.5 py-1 text-xs bg-slate-700 hover:bg-slate-600 rounded">Dequeue</button>
              </template>
              <button @click="deleteJob(jb.slug)" :disabled="jb.is_active"
                class="px-2.5 py-1 text-xs bg-rose-900/60 hover:bg-rose-800 text-rose-100 rounded disabled:opacity-40">Delete</button>
            </div>
          </div>
        </template>

        <!-- New Job card -->
        <div class="card rounded-xl p-5 border-dashed border-2 border-slate-700 flex flex-col justify-center">
          <h3 class="font-semibold mb-2">New job</h3>
          <input x-model="newJob.name" @keydown.enter="createJob()"
            placeholder="Job name (e.g. Car Ads)"
            class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 text-sm mb-2"/>
          <select x-model="newJob.preset" :disabled="!!newJob.base_on"
            class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 text-sm mb-2 disabled:opacity-50">
            <option value="">Preset: default</option>
            <template x-for="p in presetsList" :key="'np_'+p">
              <option :value="p" x-text="'Preset: ' + p"></option>
            </template>
          </select>
          <select x-model="newJob.base_on"
            class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 text-sm mb-3">
            <option value="">Start fresh from preset</option>
            <template x-for="jb in jobsList" :key="'base_'+jb.slug">
              <option :value="jb.slug" x-text="'Clone job: ' + jb.name"></option>
            </template>
          </select>
          <button @click="createJob()" class="px-3 py-2 bg-emerald-600 hover:bg-emerald-500 rounded text-sm font-medium">Create job</button>
        </div>
      </div>

      <template x-if="!jobsLoading && jobsList.length === 0">
        <div class="text-xs text-slate-500">No jobs yet — create one above to begin curating.</div>
      </template>
    </section>

    <!-- SCHEDULES (per-job cadence → scrape/curate/export) -->
    <section x-show="view === 'jobs' && active === 'schedules'" class="space-y-4">
      <div class="card rounded-xl p-5">
        <h3 class="font-semibold mb-1">Schedules</h3>
        <p class="text-xs text-slate-400 mb-3">Run a job action on a cadence (<code>@hourly</code>, <code>@daily</code>, <code>@weekly</code>, or <code>every 30m</code> / <code>every 2h</code>). The supervisor fires due schedules on its reconcile loop when <code>SCHEDULER_ENABLED=true</code>.</p>
        <div class="grid md:grid-cols-4 gap-3 items-end">
          <label class="block">
            <span class="text-xs text-slate-400">Job</span>
            <select x-model="schedules.form.slug" class="w-full bg-slate-800 border border-slate-700 rounded px-2 py-2 mt-1">
              <option value="" :selected="!schedules.form.slug">(pick a job)</option>
              <template x-for="jb in jobsList" :key="jb.slug">
                <option :value="jb.slug" :selected="schedules.form.slug === jb.slug" x-text="jb.name + ' (' + jb.slug + ')'"></option>
              </template>
            </select>
          </label>
          <label class="block">
            <span class="text-xs text-slate-400">Cadence</span>
            <select x-model="schedules.form.cadence"
              class="w-full bg-slate-800 border border-slate-700 rounded px-2 py-2 mt-1 text-sm">
              <template x-for="opt in scheduleCadences" :key="opt.value">
                <option :value="opt.value" :selected="schedules.form.cadence === opt.value" x-text="opt.label"></option>
              </template>
            </select>
          </label>
          <label class="block">
            <span class="text-xs text-slate-400">Action</span>
            <select x-model="schedules.form.action" class="w-full bg-slate-800 border border-slate-700 rounded px-2 py-2 mt-1">
              <template x-for="a in (schedules.actions.length ? schedules.actions : ['scrape','curate','export'])" :key="a">
                <option :value="a" :selected="schedules.form.action === a" x-text="a"></option>
              </template>
            </select>
          </label>
          <button @click="addSchedule()" class="px-3 py-2 bg-indigo-600 hover:bg-indigo-500 rounded text-sm">Add / update</button>
        </div>
        <div x-show="schedules.error" x-cloak class="text-xs text-rose-300 mt-2" x-text="schedules.error"></div>
      </div>

      <div class="card rounded-xl p-5">
        <div class="flex items-center justify-between mb-3">
          <h4 class="font-semibold text-sm">Active schedules</h4>
          <button @click="loadSchedules()" class="text-xs link-btn">refresh</button>
        </div>
        <div x-show="!schedules.rows.length" class="text-sm text-slate-500">No schedules yet.</div>
        <table x-show="schedules.rows.length" class="w-full text-sm">
          <thead><tr class="text-left text-xs text-slate-400 border-b border-slate-700">
            <th class="py-1">Job</th><th>Cadence</th><th>Action</th><th>Enabled</th><th>Last run</th><th></th>
          </tr></thead>
          <tbody>
            <template x-for="s in schedules.rows" :key="s.slug + '_' + s.action">
              <tr class="border-b border-slate-800">
                <td class="py-1.5 font-mono text-xs" x-text="s.slug"></td>
                <td class="font-mono text-xs" x-text="s.cadence"></td>
                <td x-text="s.action"></td>
                <td x-text="s.enabled ? 'yes' : 'no'"></td>
                <td class="text-xs text-slate-400" x-text="s.last_run || '—'"></td>
                <td class="text-right"><button @click="removeSchedule(s.slug, s.action)" class="text-xs text-rose-300 hover:text-rose-200">delete</button></td>
              </tr>
            </template>
          </tbody>
        </table>
      </div>
    </section>

    <!-- GLOBAL STATS ──────────────────────────────────────────────────────
         Aggregate analytics with a per-job filter. Reuses the Stats payload
         shape; just driven by an explicit ?job= rather than the active job. -->
    <section x-show="view === 'jobs' && active === 'gstats'" class="space-y-4">
      <div class="card rounded-xl p-5">
        <div class="flex items-center justify-between gap-3 flex-wrap">
          <div>
            <h3 class="font-semibold">Global analytics</h3>
            <p class="text-xs text-slate-400">Aggregate across all jobs, or filter to one.</p>
          </div>
          <div class="flex items-center gap-2">
            <label class="text-xs text-slate-400">Job filter</label>
            <select x-model="globalStatsJob" @change="loadGlobalStats()"
              class="bg-slate-800 border border-slate-700 rounded px-2 py-1 text-sm">
              <option value="">All jobs</option>
              <template x-for="jb in jobsList" :key="'gs_'+jb.slug">
                <option :value="jb.slug" x-text="jb.name"></option>
              </template>
            </select>
            <button @click="loadGlobalStats()" :disabled="statsLoading"
              class="px-3 py-1.5 text-xs bg-slate-700 hover:bg-slate-600 rounded disabled:opacity-60">Refresh</button>
          </div>
        </div>
        <div class="grid grid-cols-2 md:grid-cols-4 gap-3 mt-3">
          <div class="bg-slate-900/60 border border-slate-800 rounded p-3"><div class="pill text-slate-400">Total classified</div><div class="text-2xl font-mono mt-1" x-text="stats.totals?.all ?? 0"></div></div>
          <div class="bg-slate-900/60 border border-slate-800 rounded p-3"><div class="pill text-emerald-300">Kept</div><div class="text-2xl font-mono mt-1 text-emerald-200" x-text="stats.totals?.non_discard ?? 0"></div></div>
          <div class="bg-slate-900/60 border border-slate-800 rounded p-3"><div class="pill text-rose-300">Discarded</div><div class="text-2xl font-mono mt-1 text-rose-200" x-text="stats.totals?.discard ?? 0"></div></div>
          <div class="bg-slate-900/60 border border-slate-800 rounded p-3"><div class="pill text-slate-400">Sources</div><div class="text-2xl font-mono mt-1" x-text="(stats.sources?.length ?? 0)"></div></div>
        </div>
      </div>
      <div class="card rounded-xl p-5">
        <h3 class="font-semibold mb-3">Per-source platform analytics</h3>
        <div class="scroll-box"><table>
          <thead><tr>
            <th>Source</th><th class="text-right">Total</th>
            <th class="text-right">DISCARD %</th><th class="text-right">NSFW %</th>
            <th class="text-right">Avg OVR</th><th class="text-right">Avg REL</th>
          </tr></thead>
          <tbody>
            <template x-for="s in stats.sources" :key="'gs_src_'+s.source">
              <tr>
                <td class="font-mono text-xs" x-text="s.source"></td>
                <td class="text-right font-mono" x-text="s.total"></td>
                <td class="text-right font-mono" x-text="s.discard_pct + '%'"></td>
                <td class="text-right font-mono" x-text="s.nsfw_pct + '%'"></td>
                <td class="text-right font-mono" x-text="s.avg_ovr ?? '-'"></td>
                <td class="text-right font-mono" x-text="s.avg_rel ?? '-'"></td>
              </tr>
            </template>
            <template x-if="(stats.sources?.length ?? 0) === 0">
              <tr><td colspan="6" class="text-xs text-slate-500">No data yet.</td></tr>
            </template>
          </tbody>
        </table></div>
      </div>
    </section>

    <!-- OVERVIEW -->
    <section x-show="view === 'job' && active === 'overview'" class="space-y-4">
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

    <!-- STATS (job-scoped) -->
    <section x-show="view === 'job' && active === 'stats'" class="space-y-4">
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

    <!-- GALLERY (job-scoped) -->
    <section x-show="view === 'job' && active === 'gallery'" class="space-y-4">
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

        <!-- Keyboard culling legend (§3a). Click a card to focus it, then drive
             the keep/reject/recategorize hotkeys without leaving the keyboard. -->
        <div class="mb-3 rounded-lg border border-slate-800 bg-slate-900/40 px-3 py-2 text-[11px] text-slate-300">
          <div class="flex items-center justify-between">
            <span class="font-semibold text-slate-200">Keyboard culling</span>
            <button class="link-btn" @click="cullLegendOpen = !cullLegendOpen"
              x-text="cullLegendOpen ? 'hide' : 'show'"></button>
          </div>
          <div x-show="cullLegendOpen" class="mt-2 flex flex-wrap gap-x-3 gap-y-1">
            <span><kbd class="kbd">←↑↓→</kbd>/<kbd class="kbd">h j k l</kbd> move focus</span>
            <span><kbd class="kbd">k</kbd> keep <span class="text-slate-500" x-text="'(' + (cullPrimaryKeep() || '—') + ')'"></span></span>
            <span><kbd class="kbd">x</kbd> reject → DISCARD</span>
            <span><kbd class="kbd">1</kbd>–<kbd class="kbd">9</kbd> Nth category</span>
            <span><kbd class="kbd">u</kbd> undo <span class="text-slate-500" x-text="'(' + cullUndo.length + ')'"></span></span>
            <span><kbd class="kbd">Esc</kbd> clear focus</span>
          </div>
          <div x-show="cullLegendOpen && cullCategories.length" class="mt-1 text-slate-500">
            <template x-for="(cat, i) in cullCategories" :key="cat.name">
              <span class="mr-2"><span class="text-slate-400" x-text="(i+1)"></span>=<span x-text="cat.name"></span></span>
            </template>
          </div>
        </div>

        <div x-show="galleryLoading" class="text-xs text-slate-400">Loading...</div>
        <div class="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3">
          <template x-for="(c, idx) in gallery.items" :key="c.path">
            <div class="bg-slate-900/60 border rounded p-2 text-xs flex flex-col transition-shadow"
                 :data-cull-idx="idx"
                 :class="cullFocus === idx ? 'border-indigo-400 ring-2 ring-indigo-400/70' : 'border-slate-800'"
                 @click="cullFocus = idx">
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
              <div class="mt-1 flex items-center gap-3">
                <span class="link-btn" @click.stop="openModalFromCard(c)">Open</span>
                <span class="link-btn" @click.stop="findSimilar(c)" title="Find visually similar kept images (CLIP embeddings)">Find similar</span>
              </div>
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

      <!-- Find-similar results strip (CLIP embeddings; on-demand). -->
      <div x-show="sim.open" x-cloak class="card rounded-xl p-4">
        <div class="flex items-center justify-between mb-2">
          <h3 class="font-semibold">Similar to <span class="font-mono text-xs" x-text="sim.sourceName"></span></h3>
          <button @click="sim.open = false" class="text-xs px-2 py-1 bg-slate-700 hover:bg-slate-600 rounded">Close</button>
        </div>
        <div x-show="sim.loading" class="text-xs text-slate-400">Searching…</div>
        <div x-show="!sim.loading && sim.message" x-cloak class="text-xs text-amber-300" x-text="sim.message"></div>
        <div x-show="!sim.loading && !sim.message && sim.results.length === 0" x-cloak class="text-xs text-slate-500">No similar images found.</div>
        <div class="flex flex-wrap gap-3 mt-2">
          <template x-for="m in sim.results" :key="m.path">
            <div class="relative w-32">
              <img :src="m.thumb_url" loading="lazy" class="w-32 h-32 object-cover rounded border border-slate-700"/>
              <div class="text-[11px] text-slate-400 mt-1">score <span x-text="m.score"></span></div>
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

    <!-- DUPLICATES (perceptual-hash near-dupes; on-demand scan) -->
    <section x-show="view === 'job' && active === 'duplicates'" class="space-y-4">
      <div class="card rounded-xl p-5">
        <div class="flex flex-wrap items-end justify-between gap-3">
          <div>
            <h3 class="font-semibold">Near-duplicate finder</h3>
            <p class="text-xs text-slate-400 mt-1">Groups visually-similar sorted images by perceptual hash (dHash). Lower threshold = stricter (only very close matches). On-demand — it never runs on the auto-refresh.</p>
          </div>
          <div class="flex items-end gap-3">
            <label class="text-xs text-slate-400">Threshold (0-64)
              <input type="number" min="0" max="64" x-model.number="dup.threshold"
                     class="w-20 bg-slate-800 border border-slate-700 rounded px-2 py-1.5 mt-1 block"/>
            </label>
            <button @click="loadDuplicates()" :disabled="dup.loading"
                    class="px-3 py-2 bg-indigo-600 hover:bg-indigo-500 rounded text-sm disabled:opacity-50">
              <span x-text="dup.loading ? 'Scanning…' : 'Scan for duplicates'"></span>
            </button>
          </div>
        </div>
        <div x-show="dup.error" x-cloak class="text-xs text-rose-300 mt-3" x-text="dup.error"></div>
        <div x-show="dup.scanned && !dup.loading && !dup.groups.length" class="text-sm text-slate-500 mt-4">No near-duplicates found at this threshold.</div>
      </div>

      <template x-for="(group, gi) in dup.groups" :key="'dg' + gi">
        <div class="card rounded-xl p-4">
          <div class="flex items-center justify-between mb-2">
            <span class="text-xs text-slate-400" x-text="group.length + ' similar images'"></span>
          </div>
          <div class="flex flex-wrap gap-3">
            <template x-for="m in group" :key="m.path">
              <div class="relative w-32">
                <img :src="m.thumb_url" loading="lazy" class="w-32 h-32 object-cover rounded border border-slate-700"/>
                <div class="text-[11px] text-slate-400 mt-1 truncate" x-text="m.category || '—'"></div>
                <div class="flex gap-1 mt-1">
                  <button @click="keepOnly(group, m.path)" class="flex-1 px-1 py-0.5 text-[11px] bg-emerald-700 hover:bg-emerald-600 rounded" title="Keep this one, delete the rest of the group">Keep</button>
                  <button @click="deleteDuplicate(m.path)" class="flex-1 px-1 py-0.5 text-[11px] bg-rose-700 hover:bg-rose-600 rounded" title="Delete this image">Delete</button>
                </div>
              </div>
            </template>
          </div>
        </div>
      </template>
    </section>

    <!-- EXPORT (training-ready dataset profiles + HuggingFace push) -->
    <section x-show="view === 'job' && active === 'export'" class="space-y-4">
      <div class="card rounded-xl p-5">
        <h3 class="font-semibold mb-1">Export dataset</h3>
        <p class="text-xs text-slate-400 mb-3">Pack <code x-text="currentJob"></code>'s kept images into a training-ready layout under <code>data/exports/</code>. Copy-only — your sorted library is never modified.</p>
        <div class="grid md:grid-cols-2 gap-4">
          <label class="block">
            <span class="text-xs text-slate-400">Profile</span>
            <select x-model="exportForm.profile" class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1">
              <template x-for="p in exportProfiles" :key="p">
                <option :value="p" :selected="exportForm.profile === p" x-text="p"></option>
              </template>
            </select>
          </label>
          <label class="block">
            <span class="text-xs text-slate-400">Trigger word (prepended to every caption)</span>
            <input x-model="exportForm.trigger_word" placeholder="optional"
              class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1"/>
          </label>
          <label class="block">
            <span class="text-xs text-slate-400">Train/val split (0.0–0.99)</span>
            <input x-model="exportForm.train_val_split" type="number" min="0" max="0.99" step="0.05" placeholder="0 = none"
              class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1"/>
          </label>
          <label class="block">
            <span class="text-xs text-slate-400">Repeats (kohya concept folder)</span>
            <input x-model="exportForm.repeats" type="number" min="0" placeholder="0 = flat"
              class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1"/>
          </label>
          <label class="block">
            <span class="text-xs text-slate-400">Video bucketing (clip_caption)</span>
            <select x-model="exportForm.video_bucketing" class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1">
              <option value="" :selected="!exportForm.video_bucketing">(none)</option>
              <template x-for="m in videoBucketModes" :key="m">
                <option :value="m" :selected="exportForm.video_bucketing === m" x-text="m"></option>
              </template>
            </select>
          </label>
          <label class="flex items-center gap-2 mt-6">
            <input type="checkbox" x-model="exportForm.resolution_bucketing" class="accent-indigo-500"/>
            <span class="text-sm">Bucket by resolution</span>
          </label>
        </div>
        <div class="mt-4 flex items-center gap-3">
          <button @click="runDatasetExport()" :disabled="exportForm.running"
                  class="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 rounded text-sm disabled:opacity-50">
            <span x-text="exportForm.running ? 'Exporting…' : 'Run export'"></span>
          </button>
          <span x-show="exportForm.error" class="text-xs text-rose-300" x-text="exportForm.error"></span>
        </div>
        <div x-show="exportForm.summary" x-cloak class="mt-3 bg-slate-900/70 border border-slate-700 rounded p-3 text-xs font-mono whitespace-pre-wrap" x-text="exportForm.summary ? JSON.stringify(exportForm.summary, null, 2) : ''"></div>
      </div>

      <div class="card rounded-xl p-5">
        <h3 class="font-semibold mb-1">Push to HuggingFace</h3>
        <p class="text-xs text-slate-400 mb-3">Upload the kept set as a HuggingFace dataset repo. Requires <code>HF_TOKEN</code> in the environment and <code>pip install huggingface_hub</code>. The upload runs in the background — you can keep working while it finishes.</p>
        <div class="grid md:grid-cols-2 gap-4">
          <label class="block md:col-span-2">
            <span class="text-xs text-slate-400">Repo ID (namespace/name)</span>
            <input x-model="hf.repo_id" placeholder="your-name/my-dataset"
              class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1 font-mono text-xs"/>
          </label>
          <label class="flex items-center gap-2">
            <input type="checkbox" x-model="hf.private" class="accent-indigo-500"/>
            <span class="text-sm">Private repo</span>
          </label>
          <label class="flex items-center gap-2">
            <input type="checkbox" x-model="hf.include_video" class="accent-indigo-500"/>
            <span class="text-sm">Include video clips</span>
          </label>
        </div>
        <div class="mt-4 flex items-center gap-3">
          <button @click="pushToHuggingFace()" :disabled="hf.pushing"
                  class="px-4 py-2 bg-amber-600 hover:bg-amber-500 rounded text-sm disabled:opacity-50">
            <span x-text="hf.pushing ? 'Uploading…' : 'Push to HuggingFace'"></span>
          </button>
          <span x-show="hf.status && hf.pushing" x-cloak class="text-xs text-amber-300" x-text="hf.status"></span>
          <span x-show="hf.error" class="text-xs text-rose-300" x-text="hf.error"></span>
        </div>
        <div x-show="hf.result" x-cloak class="mt-3 bg-slate-900/70 border border-slate-700 rounded p-3 text-xs font-mono whitespace-pre-wrap" x-text="hf.result ? JSON.stringify(hf.result, null, 2) : ''"></div>
      </div>
    </section>

    <!-- SCRAPERS (per-job toggles via overrides; Local-<name> rows read-only) -->
    <section x-show="view === 'job' && active === 'scrapers'" class="card rounded-xl p-5">
      <div class="flex items-start justify-between gap-4 mb-3">
        <div>
          <h3 class="font-semibold">Scraper controls</h3>
          <p class="text-xs text-slate-400 mt-1">Per-job toggles (overrides on <code>scrapers.enabled</code>). The active job applies on leave / Apply. Local folders are managed in <button class="link-btn" @click="active='jobSettings'">Job Settings → Local folders</button>.</p>
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
              <div x-show="scraperTest[s.name] && !scraperTest[s.name].pending" class="text-xs mt-0.5"
                :class="scraperTest[s.name]?.ok ? 'text-emerald-400' : 'text-rose-400'"
                x-text="(scraperTest[s.name]?.ok ? '✓ ' : '✕ ') + (scraperTest[s.name]?.message || '') + (scraperTest[s.name]?.latency_ms != null ? ' (' + scraperTest[s.name].latency_ms + 'ms)' : '')"></div>
            </div>
            <!-- Togglable scrapers get a switch; Local-<name> rows are read-only status. -->
            <template x-if="s.kind !== 'local'">
              <div class="flex items-center gap-3 shrink-0">
                <button type="button" @click="testScraper(s.name)" :disabled="!!scraperTest[s.name]?.pending"
                  class="px-2 py-0.5 text-xs bg-slate-700 hover:bg-slate-600 rounded disabled:opacity-50">
                  <span x-text="scraperTest[s.name]?.pending ? 'Testing…' : 'Test'"></span>
                </button>
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
            <template x-if="s.kind === 'local'">
              <span class="pill px-2 py-0.5 rounded" :class="s.enabled ? 'bg-emerald-900/60 text-emerald-300' : 'bg-slate-800 text-slate-400'" x-text="s.enabled ? 'enabled' : 'disabled'"></span>
            </template>
          </div>
        </template>
      </div>
    </section>

    <!-- VISION (global endpoints readout + this job's captioning/scoring) -->
    <section x-show="view === 'job' && active === 'vision'" class="space-y-4">
      <!-- Vision workers — the local-LLM fleet (per-job inherit/override). One
           row per GPU/host; lmstudio + llamacpp speak the OpenAI /v1 API, ollama
           its native API. Inherits the preset's global fleet until overridden. -->
      <div class="card rounded-xl p-5" x-show="je.loaded && je.eff">
        <div class="flex items-center justify-between mb-1">
          <h3 class="font-semibold">Vision workers{{ tip('The local LLM endpoints that classify images. Add one row per GPU/host: LM Studio, llama.cpp / vLLM / koboldcpp / LocalAI (OpenAI-compatible), or Ollama. Each enabled worker runs as its own subprocess against the shared queue, so a fleet of hosts scales throughput for a big backlog.', 'one row per GPU box across a local mesh / RunPod cluster') }}</h3>
          <div class="flex items-center gap-2">
            <span x-show="!isOver('vision.workers')" class="pill px-1.5 py-0.5 rounded bg-slate-800 text-slate-400" title="inherited from the preset / global fleet">global</span>
            <button x-show="isOver('vision.workers')" @click="resetOverride('vision.workers')" class="text-xs link-btn" title="reset to the global fleet">reset ↺</button>
          </div>
        </div>
        <p class="text-xs text-slate-400 mb-3">Inherits the preset's global fleet until you customise it for this job. Leave <b>Model</b> blank to auto-detect on connect; <b>Test</b> fills the model list. Private / LAN / Tailscale URLs are fine. Restart the pipeline (or Apply) to launch changes.</p>
        <div class="space-y-2">
          <template x-for="(w, idx) in visionFleet()" :key="idx">
            <div class="bg-slate-900/60 border border-slate-800 rounded p-3">
              <div class="grid md:grid-cols-12 gap-2">
                <input :value="w.name" @change="setFleetField(idx,'name',$event.target.value)" placeholder="Name"
                       class="md:col-span-3 bg-slate-800 border border-slate-700 rounded px-2 py-1.5 text-sm"/>
                <select @change="setFleetField(idx,'provider',$event.target.value)"
                        class="md:col-span-2 bg-slate-800 border border-slate-700 rounded px-2 py-1.5 text-sm">
                  <template x-for="p in visionProviders" :key="p">
                    <option :value="p" :selected="p === w.provider" x-text="visionProviderLabels[p]"></option>
                  </template>
                </select>
                <input :value="w.base_url" @change="setFleetField(idx,'base_url',$event.target.value)" placeholder="http://host:port"
                       class="md:col-span-4 bg-slate-800 border border-slate-700 rounded px-2 py-1.5 text-sm font-mono"/>
                <select @change="setFleetField(idx,'model',$event.target.value)" title="Run Test to load the model list"
                        class="md:col-span-3 bg-slate-800 border border-slate-700 rounded px-2 py-1.5 text-sm">
                  <option value="" :selected="!w.model">(auto-detect on connect)</option>
                  <template x-for="m in fleetModelOptions(idx)" :key="m">
                    <option :value="m" :selected="m === w.model" x-text="m"></option>
                  </template>
                </select>
              </div>
              <div class="grid md:grid-cols-12 gap-2 mt-2 items-center">
                <input type="password" :value="w.api_key" @change="setFleetField(idx,'api_key',$event.target.value)" placeholder="API key (optional)"
                       autocomplete="off" class="md:col-span-4 bg-slate-800 border border-slate-700 rounded px-2 py-1.5 text-sm"/>
                <label class="md:col-span-2 inline-flex items-center gap-2 text-xs cursor-pointer">
                  <input type="checkbox" :checked="w.enabled" @change="setFleetField(idx,'enabled',$event.target.checked)" class="accent-indigo-500"/>
                  <span x-text="w.enabled ? 'Enabled' : 'Disabled'"></span>
                </label>
                <div class="md:col-span-6 flex items-center gap-2 justify-end">
                  <span class="text-xs truncate max-w-[16rem]" x-show="fleetTest[idx] && !fleetTest[idx].testing"
                        :class="fleetTest[idx]?.ok ? 'text-emerald-400' : 'text-rose-400'" x-text="fleetTest[idx]?.message"></span>
                  <button @click="testFleetWorker(idx)" :disabled="!!fleetTest[idx]?.testing"
                          class="px-2 py-1 text-xs bg-slate-700 hover:bg-slate-600 rounded disabled:opacity-50">
                    <span x-text="fleetTest[idx]?.testing ? 'Testing…' : 'Test'"></span>
                  </button>
                  <button @click="removeFleetWorker(idx)" class="px-2 py-1 text-xs bg-rose-900/60 hover:bg-rose-800 text-rose-100 rounded">Remove</button>
                </div>
              </div>
            </div>
          </template>
          <div x-show="!visionFleet().length" class="text-xs text-slate-500 italic">No vision workers — add one so images get classified.</div>
        </div>
        <button @click="addFleetWorker()" class="mt-2 px-3 py-1.5 text-sm bg-slate-700 hover:bg-slate-600 rounded">+ Add vision worker</button>
        <div class="text-xs text-slate-500 mt-3" x-text="'Running: ' + ((status.pipeline?.vision_workers || []).join(', ') || '(none)')"></div>
      </div>

      <!-- Fleet health telemetry — live probe of each worker endpoint. -->
      <div class="card rounded-xl p-5">
        <div class="flex items-center justify-between mb-2">
          <h3 class="font-semibold">Fleet health{{ tip('Live latency/reachability probe of every enabled worker endpoint in this job\'s fleet. On-demand — click Probe.') }}</h3>
          <button @click="loadVisionHealth()" :disabled="visionHealth.loading"
                  class="px-2 py-1 text-xs bg-slate-700 hover:bg-slate-600 rounded disabled:opacity-50">
            <span x-text="visionHealth.loading ? 'Probing…' : 'Probe'"></span>
          </button>
        </div>
        <div x-show="visionHealth.error" x-cloak class="text-xs text-rose-300" x-text="visionHealth.error"></div>
        <div x-show="!visionHealth.fleet.length && !visionHealth.loading" class="text-xs text-slate-500">No fleet workers to probe.</div>
        <div class="space-y-1">
          <template x-for="w in visionHealth.fleet" :key="w.id">
            <div class="flex items-center justify-between text-xs border-b border-slate-800 py-1">
              <span class="font-mono" x-text="(w.name || w.id) + ' · ' + (w.base_url || '')"></span>
              <span x-show="visionHealth.probes[w.id]"
                    :class="(visionHealth.probes[w.id] && visionHealth.probes[w.id].ok) ? 'text-emerald-400' : 'text-rose-400'"
                    x-text="visionHealth.probes[w.id] ? ((visionHealth.probes[w.id].ok ? '✓ ' + (visionHealth.probes[w.id].latency_ms ?? '?') + 'ms' : '✗ ' + (visionHealth.probes[w.id].error || 'down'))) : ''"></span>
            </div>
          </template>
        </div>
      </div>

      <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <!-- Cloud worker (Groq) — global toggle, runs alongside the local fleet. -->
        <div class="card rounded-xl p-5">
          <h3 class="font-semibold mb-2">Cloud worker — Groq{{ tip('Optional cloud vision worker (Llama-4 Scout): fast, handles NSFW, runs in parallel with your local fleet. Uses GROQ_API_KEY(S) from Settings. Global, not per-job.') }}</h3>
          <div class="flex items-center justify-between">
            <label class="inline-flex items-center gap-2 cursor-pointer">
              <input type="checkbox" :checked="groqEnabled()" @change="toggleVisionWorker('balanced-groq', $event.target.checked)" class="accent-indigo-500"/>
              <span class="text-sm" x-text="groqEnabled() ? 'Enabled' : 'Disabled'"></span>
            </label>
            <div class="flex items-center gap-2">
              <span class="text-xs" x-show="providerTest.groq?.message" :class="providerTest.groq?.ok ? 'text-emerald-400' : 'text-rose-400'" x-text="providerTest.groq?.message"></span>
              <button @click="testProvider('groq')" :disabled="!!providerTest.groq?.testing" class="px-2 py-1 text-xs bg-slate-700 hover:bg-slate-600 rounded disabled:opacity-50">
                <span x-text="providerTest.groq?.testing ? 'Testing…' : 'Test'"></span>
              </button>
            </div>
          </div>
        </div>
        <div class="card rounded-xl p-5">
          <h3 class="font-semibold mb-3">Throttle (<span x-text="throttle + '%'"></span>)</h3>
          <input type="range" min="0" max="100" step="5" x-model="throttle" @change="setThrottle()" class="w-full accent-indigo-500"/>
        </div>
      </div>

      <!-- Cloud vision workers — global toggles. Test Connection fetches the
           provider's live model catalogue into the dropdown (mirrors the local
           fleet picker). Keys + the chosen model are saved via Global Settings. -->
      <div class="card rounded-xl p-5">
        <h3 class="font-semibold mb-1">Cloud vision workers{{ tip('Optional paid cloud workers that run alongside your local fleet. Enable, click Test Connection to pull the live model list, pick a model, then Save in Global Settings (the API key + chosen model live there). Global, not per-job.') }}</h3>
        <p class="text-xs text-slate-400 mb-3">Enter the API key in <button class="link-btn" @click="view='jobs'; active='settings'">Global Settings</button>, then Test here to load models. Save in Settings to persist the model.</p>
        <div class="grid md:grid-cols-2 gap-4">
          <template x-for="cp in cloudProviders" :key="cp">
            <div class="border border-slate-700 rounded-lg p-3 space-y-2">
              <div class="flex items-center justify-between">
                <label class="inline-flex items-center gap-2 cursor-pointer">
                  <input type="checkbox" :checked="cloudEnabled(cp)" @change="toggleVisionWorker(cloudWorkerName[cp], $event.target.checked)" class="accent-indigo-500"/>
                  <span class="text-sm font-medium" x-text="cloudProviderLabels[cp]"></span>
                </label>
                <button @click="testProvider(cp)" :disabled="!!(providerTest[cp] && providerTest[cp].testing)"
                        class="px-2 py-1 text-xs bg-slate-700 hover:bg-slate-600 rounded disabled:opacity-50">
                  <span x-text="(providerTest[cp] && providerTest[cp].testing) ? 'Testing…' : 'Test Connection'"></span>
                </button>
              </div>
              <div class="text-xs" x-show="providerTest[cp] && providerTest[cp].message"
                   :class="(providerTest[cp] && providerTest[cp].ok) ? 'text-emerald-400' : 'text-rose-400'"
                   x-text="providerTest[cp] && providerTest[cp].message"></div>
              <div>
                <label class="text-xs text-slate-400 block mb-1" x-text="(cp.toUpperCase()) + ' model'"></label>
                <select @change="settings[cloudModelKey(cp)] = $event.target.value; markSettingsDirty()"
                        class="w-full bg-slate-800 border border-slate-700 rounded px-2 py-1.5 text-xs">
                  <option value="" :selected="!(settings[cloudModelKey(cp)] || '').trim()">(auto-detect)</option>
                  <template x-for="m in cloudModelOptions(cp)" :key="m">
                    <option :value="m" :selected="settings[cloudModelKey(cp)] === m" x-text="m"></option>
                  </template>
                </select>
                <p class="text-[11px] text-slate-500 mt-1" x-show="!cloudModelOptions(cp).length">Click Test Connection to load models.</p>
              </div>
            </div>
          </template>
        </div>
      </div>

      <!-- Auto-captioning + scoring — PER JOB, inherit/override (v2). Each
           field shows its EFFECTIVE value; a 'global' chip means inherited,
           'reset ↺' removes the override. Auto-saves; the active job applies
           on leave / Apply. -->
      <div class="card rounded-xl p-5" x-show="je.loaded && je.eff">
        <div class="flex items-center justify-between mb-1">
          <h3 class="font-semibold">Auto-captioning</h3>
          <span class="text-xs text-emerald-300" x-text="je.savedFlash"></span>
        </div>
        <div x-show="je.error" x-cloak class="border text-xs px-3 py-2 rounded mb-3 bg-rose-950/60 border-rose-700 text-rose-200" x-text="je.error"></div>
        <template x-if="je.eff">
        <div class="grid md:grid-cols-2 gap-4">
          <div class="space-y-3">
            <div class="flex items-center gap-2">
              <label class="flex items-center gap-3 cursor-pointer flex-1">
                <input type="checkbox" :checked="effVal('captioning.enabled')"
                       @change="setOverride('captioning.enabled', $event.target.checked)"
                       class="w-10 h-5 appearance-none bg-slate-700 rounded-full relative transition checked:bg-indigo-500 before:content-[''] before:absolute before:top-0.5 before:left-0.5 before:w-4 before:h-4 before:bg-white before:rounded-full before:transition checked:before:translate-x-5"/>
                <span class="text-sm">Enable auto-captioning{{ tip('Writes a training caption <code>.txt</code> next to each kept image, produced by the same vision call in the chosen style. Off by default — it adds output tokens per image.') }}</span>
              </label>
              <span x-show="!isOver('captioning.enabled')" class="pill px-1.5 py-0.5 rounded bg-slate-800 text-slate-400" title="inherited from preset">global</span>
              <button x-show="isOver('captioning.enabled')" @click="resetOverride('captioning.enabled')" class="text-xs link-btn" title="reset to preset">reset ↺</button>
            </div>
            <div class="flex items-center gap-2">
              <label class="flex items-center gap-3 cursor-pointer flex-1">
                <input type="checkbox" :checked="effVal('captioning.overwrite')" :disabled="!effVal('captioning.enabled')"
                       @change="setOverride('captioning.overwrite', $event.target.checked)"
                       class="w-10 h-5 appearance-none bg-slate-700 rounded-full relative transition checked:bg-indigo-500 disabled:opacity-40 before:content-[''] before:absolute before:top-0.5 before:left-0.5 before:w-4 before:h-4 before:bg-white before:rounded-full before:transition checked:before:translate-x-5"/>
                <span class="text-sm">Overwrite existing captions{{ tip('When on, replaces any caption a scraper already saved. Off keeps the source-side prompt and only fills in gaps.') }}</span>
              </label>
              <span x-show="!isOver('captioning.overwrite')" class="pill px-1.5 py-0.5 rounded bg-slate-800 text-slate-400">global</span>
              <button x-show="isOver('captioning.overwrite')" @click="resetOverride('captioning.overwrite')" class="text-xs link-btn">reset ↺</button>
            </div>
            <div class="flex items-center gap-2">
              <label class="flex items-center gap-3 cursor-pointer flex-1">
                <input type="checkbox" :checked="effVal('topic_filters.require_prompt') === false"
                       @change="setOverride('topic_filters.require_prompt', !$event.target.checked)"
                       class="w-10 h-5 appearance-none bg-slate-700 rounded-full relative transition checked:bg-indigo-500 before:content-[''] before:absolute before:top-0.5 before:left-0.5 before:w-4 before:h-4 before:bg-white before:rounded-full before:transition checked:before:translate-x-5"/>
                <span class="text-sm">Ingest images without a prompt{{ tip('Accept images that have no generation prompt/caption (e.g. plain photos). Off = require a prompt — right for AI-art, wrong for real-photo datasets.') }}</span>
              </label>
              <span x-show="!isOver('topic_filters.require_prompt')" class="pill px-1.5 py-0.5 rounded bg-slate-800 text-slate-400">global</span>
              <button x-show="isOver('topic_filters.require_prompt')" @click="resetOverride('topic_filters.require_prompt')" class="text-xs link-btn">reset ↺</button>
            </div>
          </div>
          <div>
            <div class="flex items-center justify-between mb-1">
              <label class="text-xs text-slate-400">Caption style{{ tip('Format of the generated caption. <b>SD/Flux</b>: comma-separated tag phrases. <b>Booru tags</b>: underscored tags. <b>Natural language</b>: a descriptive sentence.', 'booru tags suit anime; natural language suits photos') }}</label>
              <span x-show="!isOver('captioning.style')" class="pill px-1.5 py-0.5 rounded bg-slate-800 text-slate-400">global</span>
              <button x-show="isOver('captioning.style')" @click="resetOverride('captioning.style')" class="text-xs link-btn">reset ↺</button>
            </div>
            <select @change="setOverride('captioning.style', $event.target.value)"
                    :disabled="!effVal('captioning.enabled')"
                    class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 disabled:opacity-50"
                    :class="!isOver('captioning.style') ? 'text-slate-400' : ''">
              <template x-for="cs in captionStyles" :key="cs.id">
                <option :value="cs.id" :selected="effVal('captioning.style') === cs.id" x-text="cs.label"></option>
              </template>
            </select>
          </div>
        </div>
        </template>
      </div>

      <div class="card rounded-xl p-5" x-show="je.loaded && je.eff">
        <h3 class="font-semibold mb-3">Quality scoring</h3>
        <p class="text-xs text-slate-400 mb-3">Two 0-100 scores. <code>OVR</code> = craft, <code>REL</code> = topic relevance. Either threshold forces DISCARD when not met; 0 disables.</p>
        <template x-if="je.eff">
        <div class="grid md:grid-cols-2 gap-4">
          <div>
            <div class="flex items-center justify-between mb-1">
              <span class="text-xs text-slate-400">Minimum OVR score{{ tip('Hard floor on the model&#39;s <b>overall</b> quality score (0-100). Images below this go straight to DISCARD before category routing.', 'e.g. 0 disables the gate; 60 drops weak shots') }}</span>
              <span x-show="!isOver('scoring.ovr_min')" class="pill px-1.5 py-0.5 rounded bg-slate-800 text-slate-400">global</span>
              <button x-show="isOver('scoring.ovr_min')" @click="resetOverride('scoring.ovr_min')" class="text-xs link-btn">reset ↺</button>
            </div>
            <div class="flex items-center gap-3">
              <input type="range" min="0" max="100" step="1" :value="effVal('scoring.ovr_min')" @input="setOverride('scoring.ovr_min', Number($event.target.value))"
                     class="flex-1 accent-indigo-500 cursor-pointer"/>
              <span class="text-sm font-mono w-9 text-right" :class="isOver('scoring.ovr_min') ? 'text-indigo-300' : 'text-slate-500'" x-text="(effVal('scoring.ovr_min') ?? 0)"></span>
            </div>
          </div>
          <div>
            <div class="flex items-center justify-between mb-1">
              <span class="text-xs text-slate-400">Minimum REL score{{ tip('Hard floor on the <b>relevance</b> score — how well the image matches this job&#39;s subject (0-100).', 'e.g. 0 disables; 50-60 trims off-topic scrapes') }}</span>
              <span x-show="!isOver('scoring.rel_min')" class="pill px-1.5 py-0.5 rounded bg-slate-800 text-slate-400">global</span>
              <button x-show="isOver('scoring.rel_min')" @click="resetOverride('scoring.rel_min')" class="text-xs link-btn">reset ↺</button>
            </div>
            <div class="flex items-center gap-3">
              <input type="range" min="0" max="100" step="1" :value="effVal('scoring.rel_min')" @input="setOverride('scoring.rel_min', Number($event.target.value))"
                     class="flex-1 accent-indigo-500 cursor-pointer"/>
              <span class="text-sm font-mono w-9 text-right" :class="isOver('scoring.rel_min') ? 'text-indigo-300' : 'text-slate-500'" x-text="(effVal('scoring.rel_min') ?? 0)"></span>
            </div>
          </div>
          <div class="md:col-span-2">
            <div class="flex items-center justify-between mb-1">
              <span class="text-xs text-slate-400">Scoring notes{{ tip('Free text appended to the vision model&#39;s scoring rubric for this job — steer what &quot;quality&quot; means here.', 'e.g. Reward golden-hour light; penalise harsh on-camera flash') }}</span>
              <span x-show="!isOver('scoring.notes')" class="pill px-1.5 py-0.5 rounded bg-slate-800 text-slate-400">global</span>
              <button x-show="isOver('scoring.notes')" @click="resetOverride('scoring.notes')" class="text-xs link-btn">reset ↺</button>
            </div>
            <textarea :value="effVal('scoring.notes')" @input="setOverride('scoring.notes', $event.target.value)" rows="3"
                      placeholder="e.g. prefer golden-hour natural light, penalise heavy over-smoothing"
                      class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 font-mono text-xs" :class="!isOver('scoring.notes') ? 'text-slate-400' : ''"></textarea>
          </div>
        </div>
        </template>
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

    <!-- QUEUE (job queue strip + this job's image queue) -->
    <section x-show="view === 'job' && active === 'queue'" class="space-y-4">
      <!-- Job-queue strip: the run order across jobs, as cards. The card for
           the job you're viewing is highlighted; clicking another drills into
           that job's image queue. -->
      <div class="card rounded-xl p-4">
        <div class="flex items-center justify-between mb-3">
          <h3 class="font-semibold">Job queue</h3>
          <button @click="backToJobs()" class="text-xs link-btn">Manage run queue →</button>
        </div>
        <div class="flex flex-wrap gap-2">
          <template x-for="jb in jobsList.filter(j => j.is_active || j.queue_position !== null)" :key="'jq_'+jb.slug">
            <button @click="openJob(jb.slug)"
              class="text-left px-3 py-2 rounded border transition"
              :class="jb.slug === currentJob ? 'bg-indigo-600/30 border-indigo-400' : 'bg-slate-900/60 border-slate-800 hover:border-slate-600'">
              <div class="flex items-center gap-2">
                <span class="pill px-1.5 py-0.5 rounded" :class="jobStatusClass(jb.status)" x-text="jb.status"></span>
                <span class="font-mono text-xs" x-text="jb.slug"></span>
              </div>
              <div class="text-[11px] text-slate-400 mt-1" x-text="'queued ' + jb.queued + ' · sorted ' + jb.sorted"></div>
            </button>
          </template>
          <template x-if="jobsList.filter(j => j.is_active || j.queue_position !== null).length === 0">
            <span class="text-xs text-slate-500">No active or queued jobs.</span>
          </template>
        </div>
      </div>

      <!-- First-run welcome card: shown only when nothing has been queued OR
           classified yet for THIS job. Keeps the empty state from looking like a bug. -->
      <template x-if="(status.queue?.total ?? 0) === 0 && (status.sorted?.total ?? 0) === 0">
        <div class="card rounded-xl p-6">
          <div class="flex items-start gap-4">
            <img src="/brand/logo-transparent-dark.png" alt="" width="64" height="64"
                 class="shrink-0"/>
            <div class="flex-1">
              <h3 class="font-brand text-2xl font-medium tracking-tight">Nothing here yet</h3>
              <p class="text-sm text-slate-300 mt-1 mb-4">
                This job has no queued or sorted images. Configure its scrapers and target, activate it, then start the pipeline.
              </p>
              <div class="flex flex-wrap gap-2">
                <button @click="active = 'scrapers'" class="px-3 py-2 bg-indigo-600 hover:bg-indigo-500 rounded text-sm">Configure scrapers</button>
                <button @click="active = 'jobSettings'" class="px-3 py-2 bg-indigo-600 hover:bg-indigo-500 rounded text-sm">Job settings</button>
                <button @click="active = 'vision'" class="px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded text-sm">Vision worker</button>
              </div>
              <p class="text-xs text-slate-500 mt-4">
                Want a demo first? Run <code class="font-brand bg-slate-800 px-1.5 py-0.5 rounded">python tools/seed_demo_data.py</code> from the repo root.
              </p>
            </div>
          </div>
        </div>
      </template>

      <div class="card rounded-xl p-5">
      <h3 class="font-semibold mb-3">Queue (newest 60) <span class="text-xs font-normal text-slate-500" x-text="'· ' + currentJob"></span></h3>
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

    <!-- LOGS (job-scoped Historical) -->
    <section x-show="view === 'job' && active === 'logs'" class="card rounded-xl p-5">
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

    <!-- JOB SETTINGS (v2 inherit/override editor) ───────────────────────────
         subject is job-owned; everything else inherits from the job's preset
         and is edited as a sparse override. Each field shows its EFFECTIVE
         value with a 'global' chip (inherited) or 'reset ↺' (overridden).
         Auto-saves (debounced); the active job applies on leave / Apply. -->
    <section x-show="view === 'job' && active === 'jobSettings'" class="space-y-4">
      <!-- Header: subject + preset + save/apply state. -->
      <div class="card rounded-xl p-5" x-show="je.loaded">
        <div class="flex items-start justify-between gap-4 mb-3">
          <div>
            <h3 class="font-semibold" x-text="je.name"></h3>
            <p class="text-xs text-slate-400">
              Subject is this job's; other fields inherit from the
              <span class="font-mono text-slate-300" x-text="je.preset"></span> preset until you override them. Changes save automatically.
            </p>
          </div>
          <div class="flex items-center gap-3">
            <span class="text-xs text-emerald-300" x-text="je.savedFlash"></span>
            <template x-if="currentJob === jobsActive && je.applyPending">
              <button @click="applyActiveJob()" class="px-3 py-1.5 text-xs bg-amber-500 hover:bg-amber-400 text-slate-900 rounded font-medium"
                title="Re-project + restart the running supervisor with these edits">Apply &amp; restart</button>
            </template>
          </div>
        </div>
        <div x-show="je.error" x-cloak class="border text-xs px-3 py-2 rounded mb-3 bg-rose-950/60 border-rose-700 text-rose-200" x-text="je.error"></div>
        <template x-if="je.eff">
        <div class="grid md:grid-cols-2 gap-4">
          <label class="block">
            <span class="text-xs text-slate-400">Subject (what this job is about — always per-job)</span>
            <input :value="je.subject" @input="setSubject($event.target.value)"
                   class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1"/>
          </label>
          <label class="block">
            <span class="text-xs text-slate-400">Preset (inherited defaults)</span>
            <!-- Bind :selected on each option (not :value on the select): with
                 x-for-rendered options, :value applies before the options exist
                 and never re-applies, so the select would wrongly show the first
                 entry instead of je.preset. -->
            <select @change="setPreset($event.target.value)"
                    class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1">
              <template x-for="p in je.presets" :key="p">
                <option :value="p" :selected="p === je.preset"
                        x-text="p + (p === je.default_preset ? ' (default)' : '')"></option>
              </template>
            </select>
          </label>
        </div>
        </template>
      </div>

      <!-- Media types (inheritable): what scrapers fetch + the queue classifies. -->
      <div class="card rounded-xl p-5" x-show="je.loaded && je.eff">
        <div class="flex items-center justify-between mb-1">
          <h3 class="font-semibold">Media types</h3>
          <template x-if="isOver('media.types') || isOver('media.image_exts') || isOver('media.video_exts')">
            <button @click="resetOverride('media.types'); resetOverride('media.image_exts'); resetOverride('media.video_exts')" class="text-xs link-btn">reset ↺</button>
          </template>
        </div>
        <p class="text-xs text-slate-400 mb-3">What every scraper fetches and the queue classifies. Enabling <b>Video</b> turns on the video-classify lane (needs the <code>.[video]</code> extra) and runs image + video in parallel when both are on. Scrapers that only produce one kind sit out when it isn't selected.</p>
        <template x-if="je.eff">
          <div>
            <div class="flex gap-5 mb-3">
              <label class="flex items-center gap-2 cursor-pointer">
                <input type="checkbox" :checked="(effVal('media.types')||[]).includes('image')" @change="toggleMediaType('image', $event.target.checked)" class="accent-indigo-500"/>
                <span class="text-sm">Images</span>
              </label>
              <label class="flex items-center gap-2 cursor-pointer">
                <input type="checkbox" :checked="(effVal('media.types')||[]).includes('video')" @change="toggleMediaType('video', $event.target.checked)" class="accent-indigo-500"/>
                <span class="text-sm">Video</span>
              </label>
              <span x-show="!isOver('media.types')" class="pill px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 self-center text-xs">global</span>
            </div>
            <div class="grid md:grid-cols-2 gap-4">
              <label class="block">
                <span class="text-xs text-slate-400">Image extensions{{ tip('Comma-separated. Pillow must be able to open them.', '.jpg, .jpeg, .png, .webp') }}</span>
                <input :value="effList('media.image_exts')" @change="setOverrideList('media.image_exts', $event.target.value)" placeholder=".jpg, .jpeg, .png, .webp" class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1 font-mono text-xs"/>
              </label>
              <label class="block">
                <span class="text-xs text-slate-400">Video extensions{{ tip('Comma-separated. The .[video] ffmpeg backend must decode them.', '.mp4, .webm, .mov, .mkv, .avi') }}</span>
                <input :value="effList('media.video_exts')" @change="setOverrideList('media.video_exts', $event.target.value)" placeholder=".mp4, .webm, .mov, .mkv, .avi" class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1 font-mono text-xs"/>
              </label>
            </div>
          </div>
        </template>
      </div>

      <!-- Topic filters (inheritable). -->
      <div class="card rounded-xl p-5" x-show="je.loaded && je.eff">
        <h3 class="font-semibold mb-3">Topic filters</h3>
        <template x-if="je.eff">
        <div class="grid md:grid-cols-2 gap-4">
          <div class="md:col-span-2">
            <div class="flex items-center justify-between mb-1">
              <span class="text-xs text-slate-400">Required keywords{{ tip('Comma-separated. When a source has a prompt/caption it must contain at least one of these, or the image is skipped.', 'e.g. drone, aerial, overhead') }}</span>
              <span x-show="!isOver('topic_filters.keywords_extra')" class="pill px-1.5 py-0.5 rounded bg-slate-800 text-slate-400">global</span>
              <button x-show="isOver('topic_filters.keywords_extra')" @click="resetOverride('topic_filters.keywords_extra')" class="text-xs link-btn">reset ↺</button>
            </div>
            <input :value="effList('topic_filters.keywords_extra')" @change="setOverrideList('topic_filters.keywords_extra', $event.target.value)"
                   class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2" :class="!isOver('topic_filters.keywords_extra') ? 'text-slate-400' : ''"/>
          </div>
          <div class="md:col-span-2">
            <div class="flex items-center justify-between mb-1">
              <span class="text-xs text-slate-400">Banned keywords{{ tip('Comma-separated. Any match in the prompt/caption rejects the image.', 'e.g. nsfw, watermark, meme') }}</span>
              <span x-show="!isOver('topic_filters.banned_keywords')" class="pill px-1.5 py-0.5 rounded bg-slate-800 text-slate-400">global</span>
              <button x-show="isOver('topic_filters.banned_keywords')" @click="resetOverride('topic_filters.banned_keywords')" class="text-xs link-btn">reset ↺</button>
            </div>
            <input :value="effList('topic_filters.banned_keywords')" @change="setOverrideList('topic_filters.banned_keywords', $event.target.value)"
                   placeholder="link in bio, dm me, patreon, onlyfans..."
                   class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2" :class="!isOver('topic_filters.banned_keywords') ? 'text-slate-400' : ''"/>
          </div>
          <div class="md:col-span-2">
            <div class="flex items-center justify-between mb-1">
              <span class="text-xs text-slate-400">Generation hints{{ tip('Comma-separated. Like required keywords, but matched against the generation prompt specifically.', 'e.g. 35mm, f/1.8, golden hour') }}</span>
              <span x-show="!isOver('topic_filters.generation_hints')" class="pill px-1.5 py-0.5 rounded bg-slate-800 text-slate-400">global</span>
              <button x-show="isOver('topic_filters.generation_hints')" @click="resetOverride('topic_filters.generation_hints')" class="text-xs link-btn">reset ↺</button>
            </div>
            <input :value="effList('topic_filters.generation_hints')" @change="setOverrideList('topic_filters.generation_hints', $event.target.value)"
                   placeholder="photorealistic, cinematic, cfg, lora..."
                   class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2" :class="!isOver('topic_filters.generation_hints') ? 'text-slate-400' : ''"/>
          </div>
          <div>
            <div class="flex items-center justify-between mb-1">
              <span class="text-xs text-slate-400">Minimum prompt length{{ tip('Reject images whose prompt/caption is shorter than this many characters — filters out junk one-word captions.', 'e.g. 0 = no minimum; 40 is a sane floor') }}</span>
              <span x-show="!isOver('topic_filters.min_prompt_length')" class="pill px-1.5 py-0.5 rounded bg-slate-800 text-slate-400">global</span>
              <button x-show="isOver('topic_filters.min_prompt_length')" @click="resetOverride('topic_filters.min_prompt_length')" class="text-xs link-btn">reset ↺</button>
            </div>
            <input type="number" min="0" :value="effVal('topic_filters.min_prompt_length')" @input="setOverride('topic_filters.min_prompt_length', Number($event.target.value))"
                   class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2" :class="!isOver('topic_filters.min_prompt_length') ? 'text-slate-400' : ''"/>
          </div>
          <div class="flex items-center gap-2 mt-6">
            <label class="flex items-center gap-3 cursor-pointer flex-1">
              <input type="checkbox" :checked="effVal('topic_filters.require_prompt')" @change="setOverride('topic_filters.require_prompt', $event.target.checked)"
                     class="w-10 h-5 appearance-none bg-slate-700 rounded-full relative transition checked:bg-indigo-500 before:content-[''] before:absolute before:top-0.5 before:left-0.5 before:w-4 before:h-4 before:bg-white before:rounded-full before:transition checked:before:translate-x-5"/>
              <span class="text-sm">Require a prompt</span>
            </label>
            <span x-show="!isOver('topic_filters.require_prompt')" class="pill px-1.5 py-0.5 rounded bg-slate-800 text-slate-400">global</span>
            <button x-show="isOver('topic_filters.require_prompt')" @click="resetOverride('topic_filters.require_prompt')" class="text-xs link-btn">reset ↺</button>
          </div>
        </div>
        </template>
      </div>

      <!-- Scraper targets (inheritable lists + Discord JSON). -->
      <div class="card rounded-xl p-5" x-show="je.loaded && je.eff">
        <h3 class="font-semibold mb-3">Scraper targets</h3>
        <template x-if="je.eff">
        <div class="grid md:grid-cols-1 gap-4">
          <div>
            <div class="flex items-center justify-between mb-1">
              <span class="text-xs text-slate-400">X.com accounts{{ tip('Comma-separated handles to scrape, <b>without</b> the @.', 'e.g. dronefeed, natureshots; leave empty to scrape search results only') }}</span>
              <span x-show="!isOver('scrapers.x_accounts')" class="pill px-1.5 py-0.5 rounded bg-slate-800 text-slate-400">global</span>
              <button x-show="isOver('scrapers.x_accounts')" @click="resetOverride('scrapers.x_accounts')" class="text-xs link-btn">reset ↺</button>
            </div>
            <input :value="effList('scrapers.x_accounts')" @change="setOverrideList('scrapers.x_accounts', $event.target.value)"
                   placeholder="account1,account2" class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2" :class="!isOver('scrapers.x_accounts') ? 'text-slate-400' : ''"/>
          </div>
          <div>
            <div class="flex items-center justify-between mb-1">
              <span class="text-xs text-slate-400">Reddit subreddits{{ tip('Comma-separated subreddits to pull from (no r/ prefix).', 'e.g. drones, earthporn, aerialphotography') }}</span>
              <span x-show="!isOver('scrapers.reddit_subreddits')" class="pill px-1.5 py-0.5 rounded bg-slate-800 text-slate-400">global</span>
              <button x-show="isOver('scrapers.reddit_subreddits')" @click="resetOverride('scrapers.reddit_subreddits')" class="text-xs link-btn">reset ↺</button>
            </div>
            <input :value="effList('scrapers.reddit_subreddits')" @change="setOverrideList('scrapers.reddit_subreddits', $event.target.value)"
                   class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2" :class="!isOver('scrapers.reddit_subreddits') ? 'text-slate-400' : ''"/>
          </div>
          <div>
            <div class="flex items-center justify-between mb-1">
              <span class="text-xs text-slate-400">Civitai domains{{ tip('Tick which Civitai hosts to scrape (civitai.com and/or the civitai.red mirror).') }}</span>
              <span x-show="!isOver('scrapers.civitai_domains')" class="pill px-1.5 py-0.5 rounded bg-slate-800 text-slate-400">global</span>
              <button x-show="isOver('scrapers.civitai_domains')" @click="resetOverride('scrapers.civitai_domains')" class="text-xs link-btn">reset ↺</button>
            </div>
            <div class="flex items-center gap-5 py-1.5" :class="!isOver('scrapers.civitai_domains') ? 'opacity-70' : ''">
              <label class="inline-flex items-center gap-2 text-sm cursor-pointer">
                <input type="checkbox" :checked="civOn('civitai.com')" @change="civToggle('civitai.com', $event.target.checked)" class="accent-indigo-500"/> civitai.com
              </label>
              <label class="inline-flex items-center gap-2 text-sm cursor-pointer">
                <input type="checkbox" :checked="civOn('civitai.red')" @change="civToggle('civitai.red', $event.target.checked)" class="accent-indigo-500"/> civitai.red
              </label>
            </div>
          </div>
          <div>
            <div class="flex items-center justify-between mb-1">
              <span class="text-xs text-slate-400">Discord channels JSON{{ tip('JSON listing the channels to scrape. Each needs its <code>id</code> and <code>guild</code> id; <code>kind</code> picks how images are pulled.', 'e.g. {&quot;channels&quot;:[{&quot;id&quot;:&quot;123&quot;,&quot;guild&quot;:&quot;456&quot;,&quot;kind&quot;:&quot;png_embed&quot;}]}') }}</span>
              <span x-show="!isOver('scrapers.discord_channels_json')" class="pill px-1.5 py-0.5 rounded bg-slate-800 text-slate-400">global</span>
              <button x-show="isOver('scrapers.discord_channels_json')" @click="resetOverride('scrapers.discord_channels_json')" class="text-xs link-btn">reset ↺</button>
            </div>
            <textarea :value="effVal('scrapers.discord_channels_json')" @input="setOverride('scrapers.discord_channels_json', $event.target.value)" rows="3"
              placeholder='{"channels":[{"id":"...","name":"...","guild":"...","kind":"png_embed"}]}'
              class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 font-mono text-xs" :class="!isOver('scrapers.discord_channels_json') ? 'text-slate-400' : ''"></textarea>
          </div>
        </div>
        </template>
      </div>

      <!-- gallery-dl (overridden wholesale as scrapers.gallery_dl). -->
      <div class="card rounded-xl p-5" x-show="je.loaded && je.eff">
        <div class="flex items-center justify-between mb-3">
          <h3 class="font-semibold">gallery-dl</h3>
          <div>
            <span x-show="!isOver('scrapers.gallery_dl')" class="pill px-1.5 py-0.5 rounded bg-slate-800 text-slate-400">global</span>
            <button x-show="isOver('scrapers.gallery_dl')" @click="resetOverride('scrapers.gallery_dl')" class="text-xs link-btn">reset ↺</button>
          </div>
        </div>
        <template x-if="je.eff">
        <div class="grid md:grid-cols-2 gap-4">
          <label class="flex items-center gap-3 cursor-pointer">
            <input type="checkbox" :checked="effVal('scrapers.gallery_dl.enabled')" @change="setOverride('scrapers.gallery_dl.enabled', $event.target.checked)"
                   class="w-10 h-5 appearance-none bg-slate-700 rounded-full relative transition checked:bg-indigo-500 before:content-[''] before:absolute before:top-0.5 before:left-0.5 before:w-4 before:h-4 before:bg-white before:rounded-full before:transition checked:before:translate-x-5"/>
            <span class="text-sm">Enable gallery-dl for this job</span>
          </label>
          <label class="block">
            <span class="text-xs text-slate-400">Images per URL{{ tip('Max images to pull from each URL per run.', 'e.g. 200; keep modest to avoid huge first runs') }}</span>
            <input type="number" min="1" max="5000" :value="effVal('scrapers.gallery_dl.limit_per_url')" @input="setOverride('scrapers.gallery_dl.limit_per_url', Number($event.target.value))"
                   class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1"/>
          </label>
          <label class="block md:col-span-2">
            <span class="text-xs text-slate-400">URLs{{ tip('One gallery-dl-supported URL per line. Lines starting with <code>#</code> are ignored.', 'e.g. https://www.pixiv.net/en/users/12345') }}</span>
            <textarea :value="effUrls()" @change="setOverrideUrls($event.target.value)" rows="4"
                      placeholder="https://www.pixiv.net/users/123456&#10;https://danbooru.donmai.us/posts?tags=portrait"
                      class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1 font-mono text-xs"></textarea>
          </label>
          <label class="block">
            <span class="text-xs text-slate-400">Cookies file{{ tip('Path to a Netscape-format <code>cookies.txt</code> for sites that need a login (Pixiv, Twitter, FurAffinity).', 'export it with a browser cookies.txt extension') }}</span>
            <input :value="effVal('scrapers.gallery_dl.cookies_file')" @input="setOverride('scrapers.gallery_dl.cookies_file', $event.target.value)" placeholder="C:\\Users\\you\\cookies.txt"
                   class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1 font-mono text-xs"/>
          </label>
          <label class="block">
            <span class="text-xs text-slate-400">Custom config path{{ tip('Path to a gallery-dl JSON config to override extractor options. Advanced; leave blank for defaults.', 'see the gallery-dl docs for the config schema') }}</span>
            <input :value="effVal('scrapers.gallery_dl.config_path')" @input="setOverride('scrapers.gallery_dl.config_path', $event.target.value)"
                   class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1 font-mono text-xs"/>
          </label>
        </div>
        </template>
      </div>

      <!-- yt-dlp (overridden wholesale as scrapers.yt_dlp). Mirrors gallery-dl. -->
      <div class="card rounded-xl p-5" x-show="je.loaded && je.eff">
        <div class="flex items-center justify-between mb-3">
          <h3 class="font-semibold">yt-dlp</h3>
          <div>
            <span x-show="!isOver('scrapers.yt_dlp')" class="pill px-1.5 py-0.5 rounded bg-slate-800 text-slate-400">global</span>
            <button x-show="isOver('scrapers.yt_dlp')" @click="resetOverride('scrapers.yt_dlp')" class="text-xs link-btn">reset ↺</button>
          </div>
        </div>
        <template x-if="je.eff">
        <div class="grid md:grid-cols-2 gap-4">
          <label class="flex items-center gap-3 cursor-pointer">
            <input type="checkbox" :checked="effVal('scrapers.yt_dlp.enabled')" @change="setOverride('scrapers.yt_dlp.enabled', $event.target.checked)"
                   class="w-10 h-5 appearance-none bg-slate-700 rounded-full relative transition checked:bg-indigo-500 before:content-[''] before:absolute before:top-0.5 before:left-0.5 before:w-4 before:h-4 before:bg-white before:rounded-full before:transition checked:before:translate-x-5"/>
            <span class="text-sm">Enable yt-dlp for this job</span>
          </label>
          <label class="block">
            <span class="text-xs text-slate-400">Frames per URL{{ tip('Max frames/images to pull from each URL per run.', 'e.g. 200; keep modest to avoid huge first runs') }}</span>
            <input type="number" min="1" max="5000" :value="effVal('scrapers.yt_dlp.limit')" @input="setOverride('scrapers.yt_dlp.limit', Number($event.target.value))"
                   class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1"/>
          </label>
          <label class="block md:col-span-2">
            <span class="text-xs text-slate-400">URLs{{ tip('One yt-dlp-supported URL per line. Lines starting with <code>#</code> are ignored.', 'e.g. https://www.youtube.com/watch?v=...') }}</span>
            <textarea :value="ytUrls()" @change="setYtUrls($event.target.value)" rows="4"
                      placeholder="https://www.youtube.com/watch?v=dQw4w9WgXcQ&#10;https://vimeo.com/123456789"
                      class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1 font-mono text-xs"></textarea>
          </label>
          <label class="block md:col-span-2">
            <span class="text-xs text-slate-400">Cookies file{{ tip('Path to a Netscape-format <code>cookies.txt</code> for sites that need a login (age-gated / members-only videos).', 'export it with a browser cookies.txt extension') }}</span>
            <input :value="effVal('scrapers.yt_dlp.cookies')" @input="setOverride('scrapers.yt_dlp.cookies', $event.target.value)" placeholder="C:\\Users\\you\\cookies.txt"
                   class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1 font-mono text-xs"/>
          </label>
        </div>
        </template>
      </div>

      <!-- Local folders (multi): scrapers.local_imports list. Each row is its
           own concurrent source. -->
      <div class="card rounded-xl p-5" x-show="je.loaded && je.eff">
        <div class="flex items-center justify-between mb-3">
          <div>
            <h3 class="font-semibold">Local folders</h3>
            <p class="text-xs text-slate-400">Each enabled folder is a concurrent source, surfaced on the Scrapers tab as <code>Local-&lt;name&gt;</code>.</p>
          </div>
          <div>
            <span x-show="!isOver('scrapers.local_imports')" class="pill px-1.5 py-0.5 rounded bg-slate-800 text-slate-400">global</span>
            <button x-show="isOver('scrapers.local_imports')" @click="resetOverride('scrapers.local_imports')" class="text-xs link-btn">reset ↺</button>
          </div>
        </div>
        <template x-if="je.eff">
        <div class="space-y-2">
          <template x-for="(f, i) in localFolders()" :key="i">
            <div class="grid grid-cols-12 gap-2 items-center bg-slate-900/40 border border-slate-800 rounded p-2">
              <input :value="f.name" @input="updateLocalFolder(i, 'name', $event.target.value)" placeholder="name"
                     class="col-span-3 bg-slate-800 border border-slate-700 rounded px-2 py-1 text-xs"/>
              <input :value="f.dir" @input="updateLocalFolder(i, 'dir', $event.target.value)" placeholder="absolute folder path"
                     class="col-span-6 bg-slate-800 border border-slate-700 rounded px-2 py-1 font-mono text-xs"/>
              <label class="col-span-2 flex items-center gap-1 text-xs">
                <input type="checkbox" :checked="f.enabled" @change="updateLocalFolder(i, 'enabled', $event.target.checked)"/> on
              </label>
              <button @click="removeLocalFolder(i)" class="col-span-1 px-2 py-1 text-xs bg-rose-900/60 hover:bg-rose-800 text-rose-100 rounded">✕</button>
            </div>
          </template>
          <button @click="addLocalFolder()" class="text-xs px-3 py-1.5 bg-slate-800 hover:bg-slate-700 rounded">+ Add local folder</button>
        </div>
        </template>
      </div>

      <!-- Categories (inheritable; reject traversal/reserved at the boundary). -->
      <div class="card rounded-xl p-5" x-show="je.loaded && je.eff">
        <div class="flex items-start justify-between gap-4 mb-3">
          <div>
            <h3 class="font-semibold">Categories</h3>
            <p class="text-xs text-slate-400">Keep-buckets + hints + global judgement rules. <code>DISCARD</code>/<code>CORRUPT</code> reserved.</p>
          </div>
          <div class="flex items-center gap-2">
            <span x-show="!isOver('categories')" class="pill px-1.5 py-0.5 rounded bg-slate-800 text-slate-400">global</span>
            <button x-show="isOver('categories')" @click="resetOverride('categories')" class="text-xs link-btn">reset categories ↺</button>
          </div>
        </div>
        <template x-if="je.eff">
        <div>
          <div class="space-y-2 mb-4">
            <template x-for="(cat, idx) in (effVal('categories') || [])" :key="idx">
              <div class="flex items-start gap-2 bg-slate-900/40 border border-slate-800 rounded p-2">
                <div class="flex-1 grid md:grid-cols-3 gap-2">
                  <input :value="cat.name" @input="(() => { const l = (effVal('categories')||[]).map(c=>({...c})); l[idx].name = $event.target.value; setOverride('categories', l); })()"
                         placeholder="CategoryName" class="w-full bg-slate-800 border border-slate-700 rounded px-2 py-1 font-mono text-xs"/>
                  <textarea :value="cat.hint" @input="(() => { const l = (effVal('categories')||[]).map(c=>({...c})); l[idx].hint = $event.target.value; setOverride('categories', l); })()"
                            rows="2" placeholder="when should the model pick this?" class="md:col-span-2 w-full bg-slate-800 border border-slate-700 rounded px-2 py-1 text-xs leading-snug"></textarea>
                </div>
                <button @click="(() => setOverride('categories', (effVal('categories')||[]).filter((_,i2)=>i2!==idx)))()"
                        class="px-2 py-1 text-xs bg-rose-900/60 hover:bg-rose-800 text-rose-100 rounded shrink-0">✕</button>
              </div>
            </template>
            <button @click="(() => { const l = (effVal('categories')||[]).slice(); if (l.length>=12) return; l.push({name:'',hint:''}); setOverride('categories', l); })()"
                    class="text-xs px-3 py-1.5 bg-slate-800 hover:bg-slate-700 rounded">+ Add category</button>
          </div>
          <div class="flex items-center justify-between mb-1">
            <span class="text-xs text-slate-400">Global judgement rules</span>
            <span x-show="!isOver('category_rules')" class="pill px-1.5 py-0.5 rounded bg-slate-800 text-slate-400">global</span>
            <button x-show="isOver('category_rules')" @click="resetOverride('category_rules')" class="text-xs link-btn">reset ↺</button>
          </div>
          <textarea :value="effVal('category_rules')" @input="setOverride('category_rules', $event.target.value)" rows="6"
                    class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 font-mono text-[11px] leading-snug" :class="!isOver('category_rules') ? 'text-slate-400' : ''"></textarea>
        </div>
        </template>
      </div>
    </section>

    <!-- PRESETS MANAGER (global) ─────────────────────────────────────────────
         List/create/clone/delete/set-default + a preset editor over the same
         inheritable field set. Presets have no inheritance — values are absolute. -->
    <section x-show="view === 'jobs' && active === 'presets'" class="space-y-4">
      <div class="card rounded-xl p-5">
        <div class="flex items-center justify-between mb-3">
          <div>
            <h3 class="font-semibold">Presets</h3>
            <p class="text-xs text-slate-400">Named default bundles a job inherits from. Edits auto-save.</p>
          </div>
        </div>
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          <template x-for="p in presetsList" :key="p">
            <div class="bg-slate-900/60 border border-slate-800 rounded p-3 flex items-center justify-between gap-2"
                 :class="presetEditor.open && presetEditor.name === p ? 'ring-1 ring-indigo-500' : ''">
              <div class="min-w-0">
                <div class="font-mono text-sm truncate" x-text="p"></div>
                <div class="text-[11px] text-amber-300" x-show="p === presetsDefault">default</div>
              </div>
              <div class="flex flex-wrap gap-1 shrink-0">
                <button @click="openPreset(p)" class="px-2 py-1 text-xs bg-indigo-600 hover:bg-indigo-500 rounded">Edit</button>
                <button @click="clonePreset(p)" class="px-2 py-1 text-xs bg-slate-700 hover:bg-slate-600 rounded">Clone</button>
                <a :href="'/api/presets/' + encodeURIComponent(p) + '/export'" download class="px-2 py-1 text-xs bg-slate-700 hover:bg-slate-600 rounded">Export</a>
                <button @click="resetPreset(p)" x-show="presetsBuiltins.includes(p)"
                        title="Restore this shipped preset to its built-in defaults"
                        class="px-2 py-1 text-xs bg-amber-900/50 hover:bg-amber-800 text-amber-100 rounded">Reset</button>
                <button @click="setDefaultPreset(p)" :disabled="p === presetsDefault" class="px-2 py-1 text-xs bg-slate-700 hover:bg-slate-600 rounded disabled:opacity-40">Default</button>
                <button @click="deletePreset(p)" :disabled="p === presetsDefault" class="px-2 py-1 text-xs bg-rose-900/60 hover:bg-rose-800 text-rose-100 rounded disabled:opacity-40">Delete</button>
              </div>
            </div>
          </template>
          <div class="bg-slate-900/40 border-dashed border-2 border-slate-700 rounded p-3">
            <div class="flex items-center justify-between mb-2">
              <div class="text-sm font-semibold">New preset</div>
              <button @click="$refs.presetImport.click()" :disabled="configImporting" class="px-2 py-1 text-xs bg-slate-700 hover:bg-slate-600 rounded disabled:opacity-50"><span x-text="configImporting ? 'Importing…' : 'Import…'"></span></button>
              <input x-ref="presetImport" type="file" accept="application/json,.json" class="hidden" @change="importConfigFile($event)"/>
            </div>
            <input x-model="newPreset.name" @keydown.enter="createPreset()" placeholder="Preset name"
                   class="w-full bg-slate-800 border border-slate-700 rounded px-2 py-1 text-sm mb-2"/>
            <label class="block text-xs text-slate-400 mb-1">Start from a copy of</label>
            <select x-model="newPreset.base_on" class="w-full bg-slate-800 border border-slate-700 rounded px-2 py-1 text-sm mb-2">
              <template x-for="p in presetsList" :key="'b_'+p">
                <option :value="p" x-text="p === presetsDefault ? (p + '  (default)') : p"></option>
              </template>
            </select>
            <button @click="createPreset()" class="px-3 py-1.5 bg-emerald-600 hover:bg-emerald-500 rounded text-sm font-medium w-full">Create</button>
          </div>
        </div>
      </div>

      <!-- Preset editor -->
      <div class="card rounded-xl p-5" x-show="presetEditor.open && presetEditor.cfg">
        <div class="flex items-center justify-between mb-3">
          <h3 class="font-semibold">Editing preset: <span class="font-mono" x-text="presetEditor.name"></span></h3>
          <div class="flex items-center gap-3">
            <span class="text-xs text-emerald-300" x-text="presetEditor.savedFlash"></span>
            <button @click="closePreset()" class="px-3 py-1.5 text-xs bg-slate-700 hover:bg-slate-600 rounded">Close</button>
          </div>
        </div>
        <div x-show="presetEditor.error" x-cloak class="border text-xs px-3 py-2 rounded mb-3 bg-rose-950/60 border-rose-700 text-rose-200" x-text="presetEditor.error"></div>
        <div class="text-[11px] text-slate-500 mb-3" x-show="presetEditor.referencedBy.length"
             x-text="'Used by ' + presetEditor.referencedBy.length + ' job(s): ' + presetEditor.referencedBy.join(', ')"></div>
        <template x-if="presetEditor.cfg">
        <div class="space-y-4">
          <div class="grid md:grid-cols-2 gap-3">
            <label class="block"><span class="text-xs text-slate-400">Required keywords{{ tip('Comma-separated. A source prompt/caption must contain at least one of these or the image is skipped.', 'e.g. drone, aerial, overhead') }}</span>
              <input :value="peList('topic_filters.keywords_extra')" @change="peSetList('topic_filters.keywords_extra', $event.target.value)" class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1"/></label>
            <label class="block"><span class="text-xs text-slate-400">Banned keywords{{ tip('Comma-separated. Any match in the prompt/caption rejects the image.', 'e.g. nsfw, watermark, meme') }}</span>
              <input :value="peList('topic_filters.banned_keywords')" @change="peSetList('topic_filters.banned_keywords', $event.target.value)" class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1"/></label>
            <label class="block"><span class="text-xs text-slate-400">Generation hints{{ tip('Comma-separated. Like required keywords, but matched against the generation prompt specifically.', 'e.g. 35mm, f/1.8, golden hour') }}</span>
              <input :value="peList('topic_filters.generation_hints')" @change="peSetList('topic_filters.generation_hints', $event.target.value)" class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1"/></label>
            <label class="block"><span class="text-xs text-slate-400">Minimum prompt length</span>
              <input type="number" min="0" :value="presetEditor.cfg.topic_filters.min_prompt_length" @input="peSet('topic_filters.min_prompt_length', Number($event.target.value))" class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1"/></label>
          </div>
          <div class="border-t border-slate-800 pt-3 mt-1">
            <div class="text-xs font-semibold text-slate-300 mb-1">Media types</div>
            <p class="text-[11px] text-slate-500 mb-2">What scrapers fetch and the queue classifies. Enabling Video turns on the video-classify lane; both run in parallel when selected. Extension lists are editable.</p>
            <div class="flex gap-5 mb-2">
              <label class="flex items-center gap-2 cursor-pointer"><input type="checkbox" :checked="((presetEditor.cfg.media||{}).types||['image']).includes('image')" @change="peToggleMedia('image', $event.target.checked)" class="accent-indigo-500"/><span class="text-sm">Images</span></label>
              <label class="flex items-center gap-2 cursor-pointer"><input type="checkbox" :checked="((presetEditor.cfg.media||{}).types||['image']).includes('video')" @change="peToggleMedia('video', $event.target.checked)" class="accent-indigo-500"/><span class="text-sm">Video</span></label>
            </div>
            <div class="grid md:grid-cols-2 gap-3">
              <label class="block"><span class="text-xs text-slate-400">Image extensions</span>
                <input :value="peList('media.image_exts')" @change="peSetList('media.image_exts', $event.target.value)" placeholder=".jpg, .jpeg, .png, .webp" class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1 font-mono text-xs"/></label>
              <label class="block"><span class="text-xs text-slate-400">Video extensions</span>
                <input :value="peList('media.video_exts')" @change="peSetList('media.video_exts', $event.target.value)" placeholder=".mp4, .webm, .mov, .mkv, .avi" class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1 font-mono text-xs"/></label>
            </div>
          </div>
          <div class="grid md:grid-cols-2 gap-3">
            <label class="block md:col-span-2"><span class="text-xs text-slate-400">X.com accounts{{ tip('Comma-separated handles, <b>without</b> the @. Empty scrapes search results only.', 'e.g. dronefeed, natureshots') }}</span>
              <input :value="peList('scrapers.x_accounts')" @change="peSetList('scrapers.x_accounts', $event.target.value)" class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1"/></label>
            <label class="block md:col-span-2"><span class="text-xs text-slate-400">Reddit subreddits{{ tip('Comma-separated subreddits (no r/ prefix).', 'e.g. drones, earthporn, aerialphotography') }}</span>
              <input :value="peList('scrapers.reddit_subreddits')" @change="peSetList('scrapers.reddit_subreddits', $event.target.value)" class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1"/></label>
            <div class="block md:col-span-2"><span class="text-xs text-slate-400">Civitai domains{{ tip('Tick which Civitai hosts to scrape (civitai.com and/or the civitai.red mirror).') }}</span>
              <div class="flex items-center gap-5 py-1.5 mt-1">
                <label class="inline-flex items-center gap-2 text-sm cursor-pointer">
                  <input type="checkbox" :checked="peCivOn('civitai.com')" @change="peCivToggle('civitai.com', $event.target.checked)" class="accent-indigo-500"/> civitai.com
                </label>
                <label class="inline-flex items-center gap-2 text-sm cursor-pointer">
                  <input type="checkbox" :checked="peCivOn('civitai.red')" @change="peCivToggle('civitai.red', $event.target.checked)" class="accent-indigo-500"/> civitai.red
                </label>
              </div></div>
          </div>
          <div class="grid md:grid-cols-2 gap-3">
            <label class="block"><span class="text-xs text-slate-400">Scrapers enabled (toggles)</span>
              <div class="mt-1 flex flex-wrap gap-2">
                <template x-for="nm in scraperNames" :key="'pe_'+nm">
                  <label class="flex items-center gap-1 text-xs bg-slate-800 border border-slate-700 rounded px-2 py-1">
                    <input type="checkbox" class="appearance-none w-9 h-5 bg-slate-700 rounded-full relative transition-colors checked:bg-indigo-500 before:content-[''] before:absolute before:top-0.5 before:left-0.5 before:w-4 before:h-4 before:bg-white before:rounded-full before:transition-transform checked:before:translate-x-4 shrink-0 cursor-pointer" :checked="(presetEditor.cfg.scrapers.enabled||{})[nm] !== false"
                           @change="(() => { const e = {...(presetEditor.cfg.scrapers.enabled||{})}; e[nm] = $event.target.checked; peSet('scrapers.enabled', e); })()"/>
                    <span x-text="nm"></span>
                  </label>
                </template>
              </div>
            </label>
            <label class="flex items-center gap-2 mt-5 cursor-pointer"><input type="checkbox" class="appearance-none w-9 h-5 bg-slate-700 rounded-full relative transition-colors checked:bg-indigo-500 before:content-[''] before:absolute before:top-0.5 before:left-0.5 before:w-4 before:h-4 before:bg-white before:rounded-full before:transition-transform checked:before:translate-x-4 shrink-0 cursor-pointer" :checked="presetEditor.cfg.scrapers.gallery_dl.enabled" @change="peSet('scrapers.gallery_dl.enabled', $event.target.checked)"/><span class="text-sm">gallery-dl enabled</span></label>
            <label class="flex items-center gap-2 mt-5 cursor-pointer"><input type="checkbox" class="appearance-none w-9 h-5 bg-slate-700 rounded-full relative transition-colors checked:bg-indigo-500 before:content-[''] before:absolute before:top-0.5 before:left-0.5 before:w-4 before:h-4 before:bg-white before:rounded-full before:transition-transform checked:before:translate-x-4 shrink-0 cursor-pointer" :checked="(presetEditor.cfg.scrapers.yt_dlp||{}).enabled" @change="peSet('scrapers.yt_dlp.enabled', $event.target.checked)"/><span class="text-sm">yt-dlp enabled</span></label>
          </div>
          <label class="block"><span class="text-xs text-slate-400">gallery-dl URLs{{ tip('One gallery-dl-supported URL per line; <code>#</code> lines are ignored.', 'e.g. https://www.pixiv.net/en/users/12345') }}</span>
            <textarea :value="peUrls()" @change="peSetUrls($event.target.value)" rows="3" class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1 font-mono text-xs"></textarea></label>
          <div class="grid md:grid-cols-2 gap-3">
            <label class="block"><span class="text-xs text-slate-400">yt-dlp URLs{{ tip('One yt-dlp-supported video URL per line; <code>#</code> lines are ignored.', 'e.g. https://www.youtube.com/watch?v=...') }}</span>
              <textarea :value="peYtUrls()" @input="peSetYtUrls($event.target.value)" rows="3" class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1 font-mono text-xs"></textarea></label>
            <div class="grid grid-cols-1 gap-3">
              <label class="block"><span class="text-xs text-slate-400">yt-dlp frames per URL</span>
                <input type="number" min="1" max="5000" :value="(presetEditor.cfg.scrapers.yt_dlp||{}).limit" @input="peSet('scrapers.yt_dlp.limit', Number($event.target.value))" class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1"/></label>
              <label class="block"><span class="text-xs text-slate-400">yt-dlp cookies file</span>
                <input :value="(presetEditor.cfg.scrapers.yt_dlp||{}).cookies" @input="peSet('scrapers.yt_dlp.cookies', $event.target.value)" placeholder="C:\\Users\\you\\cookies.txt" class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1 font-mono text-xs"/></label>
            </div>
          </div>

          <div>
            <div class="text-xs text-slate-400 mb-1">Local folders</div>
            <div class="space-y-2">
              <template x-for="(f, i) in peLocalFolders()" :key="'pel_'+i">
                <div class="grid grid-cols-12 gap-2 items-center bg-slate-900/40 border border-slate-800 rounded p-2">
                  <input :value="f.name" @input="peUpdateLocal(i, 'name', $event.target.value)" placeholder="name" class="col-span-3 bg-slate-800 border border-slate-700 rounded px-2 py-1 text-xs"/>
                  <input :value="f.dir" @input="peUpdateLocal(i, 'dir', $event.target.value)" placeholder="path" class="col-span-6 bg-slate-800 border border-slate-700 rounded px-2 py-1 font-mono text-xs"/>
                  <label class="col-span-2 flex items-center gap-1 text-xs"><input type="checkbox" :checked="f.enabled" @change="peUpdateLocal(i, 'enabled', $event.target.checked)"/> on</label>
                  <button @click="peRemoveLocal(i)" class="col-span-1 px-2 py-1 text-xs bg-rose-900/60 hover:bg-rose-800 text-rose-100 rounded">✕</button>
                </div>
              </template>
              <button @click="peAddLocal()" class="text-xs px-3 py-1.5 bg-slate-800 hover:bg-slate-700 rounded">+ Add local folder</button>
            </div>
          </div>

          <div>
            <div class="text-xs text-slate-400 mb-1">Categories</div>
            <div class="space-y-2 mb-2">
              <template x-for="(cat, idx) in (presetEditor.cfg.categories || [])" :key="'pec_'+idx">
                <div class="flex items-start gap-2 bg-slate-900/40 border border-slate-800 rounded p-2">
                  <input :value="cat.name" @input="(() => { const l = presetEditor.cfg.categories.map(c=>({...c})); l[idx].name = $event.target.value; peSet('categories', l); })()" placeholder="Name" class="bg-slate-800 border border-slate-700 rounded px-2 py-1 font-mono text-xs"/>
                  <textarea :value="cat.hint" @input="(() => { const l = presetEditor.cfg.categories.map(c=>({...c})); l[idx].hint = $event.target.value; peSet('categories', l); })()" rows="2" placeholder="hint" class="flex-1 bg-slate-800 border border-slate-700 rounded px-2 py-1 text-xs"></textarea>
                  <button @click="peRemoveCategory(idx)" class="px-2 py-1 text-xs bg-rose-900/60 hover:bg-rose-800 text-rose-100 rounded">✕</button>
                </div>
              </template>
              <button @click="peAddCategory()" class="text-xs px-3 py-1.5 bg-slate-800 hover:bg-slate-700 rounded">+ Add category</button>
            </div>
            <label class="block"><span class="text-xs text-slate-400">Global judgement rules</span>
              <textarea :value="presetEditor.cfg.category_rules" @input="peSet('category_rules', $event.target.value)" rows="5" class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1 font-mono text-[11px]"></textarea></label>
          </div>

          <div class="grid md:grid-cols-3 gap-3">
            <label class="block"><span class="text-xs text-slate-400">Min OVR{{ tip('Hard floor on the <b>overall</b> quality score (0-100); below this goes to DISCARD. 0 disables.', 'e.g. 60') }}</span>
              <div class="flex items-center gap-2 mt-1">
                <input type="range" min="0" max="100" step="1" :value="presetEditor.cfg.scoring.ovr_min" @input="peSet('scoring.ovr_min', Number($event.target.value))" class="flex-1 accent-indigo-500 cursor-pointer"/>
                <span class="text-xs font-mono text-indigo-300 w-8 text-right" x-text="(presetEditor.cfg.scoring.ovr_min ?? 0)"></span>
              </div></label>
            <label class="block"><span class="text-xs text-slate-400">Min REL{{ tip('Hard floor on the <b>relevance</b> score to the subject (0-100). 0 disables.', 'e.g. 50') }}</span>
              <div class="flex items-center gap-2 mt-1">
                <input type="range" min="0" max="100" step="1" :value="presetEditor.cfg.scoring.rel_min" @input="peSet('scoring.rel_min', Number($event.target.value))" class="flex-1 accent-indigo-500 cursor-pointer"/>
                <span class="text-xs font-mono text-indigo-300 w-8 text-right" x-text="(presetEditor.cfg.scoring.rel_min ?? 0)"></span>
              </div></label>
            <label class="flex items-center gap-2 mt-5 cursor-pointer"><input type="checkbox" class="appearance-none w-9 h-5 bg-slate-700 rounded-full relative transition-colors checked:bg-indigo-500 before:content-[''] before:absolute before:top-0.5 before:left-0.5 before:w-4 before:h-4 before:bg-white before:rounded-full before:transition-transform checked:before:translate-x-4 shrink-0 cursor-pointer" :checked="presetEditor.cfg.captioning.enabled" @change="peSet('captioning.enabled', $event.target.checked)"/><span class="text-sm">Auto-caption</span></label>
          </div>

          <!-- Vision workers: the global default fleet jobs inheriting this preset use. -->
          <div class="mt-5">
            <div class="text-xs text-slate-400 mb-1">Vision workers{{ tip('The local LLM endpoints jobs inheriting this preset use to classify images. Add one per GPU/host — LM Studio, llama.cpp / vLLM / koboldcpp / LocalAI (OpenAI-compatible), or Ollama. Leave Model blank to auto-detect on Test. Private / LAN / Tailscale URLs are fine.') }}</div>
            <div class="space-y-2">
              <template x-for="(w, idx) in (presetEditor.cfg.vision?.workers || [])" :key="'pew_'+idx">
                <div class="bg-slate-800/60 border border-slate-700 rounded p-2">
                  <div class="grid md:grid-cols-12 gap-2">
                    <input :value="w.name" @change="peSetFleet(idx,'name',$event.target.value)" placeholder="Name" class="md:col-span-3 bg-slate-800 border border-slate-700 rounded px-2 py-1 text-xs"/>
                    <select @change="peSetFleet(idx,'provider',$event.target.value)" class="md:col-span-2 bg-slate-800 border border-slate-700 rounded px-2 py-1 text-xs">
                      <template x-for="p in visionProviders" :key="'pep_'+p"><option :value="p" :selected="p === w.provider" x-text="visionProviderLabels[p]"></option></template>
                    </select>
                    <input :value="w.base_url" @change="peSetFleet(idx,'base_url',$event.target.value)" placeholder="http://host:port" class="md:col-span-4 bg-slate-800 border border-slate-700 rounded px-2 py-1 text-xs font-mono"/>
                    <select @change="peSetFleet(idx,'model',$event.target.value)" title="Run Test to load the model list" class="md:col-span-3 bg-slate-800 border border-slate-700 rounded px-2 py-1 text-xs">
                      <option value="" :selected="!w.model">(auto-detect)</option>
                      <template x-for="m in peFleetModelOptions(idx)" :key="'pem_'+m"><option :value="m" :selected="m === w.model" x-text="m"></option></template>
                    </select>
                  </div>
                  <div class="grid md:grid-cols-12 gap-2 mt-1 items-center">
                    <input type="password" :value="w.api_key" @change="peSetFleet(idx,'api_key',$event.target.value)" placeholder="API key (optional)" autocomplete="off" class="md:col-span-4 bg-slate-800 border border-slate-700 rounded px-2 py-1 text-xs"/>
                    <label class="md:col-span-2 inline-flex items-center gap-1 text-xs cursor-pointer"><input type="checkbox" :checked="w.enabled" @change="peSetFleet(idx,'enabled',$event.target.checked)" class="accent-indigo-500"/><span x-text="w.enabled ? 'On' : 'Off'"></span></label>
                    <div class="md:col-span-6 flex items-center gap-2 justify-end">
                      <span class="text-[11px] truncate max-w-[12rem]" x-show="peFleetTest[idx] && !peFleetTest[idx].testing" :class="peFleetTest[idx]?.ok ? 'text-emerald-400' : 'text-rose-400'" x-text="peFleetTest[idx]?.message"></span>
                      <button @click="peTestFleet(idx)" :disabled="!!peFleetTest[idx]?.testing" class="px-2 py-0.5 text-xs bg-slate-700 hover:bg-slate-600 rounded disabled:opacity-50"><span x-text="peFleetTest[idx]?.testing ? 'Testing…' : 'Test'"></span></button>
                      <button @click="peRemoveFleet(idx)" class="px-2 py-0.5 text-xs bg-rose-900/60 hover:bg-rose-800 text-rose-100 rounded">Remove</button>
                    </div>
                  </div>
                </div>
              </template>
              <div x-show="!(presetEditor.cfg.vision?.workers || []).length" class="text-xs text-slate-500 italic">No vision workers in this preset.</div>
            </div>
            <button @click="peAddFleet()" class="mt-2 px-2 py-1 text-xs bg-slate-700 hover:bg-slate-600 rounded">+ Add vision worker</button>
          </div>
        </div>
        </template>
      </div>
    </section>

    <!-- GLOBAL SETTINGS ────────────────────────────────────────────────────
         Credentials, model endpoints, storage roots, and global UX. Per-job
         settings (topic / targets / scoring / captioning) live in Job Settings.
         Writes to .env via /api/settings. -->
    <section x-show="view === 'jobs' && active === 'settings'" class="space-y-4"
      @input="markSettingsDirty()" @change="markSettingsDirty()">

      <!-- Sticky action bar: Save/Reload for the .env form, always reachable at
           the top of the page (and pinned while scrolling). -->
      <div class="sticky top-0 z-20 py-3 bg-slate-950/90 backdrop-blur border-b border-slate-800
                  flex items-start justify-between gap-4">
        <div>
          <h3 class="font-semibold">Global settings</h3>
          <p class="text-xs text-slate-400">Credentials, endpoints, paths, and global UX. Changes write to <code>.env</code>. Stop + Start the pipeline to apply.</p>
        </div>
        <div class="flex gap-2 shrink-0">
          <button @click="reloadSettings()" class="px-3 py-1.5 text-xs bg-slate-800 hover:bg-slate-700 rounded">Reload</button>
          <button @click="saveSettings()" class="px-3 py-1.5 text-xs bg-indigo-600 hover:bg-indigo-500 rounded font-medium">Save</button>
        </div>
      </div>
      <template x-if="settingsBanner">
        <div class="border text-xs px-3 py-2 rounded"
          :class="settingsBannerOk ? 'bg-indigo-950/60 border-indigo-700 text-indigo-200'
                                   : 'bg-rose-950/60 border-rose-700 text-rose-200'"
          x-text="settingsBanner"></div>
      </template>

      <!-- Software updates — pull origin/main + restart. Mirrors the update toast. -->
      <div class="card rounded-xl p-5">
        <div class="flex items-start justify-between gap-4 mb-2">
          <div>
            <h3 class="font-semibold">Software updates</h3>
            <p class="text-xs text-slate-400">Pull the latest cull from <code>origin/main</code> and restart. Needs a clean git checkout.</p>
          </div>
          <div class="flex gap-2 shrink-0">
            <button @click="checkUpdate(true)" :disabled="!!update.checking"
              class="px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded text-sm disabled:opacity-50">
              <span x-text="update.checking ? 'Checking…' : 'Check for updates'"></span>
            </button>
            <button @click="runUpdate()" :disabled="!!update.running || update.behind === 0 || !!update.dirty"
              class="px-3 py-2 bg-amber-500 hover:bg-amber-400 text-slate-900 font-semibold rounded text-sm disabled:opacity-50"
              :title="update.dirty ? 'Commit or stash local changes first' : (update.behind === 0 ? 'Already up to date' : '')">
              <span x-text="update.running ? 'Updating…' : 'Update to latest &amp; restart'"></span>
            </button>
          </div>
        </div>
        <div class="text-xs text-slate-400 space-y-1">
          <div>Installed: <span class="font-mono text-slate-200" x-text="update.local_sha || '—'"></span>
            <template x-if="update.checked">
              <span>· latest <span class="font-mono text-slate-200" x-text="update.remote_sha || '—'"></span>
                <span x-show="update.behind > 0" class="text-amber-300" x-text="'· ' + update.behind + ' behind'"></span>
                <span x-show="update.checked && update.behind === 0 && !update.error" class="text-emerald-300">· up to date</span>
              </span>
            </template>
          </div>
          <div x-show="update.remote_subject" class="truncate text-slate-500" x-text="'→ ' + update.remote_subject"></div>
          <div x-show="update.dirty" class="text-rose-300">Working tree has uncommitted changes — commit or stash before updating.</div>
          <div x-show="update.error" class="text-rose-300" x-text="'⚠ ' + update.error"></div>
          <div x-show="update.running" class="text-slate-400">Pulling, reinstalling deps if needed, and relaunching. The dashboard will restart — refresh in a moment.</div>
        </div>
      </div>

      <!-- Backup & restore — export/import all jobs + presets, or a single preset/job. -->
      <div class="card rounded-xl p-5">
        <h3 class="font-semibold mb-2">Backup &amp; restore</h3>
        <p class="text-xs text-slate-400 mb-3">Export every job + preset to one JSON file, or import a config / preset / job export. Import is additive — existing names are skipped unless you confirm an overwrite.</p>
        <div class="flex flex-wrap gap-2">
          <a href="/api/config/export" download class="px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded text-sm">Export all config</a>
          <button @click="$refs.configImport.click()" :disabled="configImporting" class="px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded text-sm disabled:opacity-50"><span x-text="configImporting ? 'Importing…' : 'Import config / preset / job…'"></span></button>
          <input x-ref="configImport" type="file" accept="application/json,.json" class="hidden" @change="importConfigFile($event)"/>
        </div>
      </div>

      <!-- Vision workers (global default fleet) — edits the default preset every
           job inherits. A job can override it in its own Vision tab. Saves via the
           preset PUT (NOT .env), so .stop keeps it from marking the .env form dirty. -->
      <div class="card rounded-xl p-5" @change.stop @input.stop>
        <div class="flex items-start justify-between gap-4 mb-1">
          <div>
            <h3 class="font-semibold">Vision workers <span class="text-xs font-normal text-slate-500">(global default)</span></h3>
            <p class="text-xs text-slate-400">The local LLM fleet every job inherits (the <span class="font-mono" x-text="globalVision.preset || 'default'"></span> preset) — a job can override it in its own Vision tab. Pick LM Studio / llama.cpp / Ollama, set the URL, <b>Test</b> to load models (leave Model blank to auto-detect). Private / LAN / Tailscale URLs are fine.</p>
          </div>
          <button @click="loadGlobalVision()" class="px-3 py-1.5 text-xs bg-slate-800 hover:bg-slate-700 rounded shrink-0">Reload</button>
        </div>
        <div class="space-y-2 mt-2" x-show="globalVision.loaded">
          <template x-for="(w, idx) in gFleet()" :key="idx">
            <div class="bg-slate-900/60 border border-slate-800 rounded p-3">
              <div class="grid md:grid-cols-12 gap-2">
                <input :value="w.name" @change="gSetFleet(idx,'name',$event.target.value)" placeholder="Name"
                       class="md:col-span-3 bg-slate-800 border border-slate-700 rounded px-2 py-1.5 text-sm"/>
                <select @change="gSetFleet(idx,'provider',$event.target.value)"
                        class="md:col-span-2 bg-slate-800 border border-slate-700 rounded px-2 py-1.5 text-sm">
                  <template x-for="p in visionProviders" :key="'gp_'+p">
                    <option :value="p" :selected="p === w.provider" x-text="visionProviderLabels[p]"></option>
                  </template>
                </select>
                <input :value="w.base_url" @change="gSetFleet(idx,'base_url',$event.target.value)" placeholder="http://host:port"
                       class="md:col-span-4 bg-slate-800 border border-slate-700 rounded px-2 py-1.5 text-sm font-mono"/>
                <select @change="gSetFleet(idx,'model',$event.target.value)" title="Run Test to load the model list"
                        class="md:col-span-3 bg-slate-800 border border-slate-700 rounded px-2 py-1.5 text-sm">
                  <option value="" :selected="!w.model">(auto-detect on connect)</option>
                  <template x-for="m in gFleetModelOptions(idx)" :key="'gm_'+m">
                    <option :value="m" :selected="m === w.model" x-text="m"></option>
                  </template>
                </select>
              </div>
              <div class="grid md:grid-cols-12 gap-2 mt-2 items-center">
                <input type="password" :value="w.api_key" @change="gSetFleet(idx,'api_key',$event.target.value)" placeholder="API key (optional)"
                       autocomplete="off" class="md:col-span-4 bg-slate-800 border border-slate-700 rounded px-2 py-1.5 text-sm"/>
                <label class="md:col-span-2 inline-flex items-center gap-2 text-xs cursor-pointer select-none">
                  <span class="relative inline-flex h-5 w-9 shrink-0 items-center rounded-full transition-colors"
                        :class="w.enabled ? 'bg-indigo-500' : 'bg-slate-600'">
                    <input type="checkbox" class="absolute inset-0 w-full h-full opacity-0 cursor-pointer m-0"
                      :checked="w.enabled" @change="gSetFleet(idx,'enabled',$event.target.checked)"/>
                    <span class="inline-block h-4 w-4 transform rounded-full bg-white shadow transition-transform pointer-events-none"
                      :class="w.enabled ? 'translate-x-4' : 'translate-x-1'"></span>
                  </span>
                  <span x-text="w.enabled ? 'Enabled' : 'Disabled'"></span>
                </label>
                <div class="md:col-span-6 flex items-center gap-2 justify-end">
                  <span class="text-xs truncate max-w-[16rem]" x-show="globalFleetTest[idx] && !globalFleetTest[idx].testing"
                        :class="globalFleetTest[idx]?.ok ? 'text-emerald-400' : 'text-rose-400'" x-text="globalFleetTest[idx]?.message"></span>
                  <button @click="gTestFleet(idx)" :disabled="!!globalFleetTest[idx]?.testing"
                          class="px-2 py-1 text-xs bg-slate-700 hover:bg-slate-600 rounded disabled:opacity-50">
                    <span x-text="globalFleetTest[idx]?.testing ? 'Testing…' : 'Test'"></span>
                  </button>
                  <button @click="gRemoveFleet(idx)" class="px-2 py-1 text-xs bg-rose-900/60 hover:bg-rose-800 text-rose-100 rounded">Remove</button>
                </div>
              </div>
            </div>
          </template>
          <div x-show="!gFleet().length" class="text-xs text-slate-500 italic">No vision workers — add one so images get classified.</div>
        </div>
        <button @click="gAddFleet()" x-show="globalVision.loaded" class="mt-2 px-3 py-1.5 text-sm bg-slate-700 hover:bg-slate-600 rounded">+ Add vision worker</button>
        <div x-show="!globalVision.loaded" class="text-xs text-slate-500 italic mt-2">Loading…</div>
      </div>

      <!-- LM Studio runtime — part of Local vision: VRAM unload policy + per-endpoint
           timeouts for the local fleet above. Plain .env card (no @change.stop) so its
           toggle/number inputs bubble up and mark the settings form dirty. -->
      <div class="card rounded-xl p-5">
        <h3 class="font-semibold mb-1">LM Studio runtime <span class="text-xs font-normal text-slate-500">(local vision)</span></h3>
        <p class="text-xs text-slate-400 mb-3">VRAM-unload policy and per-endpoint timeouts for the local fleet above. Test the primary LM Studio endpoint or free its VRAM on demand.</p>
        <div class="flex items-center gap-3 flex-wrap mb-3">
          <button @click="testProvider('lmstudio')" :disabled="!!providerTest.lmstudio?.testing"
            class="px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded text-sm disabled:opacity-50">
            <span x-text="providerTest.lmstudio?.testing ? 'Testing…' : 'Test LM Studio'"></span>
          </button>
          <span class="text-xs" x-show="providerTest.lmstudio?.message"
            :class="providerTest.lmstudio?.ok ? 'text-emerald-300' : 'text-rose-300'"
            x-text="providerTest.lmstudio?.message"></span>
          <button @click="unloadLmStudio()" :disabled="lmstudioUnload.busy"
            class="px-3 py-2 bg-slate-700 hover:bg-slate-600 rounded text-sm disabled:opacity-50"
            title="Free VRAM by unloading every model in LM Studio. Safe to call any time — JIT load fires on the next image.">
            <span x-text="lmstudioUnload.busy ? 'Unloading…' : 'Unload LM Studio'"></span>
          </button>
          <span class="text-xs" x-show="lmstudioUnload.message"
            :class="lmstudioUnload.ok ? 'text-emerald-300' : 'text-rose-300'"
            x-text="lmstudioUnload.message"></span>
        </div>
        <div class="grid md:grid-cols-2 gap-x-8 gap-y-1">
          {{ toggle('LMSTUDIO_UNLOAD_ON_STOP', 'Unload models when the pipeline stops', 'Free GPU VRAM by unloading every LM Studio model when you Stop the pipeline. JIT reload fires on the next image.') }}
          <label class="block py-1.5">
            <span class="text-xs text-slate-300">Idle auto-unload (minutes){{ tip('Auto-unload after this many idle minutes with an empty queue. 0 = never. Ignored while the lm-keepalive worker is active.') }}</span>
            <input x-model="settings.LMSTUDIO_IDLE_UNLOAD_MINUTES" type="number" min="0" placeholder="10"
              class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-1.5 mt-1 font-mono text-xs"/>
          </label>
          <label class="block py-1.5">
            <span class="text-xs text-slate-300">Primary endpoint timeout (s){{ tip('Request timeout in seconds for the primary LM Studio endpoint.') }}</span>
            <input x-model="settings.LMSTUDIO_PRIMARY_TIMEOUT" type="number" min="1" placeholder="120"
              class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-1.5 mt-1 font-mono text-xs"/>
          </label>
          <label class="block py-1.5">
            <span class="text-xs text-slate-300">Secondary endpoint timeout (s){{ tip('Request timeout in seconds for the secondary LM Studio endpoint (only if you run a second one).') }}</span>
            <input x-model="settings.LMSTUDIO_SECONDARY_TIMEOUT" type="number" min="1" placeholder="120"
              class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-1.5 mt-1 font-mono text-xs"/>
          </label>
        </div>
      </div>

      <div class="card rounded-xl p-5">
        <h3 class="font-semibold mb-1">Credentials &amp; endpoints</h3>
        <p class="text-xs text-slate-400 mb-3">Use the <b>Save</b> button pinned at the top of this page to persist changes to <code>.env</code>.</p>
        <div class="grid md:grid-cols-2 gap-4">
          <label class="block">
            <span class="text-xs text-slate-400">Supervisor reconcile interval (seconds)</span>
            <input x-model="settings.PIPELINE_RECONCILE_SECONDS" type="number" min="1" max="3600"
                   class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1" :class="settingsErrors.PIPELINE_RECONCILE_SECONDS ? 'border-rose-600' : ''"/>
            <span class="text-[10px] text-slate-500">How quickly toggles take effect without a restart. Lower = snappier.</span>
          </label>
          <div class="mt-4 self-center">
            {{ toggle('BLUR_NSFW_THUMBS', 'Blur NSFW thumbnails by default', 'A small eye icon reveals a single image temporarily.') }}
          </div>
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
            <span class="text-xs text-slate-400">Discord token{{ tip('A bot token (from a Discord application) or a user account token. Bot tokens are safer and more rate-limit friendly; user tokens see exactly what that account sees.') }}</span>
            <input x-model="settings.DISCORD_BOT_TOKEN" type="password"
                   placeholder="paste token, no quotes"
                   class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1 font-mono text-xs"/>
            {{ testbtn('Discord-1') }}
          </label>
          <label class="block">
            <span class="text-xs text-slate-400">Auth mode{{ tip('How the token is sent: <b>auto</b> tries a Bot prefix then falls back, <b>bot</b> forces <code>Bot &lt;token&gt;</code>, <b>user</b> sends a raw user token.') }}</span>
            <select x-model="settings.DISCORD_AUTH_MODE"
                    class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1">
              <option value="auto">auto (try bot → fall back to user on 401)</option>
              <option value="bot">bot (Developer-Portal token)</option>
              <option value="user">user (personal account token)</option>
            </select>
          </label>
        </div>
      </div>

      <!-- Cloud vision providers — Groq joined with OpenAI / Anthropic / Gemini /
           OpenRouter in one section. Each row: API key + Test (pulls the live model
           list) + model dropdown + refresh. Enable a provider as a worker on the
           Vision tab; its key + chosen model save to .env here. -->
      <div class="card rounded-xl p-5">
        <h3 class="font-semibold mb-1">Cloud vision providers{{ tip('Optional paid cloud workers that run alongside your local fleet. Paste a key, click Test to pull the live model list, then pick a model. Enable the worker itself on the Vision tab; keys + chosen models save to .env.') }}</h3>
        <p class="text-xs text-slate-400 mb-3">Paste a key and <b>Test</b> to fetch that provider's live models, then choose one. Leave the model blank to auto-detect a vision-capable one on first use.</p>
        <div class="space-y-2">
          <template x-for="cp in settingsCloudProviders" :key="'setcloud_'+cp">
            <div class="bg-slate-900/60 border border-slate-800 rounded p-3">
              <div class="flex items-center justify-between mb-2">
                <span class="text-sm font-medium" x-text="cloudProviderLabels[cp]"></span>
                <span class="text-[10px] px-1.5 py-0.5 rounded bg-slate-800 text-emerald-400" x-show="cloudEnabled(cp)" x-cloak>enabled worker</span>
              </div>
              <div class="grid md:grid-cols-12 gap-2 items-center">
                <input type="password" :value="settings[cloudApiKey(cp)] || ''"
                       @input="settings[cloudApiKey(cp)] = $event.target.value"
                       :placeholder="cp === 'groq' ? 'gsk_...' : 'API key'" autocomplete="off"
                       class="md:col-span-5 bg-slate-800 border border-slate-700 rounded px-2 py-1.5 text-xs font-mono"/>
                <select :value="settings[cloudModelKey(cp)] || ''"
                        @change="settings[cloudModelKey(cp)] = $event.target.value"
                        title="Run Test to load the model list"
                        class="md:col-span-4 bg-slate-800 border border-slate-700 rounded px-2 py-1.5 text-xs">
                  <option value="">(auto-detect — Test to load)</option>
                  <template x-for="m in cloudModelOptions(cp)" :key="cp+'_'+m">
                    <option :value="m" :selected="m === settings[cloudModelKey(cp)]" x-text="m"></option>
                  </template>
                </select>
                <button @click="testProvider(cp)" :disabled="!!(providerTest[cp] && providerTest[cp].testing)"
                        class="md:col-span-2 px-2 py-1.5 text-xs bg-slate-700 hover:bg-slate-600 rounded disabled:opacity-50">
                  <span x-text="(providerTest[cp] && providerTest[cp].testing) ? 'Testing…' : 'Test'"></span>
                </button>
                <button @click="testProvider(cp)" :disabled="!!(providerTest[cp] && providerTest[cp].testing)"
                        title="Refresh model list" aria-label="Refresh model list"
                        class="md:col-span-1 px-2 py-1.5 text-sm bg-slate-800 hover:bg-slate-700 rounded disabled:opacity-50">↺</button>
              </div>
              <div class="mt-2" x-show="cp === 'groq'" x-cloak>
                <input type="password" :value="settings.GROQ_API_KEYS || ''"
                       @input="settings.GROQ_API_KEYS = $event.target.value"
                       placeholder="GROQ_API_KEYS — optional, comma-separated keys for round-robin rotation" autocomplete="off"
                       class="w-full bg-slate-800 border border-slate-700 rounded px-2 py-1.5 text-xs font-mono"/>
              </div>
              <div class="text-xs mt-2 truncate" x-show="providerTest[cp] && !providerTest[cp].testing" x-cloak
                   :class="providerTest[cp]?.ok ? 'text-emerald-300' : 'text-rose-300'" x-text="providerTest[cp]?.message"></div>
            </div>
          </template>
        </div>
      </div>

      <div class="card rounded-xl p-5">
        <h3 class="font-semibold mb-3">Features &amp; automation{{ tip('Global feature toggles for the Wave-2 pipeline add-ons. These are read by the supervisor on (re)start — stop and start the pipeline to apply.') }}</h3>
        <div class="grid md:grid-cols-2 gap-x-8 gap-y-1">
          {{ toggle('PREFILTER_ENABLED', 'Aesthetic pre-filter', 'Score images on aesthetic quality before the vision model runs and drop the worst — saves classification spend on junk. Uses the “Pre-filter min score” below.') }}
          {{ slider('PREFILTER_MIN_SCORE', 'Pre-filter min score', 0, 100, 1, 'Drop images scoring below this aesthetic score (0–100). Only applies when the aesthetic pre-filter is on.') }}
          {{ toggle('SCHEDULER_ENABLED', 'Scheduler', 'Run due per-job schedules on the supervisor reconcile loop. Configure cadences on a job’s Schedules tab.') }}
          {{ toggle('DESKTOP_NOTIFICATIONS', 'Desktop notifications', 'Fire a desktop toast when a job completes (best-effort; needs a notification backend installed).') }}
          {{ toggle('EMBEDDINGS_ENABLED', 'CLIP embeddings index', 'Build a CLIP embeddings index that powers near-duplicate detection and the “find similar” / balance tools.') }}
          {{ toggle('VIDEO_CLASSIFY_ENABLED', 'Classify video clips', 'Classify video clips from a sampled frame, not just still images.') }}
          <label class="block md:col-span-2 mt-2">
            <span class="text-xs text-slate-400">WEBHOOK_URL{{ tip('POST a JSON event here when a job completes — fan out to Slack, Discord, or a home-automation hook. Leave blank to disable.') }}</span>
            <input x-model="settings.WEBHOOK_URL" placeholder="https://hooks.example.com/..."
              class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1 font-mono text-xs"/>
          </label>
        </div>
      </div>

      <div class="card rounded-xl p-5">
        <h3 class="font-semibold mb-3">YT-DLP scraper{{ tip('Pull frames/clips from any site yt-dlp supports (YouTube, Vimeo, and 1000+ more). Enable, paste URLs (one per line or comma-separated), optionally cap per-URL items and point at a cookies.txt for gated content.') }}</h3>
        <div class="grid md:grid-cols-2 gap-4">
          <div class="self-center">
            {{ toggle('YT_DLP_ENABLED', 'yt-dlp scraper enabled', 'Enable the yt-dlp video lane for the URLs below.') }}
          </div>
          <label class="block">
            <span class="text-xs text-slate-400">YT_DLP_LIMIT — max items per URL</span>
            <input x-model="settings.YT_DLP_LIMIT" type="number" min="1" placeholder="25"
              class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1 font-mono text-xs"/>
          </label>
          <label class="block md:col-span-2">
            <span class="text-xs text-slate-400">YT_DLP_URLS — one per line or comma-separated</span>
            <textarea x-model="settings.YT_DLP_URLS" rows="2"
              placeholder="https://www.youtube.com/@channel&#10;https://vimeo.com/..."
              class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1 font-mono text-xs"></textarea>
          </label>
          <label class="block md:col-span-2">
            <span class="text-xs text-slate-400">YT_DLP_COOKIES — optional path to a Netscape cookies.txt</span>
            <input x-model="settings.YT_DLP_COOKIES" placeholder="/path/to/cookies.txt"
              class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1 font-mono text-xs"/>
          </label>
        </div>
      </div>

      <div class="card rounded-xl p-5">
        <h3 class="font-semibold mb-3">Scraper credentials</h3>
        <p class="text-xs text-slate-400 mb-3">Required only for the scrapers you've enabled on the <strong>Scrapers</strong> tab.</p>
        <div class="grid md:grid-cols-2 gap-4">
          <label class="block">
            <span class="text-xs text-slate-400">CIVITAI_API_KEY{{ tip('API key for civitai.com — get one at civitai.com under Account settings &rarr; API Keys. Needed for the Civitai-Com scraper.') }}</span>
            <input x-model="settings.CIVITAI_API_KEY" type="password"
              class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1 font-mono text-xs"/>
            {{ testbtn('Civitai-Com') }}
          </label>
          <label class="block">
            <span class="text-xs text-slate-400">CIVITAI_API_RED_KEY{{ tip('API key for the civitai.red mirror. Falls back to CIVITAI_API_KEY when left blank.') }}</span>
            <input x-model="settings.CIVITAI_API_RED_KEY" type="password"
              class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1 font-mono text-xs"/>
            {{ testbtn('Civitai-Red') }}
          </label>
          <label class="block md:col-span-2">
            <span class="text-xs text-slate-400">TWITTER_COOKIES{{ tip('Full cookie string from a logged-in X/Twitter browser session. Must include <code>auth_token</code> and <code>ct0</code>.', 'auth_token=...; ct0=...; twid=...') }}</span>
            <textarea x-model="settings.TWITTER_COOKIES" rows="2"
              placeholder="auth_token=...; ct0=...; twid=..."
              class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1 font-mono text-xs"></textarea>
            {{ testbtn('X.com') }}
          </label>
          <label class="block">
            <span class="text-xs text-slate-400">REDDIT_CLIENT_ID{{ tip('Optional. Create a &quot;script&quot; app at reddit.com/prefs/apps for higher rate limits; the scraper still works unauthenticated without it.') }}</span>
            <input x-model="settings.REDDIT_CLIENT_ID"
              class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1 font-mono text-xs"/>
          </label>
          <label class="block">
            <span class="text-xs text-slate-400">REDDIT_CLIENT_SECRET{{ tip('Optional. The secret paired with REDDIT_CLIENT_ID from reddit.com/prefs/apps.') }}</span>
            <input x-model="settings.REDDIT_CLIENT_SECRET" type="password"
              class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1 font-mono text-xs"/>
          </label>
          <label class="block md:col-span-2">
            <span class="text-xs text-slate-400">REDDIT_USER_AGENT{{ tip('Identifies your client to Reddit — use a unique, descriptive string.', 'e.g. cull/0.1 by your_username') }}</span>
            <input x-model="settings.REDDIT_USER_AGENT" placeholder="cull/0.1"
              class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1 font-mono text-xs"/>
            {{ testbtn('Reddit') }}
          </label>
          <label class="block md:col-span-2">
            <span class="text-xs text-slate-400">REDDIT_COOKIES{{ tip('Optional. Full cookie string from a logged-in reddit.com browser session. The Reddit scraper runs a real browser (Playwright); paste your cookies here to reach NSFW / gated / quarantined subreddits. Leave blank for public content.', 'reddit_session=...; token_v2=...; over18=1') }}</span>
            <textarea x-model="settings.REDDIT_COOKIES" rows="2"
              placeholder="reddit_session=...; token_v2=..."
              class="w-full bg-slate-800 border border-slate-700 rounded px-3 py-2 mt-1 font-mono text-xs"></textarea>
          </label>
        </div>
      </div>

      <div class="card rounded-xl p-5">
        <h3 class="font-semibold mb-3">Paths</h3>
        <p class="text-xs text-slate-400 mb-3">Absolute paths only (global storage roots). Pipeline must be stopped before changing these. Per-slug subfolders derive from each job's slug.</p>
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
        </div>
      </div>
    </section>

    <!-- ERRORS (job-scoped) -->
    <section x-show="view === 'job' && active === 'errors'" class="card rounded-xl p-5">
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

          <!-- Inline caption editor (§3d): edits the training <stem>.txt that
               lives next to the image. Independent of the prompt block above. -->
          <template x-if="modal.path">
            <div class="mt-5 border-t border-slate-800 pt-4">
              <div class="flex items-center justify-between mb-2">
                <h4 class="text-xs uppercase tracking-wider text-slate-400">Caption (.txt)</h4>
                <span class="text-[11px] text-slate-500" x-text="'~' + captionTokenCount(modal.caption) + ' tokens'"></span>
              </div>
              <div class="flex flex-wrap items-center gap-2 mb-2">
                <input x-model="triggerWord" placeholder="trigger word"
                  class="bg-slate-800 border border-slate-700 rounded px-2 py-1 text-xs w-40"
                  @keydown.enter.prevent="prependTriggerWord()"/>
                <button @click="prependTriggerWord()"
                  class="px-2 py-1 text-xs bg-slate-700 hover:bg-slate-600 rounded"
                  title="Prepend the trigger word to the caption">+ trigger word</button>
                <button @click="saveCaption()" :disabled="!!modal.captionSaving"
                  class="px-2 py-1 text-xs bg-emerald-600 hover:bg-emerald-500 rounded disabled:opacity-50">
                  <span x-text="modal.captionSaving ? 'Saving…' : 'Save caption'"></span>
                </button>
                <span class="text-[11px] text-emerald-300" x-show="modal.captionFlash" x-text="modal.captionFlash"></span>
              </div>
              <textarea x-model="modal.caption" rows="5"
                class="w-full bg-slate-950 border border-slate-700 rounded p-3 text-sm font-mono text-slate-200"
                placeholder="No caption yet — type one and Save."></textarea>
            </div>
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
    // ── Top-level navigation (jobs model) ────────────────────────────────
    // view 'jobs'  → landing: grid of job cards + global settings/stats.
    // view 'job'   → a single job's scoped workspace (tabs below).
    view: 'jobs',
    currentJob: null,         // slug of the job being viewed, when view==='job'
    active: 'jobs',           // active section id within the current view
    sidebarOpen: false,
    // Job-scoped tabs (shown when view==='job'). Job Settings is the per-job
    // config editor; the rest reuse today's sections filtered by currentJob.
    jobTabs: [
      {id:'logs',         label:'Historical'},
      {id:'queue',        label:'Queue'},
      {id:'gallery',      label:'Gallery'},
      {id:'duplicates',   label:'Duplicates'},
      {id:'export',       label:'Export'},
      {id:'stats',        label:'Stats'},
      {id:'overview',     label:'Overview'},
      {id:'scrapers',     label:'Scrapers'},
      {id:'vision',       label:'Vision'},
      {id:'jobSettings',  label:'Job Settings'},
      {id:'errors',       label:'Errors'},
    ],
    // Global sections (shown when view==='jobs').
    globalTabs: [
      {id:'jobs',      label:'Jobs'},
      {id:'presets',   label:'Presets'},
      {id:'schedules', label:'Schedules'},
      {id:'gstats',    label:'Global Stats'},
      {id:'settings',  label:'Global Settings'},
      {id:'faq',       label:'FAQ'},
      {id:'about',     label:'About'},
    ],
    // Jobs landing state.
    jobsList: [], jobsQueue: [], jobsActive: null, jobsLoading: false,
    newJob: { name: '', base_on: '', preset: '' },
    presetsList: [], presetsDefault: '', presetsBuiltins: [],
    // Per-job config editor (Job Settings tab) — v2 inherit/override model.
    // eff = effective (resolved) cfg; overrides = the sparse override map.
    je: {
      loaded: false, slug: null, name: '', subject: '', preset: '',
      eff: null, overrides: {}, presets: [], default_preset: '',
      savedFlash: '', saving: false, error: '', applyPending: false,
    },
    _jeTimer: null,
    _jeSaving: null,
    // Global presets manager state.
    presetEditor: { open: false, name: null, cfg: null, isDefault: false,
                    referencedBy: [], savedFlash: '', error: '', saving: false },
    _peTimer: null,
    newPreset: { name: '', base_on: '' },
    // Dynamic vision-worker fleet (per-job + preset editor).
    visionProviders: ['lmstudio', 'llamacpp', 'ollama'],
    visionProviderLabels: { lmstudio: 'LM Studio', llamacpp: 'llama.cpp / vLLM', ollama: 'Ollama' },
    fleetTest: {},     // job-tab row idx -> { testing, ok, message, models }
    peFleetTest: {},   // preset-editor row idx -> same
    // Global vision fleet (Settings tab) — edits the default preset's workers.
    globalVision: { loaded: false, preset: '', cfg: null },
    globalFleetTest: {},
    _gvTimer: null,
    // Global Stats per-job filter (null = all jobs aggregate).
    globalStatsJob: '',
    // Canonical scraper names, injected server-side from job_config.SCRAPER_NAMES
    // so this can never drift from the Python source of truth. Used by the
    // preset editor's enabled-toggles.
    scraperNames: {{ scraper_names_json|safe }},
    providers: ['balanced-groq','balanced-lm','balanced-lm-secondary','lm-autodetect'],
    provider: 'balanced-groq',
    throttle: 100,
    status: {}, scrapers: [], models: {}, visionWorkers: [],
    settings: {}, settingsBanner: '', settingsBannerOk: true,
    settingsDirty: false, settingsErrors: {}, configImporting: false,
    providerTest: {},
    lmstudioUnload: { busy: false, ok: null, message: '' },
    update: { available: false, behind: 0, local_sha: '', remote_sha: '', remote_subject: '', dismissed_sha: '', running: false, error: '', dirty: false, checking: false, checked: false },
    indexer: { in_progress: false, files_seen: 0, files_added: 0, queue_total: 0, sorted_total: 0, last_scan_at: null, scan_started_at: null },
    indexerToast: { show: false, dismissed: false },
    // ── Generic toasts + styled confirm dialog (replace window.alert/confirm) ──
    toasts: [],
    _toastSeq: 0,
    confirmDialog: { open: false, title: 'Please confirm', message: '',
                     confirmLabel: 'Confirm', cancelLabel: 'Cancel', danger: false,
                     input: false, inputValue: '', inputPlaceholder: '', _resolve: null },
    scraperTest: {},   // per-scraper Test-connection results, keyed by scraper name
    // Wave-2 surfaces (all on-demand — never on the status poll).
    dup: { loading: false, threshold: 10, groups: [], scanned: false, error: '' },
    sim: { open: false, loading: false, results: [], message: '', sourceName: '' },
    exportForm: { profile: 'kohya', trigger_word: '', train_val_split: '',
                  resolution_bucketing: false, repeats: '', video_bucketing: '',
                  running: false, summary: null, error: '' },
    exportProfiles: {{ export_profiles_json|safe }},
    videoBucketModes: {{ video_bucket_modes_json|safe }},
    // Caption styles sourced from vision_prompt.CAPTION_STYLES (server-injected)
    // so the dropdown stays in lockstep with the prompt builder + validator.
    captionStyles: ({{ caption_styles_json|safe }}).map(id => ({
      id,
      label: { sd_prompt: 'SD / Flux prompt', booru_tags: 'Booru tags',
               natural_language: 'Natural-language', motion: 'Motion / video' }[id] || id,
    })),
    hf: { repo_id: '', private: true, include_video: false, pushing: false, result: null, error: '', status: '' },
    // Cadence presets — every value here parses via scheduler.cadence_seconds().
    scheduleCadences: [
      { value: '@hourly', label: 'Hourly' },
      { value: '@daily', label: 'Daily' },
      { value: '@weekly', label: 'Weekly' },
      { value: 'every 15m', label: 'Every 15 minutes' },
      { value: 'every 30m', label: 'Every 30 minutes' },
      { value: 'every 1h', label: 'Every hour' },
      { value: 'every 2h', label: 'Every 2 hours' },
      { value: 'every 6h', label: 'Every 6 hours' },
      { value: 'every 12h', label: 'Every 12 hours' },
    ],
    schedules: { loading: false, rows: [], actions: [], error: '',
                 form: { slug: '', cadence: '@daily', action: 'scrape', enabled: true } },
    visionHealth: { loading: false, probes: {}, fleet: [], error: '' },
    notify(message, type = 'info', timeout) {
      const id = ++this._toastSeq;
      if (timeout === undefined) timeout = (type === 'error') ? 6500 : 3800;
      this.toasts.push({ id, message: String(message), type });
      if (timeout > 0) setTimeout(() => this.dismissToast(id), timeout);
      return id;
    },
    dismissToast(id) { this.toasts = this.toasts.filter(t => t.id !== id); },
    askConfirm(message, opts = {}) {
      // Cancel-resolve any dialog already open so its awaiter can't hang.
      if (this.confirmDialog.open && this.confirmDialog._resolve) this.confirmDialog._resolve(false);
      return new Promise(resolve => {
        this.confirmDialog = {
          open: true, message: String(message),
          title: opts.title || 'Please confirm',
          confirmLabel: opts.confirmLabel || 'Confirm',
          cancelLabel: opts.cancelLabel || 'Cancel',
          danger: !!opts.danger, input: false, inputValue: '', inputPlaceholder: '',
          _resolve: resolve,
        };
        this.$nextTick(() => { try { this.$refs.confirmBtn?.focus(); } catch (e) {} });
      });
    },
    // Styled replacement for window.prompt(): resolves to the string, or null on cancel.
    askPrompt(message, defaultValue = '', opts = {}) {
      if (this.confirmDialog.open && this.confirmDialog._resolve) this.confirmDialog._resolve(false);
      return new Promise(resolve => {
        this.confirmDialog = {
          open: true, message: String(message),
          title: opts.title || 'Enter a value',
          confirmLabel: opts.confirmLabel || 'OK',
          cancelLabel: opts.cancelLabel || 'Cancel',
          danger: !!opts.danger, input: true, inputValue: String(defaultValue || ''),
          inputPlaceholder: opts.placeholder || '', _resolve: resolve,
        };
        this.$nextTick(() => { try { this.$refs.dialogInput?.focus(); this.$refs.dialogInput?.select(); } catch (e) {} });
      });
    },
    _closeConfirm(result) {
      const d = this.confirmDialog;
      const resolve = d._resolve;
      const value = d.input ? (result ? d.inputValue : null) : result;
      this.confirmDialog = { ...d, open: false, _resolve: null };
      if (resolve) resolve(value);
    },
    // Live connectivity/auth test for one scraper. Sends the current Settings
    // form values (server skips masked secrets, falling back to the stored env).
    async testScraper(name, config = null) {
      this.scraperTest = { ...this.scraperTest, [name]: { pending: true } };
      // Send only the credential keys the tester reads — not paths/other settings.
      const credKeys = ['CIVITAI_API_KEY','CIVITAI_API_RED_KEY','TWITTER_COOKIES',
                        'DISCORD_BOT_TOKEN','DISCORD_AUTH_MODE',
                        'REDDIT_CLIENT_ID','REDDIT_CLIENT_SECRET','REDDIT_USER_AGENT'];
      const settings = {};
      for (const k of credKeys) if (this.settings && this.settings[k] !== undefined) settings[k] = this.settings[k];
      try {
        const r = await fetch('/api/scrapers/test', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ scraper: name, settings, config }),
        });
        const j = await r.json();
        this.scraperTest = { ...this.scraperTest, [name]: {
          pending: false, ok: !!j.ok, message: j.message || '', latency_ms: j.latency_ms } };
      } catch (e) {
        this.scraperTest = { ...this.scraperTest, [name]: { pending: false, ok: false, message: 'request failed' } };
      }
    },
    // Civitai domains as a fixed two-option multiselect (com / red), not free text.
    _domainsList(str) { return (str || '').split(',').map(s => s.trim()).filter(Boolean); },
    civOn(d) { return this._domainsList(this.effList('scrapers.civitai_domains')).includes(d); },
    civToggle(d, on) {
      let cur = this._domainsList(this.effList('scrapers.civitai_domains')).filter(x => x !== d);
      if (on) cur.push(d);
      this.setOverrideList('scrapers.civitai_domains', cur.join(','));
    },
    peCivOn(d) { return this._domainsList(this.peList('scrapers.civitai_domains')).includes(d); },
    peCivToggle(d, on) {
      let cur = this._domainsList(this.peList('scrapers.civitai_domains')).filter(x => x !== d);
      if (on) cur.push(d);
      this.peSetList('scrapers.civitai_domains', cur.join(','));
    },
    // ── Export / import. One importer dispatches by the file's "kind". ──────
    async importConfigFile(ev) {
      const file = ev.target.files && ev.target.files[0];
      ev.target.value = '';
      if (!file) return;
      let env;
      try { env = JSON.parse(await file.text()); } catch (e) { this.notify('Not valid JSON', 'error'); return; }
      this.configImporting = true;
      try {
        if (env.kind === 'cull.preset' || (env.name && env.cfg && !env.kind)) { await this._importPreset(env.name, env.cfg); return; }
        if (env.kind !== 'cull.config' && env.kind !== 'cull.job') { this.notify('Unrecognised export file (no "kind")', 'error'); return; }
        const r = await fetch('/api/config/import', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(env) });
        const j = await r.json().catch(() => ({}));
        if (!r.ok) { this.notify('Import failed: ' + (j.error || r.status), 'error'); return; }
        await this.loadPresets(); await this.loadJobs();
        this.notify('Imported ' + (j.presets_added || 0) + ' preset(s), ' + (j.jobs_added || 0) + ' job(s)'
          + (j.skipped && j.skipped.length ? ' · ' + j.skipped.length + ' skipped' : ''), 'success');
      } finally {
        this.configImporting = false;
      }
    },
    async _importPreset(name, cfg) {
      name = (name || '').trim();
      if (!name || !cfg) { this.notify('File is missing a preset name or cfg', 'error'); return; }
      let r = await fetch('/api/presets/import', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name, cfg }) });
      if (r.status === 409) {
        if (!(await this.askConfirm('Preset "' + name + '" already exists. Overwrite it?', { title: 'Overwrite preset', confirmLabel: 'Overwrite', danger: true }))) return;
        r = await fetch('/api/presets/import', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name, cfg, overwrite: true }) });
      }
      const j = await r.json().catch(() => ({}));
      if (!r.ok) { this.notify('Import failed: ' + (j.error || r.status), 'error'); return; }
      await this.loadPresets();
      this.notify('Imported preset "' + name + '".', 'success');
    },
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
             path:'', meta:null,
             // Inline caption editor (§3d) — independent of the prompt editor above.
             caption:'', captionOriginal:'', captionSaving:false, captionFlash:'' },
    // Trigger word the "+ trigger word" button prepends to a caption.
    triggerWord: '',
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

    // ── Keyboard-driven culling (§3a) ────────────────────────────────────
    // cullFocus is an index into gallery.items (-1 = nothing focused). The
    // focused card gets a visible ring and is the implicit target of the cull
    // hotkeys. cullCategories is the ordered keep-bucket taxonomy (loaded from
    // /api/categories) that powers k=keep-primary and 1..9=Nth-category.
    // cullUndo is a stack of reversible moves: each entry is
    // {path, fromCategory, toCategory}. Hard deletes are never pushed (not
    // reversible). cullBusy debounces a hotkey while a move is in flight.
    cullFocus: -1,
    cullCategories: [],   // [{name, ...}] keep buckets, in taxonomy order
    cullTerminal: [],     // ['DISCARD','CORRUPT'] — system terminal buckets
    cullUndo: [],
    cullBusy: false,
    cullLegendOpen: true,

    // Label for the active section, resolved across whichever tab set is live.
    currentTabLabel() {
      const set = this.view === 'job' ? this.jobTabs : this.globalTabs;
      const found = set.find(t => t.id === this.active);
      return found ? found.label : '';
    },

    // Query-string fragment that scopes a request to the job currently being
    // viewed. Empty when on the global (jobs) view so global endpoints aren't
    // accidentally pinned to a stale slug.
    jobParam(prefix) {
      if (this.view !== 'job' || !this.currentJob) return '';
      return (prefix || '?') + 'job=' + encodeURIComponent(this.currentJob);
    },

    async refresh() {
      // On the jobs landing view, refresh the jobs grid + the (cheap) global
      // status pill, and skip the job-scoped polls entirely.
      const j = (url) => fetch(url).then(r => r.ok ? r.json() : Promise.reject(r.status));
      if (this.view === 'jobs') {
        const [status, jobs] = await Promise.allSettled([j('/api/status'), j('/api/jobs')]);
        if (status.status === 'fulfilled') this.applyStatus(status.value);
        if (jobs.status === 'fulfilled') this.applyJobs(jobs.value);
        if (this.active === 'settings' || this.active === 'gstats') {
          const s = await j('/api/settings').catch(() => null);
          if (s && !this.settingsDirty && Object.keys(this.settings).length === 0) this.settings = s;
        }
        this.lastRefresh = new Date().toLocaleTimeString();
        return;
      }
      // Job view: scope every job-aware poll to the current job slug.
      const q = this.jobParam('?');
      const sep = q ? '&' : '?';
      const tab = this.active;
      const need = (id) => tab === id || tab === 'overview';
      const tasks = {
        status:    j('/api/status' + q),
        scrapers:  j('/api/scrapers' + q),
        workers:   j('/api/vision/workers'),
        models:    (need('vision') ? j('/api/lmstudio/models') : null),
        queue:     (need('queue')  ? j('/api/queue/files' + q + sep + 'limit=60') : null),
        history:   (need('logs')   ? j('/api/logs/history' + q + sep + 'limit=200') : null),
        activity:  (need('overview') || tab === 'vision' ? j('/api/activity' + q + sep + 'limit=12') : null),
        // Vision tab also needs GLOBAL settings for the LMStudio knobs.
        settings:  (tab === 'vision' ? j('/api/settings') : null),
      };
      const keys = Object.keys(tasks);
      const results = await Promise.allSettled(Object.values(tasks).map(p => p ?? Promise.resolve(null)));
      const out = {};
      results.forEach((r, i) => { out[keys[i]] = r.status === 'fulfilled' ? r.value : null; });
      if (out.status)    this.applyStatus(out.status);
      if (out.scrapers)  this.scrapers = out.scrapers;
      if (out.workers)   this.visionWorkers = out.workers;
      if (out.models)    this.models = out.models;
      if (out.queue)     this.queueFiles = out.queue;
      if (out.history)   this.history = out.history;
      if (out.activity)  this.activity = out.activity;
      // Only seed settings on first load OR when the user explicitly hits Reload.
      if (out.settings && !this.settingsDirty && Object.keys(this.settings).length === 0) {
        this.settings = out.settings;
      }
      this.lastRefresh = new Date().toLocaleTimeString();
    },

    // Apply a /api/status payload: pipeline pill, throttle, and the indexer
    // toast logic that rides on the same poll.
    applyStatus(status) {
      if (!status) return;
      this.status = status;
      if (status.indexer) {
        this.indexer = status.indexer;
        if (this.indexer.in_progress && !this.indexer.last_scan_at && !this.indexerToast.dismissed) {
          this.indexerToast.show = true;
        } else if (!this.indexer.in_progress) {
          this.indexerToast.show = false;
        }
      }
      this.provider = status.pipeline?.vision_worker || this.provider;
      this.throttle = status.pipeline?.throttle ?? this.throttle;
    },

    // ── Jobs landing ─────────────────────────────────────────────────────
    applyJobs(payload) {
      if (!payload) return;
      this.jobsList = payload.jobs || [];
      this.jobsQueue = payload.queue || [];
      this.jobsActive = payload.active || null;
    },
    async loadJobs() {
      this.jobsLoading = true;
      try {
        const j = await fetch('/api/jobs').then(r => r.json());
        this.applyJobs(j);
      } catch (e) { /* swallow — grid stays as-is */ }
      this.jobsLoading = false;
    },
    async createJob() {
      const name = (this.newJob.name || '').trim();
      if (!name) { this.notify('Give the job a name.', 'warn'); return; }
      const body = { name };
      if (this.newJob.base_on) body.base_on = this.newJob.base_on;
      else if (this.newJob.preset) body.preset = this.newJob.preset;
      const r = await fetch('/api/jobs', {method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify(body)});
      const j = await r.json();
      if (!r.ok) { this.notify('Create failed: ' + (j.error || r.status), 'error'); return; }
      this.newJob = { name: '', base_on: '', preset: '' };
      await this.loadJobs();
      // The create response is the v2 job detail ({job:{slug,...}}).
      this.openJob(j.job.job.slug);
    },
    async activateJob(slug) {
      const r = await fetch('/api/jobs/' + encodeURIComponent(slug) + '/activate', {method:'POST'});
      if (!r.ok) { const j = await r.json().catch(()=>({})); this.notify('Activate failed: ' + (j.error || r.status), 'error'); return; }
      await this.loadJobs();
      await this.refresh();
      this.notify('Activated "' + slug + '".', 'success');
    },
    async cloneJob(slug) {
      const name = await this.askPrompt('Name for the clone of "' + slug + '":', slug + ' copy',
                                        { title: 'Clone job', confirmLabel: 'Clone' });
      if (!name || !name.trim()) return;
      const r = await fetch('/api/jobs/' + encodeURIComponent(slug) + '/clone', {
        method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({name})});
      const j = await r.json();
      if (!r.ok) { this.notify('Clone failed: ' + (j.error || r.status), 'error'); return; }
      await this.loadJobs();
      this.notify('Cloned to "' + name.trim() + '".', 'success');
    },
    async deleteJob(slug) {
      if (!(await this.askConfirm('Delete job "' + slug + '"? Its config is removed. You\'ll be asked separately about data folders.',
                                  { title: 'Delete job', confirmLabel: 'Delete', danger: true }))) return;
      let url = '/api/jobs/' + encodeURIComponent(slug);
      if (await this.askConfirm('Also delete this job\'s queue + sorted data folders on disk? This cannot be undone.',
                                { title: 'Delete data too?', confirmLabel: 'Delete data', cancelLabel: 'Keep data', danger: true })) {
        url += '?force=1';
      }
      const r = await fetch(url, {method:'DELETE'});
      const j = await r.json().catch(()=>({}));
      if (!r.ok) { this.notify('Delete failed: ' + (j.error || r.status), 'error'); return; }
      await this.loadJobs();
      this.notify('Job "' + slug + '" deleted.', 'success');
    },
    async enqueueJob(slug) {
      await fetch('/api/jobs/' + encodeURIComponent(slug) + '/enqueue', {method:'POST'});
      await this.loadJobs();
    },
    async dequeueJob(slug) {
      await fetch('/api/jobs/' + encodeURIComponent(slug) + '/dequeue', {method:'POST'});
      await this.loadJobs();
    },
    async advanceQueue() {
      await fetch('/api/jobs/advance', {method:'POST'});
      await this.loadJobs();
      await this.refresh();
    },
    jobStatusClass(s) {
      if (s === 'running') return 'bg-emerald-900/60 text-emerald-300';
      if (s === 'active')  return 'bg-indigo-900/60 text-indigo-300';
      if (s === 'queued')  return 'bg-amber-900/60 text-amber-300';
      return 'bg-slate-800 text-slate-400';
    },

    // ── Job view navigation ──────────────────────────────────────────────
    async openJob(slug) {
      // Commit any pending edits on the job we're leaving before switching, so a
      // dangling debounce timer can't fire against the NEW slug and lose data.
      if (this.currentJob && this.currentJob !== slug && this.je.loaded) {
        await this.leaveJobEditor();
      }
      this.view = 'job';
      this.currentJob = slug;
      this.active = 'logs';            // default tab: Historical
      this.sidebarOpen = false;
      // Reset per-job tab data so we don't show the previous job's rows.
      this.fleetTest = {};             // drop stale vision-worker test results
      this.history = []; this.queueFiles = []; this.activity = [];
      this.gallery = { ...this.gallery, page: 1, items: [] };
      this.stats = { totals:{}, keywords:[], top_overall:[], top_quality:[], top_relative:[], sources:[] };
      this.je = { loaded: false, slug: null, name: '', subject: '', preset: '',
                  eff: null, overrides: {}, presets: [], default_preset: '',
                  savedFlash: '', saving: false, error: '', applyPending: false };
      this.refresh();
      this.loadJobEditor();
    },
    // Flush pending edits + apply once if the (active) job has unapplied changes.
    // Cancels the debounce timer first so it can't fire after we've moved on.
    async leaveJobEditor() {
      if (this._jeTimer) { clearTimeout(this._jeTimer); this._jeTimer = null; }
      if (this.currentJob && this.currentJob === this.jobsActive && this.je.applyPending) {
        await this.applyActiveJob();     // flushes + activates once
      } else {
        await this.flushJobSave();
      }
    },
    async backToJobs() {
      await this.leaveJobEditor();
      this.view = 'jobs';
      this.currentJob = null;
      this.active = 'jobs';
      this.sidebarOpen = false;
      this.loadJobs();
      this.refresh();
    },

    // ── Job Settings (v2 inherit/override + auto-save) ───────────────────
    // The override namespace uses topic_filters.* / scrapers.* / categories /
    // category_rules / scoring / captioning. The effective cfg exposes a v1
    // topic block, so reads map topic_filters.X → eff.topic.X. `subject` is
    // job-owned (always editable), not an override.
    async loadJobEditor() {
      if (!this.currentJob) return;
      this.je.error = '';
      try {
        const d = await fetch('/api/jobs/' + encodeURIComponent(this.currentJob)).then(r => r.json());
        this.je = {
          loaded: true, slug: d.job.slug, name: d.job.name,
          subject: d.job.subject || '', preset: d.job.preset || '',
          eff: d.effective || {}, overrides: d.overrides || {},
          presets: d.presets || [], default_preset: d.default_preset || '',
          savedFlash: '', saving: false, error: '', applyPending: this.je.applyPending,
        };
      } catch (e) {
        this.je.error = 'Failed to load job: ' + e.message;
      }
    },
    // Map an override-namespace path to its effective-cfg location.
    _effPathFor(path) {
      if (path.startsWith('topic_filters.')) return 'topic.' + path.slice('topic_filters.'.length);
      return path;   // scrapers.* / categories / category_rules / scoring / captioning map 1:1
    },
    _deepGet(obj, path) {
      return path.split('.').reduce((o, k) => (o == null ? undefined : o[k]), obj);
    },
    _deepSet(obj, path, value) {
      const keys = path.split('.'); let o = obj;
      for (let i = 0; i < keys.length - 1; i++) { if (o[keys[i]] == null || typeof o[keys[i]] !== 'object') o[keys[i]] = {}; o = o[keys[i]]; }
      o[keys[keys.length - 1]] = value;
    },
    _deepDel(obj, path) {
      const keys = path.split('.'); const stack = []; let o = obj;
      for (let i = 0; i < keys.length - 1; i++) { if (o == null || !(keys[i] in o)) return; stack.push([o, keys[i]]); o = o[keys[i]]; }
      if (o && typeof o === 'object') delete o[keys[keys.length - 1]];
      for (let i = stack.length - 1; i >= 0; i--) { const [p, k] = stack[i]; if (p[k] && typeof p[k] === 'object' && Object.keys(p[k]).length === 0) delete p[k]; }
    },
    // EFFECTIVE value at an override path (what the field displays).
    effVal(path) { return this._deepGet(this.je.eff, this._effPathFor(path)); },
    // Is this path overridden (vs inherited from the preset)?
    isOver(path) { return this._deepGet(this.je.overrides, path) !== undefined; },
    // List/url convenience views.
    effList(path) { const v = this.effVal(path); return Array.isArray(v) ? v.join(', ') : (v || ''); },
    effUrls() { const v = this.effVal('scrapers.gallery_dl.urls'); return Array.isArray(v) ? v.join('\n') : (v || ''); },
    ytUrls() { const v = this.effVal('scrapers.yt_dlp.urls'); return Array.isArray(v) ? v.join('\n') : (v || ''); },
    // Set an override (and optimistically update the effective view), debounce PUT.
    setOverride(path, value) {
      this._deepSet(this.je.overrides, path, value);
      this._deepSet(this.je.eff, this._effPathFor(path), value);
      this.scheduleJobSave();
    },
    setOverrideList(path, text) { this.setOverride(path, (text || '').split(',').map(s => s.trim()).filter(Boolean)); },
    // Media types are a fixed set {image, video}; toggling keeps at least one and
    // preserves image-before-video order.
    toggleMediaType(kind, on) {
      let cur = (this.effVal('media.types') || []).filter(x => x !== kind);
      if (on) cur.push(kind);
      if (!cur.length) cur = [kind];
      const order = ['image', 'video'];
      cur.sort((a, b) => order.indexOf(a) - order.indexOf(b));
      this.setOverride('media.types', cur);
    },
    setOverrideUrls(text) { this.setOverride('scrapers.gallery_dl.urls', (text || '').split(/[\n,]/).map(s => s.trim()).filter(s => s && !s.startsWith('#'))); },
    setYtUrls(text) { this.setOverride('scrapers.yt_dlp.urls', (text || '').split(/[\n,]/).map(s => s.trim()).filter(s => s && !s.startsWith('#'))); },
    // Reset a field back to the preset (remove the override). Re-fetch to get the
    // recomputed effective value from the preset.
    async resetOverride(path) {
      this._deepDel(this.je.overrides, path);
      await this.flushJobSave();        // persist the removal
      await this.loadJobEditor();       // refresh effective from preset
    },
    setSubject(v) { this.je.subject = v; this.scheduleJobSave(); },
    async setPreset(v) {
      this.je.preset = v;
      await this.flushJobSave();
      await this.loadJobEditor();        // effective recomputes against the new preset
    },
    // Local-imports list manager (overrides scrapers.local_imports wholesale).
    localFolders() { const v = this.effVal('scrapers.local_imports'); return Array.isArray(v) ? v : []; },
    addLocalFolder() {
      const list = (this.localFolders() || []).slice();
      list.push({ name: 'local' + (list.length || ''), dir: '', enabled: false, migrate_from: '' });
      this.setOverride('scrapers.local_imports', list);
    },
    updateLocalFolder(i, key, value) {
      const list = (this.localFolders() || []).map(f => ({ ...f }));
      if (!list[i]) return;
      list[i][key] = value;
      this.setOverride('scrapers.local_imports', list);
    },
    removeLocalFolder(i) {
      const list = (this.localFolders() || []).filter((_, idx) => idx !== i);
      this.setOverride('scrapers.local_imports', list);
    },
    // Debounced auto-save (~700ms). Marks the active job's apply as pending.
    scheduleJobSave() {
      if (this.currentJob === this.jobsActive) this.je.applyPending = true;
      if (this._jeTimer) clearTimeout(this._jeTimer);
      this._jeTimer = setTimeout(() => this.flushJobSave(), 700);
    },
    async flushJobSave() {
      if (this._jeTimer) { clearTimeout(this._jeTimer); this._jeTimer = null; }
      if (!this.je.loaded || !this.currentJob) return;
      // Serialise saves: if one is in flight, wait for it and re-run once so a
      // change made during the request isn't lost and two PUTs can't race.
      if (this._jeSaving) { await this._jeSaving; }
      const slug = this.currentJob;
      const payload = { subject: this.je.subject, preset: this.je.preset, overrides: this.je.overrides };
      this.je.saving = true; this.je.error = '';
      this._jeSaving = (async () => {
        try {
          const r = await fetch('/api/jobs/' + encodeURIComponent(slug), {
            method: 'PUT', headers: {'Content-Type':'application/json'}, body: JSON.stringify(payload)});
          const j = await r.json();
          // Ignore the result if the user navigated to a different job meanwhile.
          if (this.currentJob !== slug) return;
          if (r.ok && j.success) {
            this.je.savedFlash = 'saved ✓';
            setTimeout(() => this.je.savedFlash = '', 1800);
            this.loadJobs();
          } else {
            this.je.error = 'Save failed: ' + (j.error || r.status);
          }
        } catch (e) {
          if (this.currentJob === slug) this.je.error = 'Network error: ' + e.message;
        }
      })();
      try { await this._jeSaving; } finally { this._jeSaving = null; this.je.saving = false; }
    },
    // Apply pending edits to the ACTIVE job: flush then re-project + restart once.
    async applyActiveJob() {
      await this.flushJobSave();
      if (this.currentJob !== this.jobsActive) { this.je.applyPending = false; return; }
      await fetch('/api/jobs/' + encodeURIComponent(this.currentJob) + '/activate', { method: 'POST' });
      this.je.applyPending = false;
      this.loadJobs();
      this.refresh();
    },

    // ── Presets manager (global) ─────────────────────────────────────────
    async loadPresets() {
      try {
        const p = await fetch('/api/presets').then(r => r.json());
        this.presetsList = p.presets || [];
        this.presetsDefault = p.default || '';
        this.presetsBuiltins = p.builtins || [];
        // Default the "Start from" picker to the library default (no more
        // redundant empty "Clone the default" option). Keep a valid selection.
        if (!this.presetsList.includes(this.newPreset.base_on)) {
          this.newPreset.base_on = this.presetsDefault || (this.presetsList[0] || '');
        }
      } catch (e) { /* swallow */ }
    },
    async createPreset() {
      const name = (this.newPreset.name || '').trim();
      if (!name) { this.notify('Give the preset a name.', 'warn'); return; }
      const body = { name };
      if (this.newPreset.base_on) body.base_on = this.newPreset.base_on;
      const r = await fetch('/api/presets', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
      const j = await r.json();
      if (!r.ok) { this.notify('Create failed: ' + (j.error || r.status), 'error'); return; }
      this.newPreset = { name: '', base_on: '' };
      await this.loadPresets();
      this.openPreset(j.name);
    },
    async clonePreset(name) {
      const nn = await this.askPrompt('Name for the clone of "' + name + '":', name + ' copy',
                                      { title: 'Clone preset', confirmLabel: 'Clone' });
      if (!nn || !nn.trim()) return;
      const r = await fetch('/api/presets/' + encodeURIComponent(name) + '/clone', {
        method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({name: nn})});
      const j = await r.json();
      if (!r.ok) { this.notify('Clone failed: ' + (j.error || r.status), 'error'); return; }
      await this.loadPresets();
      this.notify('Cloned preset to "' + nn.trim() + '".', 'success');
    },
    async deletePreset(name) {
      if (!(await this.askConfirm('Delete preset "' + name + '"?',
                                  { title: 'Delete preset', confirmLabel: 'Delete', danger: true }))) return;
      const r = await fetch('/api/presets/' + encodeURIComponent(name), {method:'DELETE'});
      const j = await r.json().catch(()=>({}));
      if (!r.ok) { this.notify('Delete failed: ' + (j.error || r.status), 'error'); return; }
      if (this.presetEditor.name === name) this.presetEditor.open = false;
      await this.loadPresets();
      this.notify('Preset "' + name + '" deleted.', 'success');
    },
    async setDefaultPreset(name) {
      const r = await fetch('/api/presets/default', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({name})});
      if (!r.ok) { const j = await r.json().catch(()=>({})); this.notify('Failed: ' + (j.error || r.status), 'error'); return; }
      await this.loadPresets();
      this.notify('"' + name + '" is now the default preset.', 'success');
    },
    async resetPreset(name) {
      if (!(await this.askConfirm('Reset "' + name + '" to its shipped defaults? Your edits to this preset will be discarded.',
                                  { title: 'Reset preset', confirmLabel: 'Reset', danger: true }))) return;
      const r = await fetch('/api/presets/' + encodeURIComponent(name) + '/reset', {method:'POST'});
      const j = await r.json().catch(()=>({}));
      if (!r.ok) { this.notify('Reset failed: ' + (j.error || r.status), 'error'); return; }
      await this.loadPresets();
      if (this.presetEditor.open && this.presetEditor.name === name) await this.openPreset(name);
      this.notify('Preset "' + name + '" reset to defaults.', 'success');
    },
    async openPreset(name) {
      try {
        const d = await fetch('/api/presets/' + encodeURIComponent(name)).then(r => r.json());
        // Defensive nested defaults for deep binds.
        const cfg = d.cfg || {};
        cfg.topic_filters = cfg.topic_filters || {};
        cfg.scrapers = cfg.scrapers || {};
        cfg.scrapers.gallery_dl = cfg.scrapers.gallery_dl || {};
        cfg.scrapers.yt_dlp = cfg.scrapers.yt_dlp || {};
        if (!Array.isArray(cfg.scrapers.local_imports)) cfg.scrapers.local_imports = [];
        cfg.scoring = cfg.scoring || {};
        cfg.captioning = cfg.captioning || {};
        cfg.categories = Array.isArray(cfg.categories) ? cfg.categories : [];
        cfg.vision = cfg.vision || {};
        if (!Array.isArray(cfg.vision.workers)) cfg.vision.workers = [];
        this.peFleetTest = {};
        this.presetEditor = { open: true, name, cfg, isDefault: d.is_default,
                              referencedBy: d.referenced_by || [], savedFlash: '', error: '', saving: false };
      } catch (e) { this.notify('Failed to load preset: ' + e.message, 'error'); }
    },
    closePreset() { this.flushPresetSave(); this.presetEditor.open = false; },
    // Preset list editors mirror the job editor list helpers but write straight
    // into presetEditor.cfg (a preset has no inheritance — values are absolute).
    peList(path) { const v = this._deepGet(this.presetEditor.cfg, path); return Array.isArray(v) ? v.join(', ') : (v || ''); },
    peSetList(path, text) { this._deepSet(this.presetEditor.cfg, path, (text || '').split(',').map(s => s.trim()).filter(Boolean)); this.schedulePresetSave(); },
    peUrls() { const v = this._deepGet(this.presetEditor.cfg, 'scrapers.gallery_dl.urls'); return Array.isArray(v) ? v.join('\n') : (v || ''); },
    peSetUrls(text) { this._deepSet(this.presetEditor.cfg, 'scrapers.gallery_dl.urls', (text || '').split(/[\n,]/).map(s => s.trim()).filter(s => s && !s.startsWith('#'))); this.schedulePresetSave(); },
    peYtUrls() { const v = this._deepGet(this.presetEditor.cfg, 'scrapers.yt_dlp.urls'); return Array.isArray(v) ? v.join('\n') : (v || ''); },
    peSetYtUrls(text) { this._deepSet(this.presetEditor.cfg, 'scrapers.yt_dlp.urls', (text || '').split(/[\n,]/).map(s => s.trim()).filter(s => s && !s.startsWith('#'))); this.schedulePresetSave(); },
    peSet(path, value) { this._deepSet(this.presetEditor.cfg, path, value); this.schedulePresetSave(); },
    peToggleMedia(kind, on) {
      let cur = (((this.presetEditor.cfg || {}).media || {}).types || ['image']).filter(x => x !== kind);
      if (on) cur.push(kind);
      if (!cur.length) cur = [kind];
      const order = ['image', 'video'];
      cur.sort((a, b) => order.indexOf(a) - order.indexOf(b));
      this.peSet('media.types', cur);
    },
    // Preset-editor vision fleet (global default fleet jobs inherit).
    peFleet() { const v = this._deepGet(this.presetEditor.cfg, 'vision.workers'); return Array.isArray(v) ? v : []; },
    peFleetModelOptions(idx) {
      const w = this.peFleet()[idx] || {};
      const fetched = (this.peFleetTest[idx] && this.peFleetTest[idx].models) || [];
      const head = (w.model && !fetched.includes(w.model)) ? [w.model] : [];
      return head.concat(fetched);
    },
    peSetFleet(idx, field, val) { const l = this.peFleet().map(w => ({ ...w })); if (!l[idx]) return; l[idx][field] = val; if (field === 'provider' || field === 'base_url' || field === 'api_key') delete this.peFleetTest[idx]; this.peSet('vision.workers', l); },
    peAddFleet() { const l = this.peFleet().map(w => ({ ...w })); l.push({ id: 'w' + Date.now().toString(36), name: 'New worker', provider: 'lmstudio', base_url: 'http://127.0.0.1:1234', model: '', api_key: '', enabled: true }); this.peSet('vision.workers', l); },
    peRemoveFleet(idx) { const l = this.peFleet().map(w => ({ ...w })); l.splice(idx, 1); this.peFleetTest = {}; this.peSet('vision.workers', l); },
    async peTestFleet(idx) {
      const w = this.peFleet()[idx]; if (!w) return;
      this.peFleetTest[idx] = { testing: true, ok: null, message: 'Connecting…', models: [] };
      try {
        const j = await this._visionTest(w.provider, w.base_url, w.api_key);
        this.peFleetTest[idx] = { testing: false, ok: j.ok,
          message: (j.ok ? '✓ ' : '✗ ') + (j.message || 'unknown') + ` (${j.latency_ms}ms)`, models: j.models || [] };
        if (j.ok && (j.models || []).length && !(w.model || '').trim()) this.peSetFleet(idx, 'model', j.models[0]);
      } catch (e) { this.peFleetTest[idx] = { testing: false, ok: false, message: '✗ network error', models: [] }; }
    },
    // ── Global vision fleet (Settings tab) — edits the DEFAULT preset's workers ──
    async loadGlobalVision() {
      try {
        const lib = await fetch('/api/presets').then(r => r.json());
        const name = lib.default || 'default';
        const d = await fetch('/api/presets/' + encodeURIComponent(name)).then(r => r.json());
        const cfg = d.cfg || {};
        cfg.vision = cfg.vision || {};
        if (!Array.isArray(cfg.vision.workers)) cfg.vision.workers = [];
        this.globalVision = { loaded: true, preset: name, cfg };
        this.globalFleetTest = {};
      } catch (e) { /* swallow — card just stays empty */ }
    },
    gFleet() { const c = this.globalVision.cfg; return (c && c.vision && Array.isArray(c.vision.workers)) ? c.vision.workers : []; },
    gFleetModelOptions(idx) {
      const w = this.gFleet()[idx] || {};
      const fetched = (this.globalFleetTest[idx] && this.globalFleetTest[idx].models) || [];
      const head = (w.model && !fetched.includes(w.model)) ? [w.model] : [];
      return head.concat(fetched);
    },
    _gWrite(list) {
      if (!this.globalVision.cfg) return;
      this.globalVision.cfg.vision = this.globalVision.cfg.vision || {};
      this.globalVision.cfg.vision.workers = list;
      this._scheduleGlobalVisionSave();
    },
    gSetFleet(idx, field, val) { const l = this.gFleet().map(w => ({ ...w })); if (!l[idx]) return; l[idx][field] = val; if (field === 'provider' || field === 'base_url' || field === 'api_key') delete this.globalFleetTest[idx]; this._gWrite(l); },
    gAddFleet() { const l = this.gFleet().map(w => ({ ...w })); l.push({ id: 'w' + Date.now().toString(36), name: 'New worker', provider: 'lmstudio', base_url: 'http://127.0.0.1:1234', model: '', api_key: '', enabled: true }); this._gWrite(l); },
    gRemoveFleet(idx) { const l = this.gFleet().map(w => ({ ...w })); l.splice(idx, 1); this.globalFleetTest = {}; this._gWrite(l); },
    async gTestFleet(idx) {
      const w = this.gFleet()[idx]; if (!w) return;
      this.globalFleetTest[idx] = { testing: true, ok: null, message: 'Connecting…', models: [] };
      try {
        const j = await this._visionTest(w.provider, w.base_url, w.api_key);
        this.globalFleetTest[idx] = { testing: false, ok: j.ok,
          message: (j.ok ? '✓ ' : '✗ ') + (j.message || 'unknown') + ` (${j.latency_ms}ms)`, models: j.models || [] };
        if (j.ok && (j.models || []).length && !(w.model || '').trim()) this.gSetFleet(idx, 'model', j.models[0]);
      } catch (e) { this.globalFleetTest[idx] = { testing: false, ok: false, message: '✗ network error', models: [] }; }
    },
    _scheduleGlobalVisionSave() { if (this._gvTimer) clearTimeout(this._gvTimer); this._gvTimer = setTimeout(() => this._saveGlobalVision(), 700); },
    async _saveGlobalVision() {
      if (!this.globalVision.cfg || !this.globalVision.preset) return;
      try {
        const r = await fetch('/api/presets/' + encodeURIComponent(this.globalVision.preset),
          { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ cfg: this.globalVision.cfg }) });
        if (!r.ok) { const j = await r.json().catch(() => ({})); this.notify('Vision fleet save failed: ' + (j.error || r.status), 'error'); }
      } catch (e) { this.notify('Vision fleet save failed (network)', 'error'); }
    },
    peLocalFolders() { const v = this._deepGet(this.presetEditor.cfg, 'scrapers.local_imports'); return Array.isArray(v) ? v : []; },
    peAddLocal() { const l = this.peLocalFolders().slice(); l.push({name:'local'+(l.length||''),dir:'',enabled:false,migrate_from:''}); this.peSet('scrapers.local_imports', l); },
    peUpdateLocal(i, key, value) { const l = this.peLocalFolders().map(f=>({...f})); if (l[i]) { l[i][key]=value; this.peSet('scrapers.local_imports', l); } },
    peRemoveLocal(i) { this.peSet('scrapers.local_imports', this.peLocalFolders().filter((_,idx)=>idx!==i)); },
    peAddCategory() { const c = (this.presetEditor.cfg.categories || []).slice(); if (c.length >= 12) return; c.push({name:'',hint:''}); this.peSet('categories', c); },
    peRemoveCategory(i) { this.peSet('categories', (this.presetEditor.cfg.categories || []).filter((_,idx)=>idx!==i)); },
    schedulePresetSave() { if (this._peTimer) clearTimeout(this._peTimer); this._peTimer = setTimeout(() => this.flushPresetSave(), 700); },
    async flushPresetSave() {
      if (this._peTimer) { clearTimeout(this._peTimer); this._peTimer = null; }
      if (!this.presetEditor.open || !this.presetEditor.name) return;
      this.presetEditor.saving = true; this.presetEditor.error = '';
      try {
        const r = await fetch('/api/presets/' + encodeURIComponent(this.presetEditor.name), {
          method:'PUT', headers:{'Content-Type':'application/json'}, body: JSON.stringify({cfg: this.presetEditor.cfg})});
        const j = await r.json();
        if (r.ok && j.success) { this.presetEditor.savedFlash = 'saved ✓'; setTimeout(() => this.presetEditor.savedFlash = '', 1800); }
        else { this.presetEditor.error = 'Save failed: ' + (j.error || r.status); }
      } catch (e) { this.presetEditor.error = 'Network error: ' + e.message; }
      this.presetEditor.saving = false;
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
    // Map a vision worker name to the provider its /api/vision/test probe uses.
    workerProvider(name) {
      const n = (name || '').toLowerCase();
      if (n.includes('groq')) return 'groq';
      if (n.includes('ollama')) return 'ollama';
      if (n.includes('openai') || n.includes('compat')) return 'openai-compat';
      if (n.includes('lm')) return 'lmstudio';
      return null;
    },
    // Cloud vision providers: provider id -> the settings key holding its model.
    // Test Connection fetches the live catalogue into the <select> below.
    cloudProviders: ['openai', 'anthropic', 'gemini', 'openrouter'],
    // Settings tab unifies Groq with the four cloud providers in one section.
    settingsCloudProviders: ['groq', 'openai', 'anthropic', 'gemini', 'openrouter'],
    cloudProviderLabels: { groq: 'Groq (Llama-4 Scout)', openai: 'OpenAI (GPT vision)', anthropic: 'Anthropic Claude',
                           gemini: 'Google Gemini', openrouter: 'OpenRouter' },
    cloudWorkerName: { openai: 'openai', anthropic: 'anthropic', gemini: 'gemini', openrouter: 'openrouter' },
    cloudModelKey(provider) {
      return { groq: 'GROQ_MODEL', openai: 'OPENAI_VISION_MODEL', anthropic: 'ANTHROPIC_VISION_MODEL',
               gemini: 'GEMINI_VISION_MODEL', openrouter: 'OPENROUTER_VISION_MODEL' }[provider];
    },
    cloudApiKey(provider) {
      // Settings key whose freshly-typed value is sent so an UNSAVED key tests.
      return { groq: 'GROQ_API_KEY', openai: 'OPENAI_API_KEY', anthropic: 'ANTHROPIC_API_KEY',
               gemini: 'GEMINI_API_KEY', openrouter: 'OPENROUTER_API_KEY' }[provider];
    },
    cloudEnabled(provider) {
      return !!(this.visionWorkers.find(w => w.name === provider) || {}).enabled;
    },
    // Options for a cloud provider's model <select>: the saved model first (if it
    // isn't already in the fetched list) then everything Test discovered.
    cloudModelOptions(provider) {
      const fetched = (this.providerTest[provider] && this.providerTest[provider].models) || [];
      const cur = (this.settings[this.cloudModelKey(provider)] || '').trim();
      const head = (cur && !fetched.includes(cur)) ? [cur] : [];
      return head.concat(fetched);
    },
    async testProvider(name) {
      if (!name) return;
      this.providerTest[name] = { testing: true, message: 'Connecting...', ok: null, models: [] };
      // For a cloud provider, pass the freshly typed key + model so an unsaved
      // credential can be verified without saving first (masked secret => server
      // falls back to the stored env value).
      const body = { provider: name };
      const isCloud = this.cloudProviders.includes(name) || name === 'groq';
      if (isCloud) {
        const typedKey = (this.settings[this.cloudApiKey(name)] || '').trim();
        if (typedKey) body.api_key = typedKey;
      }
      try {
        const r = await fetch('/api/vision/test', {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(body),
        });
        const j = await r.json();
        this.providerTest[name] = {
          testing: false,
          ok: j.ok,
          models: j.models || [],
          message: (j.ok ? '✓ ' : '✗ ') + (j.message || 'unknown') + ` (${j.latency_ms}ms)`,
        };
        // Auto-fill a blank cloud model so the user isn't stuck on "" after Test.
        if (isCloud && j.ok && (j.models || []).length) {
          const k = this.cloudModelKey(name);
          if (!(this.settings[k] || '').trim()) { this.settings[k] = j.models[0]; this.markSettingsDirty(); }
        }
      } catch (e) {
        this.providerTest[name] = { testing: false, ok: false, message: '✗ network error', models: [] };
      }
    },
    // ── Vision-worker fleet (Job tab: per-job inherit/override of vision.workers) ─
    visionFleet() { const v = this.effVal('vision.workers'); return Array.isArray(v) ? v : []; },
    // Model <select> options for a row: the current model (if it's a custom value
    // not in the fetched list) followed by the models Test discovered.
    fleetModelOptions(idx) {
      const w = this.visionFleet()[idx] || {};
      const fetched = (this.fleetTest[idx] && this.fleetTest[idx].models) || [];
      const head = (w.model && !fetched.includes(w.model)) ? [w.model] : [];
      return head.concat(fetched);
    },
    _writeFleet(list) { this.setOverride('vision.workers', list); },
    setFleetField(idx, field, val) {
      const list = this.visionFleet().map(w => ({ ...w }));
      if (!list[idx]) return;
      list[idx][field] = val;
      // A changed endpoint/provider/key invalidates the last Test — drop the
      // stale "connected + models" so it can't carry over (e.g. LM Studio models
      // lingering after you switch the row to Ollama).
      if (field === 'provider' || field === 'base_url' || field === 'api_key') delete this.fleetTest[idx];
      this._writeFleet(list);
    },
    addFleetWorker() {
      const list = this.visionFleet().map(w => ({ ...w }));
      list.push({ id: 'w' + Date.now().toString(36), name: 'New worker', provider: 'lmstudio',
                  base_url: 'http://127.0.0.1:1234', model: '', api_key: '', enabled: true });
      this._writeFleet(list);
    },
    removeFleetWorker(idx) {
      const list = this.visionFleet().map(w => ({ ...w }));
      list.splice(idx, 1);
      this.fleetTest = {};          // indices shift; drop stale per-row results
      this._writeFleet(list);
    },
    async _visionTest(provider, url, api_key) {
      const r = await fetch('/api/vision/test', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ provider, url, api_key }) });
      return r.json();
    },
    async testFleetWorker(idx) {
      const w = this.visionFleet()[idx]; if (!w) return;
      this.fleetTest[idx] = { testing: true, ok: null, message: 'Connecting…', models: [] };
      try {
        const j = await this._visionTest(w.provider, w.base_url, w.api_key);
        this.fleetTest[idx] = { testing: false, ok: j.ok,
          message: (j.ok ? '✓ ' : '✗ ') + (j.message || 'unknown') + ` (${j.latency_ms}ms)`,
          models: j.models || [] };
        if (j.ok && (j.models || []).length && !(w.model || '').trim()) {
          this.setFleetField(idx, 'model', j.models[0]);   // auto-fill a blank model
        }
      } catch (e) {
        this.fleetTest[idx] = { testing: false, ok: false, message: '✗ network error', models: [] };
      }
    },
    groqEnabled() { return !!(this.visionWorkers.find(w => w.name === 'balanced-groq') || {}).enabled; },
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
    async checkUpdate(manual = false) {
      if (manual) this.update.checking = true;
      try {
        const r = await fetch('/api/update/check');
        const j = await r.json();
        if (!j.ok) {
          this.update.error = j.error || ''; this.update.checking = false; this.update.checked = true;
          if (manual) this.notify('Update check failed: ' + (j.error || 'unknown'), 'error');
          return;
        }
        const dismissed = localStorage.getItem('cull_update_dismissed_sha') || '';
        this.update = {
          ...this.update,
          available: j.behind > 0 && j.remote_sha !== dismissed,
          behind: j.behind || 0,
          local_sha: j.local_sha || '',
          remote_sha: j.remote_sha || '',
          remote_subject: j.remote_subject || '',
          dirty: !!j.dirty,
          dismissed_sha: dismissed,
          running: false,
          error: '',
          checking: false,
          checked: true,
        };
        if (manual) this.notify(j.behind > 0
          ? (j.behind + ' update' + (j.behind === 1 ? '' : 's') + ' available')
          : 'cull is up to date', j.behind > 0 ? 'info' : 'success');
      } catch (e) {
        this.update.checking = false;
        if (manual) this.notify('Update check failed (network)', 'error');
      }
    },
    dismissUpdate() {
      if (this.update.remote_sha) {
        localStorage.setItem('cull_update_dismissed_sha', this.update.remote_sha);
      }
      this.update.available = false;
    },
    async runUpdate() {
      if (!(await this.askConfirm('cull will pull the latest version, reinstall dependencies if needed, and restart. Continue?',
                                  { title: 'Update cull', confirmLabel: 'Update' }))) return;
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
    async reloadSettings() {
      if (this.settingsDirty && !(await this.askConfirm('Discard your unsaved settings changes?',
                                  { title: 'Discard changes', confirmLabel: 'Discard', danger: true }))) return;
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
      // Cancel any pending job-editor auto-save so its (pre-toggle) overrides
      // can't clobber the toggle we're about to write server-side.
      if (this._jeTimer) { clearTimeout(this._jeTimer); this._jeTimer = null; }
      const r = await fetch('/api/scrapers/toggle' + this.jobParam('?'), {method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({name, enabled})});
      const j = await r.json().catch(()=>({}));
      if (!r.ok) { this.notify(j.error || ('toggle failed: ' + r.status), 'error'); return; }
      // Resync je.overrides from the server's fresh map so a subsequent flush
      // PUTs the post-toggle overrides, never a stale snapshot.
      if (j.overrides && this.je.loaded) this.je.overrides = j.overrides;
      // Toggles write an override; the active job applies on leave / Apply.
      if (this.currentJob === this.jobsActive) this.je.applyPending = true;
      this.refresh(); this.loadJobEditor();
    },
    async scrapersBulk(action) {
      if (this._jeTimer) { clearTimeout(this._jeTimer); this._jeTimer = null; }
      const r = await fetch('/api/scrapers/bulk' + this.jobParam('?'), {method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({action})});
      const j = await r.json().catch(()=>({}));
      if (j.overrides && this.je.loaded) this.je.overrides = j.overrides;
      if (this.currentJob === this.jobsActive) this.je.applyPending = true;
      this.refresh(); this.loadJobEditor();
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
        caption: '', captionOriginal: '', captionSaving: false, captionFlash: '',
      };
      try {
        const txt = promptUrl ? await fetch(promptUrl).then(r=>r.text()) : '';
        const value = txt || '';
        this.modal.prompt = value || '(empty)';
        this.modal.promptOriginal = value;
      } catch (e) { this.modal.prompt = '(failed to load prompt)'; }
      // Load the training caption (.txt) into the inline caption editor. Separate
      // fetch from the prompt above so the two editors stay independent.
      this.loadCaption();
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
    async closeModal() {
      // Warn on unsaved prompt edits before discarding the modal.
      if (this.modal.editing && this.modal.prompt !== this.modal.promptOriginal) {
        if (!(await this.askConfirm('You have unsaved prompt edits. Discard them?',
                                    { title: 'Discard edits', confirmLabel: 'Discard', danger: true }))) return;
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
        const r = await fetch('/api/stats' + this.jobParam('?')).then(r=>r.json());
        this.stats = r;
      } catch (e) { /* fall through silently */ }
      this.statsLoading = false;
    },
    // Global Stats (jobs view): aggregate across all jobs, or one job when the
    // per-job filter is set. Uses an explicit ?job= rather than currentJob.
    async loadGlobalStats() {
      this.statsLoading = true;
      try {
        const q = this.globalStatsJob ? ('?job=' + encodeURIComponent(this.globalStatsJob)) : '';
        this.stats = await fetch('/api/stats' + q).then(r=>r.json());
      } catch (e) { /* swallow */ }
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
      // Scope the gallery to the job being viewed.
      if (this.view === 'job' && this.currentJob) params.set('job', this.currentJob);
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

    // ── Keyboard-driven culling (§3a) ────────────────────────────────────
    // The primary keep bucket is the first non-terminal category in taxonomy
    // order. Falls back to 'Keepers' only if the taxonomy somehow has none.
    cullPrimaryKeep() {
      const first = (this.cullCategories || [])[0];
      return first ? first.name : null;
    },
    // Load the ordered taxonomy so 1..9 and k map to real category names. Cheap;
    // fired once when the gallery tab opens (and refreshed on demand).
    async loadCullCategories() {
      try {
        const j = await fetch('/api/categories' + this.jobParam('?')).then(r => r.json());
        this.cullCategories = (j.active && j.active.categories) || [];
        this.cullTerminal = j.system_terminal || [];
      } catch (e) { /* leave whatever we had; hotkeys degrade gracefully */ }
    },
    // True when keyboard culling should respond: gallery tab open, no modal,
    // and the user isn't typing into a field. Keeps hotkeys from firing while
    // someone edits a filter or caption.
    cullHotkeysActive() {
      if (this.view !== 'job' || this.active !== 'gallery') return false;
      if (this.modal.open) return false;
      const el = document.activeElement;
      if (el) {
        const tag = (el.tagName || '').toLowerCase();
        if (tag === 'input' || tag === 'textarea' || tag === 'select' || el.isContentEditable) return false;
      }
      return true;
    },
    cullFocusCard(idx) {
      const n = (this.gallery.items || []).length;
      if (n === 0) { this.cullFocus = -1; return; }
      // Clamp into range and wrap is not needed; arrow handlers clamp.
      this.cullFocus = Math.max(0, Math.min(n - 1, idx));
    },
    cullClearFocus() { this.cullFocus = -1; },
    // Move focus by a delta within the flat item list (used by h/j/k/l + arrows).
    cullMoveFocus(delta) {
      const n = (this.gallery.items || []).length;
      if (n === 0) { this.cullFocus = -1; return; }
      const next = this.cullFocus < 0 ? 0 : this.cullFocus + delta;
      this.cullFocus = Math.max(0, Math.min(n - 1, next));
      this.$nextTick(() => {
        const node = document.querySelector('[data-cull-idx="' + this.cullFocus + '"]');
        if (node && node.scrollIntoView) node.scrollIntoView({ block: 'nearest' });
      });
    },
    // The currently focused card object, or null.
    cullFocused() {
      if (this.cullFocus < 0) return null;
      return (this.gallery.items || [])[this.cullFocus] || null;
    },
    // Recategorise the focused card to `target` via the sorted-move endpoint and
    // push a reversible entry onto the undo stack. Optimistically drops the card
    // from the current view so culling feels instant; a failed move re-fetches.
    async cullRecategorize(target) {
      const card = this.cullFocused();
      if (!card || !target || this.cullBusy) return;
      this.cullBusy = true;
      const focusedIdx = this.cullFocus;
      try {
        const r = await fetch('/api/gallery/recategorize', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ path: card.path, target, job: this.currentJob || undefined }),
        });
        const j = await r.json().catch(() => ({}));
        if (!r.ok || !j.success) {
          this.notify((j && j.error) || ('move failed: ' + r.status), 'error');
          return;
        }
        if (j.data && j.data.moved) {
          // Push undo BEFORE mutating the list. The new path (under the target
          // folder) is what we'd reverse.
          this.cullUndo.push({
            path: j.data.path, fromCategory: j.data.from_category, toCategory: j.data.to_category,
          });
          if (this.cullUndo.length > 100) this.cullUndo.shift();
          // Optimistically remove the card from the view and keep focus sane.
          this.gallery.items = (this.gallery.items || []).filter(it => it.path !== card.path);
          this.gallery.total = Math.max(0, (this.gallery.total || 1) - 1);
          const n = this.gallery.items.length;
          this.cullFocus = n === 0 ? -1 : Math.min(focusedIdx, n - 1);
          this.notify(card.name + ' → ' + target, 'success', 1500);
        } else {
          this.notify('Already in ' + target, 'info', 1200);
        }
      } catch (e) {
        this.notify('Network error during move', 'error');
      } finally {
        this.cullBusy = false;
      }
    },
    cullKeep() {
      const keep = this.cullPrimaryKeep();
      if (!keep) { this.notify('No keep bucket in this taxonomy', 'warn'); return; }
      this.cullRecategorize(keep);
    },
    cullReject() { this.cullRecategorize('DISCARD'); },
    // 1..9 → Nth keep bucket (1-indexed) in taxonomy order.
    cullRecategorizeNth(n) {
      const cat = (this.cullCategories || [])[n - 1];
      if (!cat) { this.notify('No category #' + n, 'warn'); return; }
      this.cullRecategorize(cat.name);
    },
    // Undo the last move by moving the image back (same endpoint, from/to swapped).
    async cullUndoLast() {
      if (this.cullBusy) return;
      const last = this.cullUndo.pop();
      if (!last) { this.notify('Nothing to undo', 'info', 1200); return; }
      this.cullBusy = true;
      try {
        const r = await fetch('/api/gallery/recategorize', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ path: last.path, target: last.fromCategory, job: this.currentJob || undefined }),
        });
        const j = await r.json().catch(() => ({}));
        if (!r.ok || !j.success) {
          this.notify((j && j.error) || 'undo failed', 'error');
          this.cullUndo.push(last);   // restore so the user can retry
          return;
        }
        this.notify('Undid → ' + last.fromCategory, 'success', 1500);
        // The card now belongs in the current view again only if the filter
        // matches; cheapest correct option is a reload.
        this.loadGallery();
      } catch (e) {
        this.notify('Network error during undo', 'error');
        this.cullUndo.push(last);
      } finally {
        this.cullBusy = false;
      }
    },
    // Single keydown entry-point bound to window. Guards against typing in
    // fields, then dispatches the cull hotkeys. Returns early (no preventDefault)
    // for unhandled keys so normal browser shortcuts still work.
    onCullKeydown(ev) {
      if (!this.cullHotkeysActive()) return;
      if (ev.metaKey || ev.ctrlKey || ev.altKey) return;   // don't hijack browser combos
      const k = ev.key;
      // Digits 1..9 → Nth category.
      if (k >= '1' && k <= '9') { ev.preventDefault(); this.cullRecategorizeNth(Number(k)); return; }
      switch (k) {
        case 'ArrowRight': case 'l': ev.preventDefault(); this.cullMoveFocus(1); return;
        case 'ArrowLeft':  case 'h': ev.preventDefault(); this.cullMoveFocus(-1); return;
        case 'ArrowDown':  case 'j': ev.preventDefault(); this.cullMoveFocus(this.cullRowStride()); return;
        case 'ArrowUp':              ev.preventDefault(); this.cullMoveFocus(-this.cullRowStride()); return;
        case 'k': ev.preventDefault(); this.cullKeep(); return;
        case 'x': ev.preventDefault(); this.cullReject(); return;
        case 'u': ev.preventDefault(); this.cullUndoLast(); return;
        case 'Escape': this.cullClearFocus(); return;
        default: return;
      }
    },
    // 'k' is BOTH "move focus down a row" (vim) AND "keep". We resolve the
    // ambiguity in favour of keep (the headline action); use ArrowDown/j to move
    // down a row. cullRowStride approximates the column count of the responsive
    // grid so j/ArrowDown jump a visual row rather than one card.
    cullRowStride() {
      const w = window.innerWidth || 1280;
      if (w >= 1024) return 6;   // lg:grid-cols-6
      if (w >= 768) return 4;    // md:grid-cols-4
      if (w >= 640) return 3;    // sm:grid-cols-3
      return 2;                  // grid-cols-2
    },

    // ── Inline caption editor (§3d) ──────────────────────────────────────
    // Approximate token count: whitespace + comma split (good enough for the UI
    // hint; not a real tokenizer). Empty/whitespace → 0.
    captionTokenCount(text) {
      const t = (text || '').trim();
      if (!t) return 0;
      return t.split(/[\s,]+/).filter(Boolean).length;
    },
    async loadCaption() {
      // Pull the current .txt for the modal's image into the caption editor.
      if (!this.modal.path) { this.modal.caption = ''; this.modal.captionOriginal = ''; return; }
      try {
        const enc = encodeURIComponent(this.modal.path);
        const j = await fetch('/api/caption?path=' + enc).then(r => r.json());
        const text = (j && j.success && j.data) ? (j.data.text || '') : '';
        this.modal.caption = text;
        this.modal.captionOriginal = text;
      } catch (e) {
        this.modal.caption = ''; this.modal.captionOriginal = '';
      }
    },
    // Prepend the trigger word to the caption (idempotent-ish: skips if the
    // caption already starts with it). Comma-separated, the LoRA-tag convention.
    prependTriggerWord() {
      const tw = (this.triggerWord || '').trim();
      if (!tw) { this.notify('Enter a trigger word first', 'warn', 1500); return; }
      const cur = this.modal.caption || '';
      if (cur.toLowerCase().startsWith(tw.toLowerCase())) { this.notify('Already prefixed', 'info', 1200); return; }
      this.modal.caption = cur ? (tw + ', ' + cur) : tw;
    },
    async saveCaption() {
      if (!this.modal.path || this.modal.captionSaving) return;
      this.modal.captionSaving = true;
      try {
        const r = await fetch('/api/caption', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ path: this.modal.path, text: this.modal.caption || '', job: this.currentJob || undefined }),
        });
        const j = await r.json().catch(() => ({}));
        if (!r.ok || !j.success) {
          this.modal.captionFlash = 'Error: ' + ((j && j.error) || r.status);
          return;
        }
        this.modal.captionOriginal = this.modal.caption;
        this.modal.captionFlash = 'Caption saved.';
        this.notify('Caption saved', 'success', 1500);
        setTimeout(() => this.modal.captionFlash = '', 2500);
      } catch (e) {
        this.modal.captionFlash = 'Network error — check console';
        console.error('saveCaption failed', e);
      } finally {
        this.modal.captionSaving = false;
      }
    },

    // Scope query string for job-aware endpoints: ?job=<slug> when viewing a
    // job, else '' (server falls back to the active job).
    jobQuery() { return this.currentJob ? ('?job=' + encodeURIComponent(this.currentJob)) : ''; },
    // ── Duplicates (perceptual-hash near-dupes) ──────────────────────────────
    async loadDuplicates() {
      this.dup.loading = true; this.dup.error = '';
      try {
        const q = this.jobQuery();
        const sep = q ? '&' : '?';
        const r = await fetch('/api/duplicates' + q + sep + 'threshold=' + (this.dup.threshold|0));
        const j = await r.json();
        if (j.success) { this.dup.groups = j.data.groups || []; this.dup.scanned = true; }
        else { this.dup.error = j.error || 'scan failed'; }
      } catch (e) { this.dup.error = 'network error'; }
      this.dup.loading = false;
    },
    async deleteDuplicate(path) {
      // Delete a sorted dupe + its sidecars (safe_inside-guarded server-side).
      try {
        const r = await fetch('/api/duplicates/delete', {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ path }),
        });
        const j = await r.json();
        if (j.success) {
          // Drop the deleted path from every group; prune singletons.
          this.dup.groups = this.dup.groups
            .map(g => g.filter(m => m.path !== path))
            .filter(g => g.length > 1);
          this.notify('Deleted 1 image', 'info');
        } else { this.notify(j.error || 'delete failed', 'error'); }
      } catch (e) { this.notify('network error', 'error'); }
    },
    async keepOnly(group, keepPath) {
      // Delete every member of the group except keepPath.
      const victims = group.filter(m => m.path !== keepPath).map(m => m.path);
      for (const p of victims) { await this.deleteDuplicate(p); }
    },

    // ── Find similar (CLIP embeddings) ───────────────────────────────────────
    async findSimilar(card) {
      // Open the results strip and fetch nearest neighbours for this image. The
      // backend degrades gracefully (no ML extra / empty index / un-indexed key)
      // by returning a friendly message instead of an error.
      this.sim = { open: true, loading: true, results: [], message: '', sourceName: card.name || '' };
      const q = this.jobQuery();
      const sep = q ? '&' : '?';
      const url = '/api/embeddings/similar' + q + sep
                + 'key=' + encodeURIComponent(card.path) + '&top_k=12';
      try {
        const j = await (await fetch(url)).json();
        const data = j.data || {};
        if (!j.success) { this.sim.message = j.error || 'similar search failed'; }
        else if (data.message && (data.results || []).length === 0) { this.sim.message = data.message; }
        else { this.sim.results = data.results || []; }
      } catch (e) { this.sim.message = 'network error'; }
      this.sim.loading = false;
    },

    // ── Dataset export ───────────────────────────────────────────────────────
    async runDatasetExport() {
      this.exportForm.running = true; this.exportForm.error = ''; this.exportForm.summary = null;
      const f = this.exportForm;
      const body = { profile: f.profile, resolution_bucketing: !!f.resolution_bucketing };
      if (f.trigger_word.trim()) body.trigger_word = f.trigger_word.trim();
      if (String(f.train_val_split).trim()) body.train_val_split = parseFloat(f.train_val_split);
      if (String(f.repeats).trim()) body.repeats = parseInt(f.repeats, 10);
      if (f.video_bucketing) body.video_bucketing = f.video_bucketing;
      try {
        const r = await fetch('/api/export/dataset', {
          method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body),
        });
        const j = await r.json();
        if (j.success) { this.exportForm.summary = j.data; this.notify('Export complete', 'info'); }
        else { this.exportForm.error = j.error || 'export failed'; }
      } catch (e) { this.exportForm.error = 'network error'; }
      this.exportForm.running = false;
    },

    // ── Push to HuggingFace ──────────────────────────────────────────────────
    async pushToHuggingFace() {
      const repo = (this.hf.repo_id || '').trim();
      if (!/^[\w.-]+\/[\w.-]+$/.test(repo)) { this.hf.error = 'repo_id must look like namespace/name'; return; }
      this.hf.pushing = true; this.hf.error = ''; this.hf.result = null; this.hf.status = '';
      try {
        const r = await fetch('/api/jobs/' + encodeURIComponent(this.currentJob) + '/export/huggingface', {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ repo_id: repo, private: !!this.hf.private, include_video: !!this.hf.include_video }),
        });
        const j = await r.json();
        // 202 + an id means the upload is running on a background thread — poll
        // the status endpoint until it finishes. A terminal result (fast failure
        // or an already-finished push) comes back inline and skips polling.
        if (r.status === 202 && j.data && j.data.id) {
          this.hf.status = 'Uploading…';
          await this._pollHfStatus(j.data.id);
        } else if (j.success) {
          this.hf.result = j.data; this.hf.status = 'done';
          this.notify('Pushed to HuggingFace', 'info');
        } else {
          this.hf.error = j.error || 'push failed'; this.hf.status = '';
        }
      } catch (e) { this.hf.error = 'network error'; this.hf.status = ''; }
      this.hf.pushing = false;
    },
    async _pollHfStatus(id) {
      // Poll the HF-export status endpoint every 2s until done/error. Best-effort
      // polling: a transient network blip just retries on the next tick.
      const url = '/api/jobs/' + encodeURIComponent(this.currentJob)
                + '/export/huggingface/status?id=' + encodeURIComponent(id);
      for (let i = 0; i < 1800; i++) {            // ~1h ceiling at 2s intervals
        await new Promise(res => setTimeout(res, 2000));
        let j;
        try { j = await (await fetch(url)).json(); } catch (e) { continue; }
        const state = j.data && j.data.state;
        if (state === 'done') {
          this.hf.result = (j.data && j.data.result) || null; this.hf.status = 'done';
          this.notify('Pushed to HuggingFace', 'info');
          return;
        }
        if (state === 'error' || j.success === false) {
          this.hf.error = j.error || 'push failed'; this.hf.status = '';
          return;
        }
        this.hf.status = 'Uploading…';
      }
      this.hf.status = '';
      this.hf.error = 'upload is taking longer than expected — check HuggingFace directly';
    },

    // ── Schedules ────────────────────────────────────────────────────────────
    async loadSchedules() {
      this.schedules.loading = true; this.schedules.error = '';
      try {
        const r = await fetch('/api/schedules');
        const j = await r.json();
        if (j.success) { this.schedules.rows = j.data.schedules || []; this.schedules.actions = j.data.actions || []; }
        else { this.schedules.error = j.error || 'load failed'; }
      } catch (e) { this.schedules.error = 'network error'; }
      this.schedules.loading = false;
    },
    async addSchedule() {
      const f = this.schedules.form;
      if (!f.slug) { this.notify('pick a job', 'error'); return; }
      try {
        const r = await fetch('/api/schedules', {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ slug: f.slug, cadence: f.cadence, action: f.action, enabled: !!f.enabled }),
        });
        const j = await r.json();
        if (j.success) { this.notify('Schedule saved', 'info'); this.loadSchedules(); }
        else { this.notify(j.error || 'save failed', 'error'); }
      } catch (e) { this.notify('network error', 'error'); }
    },
    async removeSchedule(slug, action) {
      try {
        const r = await fetch('/api/schedules?slug=' + encodeURIComponent(slug) + '&action=' + encodeURIComponent(action), { method: 'DELETE' });
        const j = await r.json();
        if (j.success) { this.loadSchedules(); } else { this.notify(j.error || 'delete failed', 'error'); }
      } catch (e) { this.notify('network error', 'error'); }
    },

    // ── Vision fleet health telemetry ────────────────────────────────────────
    async loadVisionHealth() {
      this.visionHealth.loading = true; this.visionHealth.error = '';
      try {
        const r = await fetch('/api/vision/health' + this.jobQuery());
        const j = await r.json();
        if (j.success) { this.visionHealth.probes = j.data.probes || {}; this.visionHealth.fleet = j.data.fleet || []; }
        else { this.visionHealth.error = j.error || 'probe failed'; }
      } catch (e) { this.visionHealth.error = 'network error'; }
      this.visionHealth.loading = false;
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
      this.loadJobs();
      this.loadPresets();
      this.refresh();
      // Global keyboard-cull listener. Gated entirely by cullHotkeysActive()
      // (gallery tab open, no modal, not typing) so it's inert everywhere else.
      window.addEventListener('keydown', (ev) => this.onCullKeydown(ev));
      setInterval(() => this.refresh(), 5000);
      // Lazy-load stats / gallery only when their tab is opened. Stats then
      // auto-refreshes alongside the rest of the dashboard. Gallery stays
      // user-driven because filter state shouldn't be clobbered on poll.
      this.$watch('active', (tab) => {
        if (tab === 'stats') this.loadStats();
        if (tab === 'gstats') this.loadGlobalStats();
        if (tab === 'presets') this.loadPresets();
        if (tab === 'settings') this.loadGlobalVision();
        if (tab === 'schedules') { this.loadSchedules(); if (!this.jobsList.length) this.loadJobs(); }
        // Duplicates is on-demand: the "Scan for duplicates" button is the only
        // trigger (a perceptual-hash sweep is too expensive to auto-fire).
        if (tab === 'gallery' && this.gallery.items.length === 0 && !this.galleryLoading) {
          this.loadGallery();
          this.loadGalleryInsights();
        }
        // Keyboard culling needs the ordered taxonomy; load it whenever the
        // gallery opens and reset focus so a stale index from a prior view
        // doesn't ring a card the user can't see.
        if (tab === 'gallery') { this.loadCullCategories(); this.cullFocus = -1; }
        // Job Settings + Vision both edit this job's inheritable config (v2),
        // so they need the effective cfg + overrides loaded.
        if ((tab === 'jobSettings' || tab === 'vision') && !this.je.loaded) this.loadJobEditor();
        if (tab === 'vision') this.loadVisionHealth();
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
    # Inject the canonical scraper-name list so the Alpine `scraperNames` state
    # is sourced from job_config.SCRAPER_NAMES and can never drift from it (the
    # template is server-rendered Python). The template has no other Jinja
    # mustaches; single-brace Alpine object literals are untouched by Jinja.
    return render_template_string(
        HTML_TEMPLATE,
        scraper_names_json=json.dumps(list(job_config.SCRAPER_NAMES)),
        export_profiles_json=json.dumps(list(export_profiles.PROFILES)),
        video_bucket_modes_json=json.dumps(list(export_profiles.VIDEO_BUCKET_MODES)),
        caption_styles_json=json.dumps(list(vision_prompt.CAPTION_STYLES)),
    )


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
