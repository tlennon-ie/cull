"""cull-mcp — Model Context Protocol server for cull.

Exposes cull's public APIs (job_config / paths / queue_manager / index_store /
hf_export / export_profiles) as MCP tools so a Claude Desktop / Cursor / Codex
agent can drive an end-to-end curation run over stdio.

Design invariants:

* **Never reach into private state.** Every tool goes through the same public
  APIs the dashboard and CLI use — a single source of truth for jobs, presets,
  the active pointer, and every stat/gallery query.
* **stdout is the transport.** All logging is routed to stderr (or a file via
  ``pipeline_logging``); a stray ``print()`` to stdout corrupts the JSON-RPC
  frame the MCP client is parsing.
* **Import cleanly without the SDK.** ``pip install cull`` (no ``[mcp]`` extra)
  must still be able to ``import mcp_server``; only ``main()`` fails with a
  friendly install hint. The tool functions themselves are pure Python and
  are exercised by the test suite without an MCP session.
* **Path-injection barrier.** Any tool that dereferences a caller-supplied path
  goes through :func:`_safe_media_path`, which is the moral equivalent of the
  dashboard's ``safe_inside()`` and rejects anything outside the queue / sorted
  roots.
* **Never leak credentials.** Preset/job envelopes are masked through
  :func:`_mask_secrets` before returning, mirroring the dashboard's
  ``SECRET_KEYS`` sentinel.

The console script ``cull-mcp`` (declared in ``pyproject.toml``) resolves to
:func:`main` and is what a Claude Desktop config points at.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import sys
from pathlib import Path
from typing import Any

# ── Guarded MCP SDK import ──────────────────────────────────────────────────
#
# The ``mcp`` package is declared as an *optional* extra (``pip install cull[mcp]``).
# The module must still import when the SDK is absent so the console script
# entry point (`cull-mcp`) can print a friendly install hint instead of a raw
# ImportError. All type references from the SDK live inside the ``try`` block;
# the tool implementations never reference them.
try:  # pragma: no cover - trivial guard
    from mcp.server import Server as _MCPServer  # type: ignore
    from mcp.server.stdio import stdio_server as _stdio_server  # type: ignore
    from mcp.types import (  # type: ignore
        CallToolResult as _CallToolResult,
        ImageContent as _ImageContent,
        TextContent as _TextContent,
        Tool as _Tool,
    )
    _MCP_IMPORT_ERROR: Exception | None = None
except Exception as _exc:  # pragma: no cover - only exercised without the SDK
    _MCPServer = None  # type: ignore[assignment]
    _stdio_server = None  # type: ignore[assignment]
    _Tool = None  # type: ignore[assignment]
    _TextContent = None  # type: ignore[assignment]
    _ImageContent = None  # type: ignore[assignment]
    _CallToolResult = None  # type: ignore[assignment]
    _MCP_IMPORT_ERROR = _exc

# ── cull public APIs ────────────────────────────────────────────────────────
#
# Everything below flows through these — no other module reaches into cull's
# internals from here.
import job_config
import paths
from pipeline_logging import get_logger

logger = get_logger(__name__)


# ── Constants ───────────────────────────────────────────────────────────────

SERVER_NAME = "cull"
SECRET_MASK = "********"

# Mirrors dashboard_enhanced.SECRET_KEYS — a small guard so any tool that
# returns a preset/job envelope never ships raw credentials over MCP.
SECRET_KEYS: frozenset[str] = frozenset({
    "GROQ_API_KEY", "GROQ_API_KEYS",
    "CIVITAI_API_KEY", "CIVITAI_API_RED_KEY",
    "TWITTER_COOKIES", "DISCORD_BOT_TOKEN",
    "REDDIT_CLIENT_SECRET", "REDDIT_COOKIES",
    "OPENAI_COMPAT_API_KEY", "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY", "GEMINI_API_KEY",
    "GOOGLE_API_KEY", "OPENROUTER_API_KEY",
    "HF_TOKEN", "HUGGINGFACE_TOKEN",
})

# The key inside a vision-worker fleet entry that carries the raw API key.
_FLEET_SECRET_FIELDS: frozenset[str] = frozenset({"api_key"})

# Thumbnail size returned by ``cull_sample_gallery`` for the top image.
_SAMPLE_THUMB_SIZE = 320
_SAMPLE_THUMB_QUALITY = 82


# ── Response builders ───────────────────────────────────────────────────────

def _text(payload: Any) -> "_TextContent":
    """Wrap a JSON-serialisable payload in an MCP ``TextContent`` block."""
    if isinstance(payload, str):
        text = payload
    else:
        text = json.dumps(payload, indent=2, default=str)
    return _TextContent(type="text", text=text)


def _error(message: str, **extra: Any) -> "_CallToolResult":
    """Build the standard MCP error envelope (``isError=True``).

    ``message`` is intentionally generic (no traceback); operators get the
    detail from the server-side log via ``pipeline_logging``.
    """
    payload: dict[str, Any] = {"error": message}
    payload.update(extra)
    return _CallToolResult(
        content=[_TextContent(type="text", text=json.dumps(payload, indent=2, default=str))],
        isError=True,
    )


def _ok(payload: Any) -> list["_TextContent"]:
    return [_text(payload)]


# ── Credential masking ─────────────────────────────────────────────────────

def _mask_secrets(value: Any, *, in_fleet: bool = False) -> Any:
    """Return a deep copy of ``value`` with every known secret key redacted.

    Two masking domains coexist in cull's config:

    * top-level ``SECRET_KEYS`` (env-var-style names) — dashboard settings;
    * per-fleet-entry ``api_key`` field — the vision-worker fleet uses that
      exact key name on each ``{provider, base_url, model, api_key}`` dict.

    Callers pass ``in_fleet=True`` on the fleet sub-tree so the ``api_key``
    guard fires inside a nested list of worker dicts.
    """
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, sub in value.items():
            if key in SECRET_KEYS and sub:
                cleaned[key] = SECRET_MASK
                continue
            if in_fleet and key in _FLEET_SECRET_FIELDS and sub:
                cleaned[key] = SECRET_MASK
                continue
            recurse_fleet = in_fleet or key == "workers"
            cleaned[key] = _mask_secrets(sub, in_fleet=recurse_fleet)
        return cleaned
    if isinstance(value, list):
        return [_mask_secrets(item, in_fleet=in_fleet) for item in value]
    return value


# ── Path safety (mirror of dashboard_enhanced.safe_inside) ─────────────────

def _safe_media_path(raw: str) -> Path | None:
    """Return the realpath of ``raw`` iff it lives under queue or sorted.

    The MCP surface only ever dereferences image paths that came from the
    SQLite index (or from an agent that says it did). We still refuse to open
    anything outside the queue / sorted trees so a compromised or curious
    agent cannot ask us to read arbitrary files on disk.
    """
    if not raw:
        return None
    try:
        real = os.path.realpath(str(raw))
    except (OSError, ValueError):
        return None
    for root in (paths.queue_root(), paths.sorted_root()):
        try:
            root_real = os.path.realpath(str(root))
            if os.path.commonpath([real, root_real]) == root_real:
                return Path(real)
        except (OSError, ValueError):
            continue
    return None


# ── Data-shaping helpers ────────────────────────────────────────────────────

def _job_row(job: job_config.Job, active: list[str]) -> dict[str, Any]:
    """Compact per-job row for ``cull_list_jobs`` — mirrors the CLI table."""
    counts = _job_counts(job.slug)
    return {
        "slug": job.slug,
        "name": job.name,
        "subject": job.subject,
        "preset": job.preset,
        "status": job.status,
        "active": job.slug in active,
        "queued": counts["queued"],
        "sorted": counts["sorted"],
    }


def _job_counts(slug: str) -> dict[str, int]:
    """{'queued': n, 'sorted': n} for a slug.

    Uses ``index_store`` when it is configured (dashboard-shared install), and
    falls back to a cheap filesystem walk otherwise so the MCP surface works on
    a bare CLI install that never opened the SQLite index.
    """
    try:
        import index_store
        with index_store.with_conn() as conn:
            q = conn.execute(
                "SELECT COUNT(*) FROM images WHERE status='queue' AND topic_slug = ?",
                (slug,),
            ).fetchone()[0]
            s = conn.execute(
                "SELECT COUNT(*) FROM images WHERE status='sorted' AND topic_slug = ?",
                (slug,),
            ).fetchone()[0]
            return {"queued": int(q), "sorted": int(s)}
    except Exception:
        return {
            "queued": _count_media(paths.queue_dir(slug)),
            "sorted": _count_media(paths.sorted_dir(slug)),
        }


_MEDIA_SUFFIXES: frozenset[str] = frozenset({
    ".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp",
    ".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v",
})


def _count_media(root: Path) -> int:
    """Count image / video files under ``root`` (recursive, cheap fallback)."""
    if not root.is_dir():
        return 0
    total = 0
    try:
        for path in root.rglob("*"):
            try:
                if path.is_file() and path.suffix.lower() in _MEDIA_SUFFIXES:
                    total += 1
            except OSError:
                continue
    except OSError:
        return total
    return total


def _validated_slug(raw: Any) -> str:
    """Validate + return a slug from a tool argument.

    Raises :class:`ValueError` with a clear message so :func:`call_tool` can
    turn it into an MCP error envelope. Rejects everything that ``job_config``
    would refuse (``/``, ``\\``, ``..``, uppercase, punctuation) — same guard
    the dashboard uses.
    """
    slug = str(raw or "").strip()
    if not job_config.JOB_SLUG_RE.fullmatch(slug):
        raise ValueError(
            f"invalid slug {slug!r}: use lowercase letters, digits, underscores"
        )
    return slug


# ── Tool: cull_list_jobs ────────────────────────────────────────────────────

def tool_list_jobs(_args: dict[str, Any]) -> list["_TextContent"]:
    jobs = job_config.list_jobs()
    active = job_config.get_active_slugs()
    rows = [_job_row(j, active) for j in jobs]
    return _ok({"jobs": rows, "active": active})


# ── Tool: cull_get_job ──────────────────────────────────────────────────────

def tool_get_job(args: dict[str, Any]) -> list["_TextContent"] | "_CallToolResult":
    slug = _validated_slug(args.get("slug"))
    job = job_config.get_job(slug)
    if job is None:
        return _error(f"unknown job: {slug}", slug=slug)
    return _ok({
        "slug": job.slug,
        "name": job.name,
        "status": job.status,
        "subject": job.subject,
        "preset": job.preset,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "active": slug in job_config.get_active_slugs(),
        "priority": job_config.get_job_priority(slug),
        "overrides": _mask_secrets(job.overrides),
        "effective_config": _mask_secrets(job_config.effective_config(job)),
    })


# ── Tool: cull_create_job ───────────────────────────────────────────────────

def tool_create_job(args: dict[str, Any]) -> list["_TextContent"] | "_CallToolResult":
    slug_hint = str(args.get("slug") or "").strip()
    if not slug_hint:
        return _error("slug is required")
    subject = args.get("subject")
    if subject is not None:
        subject = str(subject)
    preset = args.get("preset")
    if preset is not None:
        preset = str(preset)
    base_on = args.get("base_on")
    if base_on is not None:
        base_on = str(base_on)
    try:
        job = job_config.create_job(
            slug_hint, subject=subject, preset=preset, base_on=base_on,
        )
    except ValueError as exc:
        return _error(str(exc))
    return _ok({
        "slug": job.slug,
        "name": job.name,
        "subject": job.subject,
        "preset": job.preset,
        "status": job.status,
        "created_at": job.created_at,
    })


# ── Tool: cull_delete_job ───────────────────────────────────────────────────

def tool_delete_job(args: dict[str, Any]) -> list["_TextContent"] | "_CallToolResult":
    slug = _validated_slug(args.get("slug"))
    try:
        job_config.delete_job(slug)
    except ValueError as exc:
        return _error(str(exc), slug=slug)
    return _ok({"slug": slug, "deleted": True})


# ── Tool: cull_activate_job ────────────────────────────────────────────────

def tool_activate_job(args: dict[str, Any]) -> list["_TextContent"] | "_CallToolResult":
    slug = _validated_slug(args.get("slug"))
    exclusive = bool(args.get("exclusive", False))
    try:
        job_config.activate(slug, exclusive=exclusive)
    except ValueError as exc:
        return _error(str(exc), slug=slug)
    return _ok({"slug": slug, "exclusive": exclusive,
                "active": job_config.get_active_slugs()})


# ── Tool: cull_deactivate_job ──────────────────────────────────────────────

def tool_deactivate_job(args: dict[str, Any]) -> list["_TextContent"] | "_CallToolResult":
    slug = _validated_slug(args.get("slug"))
    job_config.deactivate(slug)
    return _ok({"slug": slug, "active": job_config.get_active_slugs()})


# ── Tool: cull_set_job_priority ────────────────────────────────────────────

def tool_set_job_priority(args: dict[str, Any]) -> list["_TextContent"] | "_CallToolResult":
    slug = _validated_slug(args.get("slug"))
    priority = args.get("priority")
    if not isinstance(priority, (int, float)):
        return _error("priority must be an integer 1-10")
    try:
        stored = job_config.set_job_priority(slug, int(priority))
    except ValueError as exc:
        return _error(str(exc), slug=slug)
    return _ok({"slug": slug, "priority": stored})


# ── Tool: cull_list_presets ────────────────────────────────────────────────

def tool_list_presets(_args: dict[str, Any]) -> list["_TextContent"]:
    lib = job_config.list_presets()
    builtin = set(job_config.builtin_preset_names())
    rows: list[dict[str, Any]] = []
    for name in sorted(lib.get("presets") or {}):
        source = "builtin" if name in builtin else "custom"
        try:
            import builtin_presets
            description = builtin_presets.preset_headline(name) if name in builtin else ""
        except Exception:
            description = ""
        rows.append({"name": name, "source": source, "description": description})
    return _ok({
        "presets": rows,
        "default": lib.get("default"),
    })


# ── Tool: cull_get_preset ──────────────────────────────────────────────────

def tool_get_preset(args: dict[str, Any]) -> list["_TextContent"] | "_CallToolResult":
    name = str(args.get("name") or "").strip()
    if not name:
        return _error("name is required")
    lib = job_config.list_presets()
    if name not in (lib.get("presets") or {}):
        return _error(f"unknown preset: {name}")
    cfg = job_config.get_preset(name)
    return _ok({
        "name": name,
        "source": "builtin" if name in job_config.builtin_preset_names() else "custom",
        "cfg": _mask_secrets(cfg),
    })


# ── Tool: cull_clone_preset ────────────────────────────────────────────────

def tool_clone_preset(args: dict[str, Any]) -> list["_TextContent"] | "_CallToolResult":
    source_name = str(args.get("source_name") or "").strip()
    new_name = str(args.get("new_name") or "").strip()
    if not source_name or not new_name:
        return _error("source_name and new_name are required")
    lib = job_config.list_presets()
    if source_name not in (lib.get("presets") or {}):
        return _error(f"unknown preset: {source_name}")
    try:
        cfg = job_config.get_preset(source_name)
        job_config.save_preset(new_name, cfg)
    except ValueError as exc:
        return _error(str(exc))
    return _ok({
        "source": source_name, "new_name": new_name, "cloned": True,
    })


# ── Tool: cull_start_pipeline / cull_stop_pipeline / cull_pipeline_status ──
#
# The MCP surface never spawns the supervisor itself — the dashboard already
# owns the subprocess handle. We surface both the actionable next step ("call
# /api/pipeline/start on the local dashboard") and, when the dashboard is up,
# the state so the agent can decide.

def _dashboard_url() -> str:
    port = os.environ.get("FLASK_PORT", "5000").strip() or "5000"
    return f"http://127.0.0.1:{port}"


def _pipeline_state_via_dashboard() -> dict[str, Any] | None:
    """Best-effort GET /api/status on the local dashboard. Returns None on any
    failure so the caller can degrade to a filesystem-only view."""
    try:
        import urllib.request
        req = urllib.request.Request(
            f"{_dashboard_url()}/api/status",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=1.5) as resp:  # noqa: S310
            data = json.loads(resp.read().decode("utf-8"))
            return data if isinstance(data, dict) else None
    except Exception:
        return None


def tool_start_pipeline(_args: dict[str, Any]) -> list["_TextContent"] | "_CallToolResult":
    state = _pipeline_state_via_dashboard()
    if state is not None:
        return _ok({
            "note": "The dashboard owns the supervisor. POST /api/pipeline/start "
                    "to start it (auth token from API_TOKEN env for PR 2's API auth).",
            "dashboard_url": _dashboard_url(),
            "current": state,
        })
    return _error(
        f"could not reach the local dashboard at {_dashboard_url()}; start it "
        "with `python pipeline_code/dashboard_enhanced.py` or `cull run`."
    )


def tool_stop_pipeline(_args: dict[str, Any]) -> list["_TextContent"] | "_CallToolResult":
    state = _pipeline_state_via_dashboard()
    if state is not None:
        return _ok({
            "note": "POST /api/pipeline/stop on the dashboard to stop it.",
            "dashboard_url": _dashboard_url(),
            "current": state,
        })
    return _error(
        f"could not reach the local dashboard at {_dashboard_url()}"
    )


def tool_pipeline_status(_args: dict[str, Any]) -> list["_TextContent"]:
    active_slugs = job_config.get_active_slugs()
    fs_totals = {
        slug: _job_counts(slug) for slug in (active_slugs or [])
    }
    state = _pipeline_state_via_dashboard() or {}
    return _ok({
        "running": bool(state.get("pipeline_running", False)),
        "active_slugs": active_slugs,
        "queue_totals": fs_totals,
        "dashboard_url": _dashboard_url() if state else None,
        "worker_health": state.get("fleet_health"),
    })


# ── Tool: cull_set_scoring / cull_get_scoring ──────────────────────────────

def tool_set_scoring(args: dict[str, Any]) -> list["_TextContent"] | "_CallToolResult":
    slug = _validated_slug(args.get("slug"))
    job = job_config.get_job(slug)
    if job is None:
        return _error(f"unknown job: {slug}")
    updated = job
    for key, arg_key in (("ovr_min", "min_ovr"), ("rel_min", "min_rel")):
        val = args.get(arg_key)
        if val is None:
            continue
        try:
            updated = job_config.set_override(updated, f"scoring.{key}", int(val))
        except (TypeError, ValueError):
            return _error(f"{arg_key} must be an integer")
    if "require_prompt" in args:
        updated = job_config.set_override(
            updated, "topic_filters.require_prompt", bool(args["require_prompt"])
        )
    if updated is job:
        return _error("no scoring fields supplied (min_ovr / min_rel / require_prompt)")
    saved = job_config.save_job(updated)
    return _ok({
        "slug": slug,
        "scoring": job_config.effective_config(saved).get("scoring", {}),
        "require_prompt": job_config.effective_config(saved)
            .get("topic", {}).get("require_prompt"),
    })


def tool_get_scoring(args: dict[str, Any]) -> list["_TextContent"] | "_CallToolResult":
    slug = _validated_slug(args.get("slug"))
    job = job_config.get_job(slug)
    if job is None:
        return _error(f"unknown job: {slug}")
    eff = job_config.effective_config(job)
    return _ok({
        "slug": slug,
        "scoring": eff.get("scoring", {}),
        "require_prompt": eff.get("topic", {}).get("require_prompt"),
    })


# ── Tool: cull_add_scraper_url ─────────────────────────────────────────────

_SCRAPER_URL_SOURCES: dict[str, str] = {
    "gallery_dl": "scrapers.gallery_dl.urls",
    "gallery-dl": "scrapers.gallery_dl.urls",
    "yt_dlp":     "scrapers.yt_dlp.urls",
    "yt-dlp":     "scrapers.yt_dlp.urls",
}


def tool_add_scraper_url(args: dict[str, Any]) -> list["_TextContent"] | "_CallToolResult":
    slug = _validated_slug(args.get("slug"))
    source = str(args.get("source") or "").strip().lower()
    url = str(args.get("url") or "").strip()
    if not url:
        return _error("url is required")
    path = _SCRAPER_URL_SOURCES.get(source)
    if path is None:
        return _error(
            f"unsupported source {source!r}; expected one of "
            f"{sorted(set(_SCRAPER_URL_SOURCES))}"
        )
    job = job_config.get_job(slug)
    if job is None:
        return _error(f"unknown job: {slug}")
    eff = job_config.effective_config(job)
    scrapers = eff.get("scrapers") or {}
    section = (scrapers.get(source.replace("-", "_"))) or {}
    urls = list(section.get("urls") or [])
    if url not in urls:
        urls.append(url)
    updated = job_config.set_override(job, path, urls)
    saved = job_config.save_job(updated)
    return _ok({
        "slug": slug, "source": source, "urls": urls,
        "count": len(urls), "job_updated_at": saved.updated_at,
    })


# ── Tool: cull_toggle_scraper ──────────────────────────────────────────────

def tool_toggle_scraper(args: dict[str, Any]) -> list["_TextContent"] | "_CallToolResult":
    slug = _validated_slug(args.get("slug"))
    name = str(args.get("name") or "").strip()
    if name not in job_config.SCRAPER_NAMES:
        return _error(
            f"unknown scraper {name!r}; valid: {list(job_config.SCRAPER_NAMES)}"
        )
    if "enabled" not in args:
        return _error("enabled is required (true/false)")
    enabled = bool(args["enabled"])
    job = job_config.get_job(slug)
    if job is None:
        return _error(f"unknown job: {slug}")
    eff_map = dict(
        (job_config.effective_config(job).get("scrapers") or {}).get("enabled") or {}
    )
    eff_map[name] = enabled
    updated = job_config.set_override(job, "scrapers.enabled", eff_map)
    job_config.save_job(updated)
    return _ok({"slug": slug, "name": name, "enabled": enabled,
                "enabled_map": eff_map})


# ── Tool: cull_stats ───────────────────────────────────────────────────────

def tool_stats(args: dict[str, Any]) -> list["_TextContent"] | "_CallToolResult":
    raw = args.get("slug")
    slug = _validated_slug(raw) if raw else None
    try:
        import index_store
    except Exception as exc:  # pragma: no cover
        return _error(f"index_store unavailable: {exc}")
    with index_store.with_conn() as conn:
        # count by status × source
        by_source: dict[str, dict[str, int]] = {"queue": {}, "sorted": {}}
        if slug is None:
            cur = conn.execute(
                "SELECT status, source, COUNT(*) FROM images GROUP BY status, source"
            )
        else:
            cur = conn.execute(
                "SELECT status, source, COUNT(*) FROM images "
                "WHERE topic_slug = ? GROUP BY status, source",
                (slug,),
            )
        for status, source, count in cur.fetchall():
            by_source.setdefault(status, {})[source] = int(count)
        # sorted by category
        if slug is None:
            cur = conn.execute(
                "SELECT category, COUNT(*) FROM images WHERE status='sorted' "
                "AND category IS NOT NULL GROUP BY category"
            )
        else:
            cur = conn.execute(
                "SELECT category, COUNT(*) FROM images WHERE status='sorted' "
                "AND category IS NOT NULL AND topic_slug = ? GROUP BY category",
                (slug,),
            )
        by_category = {row[0]: int(row[1]) for row in cur.fetchall()}
        # score histogram (bucketed 0-100 by 10s)
        params: list[Any] = []
        where = "status='sorted' AND ovr IS NOT NULL"
        if slug is not None:
            where += " AND topic_slug = ?"
            params.append(slug)
        cur = conn.execute(
            f"SELECT (ovr / 10) * 10 AS bucket, COUNT(*) FROM images "
            f"WHERE {where} GROUP BY bucket ORDER BY bucket",
            params,
        )
        ovr_hist = {int(row[0]): int(row[1]) for row in cur.fetchall()}
    return _ok({
        "slug": slug,
        "by_source": by_source,
        "by_category": by_category,
        "ovr_histogram": ovr_hist,
    })


# ── Tool: cull_sample_gallery ──────────────────────────────────────────────

def _make_thumbnail(image_path: Path, size: int) -> tuple[str, str] | None:
    """Return (base64_jpeg, mime_type) for an image path.

    Prefers the disk-cached thumbnail store when configured (dashboard-shared
    install); falls back to an in-memory Pillow resize so the CLI-only path
    still returns something the agent can see.
    """
    try:
        from PIL import Image  # local import — keeps module import cheap
    except Exception:  # pragma: no cover
        return None
    try:
        import thumb_cache
        try:
            cached = thumb_cache.get_or_create(image_path, size)
            data = cached.read_bytes()
            return base64.b64encode(data).decode("ascii"), "image/jpeg"
        except (RuntimeError, ValueError):
            pass  # not configured or path outside allowed roots — fall through
    except Exception:  # pragma: no cover
        pass
    try:
        with Image.open(image_path) as img:
            img.thumbnail((size, size))
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=_SAMPLE_THUMB_QUALITY, optimize=True)
            return base64.b64encode(buf.getvalue()).decode("ascii"), "image/jpeg"
    except Exception as exc:  # pragma: no cover
        logger.debug("thumbnail render failed for %s: %s", image_path, exc)
        return None


def tool_sample_gallery(args: dict[str, Any]) -> list[Any] | "_CallToolResult":
    slug = _validated_slug(args.get("slug"))
    category = str(args.get("category") or "Keep").strip() or "Keep"
    try:
        n = int(args.get("n", 10))
    except (TypeError, ValueError):
        return _error("n must be an integer")
    n = max(1, min(50, n))
    try:
        import index_store
    except Exception as exc:  # pragma: no cover
        return _error(f"index_store unavailable: {exc}")
    with index_store.with_conn() as conn:
        cur = conn.execute(
            "SELECT * FROM images WHERE status='sorted' AND topic_slug = ? "
            "AND category = ? ORDER BY ovr DESC NULLS LAST, mtime DESC LIMIT ?",
            (slug, category, int(n)),
        )
        rows = [index_store.IndexedImage.from_row(r) for r in cur.fetchall()]
    records: list[dict[str, Any]] = []
    for img in rows:
        records.append({
            "path": img.path,
            "category": img.category,
            "ovr": img.ovr,
            "rel": img.rel,
            "quality": img.quality,
            "prompt": (img.prompt or "")[:800],
            "source": img.source,
        })
    if not rows:
        return _ok({
            "slug": slug, "category": category, "count": 0, "items": [],
        })
    # Render a thumbnail for the top result so the agent can SEE what it's about.
    top = rows[0]
    top_path = _safe_media_path(top.path)
    blocks: list[Any] = [
        _text({
            "slug": slug, "category": category, "count": len(records),
            "items": records,
        }),
    ]
    if top_path is not None:
        thumb = _make_thumbnail(top_path, _SAMPLE_THUMB_SIZE)
        if thumb is not None:
            b64, mime = thumb
            blocks.append(_ImageContent(type="image", data=b64, mimeType=mime))
    return blocks


# ── Tool: cull_get_vision_meta ─────────────────────────────────────────────

def tool_get_vision_meta(args: dict[str, Any]) -> list["_TextContent"] | "_CallToolResult":
    raw = args.get("image_path")
    if not raw:
        return _error("image_path is required")
    safe = _safe_media_path(str(raw))
    if safe is None:
        return _error("image_path is not inside queue or sorted roots")
    meta = safe.with_suffix(safe.suffix + ".vision.json")
    if not meta.is_file():
        # Try the "<stem>.vision.json" convention alongside "<stem>.<ext>".
        alt = safe.with_name(safe.stem + ".vision.json")
        if alt.is_file():
            meta = alt
        else:
            return _error("no .vision.json sidecar for that image",
                          image_path=str(safe))
    try:
        payload = json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _error(f"could not read vision meta: {exc}")
    return _ok({
        "image_path": str(safe),
        "meta_path": str(meta),
        "meta": payload,
    })


# ── Tool: cull_export_kohya ────────────────────────────────────────────────

def tool_export_kohya(args: dict[str, Any]) -> list["_TextContent"] | "_CallToolResult":
    slug = _validated_slug(args.get("slug"))
    out_dir = str(args.get("out_dir") or "").strip()
    if not out_dir:
        return _error("out_dir is required")
    try:
        import export_profiles
    except Exception as exc:
        return _error(f"export_profiles unavailable: {exc}")
    try:
        summary = export_profiles.export_dataset(slug, "kohya", Path(out_dir))
    except (ValueError, OSError) as exc:
        return _error(f"export failed: {exc}", slug=slug)
    return _ok({
        "slug": slug,
        "profile": "kohya",
        "out_dir": summary.get("out_dir", out_dir),
        "sample_count": summary.get("sample_count", 0),
        "categories": summary.get("categories", []),
    })


# ── Tool: cull_export_hf ───────────────────────────────────────────────────

def tool_export_hf(args: dict[str, Any]) -> list["_TextContent"] | "_CallToolResult":
    slug = _validated_slug(args.get("slug"))
    repo = str(args.get("repo") or "").strip()
    if not repo or "/" not in repo:
        return _error("repo is required and must be 'namespace/name'")
    private = bool(args.get("private", True))
    try:
        import hf_export
    except Exception as exc:
        return _error(f"hf_export unavailable: {exc}")
    try:
        result = hf_export.push_to_hf(slug, repo, private=private)
    except Exception as exc:
        # ``push_to_hf`` raises MissingCredentialError (SystemExit subclass),
        # ValueError (nothing to push), RuntimeError (SDK missing) — all safe
        # to surface as a generic error message. Do NOT re-raise SystemExit —
        # that would tear the MCP server down mid-session.
        return _error(f"HF push failed: {exc}", slug=slug, repo=repo)
    repo_url = f"https://huggingface.co/datasets/{repo}"
    return _ok({
        "slug": slug,
        "repo": repo,
        "private": private,
        "uploaded": result.get("uploaded", 0),
        "categories": result.get("categories", []),
        "url": repo_url,
    })


# ── Tool registry ──────────────────────────────────────────────────────────

# name → (description, dispatch fn, JSON-schema input)
TOOLS: dict[str, tuple[str, Any, dict[str, Any]]] = {
    "cull_list_jobs": (
        "List every cull job with slug, subject, preset, active flag, and "
        "queue / sorted counts.",
        tool_list_jobs,
        {"type": "object", "properties": {}, "additionalProperties": False},
    ),
    "cull_get_job": (
        "Return one job's full effective config (preset + overrides).",
        tool_get_job,
        {
            "type": "object",
            "properties": {"slug": {"type": "string", "description": "Job slug."}},
            "required": ["slug"], "additionalProperties": False,
        },
    ),
    "cull_create_job": (
        "Create a new curation job. slug must be lowercase letters, digits, "
        "underscores. subject seeds topic.topic; preset defaults to the "
        "library default; base_on clones another job's overrides.",
        tool_create_job,
        {
            "type": "object",
            "properties": {
                "slug": {"type": "string"},
                "subject": {"type": "string"},
                "preset": {"type": "string"},
                "base_on": {"type": "string"},
            },
            "required": ["slug"], "additionalProperties": False,
        },
    ),
    "cull_delete_job": (
        "Delete a job. Refuses to delete the active job.",
        tool_delete_job,
        {
            "type": "object",
            "properties": {"slug": {"type": "string"}},
            "required": ["slug"], "additionalProperties": False,
        },
    ),
    "cull_activate_job": (
        "Mark a job active (projects env + categories). exclusive=true resets "
        "the active set to [slug]; default appends (multi-active).",
        tool_activate_job,
        {
            "type": "object",
            "properties": {
                "slug": {"type": "string"},
                "exclusive": {"type": "boolean", "default": False},
            },
            "required": ["slug"], "additionalProperties": False,
        },
    ),
    "cull_deactivate_job": (
        "Remove a job from the active set (idempotent).",
        tool_deactivate_job,
        {
            "type": "object",
            "properties": {"slug": {"type": "string"}},
            "required": ["slug"], "additionalProperties": False,
        },
    ),
    "cull_set_job_priority": (
        "Set a job's round-robin priority weight (1-10).",
        tool_set_job_priority,
        {
            "type": "object",
            "properties": {
                "slug": {"type": "string"},
                "priority": {"type": "integer", "minimum": 1, "maximum": 10},
            },
            "required": ["slug", "priority"], "additionalProperties": False,
        },
    ),
    "cull_list_presets": (
        "List all presets in the library (builtin + custom).",
        tool_list_presets,
        {"type": "object", "properties": {}, "additionalProperties": False},
    ),
    "cull_get_preset": (
        "Return a preset's full inheritable config.",
        tool_get_preset,
        {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"], "additionalProperties": False,
        },
    ),
    "cull_clone_preset": (
        "Clone a preset into a new custom name.",
        tool_clone_preset,
        {
            "type": "object",
            "properties": {
                "source_name": {"type": "string"},
                "new_name": {"type": "string"},
            },
            "required": ["source_name", "new_name"],
            "additionalProperties": False,
        },
    ),
    "cull_start_pipeline": (
        "Ask the local dashboard to start the supervisor (POST "
        "/api/pipeline/start). The dashboard owns the process handle.",
        tool_start_pipeline,
        {"type": "object", "properties": {}, "additionalProperties": False},
    ),
    "cull_stop_pipeline": (
        "Ask the local dashboard to stop the supervisor.",
        tool_stop_pipeline,
        {"type": "object", "properties": {}, "additionalProperties": False},
    ),
    "cull_pipeline_status": (
        "Return active-slug + queue counts. Includes pipeline_running / fleet "
        "health when the dashboard is up.",
        tool_pipeline_status,
        {"type": "object", "properties": {}, "additionalProperties": False},
    ),
    "cull_set_scoring": (
        "Override a job's scoring floors (min_ovr / min_rel) or require_prompt.",
        tool_set_scoring,
        {
            "type": "object",
            "properties": {
                "slug": {"type": "string"},
                "min_ovr": {"type": "integer", "minimum": 0, "maximum": 100},
                "min_rel": {"type": "integer", "minimum": 0, "maximum": 100},
                "require_prompt": {"type": "boolean"},
            },
            "required": ["slug"], "additionalProperties": False,
        },
    ),
    "cull_get_scoring": (
        "Return a job's effective scoring floors + require_prompt.",
        tool_get_scoring,
        {
            "type": "object",
            "properties": {"slug": {"type": "string"}},
            "required": ["slug"], "additionalProperties": False,
        },
    ),
    "cull_add_scraper_url": (
        "Append a URL to a scraper's per-job URL list. source=gallery_dl or "
        "yt_dlp. No-op if the URL is already listed.",
        tool_add_scraper_url,
        {
            "type": "object",
            "properties": {
                "slug": {"type": "string"},
                "source": {"type": "string",
                            "enum": ["gallery_dl", "gallery-dl",
                                     "yt_dlp", "yt-dlp"]},
                "url": {"type": "string"},
            },
            "required": ["slug", "source", "url"],
            "additionalProperties": False,
        },
    ),
    "cull_toggle_scraper": (
        "Enable or disable one scraper on a job. name must be one of "
        "X.com / Discord-1 / Civitai-Com / Civitai-Red / Web / Gallery-DL.",
        tool_toggle_scraper,
        {
            "type": "object",
            "properties": {
                "slug": {"type": "string"},
                "name": {"type": "string"},
                "enabled": {"type": "boolean"},
            },
            "required": ["slug", "name", "enabled"],
            "additionalProperties": False,
        },
    ),
    "cull_stats": (
        "Return counts by source x status, sorted-by-category, and an OVR "
        "score histogram. Scope to one job with slug, or omit for all jobs.",
        tool_stats,
        {
            "type": "object",
            "properties": {"slug": {"type": "string"}},
            "additionalProperties": False,
        },
    ),
    "cull_sample_gallery": (
        "Return the top N sorted images in a category with metadata; the top "
        "result also comes back as an ImageContent thumbnail so the agent can "
        "see it.",
        tool_sample_gallery,
        {
            "type": "object",
            "properties": {
                "slug": {"type": "string"},
                "category": {"type": "string", "default": "Keep"},
                "n": {"type": "integer", "minimum": 1, "maximum": 50,
                       "default": 10},
            },
            "required": ["slug"], "additionalProperties": False,
        },
    ),
    "cull_get_vision_meta": (
        "Return the .vision.json audit sidecar for an image path (rejects "
        "paths outside queue / sorted roots).",
        tool_get_vision_meta,
        {
            "type": "object",
            "properties": {"image_path": {"type": "string"}},
            "required": ["image_path"], "additionalProperties": False,
        },
    ),
    "cull_export_kohya": (
        "Export a job's KEPT samples in Kohya training-set layout. Writes to "
        "out_dir (created if absent).",
        tool_export_kohya,
        {
            "type": "object",
            "properties": {
                "slug": {"type": "string"},
                "out_dir": {"type": "string"},
            },
            "required": ["slug", "out_dir"], "additionalProperties": False,
        },
    ),
    "cull_export_hf": (
        "Push a job's KEPT samples to a HuggingFace dataset repo. Requires "
        "HF_TOKEN in the environment. Repo is namespace/name.",
        tool_export_hf,
        {
            "type": "object",
            "properties": {
                "slug": {"type": "string"},
                "repo": {"type": "string"},
                "private": {"type": "boolean", "default": True},
            },
            "required": ["slug", "repo"], "additionalProperties": False,
        },
    ),
}


def list_tool_names() -> list[str]:
    """Return every registered tool name — used by tests and introspection."""
    return sorted(TOOLS)


def dispatch(name: str, args: dict[str, Any]) -> Any:
    """Test-friendly synchronous dispatch. Returns whatever the tool returned;
    catches unexpected exceptions and returns a CallToolResult error envelope.
    """
    handler = TOOLS.get(name)
    if handler is None:
        return _error(f"unknown tool: {name}")
    _description, fn, _schema = handler
    try:
        return fn(args or {})
    except ValueError as exc:
        return _error(str(exc))
    except Exception as exc:  # pragma: no cover - unexpected
        logger.exception("tool %s failed", name)
        return _error(f"internal error running {name}: {exc}")


# ── Server wiring ──────────────────────────────────────────────────────────

def _build_server() -> Any:
    """Construct and wire the MCP server. Only called when the SDK is present."""
    if _MCPServer is None:
        raise RuntimeError("mcp SDK not installed")
    server = _MCPServer(SERVER_NAME)

    tools_defs = [
        _Tool(name=name, description=description, inputSchema=schema)
        for name, (description, _fn, schema) in TOOLS.items()
    ]

    @server.list_tools()
    async def _list_tools() -> list[Any]:
        return tools_defs

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any]) -> Any:
        result = dispatch(name, arguments or {})
        # ``dispatch`` returns either a list[ContentBlock] (happy path) or a
        # CallToolResult (error) — both are accepted by the SDK's call_tool
        # handler wrapper as-is.
        return result

    return server


async def _run_stdio() -> None:  # pragma: no cover - integration path
    server = _build_server()
    async with _stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main() -> int:
    """Entry point for the ``cull-mcp`` console script.

    Fails cleanly (stderr + non-zero exit) with an install hint when the
    ``mcp`` extra is not installed. Never writes to stdout — that is the MCP
    transport.
    """
    if _MCPServer is None or _stdio_server is None:
        sys.stderr.write(
            "cull-mcp requires the 'mcp' extra. Install with:\n"
            "    pip install 'cull[mcp]'\n"
            f"(underlying import error: {_MCP_IMPORT_ERROR})\n"
        )
        return 1
    try:
        asyncio.run(_run_stdio())
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        return 130
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
