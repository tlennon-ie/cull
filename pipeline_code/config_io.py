"""Portable export/import for presets and jobs — the shareable-file core.

A preset or a whole job can be shared as a single self-describing JSON file.
This module is the reusable CORE behind the dashboard's existing export/import
endpoints (it does NOT replace them — they delegate here so the envelope format
and validation live in exactly one place).

Envelopes
---------
Preset::

    {"cull_preset_version": 1, "kind": "cull.preset",
     "name": "film_photography", "preset": { ...inheritable cfg... }}

Job (carries the topic line + which preset it builds on + sparse overrides)::

    {"cull_preset_version": 1, "kind": "cull.job",
     "job": {"slug", "name", "subject", "preset", "overrides": {...}}}

Validation
----------
``import_preset`` / ``import_job`` are strict at the system boundary: unknown
top-level keys, unknown config blocks, oversized payloads and out-of-range
values all raise :class:`ValidationError` with a human-readable message. The
inheritable-config validator here mirrors the dashboard's contract (same block
shapes, same caps) but stands alone — it imports only ``job_config`` + stdlib,
so a CLI or test can reuse it without Flask.

Forward tolerance
-----------------
Older envelopes migrate forward on read: the dashboard's ``commit 9fcf2ca``
shape (``{"version": 1, "name", "cfg"}``) and the oldest bare ``{"name", "cfg"}``
are both accepted and normalised. ``cull_preset_version`` is the canonical
field; an envelope from a *newer* cull (a higher version we don't understand) is
rejected rather than silently mis-read.
"""
from __future__ import annotations

import json
import re
from typing import Any

import job_config

__all__ = [
    "PRESET_VERSION",
    "MAX_ENVELOPE_BYTES",
    "ValidationError",
    "export_preset",
    "export_preset_bytes",
    "import_preset",
    "export_job",
    "export_job_bytes",
    "import_job",
    "validate_inheritable_cfg",
]

# Bump only on a breaking envelope/shape change. Readers accept this and lower.
PRESET_VERSION = 1

# Hard ceiling on an imported envelope (serialised JSON). A shared preset/job is
# a few KB of text; anything past this is hostile or corrupt, so we refuse to
# parse-then-store it. Comfortably above a fat real preset, well below abuse.
MAX_ENVELOPE_BYTES = 512 * 1024

_KIND_PRESET = "cull.preset"
_KIND_JOB = "cull.job"

# Inheritable-config namespace (a preset body, or a job's sparse overrides). Both
# flow through the SAME validators so a bad value can never reach projection.
_INHERITABLE_KEYS: frozenset[str] = frozenset({
    "topic_filters", "scrapers", "categories", "category_rules", "scoring",
    "captioning", "vision",
})
_CAPTION_STYLES: frozenset[str] = frozenset(
    {"sd_prompt", "booru_tags", "natural_language"})
_VISION_PROVIDERS: frozenset[str] = frozenset(job_config.VISION_PROVIDERS)

# Caps mirror the dashboard validators (single contract, two call sites).
_MAX_CATEGORIES = 40
_MAX_VISION_WORKERS = 64
_MAX_LOCAL_FOLDERS = 32
_MAX_HINT = 2000
_MAX_RULES = 8000
_MAX_NOTES = 2000
_MAX_DISCORD_JSON = 20000
_MAX_PROMPT_LEN = 10000
_MAX_GALLERY_LIMIT = 5000

# A category name drives the strict JSON-schema enum; keep it a safe identifier.
_CAT_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,31}$")
_RESERVED_CATEGORIES: frozenset[str] = frozenset({"DISCARD", "CORRUPT"})
# A local-import folder ``name`` becomes a filesystem path component downstream.
_LOCAL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


class ValidationError(ValueError):
    """Raised when an envelope or its config is malformed/oversized/unknown."""


# ── small guards ─────────────────────────────────────────────────────────────

def _bad_path_str(v: Any) -> bool:
    """Refuse a stored path-ish string with traversal / NUL / newlines. The dir
    itself may legitimately be absolute; we only keep obviously-hostile values
    out of the config file (the supervisor resolves it per-agent later)."""
    return (
        not isinstance(v, str)
        or ".." in v
        or "\x00" in v
        or "\n" in v
        or "\r" in v
    )


def _require_object(value: Any, what: str) -> dict:
    if not isinstance(value, dict):
        raise ValidationError(f"{what} must be an object")
    return value


# ── inheritable-config block validators (standalone port of the dashboard) ────

def _validate_categories(cats: Any, *, required: bool) -> list[dict]:
    if cats is None and not required:
        return []
    if not isinstance(cats, list) or (required and not cats):
        raise ValidationError("categories must be a non-empty list")
    if len(cats) > _MAX_CATEGORIES:
        raise ValidationError(f"too many categories (max {_MAX_CATEGORIES})")
    seen: set[str] = set()
    cleaned: list[dict] = []
    for entry in cats:
        if not isinstance(entry, dict):
            raise ValidationError("each category must be an object with name + hint")
        name = (entry.get("name") or "").strip()
        hint = (entry.get("hint") or "").strip()
        if not _CAT_NAME_RE.match(name):
            raise ValidationError(
                f"invalid category name {name!r} (letters/digits/_ only, "
                "must start with a letter, max 32 chars)")
        if name in _RESERVED_CATEGORIES:
            raise ValidationError(f"{name!r} is reserved for the system")
        if len(hint) > _MAX_HINT:
            raise ValidationError(f"hint for {name!r} too long (max {_MAX_HINT} chars)")
        if name in seen:
            raise ValidationError(f"duplicate category name {name!r}")
        seen.add(name)
        cleaned.append({"name": name, "hint": hint})
    return cleaned


def _validate_topic_filters(tf: Any) -> dict:
    _require_object(tf, "topic_filters")
    allowed = {"keywords_extra", "banned_keywords", "generation_hints",
               "min_prompt_length", "require_prompt"}
    out: dict[str, Any] = {}
    for k, v in tf.items():
        if k not in allowed:
            raise ValidationError(f"unknown topic_filters key: {k!r}")
        if k in ("keywords_extra", "banned_keywords", "generation_hints"):
            if not isinstance(v, list) or any(not isinstance(x, str) for x in v):
                raise ValidationError(f"topic_filters.{k} must be a list of strings")
            out[k] = list(v)
        elif k == "min_prompt_length":
            iv = _as_int(v, "topic_filters.min_prompt_length")
            if iv < 0 or iv > _MAX_PROMPT_LEN:
                raise ValidationError(
                    f"topic_filters.min_prompt_length out of range (0-{_MAX_PROMPT_LEN})")
            out[k] = iv
        else:  # require_prompt
            out[k] = bool(v)
    return out


def _validate_scoring(raw: Any) -> dict:
    _require_object(raw, "scoring")
    out: dict[str, Any] = {}
    for key in ("ovr_min", "rel_min"):
        if key not in raw:
            continue
        val = _as_int(raw[key], f"scoring.{key}")
        if val < 0 or val > 100:
            raise ValidationError(f"scoring.{key} must be an integer 0-100")
        out[key] = val
    if "notes" in raw:
        if not isinstance(raw["notes"], str):
            raise ValidationError("scoring.notes must be a string")
        if len(raw["notes"]) > _MAX_NOTES:
            raise ValidationError(f"scoring.notes too long (max {_MAX_NOTES} chars)")
        out["notes"] = raw["notes"]
    return out


def _validate_captioning(cap: Any) -> dict:
    _require_object(cap, "captioning")
    allowed = {"enabled", "style", "overwrite"}
    out: dict[str, Any] = {}
    for k, v in cap.items():
        if k not in allowed:
            raise ValidationError(f"unknown captioning key: {k!r}")
        if k == "style":
            if v not in _CAPTION_STYLES:
                raise ValidationError(
                    f"captioning.style must be one of {sorted(_CAPTION_STYLES)}")
            out[k] = v
        else:
            out[k] = bool(v)
    return out


def _validate_gallery_dl(gd: Any) -> dict:
    _require_object(gd, "scrapers.gallery_dl")
    allowed = {"enabled", "urls", "limit_per_url", "cookies_file", "config_path"}
    out: dict[str, Any] = {}
    for k, v in gd.items():
        if k not in allowed:
            raise ValidationError(f"unknown gallery_dl key: {k!r}")
        if k == "enabled":
            out[k] = bool(v)
        elif k == "urls":
            if not isinstance(v, list) or any(not isinstance(x, str) for x in v):
                raise ValidationError("gallery_dl.urls must be a list of strings")
            if any("\x00" in x for x in v):
                raise ValidationError("gallery_dl.urls contains a NUL byte")
            out[k] = list(v)
        elif k == "limit_per_url":
            iv = _as_int(v, "gallery_dl.limit_per_url")
            out[k] = max(1, min(_MAX_GALLERY_LIMIT, iv))
        else:  # cookies_file / config_path
            if _bad_path_str(v):
                raise ValidationError(f"gallery_dl.{k} is not a valid path string")
            out[k] = v
    return out


def _validate_local_imports(li: Any) -> list[dict]:
    if not isinstance(li, list):
        raise ValidationError("scrapers.local_imports must be a list")
    if len(li) > _MAX_LOCAL_FOLDERS:
        raise ValidationError(f"too many local folders (max {_MAX_LOCAL_FOLDERS})")
    out: list[dict] = []
    for f in li:
        if not isinstance(f, dict):
            raise ValidationError("each local folder must be an object")
        name = f.get("name", "local")
        if not isinstance(name, str) or not _LOCAL_NAME_RE.match(name):
            raise ValidationError("local folder name must match [A-Za-z0-9_-] (1-40 chars)")
        if _bad_path_str(f.get("dir", "")):
            raise ValidationError("local folder dir is not a valid path string")
        mf = f.get("migrate_from", "")
        if not isinstance(mf, str):
            raise ValidationError("local folder migrate_from must be a string")
        out.append({
            "name": name, "dir": f.get("dir", "") or "",
            "enabled": bool(f.get("enabled", False)), "migrate_from": mf,
        })
    return out


def _validate_scrapers(s: Any) -> dict:
    _require_object(s, "scrapers")
    allowed = {"enabled", "x_accounts", "reddit_subreddits", "discord_channels_json",
               "civitai_domains", "gallery_dl", "local_imports"}
    out: dict[str, Any] = {}
    for k, v in s.items():
        if k not in allowed:
            raise ValidationError(f"unknown scrapers key: {k!r}")
        if k == "enabled":
            if not isinstance(v, dict) or any(not isinstance(b, bool) for b in v.values()):
                raise ValidationError("scrapers.enabled must be a name->bool map")
            out[k] = {n: bool(v[n]) for n in job_config.SCRAPER_NAMES if n in v}
        elif k in ("x_accounts", "reddit_subreddits", "civitai_domains"):
            if not isinstance(v, list) or any(not isinstance(x, str) for x in v):
                raise ValidationError(f"scrapers.{k} must be a list of strings")
            out[k] = list(v)
        elif k == "discord_channels_json":
            if not isinstance(v, str) or len(v) > _MAX_DISCORD_JSON:
                raise ValidationError(
                    f"scrapers.discord_channels_json must be a string (max {_MAX_DISCORD_JSON})")
            out[k] = v
        elif k == "gallery_dl":
            out[k] = _validate_gallery_dl(v)
        else:  # local_imports
            out[k] = _validate_local_imports(v)
    return out


def _validate_vision(v: Any) -> dict:
    _require_object(v, "vision")
    out: dict[str, Any] = {}
    for k, val in v.items():
        if k != "workers":
            raise ValidationError(f"unknown vision key: {k!r}")
        if not isinstance(val, list):
            raise ValidationError("vision.workers must be a list")
        if len(val) > _MAX_VISION_WORKERS:
            raise ValidationError(f"too many vision workers (max {_MAX_VISION_WORKERS})")
        workers: list[dict] = []
        seen: set[str] = set()
        for i, w in enumerate(val):
            if not isinstance(w, dict):
                raise ValidationError("each vision worker must be an object")
            provider = str(w.get("provider", "") or "").strip().lower()
            if provider not in _VISION_PROVIDERS:
                raise ValidationError(
                    f"vision worker provider must be one of {sorted(_VISION_PROVIDERS)}")
            base_url = str(w.get("base_url", "") or "").strip()
            if base_url and not re.match(r"^https?://", base_url, re.I):
                raise ValidationError("vision worker base_url must be an http(s) URL")
            wid = (str(w.get("id", "") or "").strip() or f"w{i}")[:64]
            if wid in seen:
                raise ValidationError(f"duplicate vision worker id: {wid!r}")
            seen.add(wid)
            workers.append({
                "id": wid,
                "name": str(w.get("name", "") or "").strip()[:80],
                "provider": provider,
                "base_url": base_url,
                "model": str(w.get("model", "") or "").strip()[:200],
                "api_key": str(w.get("api_key", "") or "").strip()[:400],
                "enabled": bool(w.get("enabled", True)),
            })
        out["workers"] = workers
    return out


def _as_int(v: Any, what: str) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        raise ValidationError(f"{what} must be an integer") from None


def validate_inheritable_cfg(cfg: Any, *, partial: bool) -> dict:
    """Validate a preset body (``partial=False``) or a job override map
    (``partial=True``); return a cleaned copy. Unknown top-level keys and bad
    block shapes raise :class:`ValidationError`. A full preset must carry
    categories (they drive the schema enum)."""
    _require_object(cfg, "config")
    out: dict[str, Any] = {}
    for key, value in cfg.items():
        if key not in _INHERITABLE_KEYS:
            raise ValidationError(f"unknown config key: {key!r}")
        if key == "categories":
            out["categories"] = _validate_categories(value, required=not partial)
        elif key == "category_rules":
            if not isinstance(value, str):
                raise ValidationError("category_rules must be a string")
            if len(value) > _MAX_RULES:
                raise ValidationError(f"category_rules too long (max {_MAX_RULES} chars)")
            out["category_rules"] = value
        elif key == "topic_filters":
            out["topic_filters"] = _validate_topic_filters(value)
        elif key == "scrapers":
            out["scrapers"] = _validate_scrapers(value)
        elif key == "scoring":
            out["scoring"] = _validate_scoring(value)
        elif key == "captioning":
            out["captioning"] = _validate_captioning(value)
        else:  # vision
            out["vision"] = _validate_vision(value)
    if not partial and "categories" not in out:
        raise ValidationError("categories is required for a preset")
    return out


# ── envelope load / size / version ───────────────────────────────────────────

def _coerce_to_dict(data: Any) -> dict:
    """Accept a dict, a JSON string, or raw bytes; enforce the size cap on text
    input before parsing so a giant blob can't be deserialised."""
    if isinstance(data, dict):
        _enforce_size(json.dumps(data, ensure_ascii=False))
        return data
    if isinstance(data, (bytes, bytearray)):
        try:
            text = bytes(data).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValidationError(f"payload is not valid UTF-8: {exc}") from None
    elif isinstance(data, str):
        text = data
    else:
        raise ValidationError("payload must be a dict, str, or bytes")
    _enforce_size(text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"payload is not valid JSON: {exc}") from None
    if not isinstance(parsed, dict):
        raise ValidationError("payload must be a JSON object")
    return parsed


def _enforce_size(text: str) -> None:
    size = len(text.encode("utf-8"))
    if size > MAX_ENVELOPE_BYTES:
        raise ValidationError(
            f"payload too large ({size} bytes; max {MAX_ENVELOPE_BYTES})")


def _check_version(env: dict) -> None:
    """Tolerate older envelopes; reject ones from a newer cull. Accepts the
    canonical ``cull_preset_version`` or the legacy ``version`` field; a missing
    version (oldest bare shape) is treated as v1."""
    raw = env.get("cull_preset_version", env.get("version", PRESET_VERSION))
    try:
        version = int(raw)
    except (TypeError, ValueError):
        raise ValidationError(f"invalid version field: {raw!r}") from None
    if version > PRESET_VERSION:
        raise ValidationError(
            f"envelope version {version} is newer than supported "
            f"({PRESET_VERSION}); upgrade cull to import this file")


def _require_name(env: dict, *, key: str = "name") -> str:
    name = (env.get(key) or "").strip()
    if not job_config.PRESET_NAME_RE.match(name):
        raise ValidationError(
            f"invalid preset name {name!r} (letters/digits/space/_/-, max 40)")
    return name


# ── preset export / import ───────────────────────────────────────────────────

def export_preset(name: str) -> dict:
    """Return a portable, versioned envelope for an existing preset. Raises
    :class:`ValidationError` if the preset does not exist."""
    if name not in job_config.list_presets().get("presets", {}):
        raise ValidationError(f"preset not found: {name!r}")
    return {
        "cull_preset_version": PRESET_VERSION,
        "kind": _KIND_PRESET,
        "name": name,
        "preset": job_config.get_preset(name),
    }


def export_preset_bytes(name: str) -> bytes:
    """``export_preset`` serialised to UTF-8 JSON bytes (ready to write/download)."""
    return json.dumps(export_preset(name), indent=2, ensure_ascii=False).encode("utf-8")


def import_preset(data: Any) -> tuple[str, dict]:
    """Validate an imported preset envelope (dict / JSON str / bytes) and return
    ``(name, cfg)`` where ``cfg`` is a cleaned, projection-ready preset body.

    Tolerant of older envelopes: the dashboard's ``{"version", "name", "cfg"}``
    and the oldest bare ``{"name", "cfg"}`` are normalised. Rejects unknown
    top-level keys, oversized payloads, wrong/newer versions, bad names, and any
    malformed config block with a clear message. Caller decides whether to
    overwrite an existing preset (this does not write)."""
    env = _coerce_to_dict(data)
    _check_version(env)
    body_key = "preset" if "preset" in env else "cfg"
    allowed_top = {"cull_preset_version", "version", "kind", "name", body_key}
    extra = set(env) - allowed_top
    if extra:
        raise ValidationError(f"unknown top-level key(s): {sorted(extra)}")
    kind = env.get("kind")
    if kind not in (None, _KIND_PRESET):
        raise ValidationError(f"not a preset export (kind={kind!r})")
    name = _require_name(env)
    cfg = validate_inheritable_cfg(env.get(body_key), partial=False)
    return name, cfg


# ── job export / import ──────────────────────────────────────────────────────

def export_job(slug: str) -> dict:
    """Return a portable envelope for an existing job — carries its subject, the
    preset it builds on, and its sparse overrides (NOT the resolved effective
    config, so the file stays a thin, shareable diff). Raises
    :class:`ValidationError` if the job does not exist."""
    job = job_config.get_job(slug)
    if job is None:
        raise ValidationError(f"job not found: {slug!r}")
    return {
        "cull_preset_version": PRESET_VERSION,
        "kind": _KIND_JOB,
        "job": {
            "slug": job.slug,
            "name": job.name,
            "subject": job.subject,
            "preset": job.preset,
            "overrides": job.overrides or {},
        },
    }


def export_job_bytes(slug: str) -> bytes:
    """``export_job`` serialised to UTF-8 JSON bytes."""
    return json.dumps(export_job(slug), indent=2, ensure_ascii=False).encode("utf-8")


def import_job(data: Any) -> "job_config.Job":
    """Validate an imported job envelope (dict / JSON str / bytes) and return a
    :class:`job_config.Job` (status reset to ``idle``; ``overrides`` validated
    like the dashboard PUT path). Does not persist — caller calls ``save_job``.

    Tolerant of older/bare job shapes (a top-level job dict, or a ``{"version"}``
    envelope). Rejects unknown top-level keys, a bad slug, oversized payloads,
    newer versions, and malformed overrides."""
    env = _coerce_to_dict(data)
    _check_version(env)
    if "job" in env:
        allowed_top = {"cull_preset_version", "version", "kind", "job"}
        extra = set(env) - allowed_top
        if extra:
            raise ValidationError(f"unknown top-level key(s): {sorted(extra)}")
        kind = env.get("kind")
        if kind not in (None, _KIND_JOB):
            raise ValidationError(f"not a job export (kind={kind!r})")
        jd = env["job"]
    else:
        # Bare job dict (e.g. an entry copied out of a cull.config backup).
        jd = env
    _require_object(jd, "job")
    slug = (jd.get("slug") or "").strip()
    if not job_config.JOB_SLUG_RE.match(slug):
        raise ValidationError(
            f"invalid job slug {slug!r} (lowercase letters/digits/_ only)")
    overrides = jd.get("overrides") or {}
    cleaned_overrides = validate_inheritable_cfg(overrides, partial=True)
    name = (jd.get("name") or "").strip() or slug.replace("_", " ").title()
    subject = jd.get("subject")
    if subject is not None and not isinstance(subject, str):
        raise ValidationError("job subject must be a string")
    preset = (jd.get("preset") or "").strip() or job_config.default_preset_name()
    if not job_config.PRESET_NAME_RE.match(preset):
        raise ValidationError(f"invalid preset reference {preset!r}")
    return job_config.Job.from_dict({
        "slug": slug,
        "name": name,
        "status": "idle",
        "created_at": jd.get("created_at", "") or "",
        "updated_at": jd.get("updated_at", "") or "",
        "subject": subject if subject is not None else name,
        "preset": preset,
        "overrides": cleaned_overrides,
    })
