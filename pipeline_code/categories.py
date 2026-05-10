"""User-configurable classification taxonomy.

Source of truth for which buckets the vision worker can classify into. Edit
via the dashboard's Settings -> Categories panel, or write JSON directly to
``data/cull_categories.json``. The shipped fallback (loaded automatically
when no user file exists) is the ``portrait_curation`` preset in
``cull_categories_default.json`` — six buckets tuned for influencer / model
photography, identical to cull's pre-customisation defaults.

Two distinctions:
  CATEGORIES        - keep-bucket folders that vision workers create on-write.
  ALL_CATEGORIES    - includes terminal buckets (DISCARD, CORRUPT) that the
                      supervisor + storage layer also need to know about.
  SCHEMA_CATEGORIES - the enum the JSON-schema exposes to the model. Includes
                      DISCARD; excludes CORRUPT (only the supervisor writes
                      that bucket as a safety net for unreadable images).

Dynamic readers:
  get_active(), get_presets(), set_active(), get_categories(),
  get_category_hints(), get_global_rules(), get_schema_categories().
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

# ── Paths ────────────────────────────────────────────────────────────────────
# User state lives under the data/ root (gitignored). Default presets ship
# with the codebase. Both paths are overridable via env so the dashboard's
# safe_inside() guard can pin them.

_BASE_DIR_ENV = os.environ.get("PIPELINE_BASE_DIR")
_DEFAULT_BASE = Path(__file__).resolve().parent.parent / "data"
ACTIVE_PATH: Path = Path(_BASE_DIR_ENV or _DEFAULT_BASE) / "cull_categories.json"
DEFAULT_PRESETS_PATH: Path = Path(__file__).resolve().parent / "cull_categories_default.json"

# Always-present terminal buckets — system-required, not user-editable.
SYSTEM_TERMINAL: tuple[str, ...] = ("DISCARD", "CORRUPT")

_lock = threading.Lock()
_cache: dict[str, Any] | None = None
_cache_mtime: float = 0.0


# ── Loaders ──────────────────────────────────────────────────────────────────

def _load_default() -> dict[str, Any]:
    """Load the shipped default presets file."""
    return json.loads(DEFAULT_PRESETS_PATH.read_text(encoding="utf-8"))


def _default_active() -> dict[str, Any]:
    """The active payload to use when no user file exists."""
    defaults = _load_default()
    preset_name = defaults.get("default") or next(iter(defaults["presets"]))
    preset = defaults["presets"][preset_name]
    return {
        "preset": preset_name,
        "categories": list(preset["categories"]),
        "global_rules": preset.get("global_rules", ""),
    }


def _read_active_file() -> dict[str, Any] | None:
    """Read the user's active-categories file. Returns None on any error so
    callers fall back to the shipped default rather than crashing."""
    if not ACTIVE_PATH.exists():
        return None
    try:
        payload = json.loads(ACTIVE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    # Minimal sanity: must have a categories list.
    if not isinstance(payload, dict) or not isinstance(payload.get("categories"), list):
        return None
    payload.setdefault("preset", "custom")
    payload.setdefault("global_rules", "")
    return payload


def _refresh_cache() -> None:
    """Recompute _cache from disk if the active file has changed since last read."""
    global _cache, _cache_mtime
    try:
        mtime = ACTIVE_PATH.stat().st_mtime
    except OSError:
        mtime = 0.0
    if _cache is None or mtime != _cache_mtime:
        _cache = _read_active_file() or _default_active()
        _cache_mtime = mtime


# ── Public API ───────────────────────────────────────────────────────────────

def get_active() -> dict[str, Any]:
    """Return the user's active taxonomy. Cached, refreshed on file mtime change."""
    with _lock:
        _refresh_cache()
        # Return a defensive copy so callers can't mutate the cache.
        return json.loads(json.dumps(_cache))


def get_presets() -> dict[str, Any]:
    """Return the shipped preset library."""
    return _load_default()


def set_active(payload: dict[str, Any]) -> None:
    """Persist a new active taxonomy to disk and bust the cache.

    Caller is responsible for validation; this just serialises.
    """
    global _cache, _cache_mtime
    ACTIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    ACTIVE_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with _lock:
        _cache = json.loads(json.dumps(payload))
        try:
            _cache_mtime = ACTIVE_PATH.stat().st_mtime
        except OSError:
            _cache_mtime = 0.0


def get_categories() -> tuple[str, ...]:
    """Just the keep-bucket category names."""
    return tuple(c["name"] for c in get_active()["categories"])


def get_terminal_categories() -> tuple[str, ...]:
    return SYSTEM_TERMINAL


def get_all_categories() -> tuple[str, ...]:
    return get_categories() + SYSTEM_TERMINAL


def get_schema_categories() -> tuple[str, ...]:
    """Categories the vision model is allowed to return — keep buckets + DISCARD.
    Excludes CORRUPT (only the supervisor writes that bucket)."""
    return get_categories() + ("DISCARD",)


def get_category_hints() -> list[tuple[str, str]]:
    """[(name, hint), ...] for prompt building."""
    return [(c["name"], c.get("hint", "")) for c in get_active()["categories"]]


def get_global_rules() -> str:
    """The freeform STRICT JUDGEMENT RULES block injected into the prompt."""
    return get_active().get("global_rules", "")


def category_pipe_string() -> str:
    """Pipe-separated category list for embedding in the model prompt text."""
    return "|".join(get_schema_categories())


# ── Backward-compat module-level constants ───────────────────────────────────
# Older code reads ``from categories import CATEGORIES``. We expose them as
# module attributes so existing imports keep working at process startup; the
# values reflect the active taxonomy at *import time*. Workers respawn on
# taxonomy change (via the supervisor's mtime watch) so their copies stay
# fresh. Long-lived processes like the dashboard should call get_*() instead.
CATEGORIES: tuple[str, ...] = get_categories()
TERMINAL_CATEGORIES: tuple[str, ...] = SYSTEM_TERMINAL
ALL_CATEGORIES: tuple[str, ...] = get_all_categories()
SCHEMA_CATEGORIES: tuple[str, ...] = get_schema_categories()


__all__ = [
    "ACTIVE_PATH",
    "DEFAULT_PRESETS_PATH",
    "SYSTEM_TERMINAL",
    "CATEGORIES",
    "TERMINAL_CATEGORIES",
    "ALL_CATEGORIES",
    "SCHEMA_CATEGORIES",
    "get_active",
    "get_presets",
    "set_active",
    "get_categories",
    "get_terminal_categories",
    "get_all_categories",
    "get_schema_categories",
    "get_category_hints",
    "get_global_rules",
    "category_pipe_string",
]
