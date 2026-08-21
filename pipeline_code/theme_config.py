"""Theme registry — builtin, community, and user custom colour schemes.

Mirrors the presets pattern (see ``builtin_presets.py`` + the ``/api/presets``
routes in ``dashboard_enhanced.py``) so operators can pick a shipped theme,
customise it in the dashboard, save it locally under ``data/themes/``, and
publish it to the ``themes/community/`` folder via the same git-add / commit /
push path the preset marketplace uses.

Design notes
------------
- A *theme* is a small JSON envelope: ``{name, font_family, vars}``. ``vars`` is
  a flat dict mapping CSS custom-property names (``--color-bg`` etc.) to strings.
- Three sources, precedence user > community > builtin:
    * builtin  — ``themes/builtin/<slug>.theme.json`` (checked in with cull)
    * community — ``themes/community/<slug>.theme.json`` (contributed via
      publish; ships as JSON only, no thumbnails hard-required)
    * user     — ``data/themes/<slug>.json`` (per-install, gitignored)
- Slugs match ``^[a-z0-9_-]{1,40}$`` (a superset of ``JOB_SLUG_RE`` because we
  allow hyphens for theme names like ``ai-slop``).
- CSS variable keys are constrained to :data:`THEME_VAR_KEYS`; unknown keys are
  dropped on write. Values are capped to :data:`_MAX_VALUE_LEN` and rejected if
  they contain any character that could break out of a ``style`` attribute
  (``<``, ``>``, ``;``, backticks, ``url(``, ``javascript:``, etc.).
- The dashboard reads themes at request time (no cache) — small (<3 KB) JSON,
  so simplicity beats cleverness.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import paths as _paths


# ── Constants ────────────────────────────────────────────────────────────────

# Slug validation. Superset of JOB_SLUG_RE because ``ai-slop`` and similar
# hyphenated names are natural for theme identifiers.
THEME_NAME_RE = re.compile(r"^[a-z0-9_-]{1,40}$")

# The complete set of CSS custom properties a theme MAY set. Any key not in
# this tuple is silently dropped on write — this is the schema. Keep in
# lockstep with the ``:root`` block in ``dashboard_enhanced.py`` (T3 #22
# theme system) and with the colour picker list in the editor UI.
THEME_VAR_KEYS: tuple[str, ...] = (
    "--color-bg",
    "--color-bg-elev",
    "--color-fg",
    "--color-fg-muted",
    "--color-surface",
    "--color-surface-alt",
    "--color-border",
    "--color-border-strong",
    "--color-accent",
    "--color-accent-hover",
    "--color-accent-fg",
    "--color-success",
    "--color-success-fg",
    "--color-warn",
    "--color-warn-fg",
    "--color-danger",
    "--color-danger-fg",
    "--color-input-bg",
    "--color-input-fg",
    "--color-input-placeholder",
    "--color-pill-bg",
    "--color-pill-fg",
)

# The three shipped themes that live in the CSS itself, not as JSON. They are
# listed here so the dashboard can surface them as read-only "core" cards in
# the theme picker; users can clone them but cannot delete or overwrite.
CORE_THEMES: tuple[str, ...] = ("dark", "light", "hc")

# Font family is a free-form string, but we cap it and forbid characters that
# would allow CSS injection.
_MAX_VALUE_LEN = 128
_MAX_FONT_LEN = 200
_FONT_FAMILY_RE = re.compile(r"^[A-Za-z0-9 ,\"'\-_.()/]{0,200}$")

# CSS value guard — a colour value is expected to be a hex code, rgb/rgba,
# hsl/hsla, oklch(), or a keyword. We accept anything short of characters
# that could terminate an attribute / begin a script.
_UNSAFE_CSS_VALUE_RE = re.compile(r"[<>;`]|url\s*\(|javascript:|expression\s*\(", re.IGNORECASE)


# ── Paths ────────────────────────────────────────────────────────────────────

def builtin_themes_dir() -> Path:
    """``<repo>/themes/builtin/`` — shipped starter themes.

    Never written to at runtime. Resolved off the workspace root that owns
    ``paths.base_dir()``'s parent by default.
    """
    return _repo_root() / "themes" / "builtin"


def community_themes_dir() -> Path:
    """``<repo>/themes/community/`` — user-contributed themes, publish target."""
    return _repo_root() / "themes" / "community"


def community_thumbnail_dir() -> Path:
    """``<repo>/themes/thumbnails/`` — optional preview images per theme."""
    return _repo_root() / "themes" / "thumbnails"


def user_themes_dir() -> Path:
    """``<data>/themes/`` — user-created themes (per install, not versioned)."""
    return _paths.base_dir() / "themes"


def _repo_root() -> Path:
    """Best-effort workspace root discovery.

    Uses ``WORKSPACE_ROOT`` env var when set (the dashboard exports it), else
    walks up from this file. Kept private because callers should reach for the
    concrete ``builtin_themes_dir`` / ``community_themes_dir`` helpers above.
    """
    raw = os.environ.get("WORKSPACE_ROOT", "").strip()
    if raw:
        return Path(raw).resolve()
    return Path(__file__).resolve().parent.parent


# ── Slug / validation ────────────────────────────────────────────────────────

def slugify(name: str) -> str:
    """Same shape as :func:`job_config.slugify` but preserves hyphens.

    Theme names like ``ai-slop`` should round-trip; the presets slugify would
    turn that into ``ai_slop`` and we lose identity across sessions. Everything
    else becomes ``_``.
    """
    lowered = (name or "").lower().strip()
    return re.sub(r"[^a-z0-9_-]+", "_", lowered).strip("_-")


def is_valid_name(name: str) -> bool:
    return bool(name and THEME_NAME_RE.match(name))


def _is_safe_value(value: str) -> bool:
    if not isinstance(value, str):
        return False
    if len(value) > _MAX_VALUE_LEN:
        return False
    return not _UNSAFE_CSS_VALUE_RE.search(value)


def _is_safe_font(value: str) -> bool:
    if not isinstance(value, str):
        return False
    if len(value) > _MAX_FONT_LEN:
        return False
    return bool(_FONT_FAMILY_RE.match(value))


# ── Envelope shape ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Theme:
    """A validated theme envelope. Immutable — use :meth:`with_updates`."""

    name: str
    font_family: str
    vars: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "font_family": self.font_family,
            # copy so callers can't mutate the frozen instance's dict.
            "vars": dict(self.vars),
        }


def _sanitize_theme(name: str, payload: Any) -> Theme:
    """Validate + coerce a raw dict into a :class:`Theme`.

    Unknown ``vars`` keys are dropped. Bad values (too long / injection
    characters) are dropped. Font family failing the regex falls back to the
    empty string ("theme does not set a font"). Raises :class:`ValueError`
    only when the shape is unrecoverable (not a dict; empty name).
    """
    if not is_valid_name(name):
        raise ValueError(f"invalid theme name: {name!r}")
    if not isinstance(payload, dict):
        raise ValueError("theme payload must be a dict")
    raw_vars = payload.get("vars") or {}
    if not isinstance(raw_vars, dict):
        raw_vars = {}
    clean_vars: dict[str, str] = {}
    for key in THEME_VAR_KEYS:
        val = raw_vars.get(key)
        if val is None:
            continue
        if isinstance(val, (int, float)):
            val = str(val)
        if not isinstance(val, str):
            continue
        val = val.strip()
        if not val:
            continue
        if _is_safe_value(val):
            clean_vars[key] = val
    font = payload.get("font_family") or ""
    if not isinstance(font, str) or not _is_safe_font(font.strip()):
        font = ""
    return Theme(name=name, font_family=font.strip(), vars=clean_vars)


# ── Public API ───────────────────────────────────────────────────────────────

def list_themes() -> list[dict[str, Any]]:
    """Return every theme visible to the dashboard.

    Merges builtin + community + user with the precedence rule described in
    the module docstring: if the same slug appears in multiple layers the
    user copy wins. Each entry has ``{name, source, path, font_family, vars}``.
    Never fails — a missing / unreadable directory contributes nothing.
    """
    merged: dict[str, dict[str, Any]] = {}
    # Walk lowest precedence first so higher-precedence layers overwrite.
    for source, folder in (
        ("builtin", builtin_themes_dir()),
        ("community", community_themes_dir()),
        ("user", user_themes_dir()),
    ):
        for name, payload, path in _iter_theme_dir(folder):
            try:
                theme = _sanitize_theme(name, payload)
            except ValueError:
                continue
            merged[name] = {
                "name": theme.name,
                "source": source,
                "path": str(path),
                "font_family": theme.font_family,
                "vars": theme.vars,
            }
    # Stable ordering: builtin cores first (in a fixed order), then everything
    # else alphabetically. The dashboard displays them left-to-right; keeping
    # the anchor themes at the front reduces card thrash.
    def _sort_key(item: dict[str, Any]) -> tuple[int, str]:
        name = item["name"]
        # Preserve the CORE_THEMES ordering for the canonical trio when they
        # ship as JSON. All other names sort alphabetically.
        for rank, core in enumerate(CORE_THEMES):
            if name == core:
                return (0, f"{rank:02d}")
        return (1, name)
    return sorted(merged.values(), key=_sort_key)


def read_theme(name: str) -> dict[str, Any] | None:
    """Return the effective theme for ``name`` (user > community > builtin).

    ``None`` when nothing is found. Layer chosen is reflected in the returned
    ``source`` field so the UI can label "Custom" vs "Built-in".
    """
    if not is_valid_name(name):
        return None
    for source, folder in (
        ("user", user_themes_dir()),
        ("community", community_themes_dir()),
        ("builtin", builtin_themes_dir()),
    ):
        payload, path = _read_one(folder, name)
        if payload is None:
            continue
        try:
            theme = _sanitize_theme(name, payload)
        except ValueError:
            continue
        return {
            "name": theme.name,
            "source": source,
            "path": str(path),
            "font_family": theme.font_family,
            "vars": theme.vars,
        }
    return None


def write_theme(name: str, payload: dict[str, Any]) -> Path:
    """Persist a user theme to ``data/themes/<slug>.json``.

    Payload is sanitised via :func:`_sanitize_theme`. Returns the written
    path. Raises :class:`ValueError` on an invalid name / non-dict payload.
    """
    theme = _sanitize_theme(name, payload)
    folder = user_themes_dir()
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / f"{theme.name}.json"
    tmp = dest.with_name(dest.name + ".tmp")
    tmp.write_text(
        json.dumps(theme.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(tmp, dest)
    return dest


def delete_theme(name: str) -> bool:
    """Remove the user override for ``name``. Returns True iff a file was deleted.

    Never touches builtin or community shipped files — those are read-only
    from the dashboard's POV.
    """
    if not is_valid_name(name):
        return False
    dest = user_themes_dir() / f"{name}.json"
    if not dest.is_file():
        return False
    try:
        dest.unlink()
        return True
    except OSError:
        return False


def is_builtin(name: str) -> bool:
    """True iff ``name`` ships in ``themes/builtin/`` OR is a hard-coded CSS core.

    The CSS-only core themes (``dark`` / ``light`` / ``hc``) also count so the
    UI can guard the Delete button against wiping the fallback set.
    """
    if name in CORE_THEMES:
        return True
    if not is_valid_name(name):
        return False
    return (builtin_themes_dir() / f"{name}.theme.json").is_file()


def is_community(name: str) -> bool:
    if not is_valid_name(name):
        return False
    return (community_themes_dir() / f"{name}.theme.json").is_file()


def community_file_for(name: str) -> Path:
    """Where a published community theme lands. Callers create parent dirs."""
    return community_themes_dir() / f"{name}.theme.json"


def user_file_for(name: str) -> Path:
    return user_themes_dir() / f"{name}.json"


# ── Directory-walking helpers ────────────────────────────────────────────────

def _iter_theme_dir(folder: Path):
    """Yield ``(slug, payload_dict, path)`` for every valid theme file in ``folder``.

    Silently skips folders that don't exist, files that fail JSON parse, or
    filenames that don't match ``THEME_NAME_RE``. Never raises.
    """
    if not folder.exists() or not folder.is_dir():
        return
    for path in sorted(folder.iterdir()):
        if not path.is_file():
            continue
        stem = _theme_stem(path.name)
        if stem is None:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        yield stem, payload, path


def _read_one(folder: Path, name: str) -> tuple[dict | None, Path | None]:
    """Return the raw dict for a specific theme name, or (None, None)."""
    for suffix in (".json", ".theme.json"):
        candidate = folder / f"{name}{suffix}"
        if candidate.is_file():
            try:
                return json.loads(candidate.read_text(encoding="utf-8")), candidate
            except (OSError, json.JSONDecodeError):
                return None, None
    return None, None


def _theme_stem(filename: str) -> str | None:
    """Extract the theme slug from a filename.

    Accepts both ``<slug>.json`` (user themes) and ``<slug>.theme.json``
    (community + builtin). Returns None for anything else so the walker skips
    stray files.
    """
    name = filename.lower()
    if name.endswith(".theme.json"):
        stem = name[: -len(".theme.json")]
    elif name.endswith(".json"):
        stem = name[: -len(".json")]
    else:
        return None
    return stem if is_valid_name(stem) else None


__all__ = [
    "THEME_NAME_RE",
    "THEME_VAR_KEYS",
    "CORE_THEMES",
    "Theme",
    "builtin_themes_dir",
    "community_themes_dir",
    "community_thumbnail_dir",
    "user_themes_dir",
    "slugify",
    "is_valid_name",
    "list_themes",
    "read_theme",
    "write_theme",
    "delete_theme",
    "is_builtin",
    "is_community",
    "community_file_for",
    "user_file_for",
]
