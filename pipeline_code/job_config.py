"""Per-job configuration v2 — preset library + inherit-by-default overrides.

A **Preset** is a named bundle of the *inheritable* config (topic filters,
scraper targets/toggles, categories + rules, scoring, captioning, local-import
folders). Presets live in a global library at ``data/jobs/_presets.json``.

A **Job** (``data/jobs/<slug>.json``) references a preset and stores a *sparse*
``overrides`` map plus its always-per-job ``subject`` (the topic line). The
effective config is ``preset ⊕ overrides`` with ``subject`` injected as
``topic.topic``. Editing a field writes it to ``overrides``; resetting removes
it (inherit-by-default).

Activating a job still **projects** the effective config down into the existing
runtime contracts — env vars (``resolve_env``, merged over ``.env`` when the
supervisor spawns children) and ``cull_categories.json`` (``project_categories``).
Workers/scrapers are unchanged.

v1 job files (flat ``topic``/``scrapers``/``categories``) auto-upgrade to v2 on
read: their config becomes the job's ``overrides`` (so effective config is
unchanged) and the two legacy single-folder local sources fold into the
``local_imports`` list. See ``docs/jobs-presets-design.md``.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import builtin_presets
import categories
import paths
from pipeline_logging import get_logger

logger = get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

JOB_SLUG_RE = re.compile(r"^[a-z0-9_]+$")
PRESET_NAME_RE = re.compile(r"^[A-Za-z0-9 _-]{1,40}$")
JOB_STATUSES: tuple[str, ...] = ("idle", "queued", "running", "paused", "done")

# Canonical scraper names (single source of truth shared with the dashboard).
# Local folders are managed via scrapers.local_imports, not as a scraper name.
SCRAPER_NAMES: tuple[str, ...] = (
    "X.com", "Discord-1", "Civitai-Com", "Civitai-Red", "Web", "Gallery-DL",
)

# Non-canonical scrapers that still participate in per-job priority + ordering.
# Kept out of SCRAPER_NAMES (the enable-toggle contract, load-bearing) so
# nothing that consumes SCRAPER_NAMES for the on/off map has to grow with new
# opt-in feeders. run_pipeline gates YT-DLP on YT_DLP_ENABLED + URLs.
PRIORITY_EXTRA_NAMES: tuple[str, ...] = ("YT-DLP",)

# Full ordered set the priority block covers (spawn order + weights). Callers
# that need "every scraper the user can reorder" use this; toggling still uses
# SCRAPER_NAMES.
PRIORITY_NAMES: tuple[str, ...] = SCRAPER_NAMES + PRIORITY_EXTRA_NAMES

# Priority weight bounds (1-10). Higher = more turns in the round-robin.
PRIORITY_WEIGHT_MIN: int = 1
PRIORITY_WEIGHT_MAX: int = 10
PRIORITY_WEIGHT_DEFAULT: int = 5

_INDEX_FILENAME = "_index.json"
_PRESETS_FILENAME = "_presets.json"
_MISSING = object()


# ── Slug / name / time / IO helpers ──────────────────────────────────────────

def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (name or "").lower()).strip("_")


def _titleize(slug: str) -> str:
    return slug.replace("_", " ").title()


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively overlay ``override`` onto ``base`` (new dicts; lists replace)."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        elif v is not None:
            out[k] = copy.deepcopy(v)
    return out


def _clone(value: Any) -> Any:
    return json.loads(json.dumps(value))


# ── Dotted-path helpers (operate on the override / cfg namespace) ────────────

def _get_path(d: dict, dotted: str) -> Any:
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return _MISSING
        cur = cur[part]
    return cur


def _set_path(d: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur = d
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def _del_path(d: dict, dotted: str) -> None:
    parts = dotted.split(".")
    stack: list[tuple[dict, str]] = []
    cur: Any = d
    for part in parts[:-1]:
        if not isinstance(cur, dict) or part not in cur:
            return
        stack.append((cur, part))
        cur = cur[part]
    if isinstance(cur, dict):
        cur.pop(parts[-1], None)
    for parent, key in reversed(stack):       # prune emptied parents
        child = parent.get(key)
        if isinstance(child, dict) and not child:
            parent.pop(key, None)


# ── Paths ────────────────────────────────────────────────────────────────────

def jobs_dir() -> Path:
    return paths.jobs_dir()


def _job_path(slug: str) -> Path:
    # ``fullmatch`` anchors both ends so the slug cannot contain a path separator
    # or '..' traversal — this is the path-injection barrier for the per-job file.
    if not JOB_SLUG_RE.fullmatch(slug or ""):
        raise ValueError(f"invalid slug: {slug!r}")
    return jobs_dir() / f"{slug}.json"


def _index_path() -> Path:
    return jobs_dir() / _INDEX_FILENAME


def _presets_path() -> Path:
    return jobs_dir() / _PRESETS_FILENAME


# ── Default inheritable config (preset shape) ────────────────────────────────

def _default_categories() -> tuple[list[dict], str]:
    try:
        presets = categories.get_presets()
        name = presets.get("default") or next(iter(presets["presets"]))
        preset = presets["presets"][name]
        return [dict(c) for c in preset.get("categories", [])], preset.get("global_rules", "")
    except Exception:  # pragma: no cover - never block creation
        return [], ""


# Local vision-worker providers a fleet instance may use. lmstudio + llamacpp
# both speak the OpenAI /v1 API (one worker script); ollama uses its native API.
VISION_PROVIDERS: tuple[str, ...] = ("lmstudio", "llamacpp", "ollama")

# Shipped default fleet — "one local LLM initially" (LM Studio on localhost,
# model auto-detected on connect). Inherited by every preset/job until overridden.
def _default_vision_workers() -> list[dict]:
    return [{
        "id": "local-lm", "name": "LM Studio", "provider": "lmstudio",
        "base_url": "http://127.0.0.1:1234", "model": "", "api_key": "",
        "enabled": True,
    }]


def _default_preset_cfg() -> dict:
    cats, rules = _default_categories()
    return {
        "topic_filters": {
            "keywords_extra": [], "banned_keywords": [], "generation_hints": [],
            "min_prompt_length": 0, "require_prompt": True,
        },
        "scrapers": {
            "enabled": {n: True for n in SCRAPER_NAMES},
            "x_accounts": [], "reddit_subreddits": [], "discord_channels_json": "",
            "civitai_domains": [],
            "gallery_dl": {"enabled": False, "urls": [], "limit_per_url": 200,
                            "cookies_file": "", "config_path": "", "config_json": ""},
            "yt_dlp": {"enabled": False, "urls": [], "limit": 200, "cookies": ""},
            "local_imports": [],
            # Kohya training-set feeder. One per-job dataset root that follows the
            # ``<repeats>_<concept>`` subfolder convention. See feed_kohya_folder.py.
            "kohya_import": {
                "enabled": False, "dir": "", "name": "kohya",
                "move": False, "allow_flat": False,
            },
            # Per-job scraper priority: `order` fixes supervisor spawn order
            # (top = fires first for a queue slot) and `weights` (1-10, default 5)
            # scale each scraper's round-robin turns. Projected to
            # SCRAPER_PRIORITY_JSON by resolve_env; unknown names are ignored and
            # missing names get the default weight so a fresh install "just works"
            # in PRIORITY_NAMES order.
            "priority": {
                "order": list(PRIORITY_NAMES),
                "weights": {n: PRIORITY_WEIGHT_DEFAULT for n in PRIORITY_NAMES},
            },
        },
        "categories": cats,
        "category_rules": rules,
        "scoring": {"ovr_min": 0, "rel_min": 0, "notes": ""},
        "captioning": {"enabled": False, "style": "sd_prompt", "overwrite": False},
        # Which media the scrapers fetch + the queue pops. ``types`` is a subset of
        # {image, video}; the ext lists are editable so you aren't limited to one
        # format each. Projected to MEDIA_* env by resolve_env; media_policy.py is
        # the runtime source of truth every scraper consults.
        "media": {
            "types": ["image"],
            "image_exts": [".jpg", ".jpeg", ".png", ".webp", ".gif"],
            "video_exts": [".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"],
        },
        "vision": {"workers": _default_vision_workers()},
    }


# ── Preset library ───────────────────────────────────────────────────────────

# Records, per builtin preset, the signature of the shipped content we last
# reconciled against. A builtin still matching its baseline is "unmodified" and
# safe to refresh to a newer ship; a diverged one is a user edit we keep.
_BASELINE_KEY = "_builtin_baselines"

# Taste-bearing leaves a *legacy* (pre-baseline) builtin gets filled from the
# ship when blank — exactly the keyword / scraper-target / scoring-floor fields
# that shipped empty before seeding. Categories, rules and captioning are left
# to the user (only replaced wholesale for a brand-new or provably-unmodified
# builtin), so a customised legacy preset is never clobbered.
_SEED_FILL_PATHS: tuple[tuple[str, str], ...] = (
    ("topic_filters", "keywords_extra"),
    ("topic_filters", "banned_keywords"),
    ("topic_filters", "generation_hints"),
    ("topic_filters", "min_prompt_length"),
    ("scrapers", "x_accounts"),
    ("scrapers", "reddit_subreddits"),
    ("scrapers", "civitai_domains"),
    ("scoring", "ovr_min"),
    ("scoring", "rel_min"),
)


def _preset_signature(cfg: dict) -> str:
    """Stable content hash of a preset (or library), for change detection."""
    blob = json.dumps(cfg, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _is_blank(v: Any) -> bool:
    """A seed field counts as needing a fill when empty or a zero scoring floor."""
    return v is None or v == "" or v == [] or v == {} or v == 0


def _scoped_seed_fill(on_disk: dict, shipped: dict) -> dict:
    """Copy of ``on_disk`` with only its BLANK seed leaves filled from ``shipped``.
    Non-empty user content (categories, rules, custom keywords) is preserved."""
    merged = copy.deepcopy(on_disk)
    for sect, key in _SEED_FILL_PATHS:
        section = merged.get(sect)
        cur = section.get(key, _MISSING) if isinstance(section, dict) else _MISSING
        if cur is _MISSING or _is_blank(cur):
            ship_val = (shipped.get(sect) or {}).get(key, _MISSING)
            if ship_val is not _MISSING and not _is_blank(ship_val):
                merged.setdefault(sect, {})[key] = copy.deepcopy(ship_val)
    return merged


def _merge_builtin_presets(lib: dict) -> dict:
    """Ensure every shipped builtin preset is present and reconciled with the
    library — a managed-defaults merge that never touches a user-CREATED preset.

    Per builtin:
      * absent            -> insert the ship, record its baseline.
      * baseline == on-disk (unmodified) -> refresh wholesale to the latest ship.
      * baseline != on-disk (user edited) -> keep the user's version.
      * no baseline (legacy file) -> fill only BLANK seed leaves from the ship
        (new keywords / subreddits / scoring floors appear without clobbering a
        customisation), then record the ship baseline so later edits are detected.
    """
    presets = lib.setdefault("presets", {})
    baselines = lib.setdefault(_BASELINE_KEY, {})
    for name, cfg in builtin_presets.builtin_library()["presets"].items():
        shipped = copy.deepcopy(cfg)
        shipped_sig = _preset_signature(shipped)
        existing = presets.get(name)
        if existing is None:
            presets[name] = shipped
            baselines[name] = shipped_sig
            continue
        recorded = baselines.get(name)
        if recorded is None:
            presets[name] = _scoped_seed_fill(existing, shipped)
            baselines[name] = shipped_sig
        elif _preset_signature(existing) == recorded:
            presets[name] = shipped
            baselines[name] = shipped_sig
        # else: user-edited builtin -> keep as-is (baseline unchanged)
    lib.setdefault("default", builtin_presets.DEFAULT_PRESET)
    return lib


def _read_presets_raw(path: Path) -> dict | None:
    """Parse the on-disk presets file ONCE. Returns the raw dict, or None when
    the file is absent/unreadable/structurally invalid (caller falls back to the
    builtin library). Read exactly once per call to avoid a TOCTOU re-read."""
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("presets file %s is unreadable; using built-in defaults", path)
        return None
    if isinstance(data, dict) and isinstance(data.get("presets"), dict) and data["presets"]:
        return data
    return None


#: Soft-warn ceiling for user-editable UI (mirror of dashboard cap). Presets
#: above this still load — the dashboard's save path will refuse a new write
#: above the cap, so the warning surfaces the mismatch on legacy files.
_SOFT_CATEGORY_LIMIT = 12
_LOAD_WARN_ONCE: set[str] = set()


def _warn_oversized_presets(lib: dict) -> None:
    """Log once per preset that exceeds the soft UI category cap.

    On upgrade, users who hand-edited ``_presets.json`` before the wave dropped
    the dashboard save cap to 12 might have >12 categories on disk. The
    library still loads fine (preserving their data), but new dashboard saves
    will refuse to persist a change back — surface the mismatch on read so
    they can prune before hitting a confusing "max 12 categories" error.
    """
    for name, cfg in (lib.get("presets") or {}).items():
        if not isinstance(cfg, dict):
            continue
        cats = cfg.get("categories") or []
        if isinstance(cats, list) and len(cats) > _SOFT_CATEGORY_LIMIT:
            token = f"cats:{name}"
            if token in _LOAD_WARN_ONCE:
                continue
            _LOAD_WARN_ONCE.add(token)
            logger.warning(
                "preset %r has %d categories (>%d recommended). It will load, "
                "but the dashboard editor caps new saves at %d — trim it there "
                "or edit _presets.json directly before saving via the UI.",
                name, len(cats), _SOFT_CATEGORY_LIMIT, _SOFT_CATEGORY_LIMIT,
            )


def _read_presets() -> dict:
    data = _read_presets_raw(_presets_path())
    if data is None:
        return _merge_builtin_presets(builtin_presets.builtin_library())
    data.setdefault("default", next(iter(data["presets"])))
    lib = _merge_builtin_presets(data)
    _warn_oversized_presets(lib)
    return lib


def _write_presets(lib: dict) -> None:
    _atomic_write_json(_presets_path(), lib)


def list_presets() -> dict:
    # Single read of the file (no TOCTOU re-read): seed on first access, else
    # durably persist any builtin presets merged into an older library.
    path = _presets_path()
    raw = _read_presets_raw(path)
    if raw is None:
        lib = _merge_builtin_presets(builtin_presets.builtin_library())
        _write_presets(lib)                    # seed + record baselines on first access
        _warn_oversized_presets(lib)
        return lib
    raw.setdefault("default", next(iter(raw["presets"])))
    before = _preset_signature(raw)            # snapshot pre-merge (content, not just names)
    lib = _merge_builtin_presets(raw)
    if _preset_signature(lib) != before:
        _write_presets(lib)                    # persist refreshed/seeded builtins + baselines
    _warn_oversized_presets(lib)
    return lib


def default_preset_name() -> str:
    return _read_presets().get("default", "default")


def get_preset(name: str) -> dict:
    """Return a TOTAL preset cfg (default shape ⊕ stored), falling back to the
    library default for an unknown name."""
    lib = _read_presets()
    cfg = lib["presets"].get(name) or lib["presets"].get(lib.get("default", "default")) \
        or _default_preset_cfg()
    return _deep_merge(_default_preset_cfg(), cfg)


def save_preset(name: str, cfg: dict) -> None:
    if not PRESET_NAME_RE.match(name):
        raise ValueError(f"invalid preset name: {name!r}")
    lib = list_presets()
    lib["presets"][name] = _clone(cfg)
    _write_presets(lib)


def delete_preset(name: str) -> None:
    lib = list_presets()
    if name not in lib["presets"]:
        raise ValueError(f"unknown preset: {name}")
    if name == lib.get("default"):
        raise ValueError(f"cannot delete the default preset: {name}")
    if any(j.preset == name for j in list_jobs()):
        raise ValueError(f"preset {name!r} is referenced by a job")
    lib["presets"].pop(name, None)
    _write_presets(lib)


def set_default_preset(name: str) -> None:
    lib = list_presets()
    if name not in lib["presets"]:
        raise ValueError(f"unknown preset: {name}")
    lib["default"] = name
    _write_presets(lib)


def builtin_preset_names() -> tuple[str, ...]:
    """Names of the presets cull ships (the ones a Reset-to-defaults applies to)."""
    return builtin_presets.PRESET_NAMES


def reset_preset_to_builtin(name: str) -> dict:
    """Restore a builtin preset to its shipped definition and re-baseline it, so
    a user can force a stale/edited builtin back to the library. Raises
    ValueError for a name that isn't a shipped builtin."""
    shipped = builtin_presets.builtin_library()["presets"].get(name)
    if shipped is None:
        raise ValueError(f"{name!r} is not a builtin preset")
    lib = list_presets()
    lib["presets"][name] = copy.deepcopy(shipped)
    lib.setdefault(_BASELINE_KEY, {})[name] = _preset_signature(shipped)
    _write_presets(lib)
    return get_preset(name)


# ── v1 → v2 conversion ───────────────────────────────────────────────────────

def _v1_scrapers_to_cfg(v1: dict) -> dict:
    enabled_in = v1.get("enabled", {}) or {}
    enabled = {n: bool(enabled_in.get(n, True)) for n in SCRAPER_NAMES}
    local_imports: list[dict] = []
    li = v1.get("local_import")
    if isinstance(li, dict) and (li.get("dir") or li.get("enabled")):
        local_imports.append({
            "name": li.get("name", "local") or "local", "dir": li.get("dir", "") or "",
            "enabled": bool(li.get("enabled", False)), "migrate_from": li.get("migrate_from", "") or "",
        })
    gd = v1.get("gallery_dl", {}) or {}
    return {
        "enabled": enabled,
        "x_accounts": list(v1.get("x_accounts", []) or []),
        "reddit_subreddits": list(v1.get("reddit_subreddits", []) or []),
        "discord_channels_json": v1.get("discord_channels_json", "") or "",
        "civitai_domains": list(v1.get("civitai_domains", []) or []),
        "gallery_dl": {
            "enabled": bool(gd.get("enabled", False)), "urls": list(gd.get("urls", []) or []),
            "limit_per_url": int(gd.get("limit_per_url", 200) or 200),
            "cookies_file": gd.get("cookies_file", "") or "", "config_path": gd.get("config_path", "") or "",
        },
        "local_imports": local_imports,
    }


def _v1_job_to_cfg(d: dict) -> dict:
    topic = d.get("topic", {}) or {}
    return {
        "topic_filters": {
            "keywords_extra": list(topic.get("keywords_extra", []) or []),
            "banned_keywords": list(topic.get("banned_keywords", []) or []),
            "generation_hints": list(topic.get("generation_hints", []) or []),
            "min_prompt_length": int(topic.get("min_prompt_length", 0) or 0),
            "require_prompt": bool(topic.get("require_prompt", True)),
        },
        "scrapers": _v1_scrapers_to_cfg(d.get("scrapers", {}) or {}),
        "categories": _clone(d.get("categories", []) or []),
        "category_rules": d.get("category_rules", "") or "",
        "scoring": {
            "ovr_min": int((d.get("scoring", {}) or {}).get("ovr_min", 0) or 0),
            "rel_min": int((d.get("scoring", {}) or {}).get("rel_min", 0) or 0),
            "notes": (d.get("scoring", {}) or {}).get("notes", "") or "",
        },
        "captioning": _clone(d.get("captioning", {}) or
                             {"enabled": False, "style": "sd_prompt", "overwrite": False}),
    }


# ── Job model ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Job:
    slug: str
    name: str
    status: str
    created_at: str
    updated_at: str
    subject: str = ""
    preset: str = "default"
    overrides: dict = field(default_factory=dict)

    def with_updates(self, **changes: Any) -> "Job":
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> dict:
        return {
            "slug": self.slug, "name": self.name, "status": self.status,
            "created_at": self.created_at, "updated_at": self.updated_at,
            "subject": self.subject, "preset": self.preset, "overrides": self.overrides,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Job":
        slug = d.get("slug", "")
        name = d.get("name", "") or _titleize(slug)
        # v2 marker: "overrides" is the distinctive field. A stray "subject" or
        # "preset" alone on an otherwise-v1 file must NOT mask the v1 config.
        if "overrides" in d or ("preset" in d and "subject" in d):
            return cls(
                slug=slug, name=name, status=d.get("status", "idle"),
                created_at=d.get("created_at", ""), updated_at=d.get("updated_at", ""),
                subject=d.get("subject", name), preset=d.get("preset") or default_preset_name(),
                overrides=d.get("overrides") or {},
            )
        # legacy v1 file → upgrade in memory
        subject = (d.get("topic", {}) or {}).get("topic", name)
        return cls(
            slug=slug, name=name, status=d.get("status", "idle"),
            created_at=d.get("created_at", ""), updated_at=d.get("updated_at", ""),
            subject=subject, preset="default", overrides=_v1_job_to_cfg(d),
        )


def _make_job(slug: str, name: str, *, subject: str | None = None, preset: str | None = None,
              overrides: dict | None = None, status: str = "idle") -> Job:
    now = _now_iso()
    return Job(
        slug=slug, name=name, status=status, created_at=now, updated_at=now,
        subject=subject if subject is not None else name,
        preset=preset or default_preset_name(), overrides=overrides or {},
    )


# ── Effective config + override editing ──────────────────────────────────────

_TOPIC_FILTER_KEYS = ("keywords_extra", "banned_keywords", "generation_hints",
                      "min_prompt_length", "require_prompt")


def _resolved_cfg(job: Job) -> dict:
    return _deep_merge(get_preset(job.preset), job.overrides or {})


def effective_config(job: Job) -> dict:
    """preset ⊕ overrides, reshaped to the v1-style config the runtime mappers
    consume (topic block = subject + topic_filters)."""
    cfg = _resolved_cfg(job)
    tf = cfg.get("topic_filters", {})
    return {
        "topic": {"topic": job.subject, **{k: tf.get(k) for k in _TOPIC_FILTER_KEYS}},
        "scrapers": cfg.get("scrapers", {}),
        "categories": cfg.get("categories", []),
        "category_rules": cfg.get("category_rules", ""),
        "scoring": cfg.get("scoring", {}),
        "captioning": cfg.get("captioning", {}),
        "media": cfg.get("media", {}),
        "vision": cfg.get("vision", {}),
    }


def is_overridden(job: Job, path: str) -> bool:
    return _get_path(job.overrides or {}, path) is not _MISSING


def set_override(job: Job, path: str, value: Any) -> Job:
    ov = copy.deepcopy(job.overrides or {})
    _set_path(ov, path, _clone(value))
    return job.with_updates(overrides=ov)


def reset_override(job: Job, path: str) -> Job:
    ov = copy.deepcopy(job.overrides or {})
    _del_path(ov, path)
    return job.with_updates(overrides=ov)


# ── CRUD ─────────────────────────────────────────────────────────────────────

def _read_job_file(path: Path) -> Job | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or not payload.get("slug"):
        return None
    return Job.from_dict(payload)


def get_job(slug: str) -> Job | None:
    if not JOB_SLUG_RE.fullmatch(slug or ""):
        return None
    path = _job_path(slug)
    return _read_job_file(path) if path.is_file() else None


def list_jobs() -> list[Job]:
    d = jobs_dir()
    if not d.is_dir():
        return []
    jobs: list[Job] = []
    for p in sorted(d.glob("*.json")):
        if p.name in (_INDEX_FILENAME, _PRESETS_FILENAME):
            continue
        job = _read_job_file(p)
        if job is not None:
            jobs.append(job)
    idx = get_index()
    # Multi-active: active jobs come first (in the order they were activated),
    # then the queue. Legacy readers that expected a single active still see
    # the head at index 0 because get_active_slug() = active[0].
    order = list(idx["active"]) + list(idx["queue"])
    pos = {slug: i for i, slug in enumerate(order)}
    jobs.sort(key=lambda j: (pos.get(j.slug, len(order)), j.name.lower()))
    return jobs


def save_job(job: Job) -> Job:
    if not JOB_SLUG_RE.match(job.slug):
        raise ValueError(f"invalid slug: {job.slug!r}")
    saved = job.with_updates(updated_at=_now_iso())
    _atomic_write_json(_job_path(saved.slug), saved.to_dict())
    return saved


def create_job(name: str, *, subject: str | None = None, preset: str | None = None,
               base_on: str | None = None) -> Job:
    slug = slugify(name)
    if not slug:
        raise ValueError(f"name produces empty slug: {name!r}")
    if get_job(slug) is not None:
        raise ValueError(f"job already exists: {slug}")
    if base_on:
        src = get_job(base_on)
        if src is None:
            raise ValueError(f"base_on job not found: {base_on}")
        job = _make_job(slug, name, subject=subject if subject is not None else src.subject,
                        preset=src.preset, overrides=_clone(src.overrides))
    else:
        job = _make_job(slug, name, subject=subject, preset=preset)
    return save_job(job)


def delete_job(slug: str) -> None:
    # Sanitise before building any path: rejects '/', '\\' and '..' traversal.
    if not JOB_SLUG_RE.fullmatch(slug or ""):
        raise ValueError(f"invalid slug: {slug!r}")
    if slug in get_active_slugs():
        raise ValueError(f"cannot delete the active job: {slug}")
    path = _job_path(slug)
    if path.is_file():
        path.unlink()
    idx = get_index()
    changed = False
    queue = list(idx["queue"])
    if slug in queue:
        queue = [s for s in queue if s != slug]
        changed = True
    priority = dict(idx.get("priority") or {})
    if slug in priority:
        priority.pop(slug, None)
        changed = True
    if changed:
        _save_index({**idx, "queue": queue, "priority": priority})


# ── Index: queue order + active pointer(s) + per-job priority ────────────────
#
# Historic shape:  {"active": <slug>|null, "queue": [...]}
# Multi-active v2: {"active": [<slug>, ...], "queue": [...], "priority": {slug: 1-10}}
#
# ``get_index()`` always returns the CANONICAL multi-active shape so the rest
# of the module stops having to care which on-disk shape a reader landed on.
# Writes go back through ``_save_index`` in the canonical shape.

def _canonical_active(raw: Any) -> list[str]:
    """Coerce the on-disk ``active`` field to a canonical list of slugs.

    Accepts the historic single-string form and the new list form; drops
    duplicates while preserving insertion order.
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw] if raw else []
    if isinstance(raw, list):
        out: list[str] = []
        seen: set[str] = set()
        for item in raw:
            if isinstance(item, str) and item and item not in seen:
                out.append(item)
                seen.add(item)
        return out
    return []


def _canonical_priority(raw: Any) -> dict[str, int]:
    """Coerce the on-disk ``priority`` field to a canonical clamped map."""
    if not isinstance(raw, dict):
        return {}
    out: dict[str, int] = {}
    for slug, w in raw.items():
        if not isinstance(slug, str) or not JOB_SLUG_RE.fullmatch(slug):
            continue
        try:
            weight = int(w)
        except (TypeError, ValueError):
            weight = PRIORITY_WEIGHT_DEFAULT
        out[slug] = max(PRIORITY_WEIGHT_MIN, min(PRIORITY_WEIGHT_MAX, weight))
    return out


def get_index() -> dict:
    """Return the canonical multi-active index shape.

    Migration is on-read: legacy ``{"active": "slug", "queue": [...]}`` files
    are coerced into ``{"active": ["slug"], "queue": [...], "priority": {}}``
    without a disk write. The next ``_save_index`` re-persists in the new shape.
    """
    p = _index_path()
    empty = {"active": [], "queue": [], "priority": {}}
    if not p.is_file():
        return empty
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return empty
    if not isinstance(data, dict):
        return empty
    active = _canonical_active(data.get("active"))
    queue_raw = data.get("queue")
    queue = [s for s in queue_raw if isinstance(s, str)] if isinstance(queue_raw, list) else []
    return {
        "active": active,
        "queue": queue,
        "priority": _canonical_priority(data.get("priority")),
    }


def _save_index(idx: dict) -> None:
    """Persist the canonical multi-active shape (atomic tmpfile+os.replace)."""
    payload = {
        "active": list(idx.get("active") or []),
        "queue": list(idx.get("queue") or []),
        "priority": dict(idx.get("priority") or {}),
        "updated_at": _now_iso(),
    }
    _atomic_write_json(_index_path(), payload)


def get_active_slugs() -> list[str]:
    """Canonical multi-active reader: every slug currently marked to run."""
    return list(get_index()["active"])


def get_active_slug() -> str | None:
    """Single-slug view for legacy callers — the head of the active list.

    Returns None when nothing is active. Every existing single-active caller
    keeps working because a one-slug active list still yields that one slug
    here.
    """
    active = get_active_slugs()
    return active[0] if active else None


def set_active(slug: str | None) -> None:
    """Replace the active set with ``[slug]`` (or clear it when ``None``).

    Legacy single-active setter kept for callers that own the entire active
    pointer (tests, ``activate(exclusive=True)``, "stop everything" paths).
    """
    if slug is not None and get_job(slug) is None:
        raise ValueError(f"unknown job: {slug}")
    idx = get_index()
    queue = [s for s in idx["queue"] if s != slug] if slug is not None else idx["queue"]
    new_active = [slug] if slug is not None else []
    _save_index({**idx, "active": new_active, "queue": queue})


def set_active_slugs(slugs: list[str]) -> None:
    """Replace the entire active set with the given list (order preserved).

    Every slug is validated before any write — an unknown slug raises with the
    index untouched.
    """
    seen: set[str] = set()
    clean: list[str] = []
    for s in slugs:
        if not isinstance(s, str) or s in seen:
            continue
        if get_job(s) is None:
            raise ValueError(f"unknown job: {s}")
        clean.append(s)
        seen.add(s)
    idx = get_index()
    queue = [s for s in idx["queue"] if s not in seen]
    _save_index({**idx, "active": clean, "queue": queue})


def set_queue(order: list[str]) -> None:
    for slug in order:
        if get_job(slug) is None:
            raise ValueError(f"unknown job: {slug}")
    _save_index({**get_index(), "queue": list(order)})


def enqueue(slug: str) -> None:
    if get_job(slug) is None:
        raise ValueError(f"unknown job: {slug}")
    idx = get_index()
    if slug not in idx["active"] and slug not in idx["queue"]:
        _save_index({**idx, "queue": [*idx["queue"], slug]})


def dequeue(slug: str) -> None:
    idx = get_index()
    if slug in idx["queue"]:
        _save_index({**idx, "queue": [s for s in idx["queue"] if s != slug]})


def advance() -> str | None:
    """Pop the queue head and ADD it to the active set.

    Multi-active semantics: advance no longer replaces the active pointer —
    it augments it. Returns the slug that was promoted, or ``None`` when the
    queue is empty (or every queued entry was orphaned).
    """
    idx = get_index()
    queue = list(idx["queue"])
    active = list(idx["active"])
    new_active: str | None = None
    while queue:
        candidate = queue.pop(0)
        if get_job(candidate) is None:
            logger.warning("advance: skipping orphaned queued slug %r", candidate)
            continue
        if candidate not in active:
            active.append(candidate)
        new_active = candidate
        break
    _save_index({**idx, "active": active, "queue": queue})
    if new_active is not None:
        job = get_job(new_active)
        if job is not None:
            project_categories(job)
    return new_active


# ── Per-job priority (1-10 — drives the queue round-robin weighting) ─────────

def get_job_priority(slug: str) -> int:
    """Read this job's priority weight. Defaults to ``PRIORITY_WEIGHT_DEFAULT``
    (5) when unset — every job starts equal-weight so a fresh install just works.
    """
    return int(get_index().get("priority", {}).get(slug, PRIORITY_WEIGHT_DEFAULT))


def set_job_priority(slug: str, weight: int) -> int:
    """Clamp ``weight`` to [1, 10] and persist it against ``slug``.

    Returns the clamped weight actually stored so callers (dashboard) can echo
    it back to the user when they typed something out of range.
    """
    if not JOB_SLUG_RE.fullmatch(slug or ""):
        raise ValueError(f"invalid slug: {slug!r}")
    if get_job(slug) is None:
        raise ValueError(f"unknown job: {slug}")
    try:
        w = int(weight)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"weight must be an int: {weight!r}") from exc
    w = max(PRIORITY_WEIGHT_MIN, min(PRIORITY_WEIGHT_MAX, w))
    idx = get_index()
    priority = dict(idx.get("priority") or {})
    priority[slug] = w
    _save_index({**idx, "priority": priority})
    return w


def deactivate(slug: str) -> None:
    """Remove ``slug`` from the active set (no-op if absent). Idempotent."""
    idx = get_index()
    active = [s for s in idx["active"] if s != slug]
    if len(active) != len(idx["active"]):
        _save_index({**idx, "active": active})


# ── Projection: job → runtime contracts ──────────────────────────────────────

def _b(v: Any) -> str:
    return "true" if v else "false"


def _csv(xs: Any) -> str:
    if not xs:
        return ""
    if isinstance(xs, str):
        return xs
    return ",".join(str(x) for x in xs)


def clean_vision_fleet(workers: Any) -> list[dict]:
    """Normalise an effective ``vision.workers`` list into the JSON the supervisor
    fans out: only ENABLED instances with a real base_url and a known provider,
    each a flat ``{id, name, provider, base_url, model, api_key}`` dict."""
    out: list[dict] = []
    if not isinstance(workers, list):
        return out
    for i, w in enumerate(workers):
        if not isinstance(w, dict) or not w.get("enabled", True):
            continue
        provider = str(w.get("provider", "") or "").strip().lower()
        base_url = str(w.get("base_url", "") or "").strip()
        if provider not in VISION_PROVIDERS or not base_url:
            continue
        wid = str(w.get("id", "") or f"w{i}").strip() or f"w{i}"
        out.append({
            "id": wid,
            "name": str(w.get("name", "") or wid).strip() or wid,
            "provider": provider,
            "base_url": base_url,
            "model": str(w.get("model", "") or "").strip(),
            "api_key": str(w.get("api_key", "") or "").strip(),
        })
    return out


def _project_active_learning_exemplars(slug: str, eff: dict) -> str:
    """Best-effort: project the job's accumulated active-learning exemplars into
    a JSON map ``{category: [exemplar, ...]}`` for the classification prompt's
    few-shot block. Positives are the EFFECTIVE keep buckets; negatives are the
    terminal categories. Returns ``"{}"`` when there is no signal or on ANY error
    — this is an opportunistic enrichment that must never break projection.

    The env var ``ACTIVE_LEARNING_EXEMPLARS_JSON`` it feeds is an internal
    projection (never user-set); ``vision_prompt`` reads it and injects the block
    only when it is non-empty, so an absent/empty value leaves the prompt
    byte-identical to today."""
    try:
        import active_learning
    except Exception:  # pragma: no cover - active_learning always importable here
        return "{}"
    try:
        cats = eff.get("categories")
        keep = [c.get("name", "") for c in cats if isinstance(c, dict)] \
            if isinstance(cats, list) else []
        terminal = sorted(active_learning.TERMINAL)
        out: dict[str, list] = {}
        # Keep buckets first (positives), then terminal (negatives); dedupe names.
        for cat in [*keep, *terminal]:
            name = (cat or "").strip()
            if not name or name in out:
                continue
            try:
                ex = active_learning.exemplars(slug, name)
            except Exception:
                ex = []
            if ex:
                out[name] = ex
        return json.dumps(out)
    except Exception:  # pragma: no cover - never block projection
        return "{}"


def clean_scraper_priority(raw: Any) -> dict[str, Any]:
    """Normalise a ``scrapers.priority`` blob into a canonical projected shape.

    Return value is always ``{"order": [str, ...], "weights": {str: int}}``:
      * ``order`` = user-supplied order (only names in PRIORITY_NAMES), then
        the remaining PRIORITY_NAMES appended alphabetically so every known
        scraper is deterministically covered.
      * ``weights`` = clamped int 1-10 for every PRIORITY_NAME, filling
        missing/invalid entries with PRIORITY_WEIGHT_DEFAULT.

    Unknown names are dropped defensively — the runtime never spawns them, and
    keeping them out of the projected env keeps SCRAPER_PRIORITY_JSON tight.
    Callers may pass ``None`` or a partial dict; we always return a full shape.
    """
    known = set(PRIORITY_NAMES)
    raw_dict = raw if isinstance(raw, dict) else {}
    order_in = raw_dict.get("order")
    weights_in = raw_dict.get("weights")

    seen: set[str] = set()
    order: list[str] = []
    if isinstance(order_in, list):
        for n in order_in:
            if isinstance(n, str) and n in known and n not in seen:
                order.append(n)
                seen.add(n)
    # Append any known names the user hasn't ordered (deterministic tail).
    for n in PRIORITY_NAMES:
        if n not in seen:
            order.append(n)

    weights: dict[str, int] = {}
    raw_weights = weights_in if isinstance(weights_in, dict) else {}
    for n in PRIORITY_NAMES:
        try:
            w = int(raw_weights.get(n, PRIORITY_WEIGHT_DEFAULT))
        except (TypeError, ValueError):
            w = PRIORITY_WEIGHT_DEFAULT
        weights[n] = max(PRIORITY_WEIGHT_MIN, min(PRIORITY_WEIGHT_MAX, w))
    return {"order": order, "weights": weights}


def resolve_env(job: Job) -> dict[str, str]:
    """Flatten the EFFECTIVE config into the existing env-var names. Every key is
    always emitted so a job fully overrides any stale value in the global .env."""
    eff = effective_config(job)

    def _d(v: Any) -> dict:
        return v if isinstance(v, dict) else {}

    t = _d(eff.get("topic"))
    s = _d(eff.get("scrapers"))
    gd = _d(s.get("gallery_dl"))
    yt = _d(s.get("yt_dlp"))
    ko = _d(s.get("kohya_import"))
    sc = _d(eff.get("scoring"))
    cap = _d(eff.get("captioning"))
    md = _d(eff.get("media"))
    media_types_list = [x for x in (md.get("types") or ["image"])
                        if x in ("image", "video")] or ["image"]
    enabled = _d(s.get("enabled"))
    vision = _d(eff.get("vision"))
    fleet = clean_vision_fleet(vision.get("workers"))
    priority = clean_scraper_priority(s.get("priority"))
    # Only known scrapers contribute to SCRAPER_DISABLED (ignore stale/unknown names).
    disabled = sorted(n for n in SCRAPER_NAMES if not enabled.get(n, True))
    local_list = s.get("local_imports") if isinstance(s.get("local_imports"), list) else []
    local = [
        {"name": f.get("name", "local") or "local", "dir": f.get("dir", "") or "",
         "migrate_from": f.get("migrate_from", "") or ""}
        for f in local_list if isinstance(f, dict) and f.get("enabled")
    ]
    return {
        "PIPELINE_SLUG": job.slug,
        "PIPELINE_TOPIC": str(t.get("topic", "")),
        "TOPIC_KEYWORDS_EXTRA": _csv(t.get("keywords_extra")),
        "TOPIC_BANNED_KEYWORDS": _csv(t.get("banned_keywords")),
        "TOPIC_GENERATION_HINTS": _csv(t.get("generation_hints")),
        "MIN_PROMPT_LENGTH": str(int(t.get("min_prompt_length", 0) or 0)),
        "REQUIRE_PROMPT": _b(t.get("require_prompt", True)),
        "SCRAPER_DISABLED": ",".join(disabled),
        "X_ACCOUNTS": _csv(s.get("x_accounts")),
        "REDDIT_SUBREDDITS": _csv(s.get("reddit_subreddits")),
        "DISCORD_CHANNELS_JSON": str(s.get("discord_channels_json", "") or ""),
        "CIVITAI_DOMAINS": _csv(s.get("civitai_domains")),
        "GALLERY_DL_ENABLED": _b(gd.get("enabled", False)),
        "GALLERY_DL_URLS": "\n".join(gd.get("urls", []) or []),
        "GALLERY_DL_LIMIT_PER_URL": str(int(gd.get("limit_per_url", 200) or 200)),
        "GALLERY_DL_CONFIG_PATH": str(gd.get("config_path", "") or ""),
        # Per-job custom gallery-dl config (inline JSON), applied ON TOP of the
        # global GALLERY_DL_CONFIG_JSON. Distinct env name so both coexist/merge.
        "GALLERY_DL_CONFIG_JSON_JOB": str(gd.get("config_json", "") or ""),
        "YT_DLP_ENABLED": _b(yt.get("enabled", False)),
        "YT_DLP_URLS": "\n".join(yt.get("urls", []) or []),
        "YT_DLP_LIMIT": str(int(yt.get("limit", 200) or 200)),
        "YT_DLP_COOKIES": str(yt.get("cookies", "") or ""),
        # Kohya training-set feeder — a single per-job dataset root, projected as
        # KOHYA_* env vars the (unchanged) feed_kohya_folder.py reads at spawn.
        "KOHYA_IMPORT_ENABLED": _b(ko.get("enabled", False)),
        "KOHYA_IMPORT_DIR": str(ko.get("dir", "") or ""),
        "KOHYA_IMPORT_NAME": str(ko.get("name", "") or "kohya") or "kohya",
        "KOHYA_MOVE": _b(ko.get("move", False)),
        "KOHYA_ALLOW_FLAT": _b(ko.get("allow_flat", False)),
        "LOCAL_IMPORTS_JSON": json.dumps(local),
        # Priority (order + per-scraper weight) the supervisor consumes at spawn
        # time to fix which scraper starts first + how many round-robin turns
        # each gets. See run_pipeline.compute_desired_agents for the reader.
        "SCRAPER_PRIORITY_JSON": json.dumps(priority),
        "VISION_OVR_MIN_SCORE": str(int(sc.get("ovr_min", 0) or 0)),
        "VISION_REL_MIN_SCORE": str(int(sc.get("rel_min", 0) or 0)),
        "VISION_SCORE_NOTES": str(sc.get("notes", "") or ""),
        "VISION_WORKERS_JSON": json.dumps(fleet),
        "AUTO_CAPTION_ENABLED": _b(cap.get("enabled", False)),
        "AUTO_CAPTION_STYLE": str(cap.get("style", "sd_prompt") or "sd_prompt"),
        "AUTO_CAPTION_OVERWRITE": _b(cap.get("overwrite", False)),
        "ACTIVE_LEARNING_EXEMPLARS_JSON": _project_active_learning_exemplars(job.slug, eff),
        "MEDIA_TYPES": ",".join(media_types_list),
        "MEDIA_IMAGE_EXTS": _csv(md.get("image_exts")),
        "MEDIA_VIDEO_EXTS": _csv(md.get("video_exts")),
        # Selecting video turns the classify lane on so the queue actually pops
        # clips; image-only jobs inherit the global .env toggle (key omitted).
        **({"VIDEO_CLASSIFY_ENABLED": "true"} if "video" in media_types_list else {}),
        # Per-job gallery-dl cookies file OVERRIDES the global .env default when
        # set; emitted only when non-empty so an unset job falls through to the
        # global GALLERY_DL_COOKIES_FILE from Settings.
        **({"GALLERY_DL_COOKIES_FILE": str(gd.get("cookies_file", "") or "").strip()}
           if str(gd.get("cookies_file", "") or "").strip() else {}),
    }


def project_categories(job: Job) -> None:
    eff = effective_config(job)
    cats = eff.get("categories")
    cats = cats if isinstance(cats, list) else []
    categories.set_active({
        "preset": "custom",
        "categories": [dict(c) for c in cats if isinstance(c, dict)],
        "global_rules": str(eff.get("category_rules", "") or ""),
    })


def activate(slug: str, *, exclusive: bool = False) -> None:
    """Mark ``slug`` active (additive by default) and project its taxonomy.

    ``exclusive=True`` restores the historic single-active behaviour: the
    active set is reset to ``[slug]``. The default (``exclusive=False``)
    APPENDS ``slug`` to the active set so multiple jobs can run in parallel.
    A slug already in the active set is a no-op on the pointer but still
    re-projects its taxonomy so a preset edit takes effect.

    Note: ``project_categories`` writes ``cull_categories.json`` from the
    passed job, so in multi-active mode the LAST activated job "wins" the
    file. The supervisor's shared vision fleet reads schema per-image from
    the taxonomy on disk; per-job classification hints stay tight because
    each per-slug worker fanned out by the supervisor also re-projects its
    own env before use. See CLAUDE.md Jobs model.
    """
    job = get_job(slug)
    if job is None:
        raise ValueError(f"unknown job: {slug}")
    if exclusive:
        set_active(slug)
    else:
        idx = get_index()
        active = list(idx["active"])
        if slug not in active:
            # Preserve insertion order — re-activating an already-active slug
            # is a no-op on the pointer (idempotent) so the head stays put.
            active.append(slug)
        queue = [s for s in idx["queue"] if s != slug]
        _save_index({**idx, "active": active, "queue": queue})
    project_categories(job)


# ── Migration (legacy .env → v2) ─────────────────────────────────────────────

def _csv_to_list(raw: str | None) -> list[str]:
    return [x.strip() for x in (raw or "").split(",") if x.strip()]


def _int_env(key: str, default: int = 0) -> int:
    try:
        return int(os.environ.get(key, "").strip() or default)
    except ValueError:
        return default


def _bool_env(key: str, default: bool = False) -> bool:
    raw = os.environ.get(key, "").strip().lower()
    return raw in ("1", "true", "yes", "on") if raw else default


def _env_to_cfg() -> dict:
    cfg = _default_preset_cfg()
    disabled = set(_csv_to_list(os.environ.get("SCRAPER_DISABLED")))
    urls_raw = os.environ.get("GALLERY_DL_URLS", "") or ""
    urls = [u.strip() for u in re.split(r"[\n,]", urls_raw)
            if u.strip() and not u.strip().startswith("#")]
    local_imports: list[dict] = []
    if os.environ.get("LOCAL_IMPORT_DIR") or _bool_env("LOCAL_IMPORT_ENABLED"):
        local_imports.append({
            "name": os.environ.get("LOCAL_IMPORT_NAME", "local") or "local",
            "dir": os.environ.get("LOCAL_IMPORT_DIR", "") or "",
            "enabled": _bool_env("LOCAL_IMPORT_ENABLED", False),
            "migrate_from": os.environ.get("LOCAL_IMPORT_MIGRATE_FROM", "") or "",
        })
    cfg["topic_filters"] = {
        "keywords_extra": _csv_to_list(os.environ.get("TOPIC_KEYWORDS_EXTRA")),
        "banned_keywords": _csv_to_list(os.environ.get("TOPIC_BANNED_KEYWORDS")),
        "generation_hints": _csv_to_list(os.environ.get("TOPIC_GENERATION_HINTS")),
        "min_prompt_length": _int_env("MIN_PROMPT_LENGTH", 0),
        "require_prompt": _bool_env("REQUIRE_PROMPT", True),
    }
    cfg["scrapers"] = {
        "enabled": {n: (n not in disabled) for n in SCRAPER_NAMES},
        "x_accounts": _csv_to_list(os.environ.get("X_ACCOUNTS")),
        "reddit_subreddits": _csv_to_list(os.environ.get("REDDIT_SUBREDDITS")),
        "discord_channels_json": os.environ.get("DISCORD_CHANNELS_JSON", "") or "",
        "civitai_domains": _csv_to_list(os.environ.get("CIVITAI_DOMAINS")),
        "gallery_dl": {
            "enabled": _bool_env("GALLERY_DL_ENABLED", False), "urls": urls,
            "limit_per_url": _int_env("GALLERY_DL_LIMIT_PER_URL", 200),
            "cookies_file": os.environ.get("GALLERY_DL_COOKIES_FILE", "") or "",
            "config_path": os.environ.get("GALLERY_DL_CONFIG_PATH", "") or "",
        },
        "local_imports": local_imports,
    }
    try:
        active_tax = categories.get_active()
        if active_tax.get("categories"):
            cfg["categories"] = [dict(c) for c in active_tax["categories"]]
            cfg["category_rules"] = active_tax.get("global_rules", "")
    except Exception:  # pragma: no cover - defensive
        pass
    cfg["scoring"] = {
        "ovr_min": _int_env("VISION_OVR_MIN_SCORE", 0),
        "rel_min": _int_env("VISION_REL_MIN_SCORE", 0),
        "notes": os.environ.get("VISION_SCORE_NOTES", "") or "",
    }
    cfg["captioning"] = {
        "enabled": _bool_env("AUTO_CAPTION_ENABLED", False),
        "style": os.environ.get("AUTO_CAPTION_STYLE", "sd_prompt") or "sd_prompt",
        "overwrite": _bool_env("AUTO_CAPTION_OVERWRITE", False),
    }
    return cfg


def migrate_env_to_default_job() -> Job | None:
    """If no jobs exist, seed the default preset and build a job from the current
    environment, adopting PIPELINE_SLUG so existing data dirs are inherited. The
    env config becomes the job's overrides (effective == legacy behaviour).
    Idempotent."""
    if list_jobs():
        return None
    list_presets()                              # seed default preset
    slug = (os.environ.get("PIPELINE_SLUG", "").strip() or "default")
    if not JOB_SLUG_RE.match(slug):
        slug = slugify(slug) or "default"
    subject = os.environ.get("PIPELINE_TOPIC", "").strip() or _titleize(slug)
    saved = save_job(_make_job(slug, subject, subject=subject, preset="default", overrides=_env_to_cfg()))
    set_active(slug)
    logger.info("migrated legacy .env into job %r (active)", slug)
    return saved


def _discovery_root(env_key: str, default_name: str) -> Path:
    raw = os.environ.get(env_key, "").strip()
    p = Path(raw) if raw else (paths.base_dir() / default_name)
    if p.name == default_name:
        return p
    if p.parent.name == default_name:
        return p.parent
    logger.warning("%s=%r doesn't match expected '.../%s' layout; discovery may be inaccurate",
                   env_key, raw, default_name)
    return p


def discover_data_slugs() -> list[str]:
    slugs: set[str] = set()
    roots = {
        _discovery_root("PIPELINE_QUEUE", "queue"),
        _discovery_root("PIPELINE_SORTED", "sorted"),
        paths.base_dir() / "queue", paths.base_dir() / "sorted",
    }
    for root in roots:
        try:
            if root.is_dir():
                for child in root.iterdir():
                    if child.is_dir() and not child.name.startswith("."):
                        slugs.add(child.name)
        except OSError:
            continue
    return sorted(slugs)


def migrate_existing_data() -> list[Job]:
    created: list[Job] = []
    env_job = migrate_env_to_default_job()
    if env_job is not None:
        created.append(env_job)
    existing = {j.slug for j in list_jobs()}
    for slug in discover_data_slugs():
        if slug in existing or not JOB_SLUG_RE.match(slug):
            continue
        created.append(save_job(_make_job(slug, _titleize(slug), preset="default")))
        existing.add(slug)
        logger.info("adopted existing on-disk data as job %r", slug)
    if get_active_slug() is None:
        primary = os.environ.get("PIPELINE_SLUG", "").strip() or "default"
        if get_job(primary) is not None:
            set_active(primary)
        elif created:
            set_active(created[0].slug)
    return created


def _legacy_vision_workers_from_env() -> list[dict]:
    """Build fleet instances from the legacy single-endpoint vision env vars."""
    out: list[dict] = []

    def add(provider: str, name: str, url_key: str, model_key: str,
            key_key: str | None = None) -> None:
        url = (os.environ.get(url_key, "") or "").strip()
        if not url:
            return
        out.append({
            "id": re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or provider,
            "name": name, "provider": provider, "base_url": url,
            "model": (os.environ.get(model_key, "") or "").strip(),
            "api_key": (os.environ.get(key_key, "") or "").strip() if key_key else "",
            "enabled": True,
        })

    add("lmstudio", "LM Studio Primary", "LMSTUDIO_PRIMARY_URL", "LMSTUDIO_PRIMARY_MODEL")
    add("lmstudio", "LM Studio Secondary", "LMSTUDIO_SECONDARY_URL", "LMSTUDIO_SECONDARY_MODEL")
    add("ollama", "Ollama", "OLLAMA_URL", "OLLAMA_MODEL")
    add("llamacpp", "OpenAI-compatible", "OPENAI_COMPAT_URL", "OPENAI_COMPAT_MODEL",
        "OPENAI_COMPAT_API_KEY")
    return out


def migrate_legacy_vision_to_fleet() -> bool:
    """One-shot: fold legacy single-endpoint vision env (LMSTUDIO_*/OLLAMA_*/
    OPENAI_COMPAT_*) into the default preset's ``vision.workers`` fleet so an
    upgrade keeps the user's working endpoints. Idempotent — only acts while the
    default preset still carries the shipped localhost default. Returns True when
    it changed the fleet."""
    legacy = _legacy_vision_workers_from_env()
    if not legacy:
        return False
    default_name = default_preset_name()
    cfg = get_preset(default_name)
    current = (cfg.get("vision") or {}).get("workers")
    if clean_vision_fleet(current) != clean_vision_fleet(_default_vision_workers()):
        return False                       # user already configured a fleet
    cfg.setdefault("vision", {})["workers"] = legacy
    save_preset(default_name, cfg)
    logger.info("migrated %d legacy vision endpoint(s) into the %r preset fleet",
                len(legacy), default_name)
    return True


__all__ = [
    "Job", "JOB_SLUG_RE", "PRESET_NAME_RE", "JOB_STATUSES", "SCRAPER_NAMES",
    "PRIORITY_EXTRA_NAMES", "PRIORITY_NAMES",
    "PRIORITY_WEIGHT_MIN", "PRIORITY_WEIGHT_MAX", "PRIORITY_WEIGHT_DEFAULT",
    "slugify", "jobs_dir",
    "list_presets", "get_preset", "save_preset", "delete_preset",
    "set_default_preset", "default_preset_name",
    "effective_config", "is_overridden", "set_override", "reset_override",
    "get_job", "list_jobs", "save_job", "create_job", "delete_job",
    "get_index", "get_active_slug", "get_active_slugs", "set_active", "set_active_slugs",
    "deactivate", "set_job_priority", "get_job_priority",
    "set_queue", "enqueue", "dequeue", "advance",
    "resolve_env", "project_categories", "activate", "clean_scraper_priority",
    "migrate_env_to_default_job", "discover_data_slugs", "migrate_existing_data",
    "migrate_legacy_vision_to_fleet", "clean_vision_fleet", "VISION_PROVIDERS",
    "reset_preset_to_builtin", "builtin_preset_names",
]
