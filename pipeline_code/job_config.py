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
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import categories
import paths
from pipeline_logging import get_logger

logger = get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

JOB_SLUG_RE = re.compile(r"^[a-z0-9_]+$")
PRESET_NAME_RE = re.compile(r"^[A-Za-z0-9 _-]{1,40}$")
JOB_STATUSES: tuple[str, ...] = ("idle", "queued", "running", "paused", "done")

# Canonical scraper names (single source of truth shared with the dashboard).
# ZFF-Local is gone — local folders are managed via scrapers.local_imports.
SCRAPER_NAMES: tuple[str, ...] = (
    "X.com", "Discord-1", "Civitai-Com", "Civitai-Red", "Web", "Gallery-DL",
)

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
    if not JOB_SLUG_RE.match(slug):
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
                            "cookies_file": "", "config_path": ""},
            "local_imports": [],
        },
        "categories": cats,
        "category_rules": rules,
        "scoring": {"ovr_min": 0, "rel_min": 0, "notes": ""},
        "captioning": {"enabled": False, "style": "sd_prompt", "overwrite": False},
    }


# ── Preset library ───────────────────────────────────────────────────────────

def _read_presets() -> dict:
    p = _presets_path()
    if p.is_file():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("presets"), dict) and data["presets"]:
                data.setdefault("default", next(iter(data["presets"])))
                return data
        except (OSError, json.JSONDecodeError):
            logger.warning("presets file %s is unreadable; using built-in defaults", p)
    return {"default": "default", "presets": {"default": _default_preset_cfg()}}


def _write_presets(lib: dict) -> None:
    _atomic_write_json(_presets_path(), lib)


def list_presets() -> dict:
    lib = _read_presets()
    if not _presets_path().is_file():
        _write_presets(lib)                    # seed on first access
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
    zf = v1.get("zforfree")
    if isinstance(zf, dict) and zf.get("local_src"):
        local_imports.append({
            "name": "zforfree", "dir": zf.get("local_src", "") or "",
            "enabled": bool(zf.get("local_enabled", False)), "migrate_from": "",
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
    if not JOB_SLUG_RE.match(slug or ""):
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
    order = ([idx["active"]] if idx["active"] else []) + list(idx["queue"])
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
    if slug == get_active_slug():
        raise ValueError(f"cannot delete the active job: {slug}")
    path = _job_path(slug)
    if path.is_file():
        path.unlink()
    idx = get_index()
    if slug in idx["queue"]:
        _save_index({**idx, "queue": [s for s in idx["queue"] if s != slug]})


# ── Index: queue order + active pointer ──────────────────────────────────────

def get_index() -> dict:
    p = _index_path()
    if not p.is_file():
        return {"active": None, "queue": []}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"active": None, "queue": []}
    if not isinstance(data, dict):
        return {"active": None, "queue": []}
    data.setdefault("active", None)
    data.setdefault("queue", [])
    if not isinstance(data["queue"], list):
        data["queue"] = []
    return data


def _save_index(idx: dict) -> None:
    _atomic_write_json(_index_path(), {**idx, "updated_at": _now_iso()})


def get_active_slug() -> str | None:
    return get_index()["active"]


def set_active(slug: str | None) -> None:
    if slug is not None and get_job(slug) is None:
        raise ValueError(f"unknown job: {slug}")
    idx = get_index()
    queue = [s for s in idx["queue"] if s != slug] if slug is not None else idx["queue"]
    _save_index({**idx, "active": slug, "queue": queue})


def set_queue(order: list[str]) -> None:
    for slug in order:
        if get_job(slug) is None:
            raise ValueError(f"unknown job: {slug}")
    _save_index({**get_index(), "queue": list(order)})


def enqueue(slug: str) -> None:
    if get_job(slug) is None:
        raise ValueError(f"unknown job: {slug}")
    idx = get_index()
    if slug != idx["active"] and slug not in idx["queue"]:
        _save_index({**idx, "queue": [*idx["queue"], slug]})


def dequeue(slug: str) -> None:
    idx = get_index()
    if slug in idx["queue"]:
        _save_index({**idx, "queue": [s for s in idx["queue"] if s != slug]})


def advance() -> str | None:
    idx = get_index()
    queue = list(idx["queue"])
    new_active: str | None = None
    while queue:
        candidate = queue.pop(0)
        if get_job(candidate) is not None:
            new_active = candidate
            break
        logger.warning("advance: skipping orphaned queued slug %r", candidate)
    _save_index({**idx, "active": new_active, "queue": queue})
    if new_active is not None:
        job = get_job(new_active)
        if job is not None:
            project_categories(job)
    return new_active


# ── Projection: job → runtime contracts ──────────────────────────────────────

def _b(v: Any) -> str:
    return "true" if v else "false"


def _csv(xs: Any) -> str:
    if not xs:
        return ""
    if isinstance(xs, str):
        return xs
    return ",".join(str(x) for x in xs)


def resolve_env(job: Job) -> dict[str, str]:
    """Flatten the EFFECTIVE config into the existing env-var names. Every key is
    always emitted so a job fully overrides any stale value in the global .env."""
    eff = effective_config(job)

    def _d(v: Any) -> dict:
        return v if isinstance(v, dict) else {}

    t = _d(eff.get("topic"))
    s = _d(eff.get("scrapers"))
    gd = _d(s.get("gallery_dl"))
    sc = _d(eff.get("scoring"))
    cap = _d(eff.get("captioning"))
    enabled = _d(s.get("enabled"))
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
        "GALLERY_DL_COOKIES_FILE": str(gd.get("cookies_file", "") or ""),
        "GALLERY_DL_CONFIG_PATH": str(gd.get("config_path", "") or ""),
        "LOCAL_IMPORTS_JSON": json.dumps(local),
        "VISION_OVR_MIN_SCORE": str(int(sc.get("ovr_min", 0) or 0)),
        "VISION_REL_MIN_SCORE": str(int(sc.get("rel_min", 0) or 0)),
        "VISION_SCORE_NOTES": str(sc.get("notes", "") or ""),
        "AUTO_CAPTION_ENABLED": _b(cap.get("enabled", False)),
        "AUTO_CAPTION_STYLE": str(cap.get("style", "sd_prompt") or "sd_prompt"),
        "AUTO_CAPTION_OVERWRITE": _b(cap.get("overwrite", False)),
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


def activate(slug: str) -> None:
    job = get_job(slug)
    if job is None:
        raise ValueError(f"unknown job: {slug}")
    set_active(slug)
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
    if os.environ.get("ZFORFREE_LOCAL_SRC"):
        local_imports.append({
            "name": "zforfree", "dir": os.environ.get("ZFORFREE_LOCAL_SRC", "") or "",
            "enabled": _bool_env("ZFORFREE_LOCAL_ENABLED", False), "migrate_from": "",
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


__all__ = [
    "Job", "JOB_SLUG_RE", "PRESET_NAME_RE", "JOB_STATUSES", "SCRAPER_NAMES",
    "slugify", "jobs_dir",
    "list_presets", "get_preset", "save_preset", "delete_preset",
    "set_default_preset", "default_preset_name",
    "effective_config", "is_overridden", "set_override", "reset_override",
    "get_job", "list_jobs", "save_job", "create_job", "delete_job",
    "get_index", "get_active_slug", "set_active", "set_queue", "enqueue", "dequeue", "advance",
    "resolve_env", "project_categories", "activate",
    "migrate_env_to_default_job", "discover_data_slugs", "migrate_existing_data",
]
