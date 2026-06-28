"""Per-job configuration — the jobs model keystone.

A **Job** is a named curation target with a self-contained config bundle stored
at ``data/jobs/<slug>.json``. The ordered queue + active pointer live in
``data/jobs/_index.json``. ``slug`` is the identity and the namespace key reused
by the queue / sorted / seen / logs / index layers that are *already* per-slug.

The unifying idea is **projection**: a job is the source of truth, and
*activating* it projects its config down into the two contracts the runtime
already consumes —

1. env vars (``resolve_env``) merged over the global ``.env`` when the supervisor
   spawns scrapers + vision workers (so worker code is unchanged), and
2. the active taxonomy file ``cull_categories.json`` (``project_categories``),
   which ``categories.py`` hot-reloads and ``vision_prompt`` builds the schema from.

Global concerns (credentials, model endpoints) stay in ``.env`` and are NOT
stored here — job files never hold secrets and are safe to share.

See ``docs/jobs-model-design.md`` for the full contract.
"""
from __future__ import annotations

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
JOB_STATUSES: tuple[str, ...] = ("idle", "queued", "running", "paused", "done")

# Canonical scraper names — the single source of truth shared with the dashboard
# Scrapers tab. Keep in sync with dashboard ``_STATIC_SCRAPERS``.
SCRAPER_NAMES: tuple[str, ...] = (
    "X.com", "Discord-1", "Civitai-Com", "Civitai-Red", "Web", "ZFF-Local", "Gallery-DL",
)

_INDEX_FILENAME = "_index.json"


# ── Slug helpers ─────────────────────────────────────────────────────────────

def slugify(name: str) -> str:
    """Lowercase, collapse non-alphanumerics to underscores, trim. Mirrors the
    ``run_pipeline.topic_slug`` rule so slugs match historical data dirs."""
    return re.sub(r"[^a-z0-9]+", "_", (name or "").lower()).strip("_")


def _titleize(slug: str) -> str:
    return slug.replace("_", " ").title()


# ── Time / IO helpers ────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively overlay ``override`` onto ``base``. A ``None`` leaf is skipped
    (keeps the base default) so partial/old job files forward-merge cleanly;
    an explicit ``[]`` or ``""`` is a real value and DOES replace the default."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        elif v is not None:
            out[k] = v
    return out


# ── Defaults ─────────────────────────────────────────────────────────────────

def _default_categories() -> tuple[list[dict], str]:
    """Categories + global rules for a brand-new job: the shipped default preset."""
    try:
        presets = categories.get_presets()
        name = presets.get("default") or next(iter(presets["presets"]))
        preset = presets["presets"][name]
        return [dict(c) for c in preset.get("categories", [])], preset.get("global_rules", "")
    except Exception:  # pragma: no cover - defensive: never block job creation
        return [], ""


def _default_job_data(name: str, slug: str) -> dict:
    cats, rules = _default_categories()
    return {
        "topic": {
            "topic": name,
            "keywords_extra": [],
            "banned_keywords": [],
            "generation_hints": [],
            "min_prompt_length": 0,
            "require_prompt": True,
        },
        "scrapers": {
            "enabled": {n: True for n in SCRAPER_NAMES},
            "x_accounts": [],
            "reddit_subreddits": [],
            "discord_channels_json": "",
            "civitai_domains": [],
            "gallery_dl": {
                "enabled": False, "urls": [], "limit_per_url": 200,
                "cookies_file": "", "config_path": "",
            },
            "local_import": {"enabled": False, "dir": "", "name": "local", "migrate_from": ""},
            "zforfree": {"local_enabled": False, "web_enabled": False, "local_src": ""},
        },
        "categories": cats,
        "category_rules": rules,
        "scoring": {"ovr_min": 0, "rel_min": 0, "notes": ""},
        "captioning": {"enabled": False, "style": "sd_prompt", "overwrite": False},
    }


# ── Job model ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Job:
    slug: str
    name: str
    status: str
    created_at: str
    updated_at: str
    topic: dict = field(default_factory=dict)
    scrapers: dict = field(default_factory=dict)
    categories: list = field(default_factory=list)
    category_rules: str = ""
    scoring: dict = field(default_factory=dict)
    captioning: dict = field(default_factory=dict)

    def with_updates(self, **changes: Any) -> "Job":
        """Return a new Job with ``changes`` applied (immutable update)."""
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> dict:
        return {
            "slug": self.slug,
            "name": self.name,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "topic": self.topic,
            "scrapers": self.scrapers,
            "categories": self.categories,
            "category_rules": self.category_rules,
            "scoring": self.scoring,
            "captioning": self.captioning,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Job":
        slug = d.get("slug", "")
        name = d.get("name", "") or _titleize(slug)
        merged = _deep_merge(_default_job_data(name, slug), {
            k: v for k, v in d.items()
            if k in ("topic", "scrapers", "categories", "category_rules", "scoring", "captioning")
        })
        return cls(
            slug=slug,
            name=name,
            status=d.get("status", "idle"),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
            topic=merged["topic"],
            scrapers=merged["scrapers"],
            categories=merged["categories"],
            category_rules=merged["category_rules"],
            scoring=merged["scoring"],
            captioning=merged["captioning"],
        )


def _make_job(slug: str, name: str, *, status: str = "idle", base_data: dict | None = None) -> Job:
    now = _now_iso()
    data = base_data if base_data is not None else _default_job_data(name, slug)
    return Job(
        slug=slug, name=name, status=status, created_at=now, updated_at=now,
        topic=data["topic"], scrapers=data["scrapers"], categories=data["categories"],
        category_rules=data["category_rules"], scoring=data["scoring"], captioning=data["captioning"],
    )


# ── Paths ────────────────────────────────────────────────────────────────────

def jobs_dir() -> Path:
    return paths.jobs_dir()


def _job_path(slug: str) -> Path:
    """Path to a job file. Rejects any slug that isn't ``[a-z0-9_]+`` so a
    malicious/garbled slug can never escape ``jobs_dir()`` (path traversal)."""
    if not JOB_SLUG_RE.match(slug):
        raise ValueError(f"invalid slug: {slug!r}")
    return jobs_dir() / f"{slug}.json"


def _index_path() -> Path:
    return jobs_dir() / _INDEX_FILENAME


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
        return None                      # malformed lookup is a clean miss
    path = _job_path(slug)
    if not path.is_file():
        return None
    return _read_job_file(path)


def list_jobs() -> list[Job]:
    d = jobs_dir()
    if not d.is_dir():
        return []
    jobs: list[Job] = []
    for p in sorted(d.glob("*.json")):
        if p.name == _INDEX_FILENAME:
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
    """Persist a job (atomic), bumping ``updated_at``. Returns the saved Job."""
    if not JOB_SLUG_RE.match(job.slug):
        raise ValueError(f"invalid slug: {job.slug!r}")
    saved = job.with_updates(updated_at=_now_iso())
    _atomic_write_json(_job_path(saved.slug), saved.to_dict())
    return saved


def create_job(name: str, *, base_on: str | None = None, **overrides: Any) -> Job:
    slug = slugify(name)
    if not slug:
        raise ValueError(f"name produces empty slug: {name!r}")
    if get_job(slug) is not None:
        raise ValueError(f"job already exists: {slug}")
    if base_on:
        src = get_job(base_on)
        if src is None:
            raise ValueError(f"base_on job not found: {base_on}")
        job = _make_job(slug, name, base_data={
            "topic": json.loads(json.dumps(src.topic)),
            "scrapers": json.loads(json.dumps(src.scrapers)),
            "categories": json.loads(json.dumps(src.categories)),
            "category_rules": src.category_rules,
            "scoring": json.loads(json.dumps(src.scoring)),
            "captioning": json.loads(json.dumps(src.captioning)),
        })
    else:
        job = _make_job(slug, name)
    if overrides:
        job = job.with_updates(**overrides)
    return save_job(job)


def delete_job(slug: str) -> None:
    if slug == get_active_slug():
        raise ValueError(f"cannot delete the active job: {slug}")
    path = _job_path(slug)
    if path.is_file():
        path.unlink()
    # drop from queue if present
    idx = get_index()
    if slug in idx["queue"]:
        idx["queue"] = [s for s in idx["queue"] if s != slug]
        _save_index(idx)


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
    # don't mutate the caller's dict — write a stamped copy
    _atomic_write_json(_index_path(), {**idx, "updated_at": _now_iso()})


def get_active_slug() -> str | None:
    return get_index()["active"]


def set_active(slug: str | None) -> None:
    if slug is not None and get_job(slug) is None:
        raise ValueError(f"unknown job: {slug}")
    idx = get_index()
    idx["active"] = slug
    if slug is not None:
        idx["queue"] = [s for s in idx["queue"] if s != slug]
    _save_index(idx)


def set_queue(order: list[str]) -> None:
    for slug in order:
        if get_job(slug) is None:
            raise ValueError(f"unknown job: {slug}")
    idx = get_index()
    idx["queue"] = list(order)
    _save_index(idx)


def enqueue(slug: str) -> None:
    if get_job(slug) is None:
        raise ValueError(f"unknown job: {slug}")
    idx = get_index()
    if slug != idx["active"] and slug not in idx["queue"]:
        idx["queue"].append(slug)
        _save_index(idx)


def dequeue(slug: str) -> None:
    idx = get_index()
    if slug in idx["queue"]:
        idx["queue"] = [s for s in idx["queue"] if s != slug]
        _save_index(idx)


def advance() -> str | None:
    """Promote the next *existing* job in the queue to active and return it.

    Orphaned slugs (file removed) are skipped, never promoted — a dangling
    active pointer would leave the supervisor with no config to spawn from.
    Returns None when the queue holds no live jobs.
    """
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
    """Flatten a job into the existing env-var names the runtime consumes.

    Every key is always emitted (empty string when unset) so a job's config
    fully overrides any stale value in the global ``.env`` at spawn time.
    """
    t = job.topic
    s = job.scrapers
    gd = s.get("gallery_dl", {})
    li = s.get("local_import", {})
    zf = s.get("zforfree", {})
    sc = job.scoring
    cap = job.captioning
    disabled = sorted(n for n, on in s.get("enabled", {}).items() if not on)
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
        "GALLERY_DL_LIMIT_PER_URL": str(int(gd.get("limit_per_url", 200) or 0)),
        "GALLERY_DL_COOKIES_FILE": str(gd.get("cookies_file", "") or ""),
        "GALLERY_DL_CONFIG_PATH": str(gd.get("config_path", "") or ""),
        "LOCAL_IMPORT_ENABLED": _b(li.get("enabled", False)),
        "LOCAL_IMPORT_DIR": str(li.get("dir", "") or ""),
        "LOCAL_IMPORT_NAME": str(li.get("name", "local") or "local"),
        "LOCAL_IMPORT_MIGRATE_FROM": str(li.get("migrate_from", "") or ""),
        "ZFORFREE_LOCAL_ENABLED": _b(zf.get("local_enabled", False)),
        "ZFORFREE_WEB_ENABLED": _b(zf.get("web_enabled", False)),
        "ZFORFREE_LOCAL_SRC": str(zf.get("local_src", "") or ""),
        "VISION_OVR_MIN_SCORE": str(int(sc.get("ovr_min", 0) or 0)),
        "VISION_REL_MIN_SCORE": str(int(sc.get("rel_min", 0) or 0)),
        "VISION_SCORE_NOTES": str(sc.get("notes", "") or ""),
        "AUTO_CAPTION_ENABLED": _b(cap.get("enabled", False)),
        "AUTO_CAPTION_STYLE": str(cap.get("style", "sd_prompt") or "sd_prompt"),
        "AUTO_CAPTION_OVERWRITE": _b(cap.get("overwrite", False)),
    }


def project_categories(job: Job) -> None:
    """Write the job's taxonomy into the active ``cull_categories.json`` so the
    vision workers + dashboard pick it up (they already watch that file)."""
    categories.set_active({
        "preset": "custom",
        "categories": [dict(c) for c in (job.categories or [])],
        "global_rules": job.category_rules or "",
    })


def activate(slug: str) -> None:
    """Make ``slug`` the active job and project its taxonomy. The supervisor's
    ``_index.json`` + ``cull_categories.json`` mtime watch does the rest."""
    job = get_job(slug)
    if job is None:
        raise ValueError(f"unknown job: {slug}")
    set_active(slug)
    project_categories(job)


# ── Migration ────────────────────────────────────────────────────────────────

def _csv_to_list(raw: str | None) -> list[str]:
    return [x.strip() for x in (raw or "").split(",") if x.strip()]


def _int_env(key: str, default: int = 0) -> int:
    try:
        return int(os.environ.get(key, "").strip() or default)
    except ValueError:
        return default


def _bool_env(key: str, default: bool = False) -> bool:
    raw = os.environ.get(key, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _env_job_data(name: str) -> dict:
    """Build a job config dict from the current global environment (legacy .env)."""
    data = _default_job_data(name, slugify(name) or "default")
    disabled = set(_csv_to_list(os.environ.get("SCRAPER_DISABLED")))
    active_tax = {}
    try:
        active_tax = categories.get_active()
    except Exception:  # pragma: no cover - defensive
        active_tax = {}
    urls_raw = os.environ.get("GALLERY_DL_URLS", "") or ""
    urls = [u.strip() for u in re.split(r"[\n,]", urls_raw) if u.strip() and not u.strip().startswith("#")]
    data["topic"] = {
        "topic": os.environ.get("PIPELINE_TOPIC", name) or name,
        "keywords_extra": _csv_to_list(os.environ.get("TOPIC_KEYWORDS_EXTRA")),
        "banned_keywords": _csv_to_list(os.environ.get("TOPIC_BANNED_KEYWORDS")),
        "generation_hints": _csv_to_list(os.environ.get("TOPIC_GENERATION_HINTS")),
        "min_prompt_length": _int_env("MIN_PROMPT_LENGTH", 0),
        "require_prompt": _bool_env("REQUIRE_PROMPT", True),
    }
    data["scrapers"] = {
        "enabled": {n: (n not in disabled) for n in SCRAPER_NAMES},
        "x_accounts": _csv_to_list(os.environ.get("X_ACCOUNTS")),
        "reddit_subreddits": _csv_to_list(os.environ.get("REDDIT_SUBREDDITS")),
        "discord_channels_json": os.environ.get("DISCORD_CHANNELS_JSON", "") or "",
        "civitai_domains": _csv_to_list(os.environ.get("CIVITAI_DOMAINS")),
        "gallery_dl": {
            "enabled": _bool_env("GALLERY_DL_ENABLED", False),
            "urls": urls,
            "limit_per_url": _int_env("GALLERY_DL_LIMIT_PER_URL", 200),
            "cookies_file": os.environ.get("GALLERY_DL_COOKIES_FILE", "") or "",
            "config_path": os.environ.get("GALLERY_DL_CONFIG_PATH", "") or "",
        },
        "local_import": {
            "enabled": _bool_env("LOCAL_IMPORT_ENABLED", False),
            "dir": os.environ.get("LOCAL_IMPORT_DIR", "") or "",
            "name": os.environ.get("LOCAL_IMPORT_NAME", "local") or "local",
            "migrate_from": os.environ.get("LOCAL_IMPORT_MIGRATE_FROM", "") or "",
        },
        "zforfree": {
            "local_enabled": _bool_env("ZFORFREE_LOCAL_ENABLED", False),
            "web_enabled": _bool_env("ZFORFREE_WEB_ENABLED", False),
            "local_src": os.environ.get("ZFORFREE_LOCAL_SRC", "") or "",
        },
    }
    if active_tax.get("categories"):
        data["categories"] = [dict(c) for c in active_tax["categories"]]
        data["category_rules"] = active_tax.get("global_rules", "")
    data["scoring"] = {
        "ovr_min": _int_env("VISION_OVR_MIN_SCORE", 0),
        "rel_min": _int_env("VISION_REL_MIN_SCORE", 0),
        "notes": os.environ.get("VISION_SCORE_NOTES", "") or "",
    }
    data["captioning"] = {
        "enabled": _bool_env("AUTO_CAPTION_ENABLED", False),
        "style": os.environ.get("AUTO_CAPTION_STYLE", "sd_prompt") or "sd_prompt",
        "overwrite": _bool_env("AUTO_CAPTION_OVERWRITE", False),
    }
    return data


def migrate_env_to_default_job() -> Job | None:
    """If no jobs exist yet, build one from the current environment and activate
    it. The slug ADOPTS ``PIPELINE_SLUG`` (falling back to ``default``) so the
    job inherits any existing ``data/queue/<slug>`` + ``data/sorted/<slug>``.
    Idempotent — returns None when jobs already exist.
    """
    if list_jobs():
        return None
    slug = (os.environ.get("PIPELINE_SLUG", "").strip() or "default")
    if not JOB_SLUG_RE.match(slug):
        slug = slugify(slug) or "default"
    name = os.environ.get("PIPELINE_TOPIC", "").strip() or _titleize(slug)
    job = _make_job(slug, name, base_data=_env_job_data(name))
    job = save_job(job)
    set_active(slug)
    logger.info("migrated legacy .env into job %r (active)", slug)
    return get_job(slug)


def _discovery_root(env_key: str, default_name: str) -> Path:
    """The multi-slug parent dir for queue/ or sorted/.

    ``PIPELINE_QUEUE`` / ``PIPELINE_SORTED`` may point at either the plain root
    (``.../queue``) or a slug-included dir (``.../queue/<slug>``). The true
    parent is the folder literally named ``queue`` / ``sorted``; anchor on it so
    we never mistake a slug's *category* children for slugs.
    """
    raw = os.environ.get(env_key, "").strip()
    p = Path(raw) if raw else (paths.base_dir() / default_name)
    if p.name == default_name:
        return p
    if p.parent.name == default_name:
        return p.parent
    logger.warning(
        "%s=%r doesn't match the expected '.../%s' or '.../%s/<slug>' layout; "
        "slug discovery may be inaccurate",
        env_key, raw, default_name, default_name,
    )
    return p


def discover_data_slugs() -> list[str]:
    """Slugs that already have on-disk queue/ or sorted/ folders (upgraders)."""
    slugs: set[str] = set()
    roots = {
        _discovery_root("PIPELINE_QUEUE", "queue"),
        _discovery_root("PIPELINE_SORTED", "sorted"),
        paths.base_dir() / "queue",
        paths.base_dir() / "sorted",
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
    """One-shot upgrader: ensure the env/active job exists, then adopt every
    other slug already present on disk as its own job. Idempotent — returns the
    jobs it created this run (empty when nothing new)."""
    created: list[Job] = []
    env_job = migrate_env_to_default_job()
    if env_job is not None:
        created.append(env_job)
    existing = {j.slug for j in list_jobs()}
    for slug in discover_data_slugs():
        if slug in existing or not JOB_SLUG_RE.match(slug):
            continue
        job = save_job(_make_job(slug, _titleize(slug)))
        created.append(job)
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
    "Job",
    "JOB_SLUG_RE",
    "JOB_STATUSES",
    "SCRAPER_NAMES",
    "slugify",
    "jobs_dir",
    "get_job",
    "list_jobs",
    "save_job",
    "create_job",
    "delete_job",
    "get_index",
    "get_active_slug",
    "set_active",
    "set_queue",
    "enqueue",
    "dequeue",
    "advance",
    "resolve_env",
    "project_categories",
    "activate",
    "migrate_env_to_default_job",
    "discover_data_slugs",
    "migrate_existing_data",
]
