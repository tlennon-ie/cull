"""Single source of truth for filesystem paths.

Computes repo-relative defaults so a fresh clone runs without editing `.env`.
Every hardcoded absolute-path default elsewhere is gone - this module
replaces them.

Admin overrides flow from the dashboard Settings tab → writes to `.env` →
these helpers re-read them each call, so updates take effect on pipeline restart.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

# pipeline_code/ sits under the repo root; the repo root is where .env lives.
REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR: Path = REPO_ROOT / "data"

# Job-slug charset (mirrors ``job_config.JOB_SLUG_RE``). It deliberately
# excludes the path separators ``/`` and ``\`` and the dot ``.``, so a string
# that passes this gate cannot contain a directory traversal component. That
# makes ``validate_slug`` a path-injection *barrier* the CodeQL py/path-injection
# query recognises: every slug→filesystem-path flow is sanitised here first.
_SAFE_SLUG_RE = re.compile(r"^[a-z0-9_]+$")


def validate_slug(slug: str) -> str:
    """Return ``slug`` iff it matches the safe job-slug charset; else raise.

    The charset (``^[a-z0-9_]+$``) excludes ``/``, ``\\`` and ``.``, so the
    returned value can never contain a path separator or ``..`` traversal — this
    is the shared sanitiser callers run a slug (or any user-derived key) through
    immediately before using it to build a filesystem path.

    Raises:
        ValueError: when ``slug`` is empty or contains any character outside the
            safe charset.
    """
    s = str(slug or "")
    if not _SAFE_SLUG_RE.fullmatch(s):
        raise ValueError(f"invalid job slug: {slug!r}")
    return s


def base_dir() -> Path:
    """Parent that holds queue/, sorted/, logs/.

    Override via `PIPELINE_BASE_DIR`. Default: `<repo>/data`.
    """
    raw = os.environ.get("PIPELINE_BASE_DIR", "").strip()
    return Path(raw) if raw else DEFAULT_DATA_DIR


def _normalised(env_key: str, default_name: str, slug: str) -> Path:
    """Resolve a per-slug directory from env, tolerating both ".../<name>" and
    ".../<name>/<slug>" styles so dashboard-edited values Just Work."""
    raw = os.environ.get(env_key, "").strip()
    root = Path(raw) if raw else (base_dir() / default_name)
    return root if root.name == slug else root / slug


def queue_dir(slug: str | None = None) -> Path:
    slug = slug or os.environ.get("PIPELINE_SLUG", "default")
    return _normalised("PIPELINE_QUEUE", "queue", slug)


def sorted_dir(slug: str | None = None) -> Path:
    slug = slug or os.environ.get("PIPELINE_SLUG", "default")
    return _normalised("PIPELINE_SORTED", "sorted", slug)


def log_dir() -> Path:
    raw = os.environ.get("LOG_DIR", "").strip()
    return Path(raw) if raw else (base_dir() / "logs")


def jobs_dir() -> Path:
    """Directory holding per-job config bundles (`<slug>.json`) and `_index.json`.

    Always under the global base dir — jobs are a global registry, not per-slug.
    """
    return base_dir() / "jobs"


def queue_root() -> Path:
    """Parent of all per-slug queue folders (i.e. without the slug suffix)."""
    raw = os.environ.get("PIPELINE_QUEUE", "").strip()
    return Path(raw) if raw else (base_dir() / "queue")


def sorted_root() -> Path:
    raw = os.environ.get("PIPELINE_SORTED", "").strip()
    return Path(raw) if raw else (base_dir() / "sorted")


def local_import_dir() -> Path | None:
    """Optional admin-configured folder to mirror into the queue.

    Returns None if `LOCAL_IMPORT_DIR` isn't set. Enables generic local-folder
    ingestion.
    """
    raw = os.environ.get("LOCAL_IMPORT_DIR", "").strip()
    return Path(raw) if raw else None


def local_import_name() -> str:
    """Label / source name for items imported from `LOCAL_IMPORT_DIR`."""
    return os.environ.get("LOCAL_IMPORT_NAME", "local").strip() or "local"


__all__ = [
    "REPO_ROOT",
    "DEFAULT_DATA_DIR",
    "validate_slug",
    "base_dir",
    "queue_dir",
    "sorted_dir",
    "queue_root",
    "sorted_root",
    "log_dir",
    "jobs_dir",
    "local_import_dir",
    "local_import_name",
]
