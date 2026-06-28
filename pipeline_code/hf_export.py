"""hf_export.py - push a curated cull dataset to a PRIVATE HuggingFace dataset repo.

cull sorts kept images into ``data/sorted/<slug>/<category>/<source>/`` next to a
``<name>.txt`` caption and a ``<name>.vision.json`` audit record. HuggingFace's
dataset loaders (``datasets.load_dataset("imagefolder", ...)`` /
``"videofolder"``) want a flat-ish media folder plus a ``metadata.jsonl`` whose
``file_name`` column links each row to its media file, with any extra columns
(caption, category, score, ...) riding alongside. See
https://huggingface.co/docs/hub/en/datasets-image .

This module bridges the two:

  1. read the active taxonomy (``categories``) and export ONLY keep buckets —
     terminal ``DISCARD`` / ``CORRUPT`` folders are never published;
  2. COPY each kept sample (and, when ``include_video`` is set, video files) into
     a throwaway staging dir and write ``metadata.jsonl``;
  3. ``HfApi.create_repo(..., repo_type="dataset", private=...)`` then
     ``upload_folder(...)`` — all network calls live behind a gated import so the
     module imports cleanly even when ``huggingface_hub`` is not installed.

The HF token is read from the call argument, then ``HF_TOKEN`` /
``HUGGINGFACE_TOKEN`` in the environment (via ``credentials``). It is handed to
``HfApi`` in memory only and is NEVER written to disk.

Conventions honoured:
  * paths come from ``paths.sorted_dir(slug)`` — no hardcoded absolute paths;
  * the kept-bucket / terminal split comes from ``categories`` — no inlined list;
  * the token lookup goes through ``credentials`` like every scraper.

Ad-hoc run (after ``pip install huggingface_hub`` and exporting ``HF_TOKEN``)::

    python pipeline_code/hf_export.py --slug default --repo me/my-dataset
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator

import categories
import credentials
from paths import sorted_dir
from pipeline_logging import get_logger

logger = get_logger(__name__)

# Media extensions we recognise when staging a folder. Images always export;
# videos only when ``include_video=True`` is passed to ``push_to_hf``.
IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"}
)
VIDEO_EXTENSIONS: frozenset[str] = frozenset(
    {".mp4", ".mov", ".webm", ".mkv", ".avi"}
)

# Sidecar suffixes that are NOT samples themselves — used to skip them while
# walking the sorted tree.
_SIDECAR_SUFFIXES: tuple[str, ...] = (".txt", ".vision.json", ".meta.json")

METADATA_FILENAME = "metadata.jsonl"


# ── token resolution ─────────────────────────────────────────────────────────

def _resolve_token(token: str | None) -> str:
    """Return the HF token to authenticate with.

    Order of precedence:
      1. the explicit ``token`` argument (already in memory);
      2. ``HF_TOKEN`` in the environment;
      3. ``HUGGINGFACE_TOKEN`` in the environment.

    Raises ``credentials.MissingCredentialError`` (a ``SystemExit`` subclass)
    with a clear message if none is set. The token is never persisted.
    """
    if token and token.strip():
        return token.strip()
    env_token = credentials.get_keylist("HF_TOKEN", "HUGGINGFACE_TOKEN")
    if env_token:
        return env_token
    # Surfaces a consistent, supervisor-aware error message naming HF_TOKEN.
    return credentials.get_required("HF_TOKEN", scraper="hf_export")


# ── category selection ───────────────────────────────────────────────────────

def _kept_categories(categories_arg: Iterable[str] | None) -> list[str]:
    """Resolve which category folders to export.

    Always intersects with the active taxonomy's keep buckets
    (``categories.get_categories()``) so terminal buckets (DISCARD / CORRUPT)
    can never leak in, even if the caller passes them explicitly.

    ``None`` => every keep bucket. A list => that subset (terminal entries
    silently dropped). Order follows the taxonomy for determinism.
    """
    keep = list(categories.get_categories())
    if categories_arg is None:
        return keep
    requested = {c.strip() for c in categories_arg if c and c.strip()}
    return [c for c in keep if c in requested]


# ── staging ──────────────────────────────────────────────────────────────────

def _allowed_extensions(include_video: bool) -> frozenset[str]:
    return IMAGE_EXTENSIONS | VIDEO_EXTENSIONS if include_video else IMAGE_EXTENSIONS


def _is_sidecar(path: Path) -> bool:
    """True for the .txt / .vision.json / .meta.json companions of a sample."""
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in _SIDECAR_SUFFIXES)


def _read_caption(image_path: Path) -> str:
    """Read the sibling ``<stem>.txt`` caption, or '' when absent/unreadable."""
    txt_path = image_path.with_suffix(".txt")
    if not txt_path.exists():
        return ""
    try:
        return txt_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        logger.warning("could not read caption %s: %s", txt_path, exc)
        return ""


def _read_score(image_path: Path) -> int | float | None:
    """Mine a numeric score from the sibling ``<stem>.vision.json``.

    Prefers ``OVR_Quality_Score`` (topic-independent craft quality), then
    ``REL_Quality_Score``, then the legacy ``quality_score``. Returns ``None``
    when no audit record exists or it is unparseable.
    """
    vision_path = image_path.with_suffix(".vision.json")
    if not vision_path.exists():
        return None
    try:
        record = json.loads(vision_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("could not read audit record %s: %s", vision_path, exc)
        return None
    if not isinstance(record, dict):
        return None
    for key in ("OVR_Quality_Score", "REL_Quality_Score", "quality_score"):
        value = record.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
    return None


def _read_source(image_path: Path, sorted_root: Path) -> str:
    """Derive the source name from the sorted layout.

    Layout is ``<sorted>/<category>/<source>/<name>``, so the source is the
    immediate parent folder of the image. Falls back to '' if the path is not
    nested as expected.
    """
    try:
        rel = image_path.relative_to(sorted_root)
    except ValueError:
        return ""
    # rel.parts == (category, source, filename) in the canonical layout.
    return rel.parts[1] if len(rel.parts) >= 3 else ""


def _iter_samples(
    sorted_root: Path,
    kept: list[str],
    include_video: bool,
) -> Iterator[tuple[Path, str]]:
    """Yield ``(image_path, category)`` for every exportable media file under
    the kept category folders. Sidecars are skipped."""
    allowed = _allowed_extensions(include_video)
    for category in kept:
        category_dir = sorted_root / category
        if not category_dir.is_dir():
            continue
        for path in sorted(category_dir.rglob("*")):
            if not path.is_file() or _is_sidecar(path):
                continue
            if path.suffix.lower() in allowed:
                yield path, category


def _stage_dataset(
    slug: str,
    staging_dir: Path,
    kept: list[str],
    include_video: bool,
) -> list[dict[str, Any]]:
    """COPY every kept sample into ``staging_dir`` and write ``metadata.jsonl``.

    Returns the list of metadata rows (one per staged file). The originals under
    ``data/sorted`` are never moved or modified. ``file_name`` values are POSIX
    relative paths so the HF dataset viewer / loader resolves them.
    """
    sorted_root = sorted_dir(slug)
    staging_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    used_names: set[str] = set()

    for image_path, category in _iter_samples(sorted_root, kept, include_video):
        # Lay files out under data/<category>/ in staging to mirror a clean
        # imagefolder/videofolder split, de-duplicating names defensively.
        rel_name = _unique_relpath(category, image_path.name, used_names)
        dest = staging_dir / rel_name
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(image_path, dest)
        except OSError as exc:
            logger.warning("skipping %s: copy failed (%s)", image_path, exc)
            continue

        row: dict[str, Any] = {
            "file_name": rel_name,
            "caption": _read_caption(image_path),
            "category": category,
        }
        score = _read_score(image_path)
        if score is not None:
            row["score"] = score
        source = _read_source(image_path, sorted_root)
        if source:
            row["source"] = source
        rows.append(row)

    _write_metadata(staging_dir, rows)
    return rows


def _unique_relpath(category: str, filename: str, used: set[str]) -> str:
    """Return a POSIX-relative ``<category>/<filename>`` unique within ``used``."""
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    candidate = f"{category}/{filename}"
    counter = 1
    while candidate in used:
        candidate = f"{category}/{stem}_{counter}{suffix}"
        counter += 1
    used.add(candidate)
    return candidate


def _write_metadata(staging_dir: Path, rows: list[dict[str, Any]]) -> None:
    """Write the JSON-Lines metadata file that HF datasets read."""
    meta_path = staging_dir / METADATA_FILENAME
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    meta_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


# ── HuggingFace upload (gated import) ────────────────────────────────────────

def _load_hf_api(token: str):
    """Import ``huggingface_hub`` lazily and return an authenticated ``HfApi``.

    The import is gated so the module loads without the package; the actual
    failure (with a clear, actionable message) only happens here, at upload
    time. Re-importing on each call lets tests inject a fake module.
    """
    try:
        import huggingface_hub  # noqa: WPS433 (intentional lazy import)
    except ImportError as exc:  # pragma: no cover - exercised via sys.modules=None
        raise RuntimeError(
            "huggingface_hub is required to push datasets. "
            "Install it with `pip install huggingface_hub` "
            "(declare it in requirements.txt)."
        ) from exc
    # A monkeypatched ``sys.modules['huggingface_hub'] = None`` surfaces here as
    # a module object that is falsy/None rather than an ImportError.
    if huggingface_hub is None or not hasattr(huggingface_hub, "HfApi"):
        raise RuntimeError(
            "huggingface_hub is not importable. "
            "Install it with `pip install huggingface_hub`."
        )
    return huggingface_hub.HfApi(token=token)


def _upload(
    api: Any,
    repo_id: str,
    folder_path: Path,
    private: bool,
    token: str,
) -> None:
    """Create the dataset repo (idempotently) then upload the staged folder."""
    api.create_repo(
        repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
        token=token,
    )
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(folder_path),
        token=token,
        commit_message="cull: dataset export",
    )


# ── public API ───────────────────────────────────────────────────────────────

def push_to_hf(
    slug: str,
    repo_id: str,
    private: bool = True,
    include_video: bool = False,
    token: str | None = None,
    categories: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Push the curated dataset for ``slug`` to a HuggingFace dataset repo.

    Args:
        slug: the cull job slug whose ``data/sorted/<slug>`` tree to export.
        repo_id: the target ``namespace/name`` dataset repo.
        private: create the repo as private (default True).
        include_video: also export video files (.mp4/.mov/.webm/.mkv/.avi).
        token: HF token; falls back to ``HF_TOKEN`` / ``HUGGINGFACE_TOKEN`` env.
        categories: restrict the export to these keep buckets (terminal buckets
            are always excluded). ``None`` exports every keep bucket.

    Returns a result dict::

        {"repo_id", "repo_type", "private", "slug", "uploaded", "categories"}

    Raises:
        credentials.MissingCredentialError: no token available.
        ValueError: no exportable samples were found.
        RuntimeError: ``huggingface_hub`` is not installed.
    """
    # 1. Resolve the token FIRST so we fail fast and never stage on a misconfig.
    resolved_token = _resolve_token(token)

    kept = _kept_categories(categories)
    if not kept:
        raise ValueError(
            "No keep categories selected for export (after excluding "
            "terminal buckets). Nothing to push."
        )

    # 2. Stage into a throwaway temp dir; always clean it up. The token is never
    #    written here — only the copied media + metadata.jsonl.
    staging_dir = Path(tempfile.mkdtemp(prefix=f"cull_hf_{slug}_"))
    try:
        rows = _stage_dataset(slug, staging_dir, kept, include_video)
        if not rows:
            raise ValueError(
                f"No samples found under data/sorted/{slug} for categories "
                f"{kept}. Nothing to push."
            )

        # 3. Upload behind the gated import.
        api = _load_hf_api(resolved_token)
        _upload(api, repo_id, staging_dir, private, resolved_token)

        logger.info(
            "pushed %d samples from %s to %s (private=%s)",
            len(rows), slug, repo_id, private,
        )
        return {
            "repo_id": repo_id,
            "repo_type": "dataset",
            "private": private,
            "slug": slug,
            "uploaded": len(rows),
            "categories": kept,
        }
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)


# ── CLI ──────────────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Push a curated cull dataset to a private HuggingFace dataset repo.",
    )
    parser.add_argument("--slug", default="default", help="cull job slug to export")
    parser.add_argument("--repo", required=True, help="target namespace/name dataset repo")
    parser.add_argument(
        "--public",
        action="store_true",
        help="create the repo as public (default: private)",
    )
    parser.add_argument(
        "--include-video",
        action="store_true",
        help="also export video files",
    )
    parser.add_argument(
        "--categories",
        default=None,
        help="comma-separated subset of keep buckets to export",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    cats = (
        [c for c in args.categories.split(",") if c.strip()]
        if args.categories
        else None
    )
    try:
        result = push_to_hf(
            slug=args.slug,
            repo_id=args.repo,
            private=not args.public,
            include_video=args.include_video,
            categories=cats,
        )
    except (ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", flush=True)
        return 1
    print(json.dumps(result, indent=2), flush=True)
    return 0


__all__ = [
    "push_to_hf",
    "IMAGE_EXTENSIONS",
    "VIDEO_EXTENSIONS",
    "METADATA_FILENAME",
]


if __name__ == "__main__":
    sys.exit(main())
