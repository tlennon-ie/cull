"""embed_sorted.py - Backfill CLIP embeddings for a job's sorted set.

The Duplicates / semantic-search / dataset-balance features all read the
per-slug vector store written by :class:`embeddings_index.EmbeddingIndex`. That
store is normally populated incrementally as images are curated; this tool is the
**backfill**: it walks an existing ``data/sorted/<slug>/`` tree and embeds every
kept image that isn't already in the index, so those features have vectors to
work with on a dataset that predates them (or after a re-import).

Only **kept** category folders are embedded — the terminal buckets (DISCARD /
CORRUPT, per :mod:`categories`) are skipped, because there's no value in finding
"more like this" for an image you already threw away.

The heavy model stack (``torch`` + ``open_clip_torch``) lives behind the
project's optional ``[ml]`` extra and is import-guarded inside
:func:`embeddings.embed_image`. This tool checks
:func:`embeddings.open_clip_available` up front and exits with a clear, actionable
message rather than crashing deep in a worker when the extra is absent.

Run from the repo root (inside the venv)::

    python tools/embed_sorted.py --slug my_job             # embed everything new
    python tools/embed_sorted.py --slug my_job --limit 500 # cap this run
    python tools/embed_sorted.py --slug my_job --force      # re-embed all kept images

Re-running without ``--force`` is cheap: already-embedded images are skipped, so
it only ever does the new work.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_CODE = REPO_ROOT / "pipeline_code"
if str(PIPELINE_CODE) not in sys.path:
    sys.path.insert(0, str(PIPELINE_CODE))

import categories  # noqa: E402  (after sys.path insert)
import embeddings  # noqa: E402
from embeddings_index import EmbeddingIndex  # noqa: E402
from paths import sorted_dir  # noqa: E402
from paths import validate_slug as _validate_slug  # noqa: E402

# Image extensions cull curates. Mirrors feed_local_folder.IMAGE_SUFFIXES /
# hf_export.IMAGE_EXTENSIONS — the sorted tree only ever holds these for images.
IMAGE_SUFFIXES: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".webp"})

# Sidecars that sit next to every sorted image. Never embed these.
_SIDECAR_SUFFIXES: tuple[str, ...] = (".txt", ".vision.json", ".meta.json")

# How often to report progress and persist the index, in images embedded.
_FLUSH_EVERY: int = 50

# Actionable early-out when the optional ML stack is missing. Worded exactly as
# the task spec / dashboard surface expects so a user knows the one command to run.
_ML_MISSING_MSG = "Embeddings need the ML extra: pip install -e .[ml]"


def _is_sidecar(path: Path) -> bool:
    """True for the .txt / .vision.json / .meta.json companions of an image."""
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in _SIDECAR_SUFFIXES)


def iter_kept_images(slug: str) -> Iterator[Path]:
    """Yield every image file under the KEPT category folders for ``slug``.

    Layout is ``<sorted>/<category>/<source>/<name>`` (see hf_export). We only
    descend the keep buckets from :func:`categories.get_categories` — the
    terminal DISCARD / CORRUPT folders are deliberately never visited. Sidecars
    and non-image files are skipped. Ordering is deterministic (sorted) so a
    ``--limit`` run is reproducible.
    """
    # Sanitise the slug before it selects the sorted-tree root to walk.
    root = sorted_dir(_validate_slug(slug))
    for category in categories.get_categories():
        category_dir = root / category
        if not category_dir.is_dir():
            continue
        for path in sorted(category_dir.rglob("*")):
            if not path.is_file() or _is_sidecar(path):
                continue
            if path.suffix.lower() in IMAGE_SUFFIXES:
                yield path


def embed_slug(
    slug: str,
    *,
    limit: int | None = None,
    force: bool = False,
) -> int:
    """Embed the kept sorted images for ``slug`` into its EmbeddingIndex.

    Skips images already present in the index unless ``force`` is set. Flushes
    the index periodically (so a long run survives a crash) and once at the end.
    Returns the number of images newly embedded this run.
    """
    index = EmbeddingIndex(slug=slug)
    print(
        f"embed_sorted: slug={slug!r}  sorted={sorted_dir(slug)}  "
        f"already indexed={len(index)}  force={force}"
        + (f"  limit={limit}" if limit is not None else ""),
        flush=True,
    )

    embedded = 0
    skipped = 0
    failed = 0
    seen = 0

    for image_path in iter_kept_images(slug):
        if limit is not None and embedded >= limit:
            print(f"  reached --limit {limit}; stopping", flush=True)
            break
        seen += 1
        key = str(image_path)

        if not force and key in index:
            skipped += 1
            continue

        try:
            vector = embeddings.embed_image(image_path)
        except (ValueError, RuntimeError, OSError) as exc:
            # One unreadable/odd image must not abort the whole backfill.
            failed += 1
            print(f"  [skip] could not embed {key}: {exc}", flush=True)
            continue

        index.add(key, vector)
        embedded += 1
        if embedded % _FLUSH_EVERY == 0:
            index.flush()
            print(
                f"  embedded {embedded} (skipped {skipped}, failed {failed})",
                flush=True,
            )

    index.flush()
    print(
        f"done. embedded={embedded} skipped={skipped} failed={failed} "
        f"scanned={seen} index_size={len(index)}",
        flush=True,
    )
    return embedded


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill CLIP embeddings for a job's sorted set so the "
            "Duplicates / semantic-search / balance features have vectors."
        ),
    )
    parser.add_argument(
        "--slug",
        required=True,
        help="job slug whose data/sorted/<slug>/ tree to embed",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap the number of images embedded this run (default: no cap)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-embed images even if they're already in the index",
    )
    args = parser.parse_args(argv)

    # Fail fast with an actionable message when the optional ML stack is absent,
    # rather than letting embed_image() raise mid-walk.
    if not embeddings.open_clip_available():
        print(_ML_MISSING_MSG, file=sys.stderr, flush=True)
        return 1

    if args.limit is not None and args.limit < 0:
        print("--limit must be >= 0", file=sys.stderr, flush=True)
        return 2

    # Reject a malformed slug up front with a clear, actionable CLI error rather
    # than letting it reach a path builder.
    try:
        _validate_slug(args.slug)
    except ValueError as exc:
        print(str(exc), file=sys.stderr, flush=True)
        return 2

    embed_slug(args.slug, limit=args.limit, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
