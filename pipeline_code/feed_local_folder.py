"""
feed_local_folder.py - Mirror an arbitrary admin-configured local folder into the queue.

Admins set via the dashboard:

    LOCAL_IMPORT_DIR       path to a folder on disk containing `<n>.<ext>` + `<n>.txt` pairs
    LOCAL_IMPORT_NAME      source label (default "local") - used as the queue subfolder
                           and for already-sorted dedup
    LOCAL_IMPORT_ENABLED   "true" to enable; anything else = skip

For each image/txt pair found, the feeder checks (in order) that the stem is
not already in the `seen_<name>_<slug>.json` dedup file, not already queued,
and not already present under `<sorted>/<slug>/*/<name>/`. Anything new is
copied into `<queue>/<slug>/<name>/` via queue_manager.save_to_queue.

No path is hardcoded - paths.py supplies defaults.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from credentials import get_optional
from paths import base_dir, local_import_dir, local_import_name, queue_dir, sorted_dir, sorted_root
from pipeline_logging import get_logger
from queue_manager import save_to_queue
from seen_store import MigrationSpec, SeenStore
from topic_filter import prompt_optional
import media_policy

logger = get_logger(__name__)

SLUG: str = get_optional("PIPELINE_SLUG", "default")
SRC_DIR: Path | None = local_import_dir()
SOURCE_NAME: str = local_import_name()
QUEUE_DIR: Path = queue_dir(SLUG)
SORTED_DIR: Path = sorted_dir(SLUG)

# Migration: read legacy seen files / scan legacy sorted subfolders so admins
# switching from a prior feeder to this generic importer don't re-queue items
# they already processed. Configure per-folder with:
#   LOCAL_IMPORT_MIGRATE_FROM=<seen_prefix>:<stem_prefix>:<sorted_src>,<...>,
_DEFAULT_MIGRATIONS = ""
_MIGRATION_RAW = get_optional("LOCAL_IMPORT_MIGRATE_FROM", _DEFAULT_MIGRATIONS)
_MIGRATIONS: list[MigrationSpec] = MigrationSpec.parse_env(
    _MIGRATION_RAW, slug=SLUG, base=base_dir(), sorted_root=sorted_root() / SLUG,
)

MIN_PROMPT_LENGTH: int = int(get_optional("MIN_PROMPT_LENGTH", "40"))
MIN_IMAGE_BYTES: int = 5000
IMAGE_SUFFIXES: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp")
# Video containers a local folder may hold. Local folders can now be "video
# folders" too, so the feeder picks these up alongside images.
VIDEO_SUFFIXES: tuple[str, ...] = (".mp4", ".mov", ".webm", ".mkv", ".avi")
# The full accepted set. Existing image behaviour is unchanged; this just
# widens what counts as an ingestible media file.
MEDIA_SUFFIXES: tuple[str, ...] = IMAGE_SUFFIXES + VIDEO_SUFFIXES
ENABLED: bool = get_optional("LOCAL_IMPORT_ENABLED", "false").lower() == "true"


def _match_in(source_dir: Path, filename_prefix: str, stem: str) -> bool:
    """True if any file starting with `<filename_prefix>_<stem>` exists in source_dir."""
    if not source_dir.exists():
        return False
    # Vision worker appends `_<timestamp>_<rand>` to the original stem when
    # moving to sorted, so we match by prefix instead of exact filename.
    target_prefix = f"{filename_prefix}_{stem}"
    return any(f.name.startswith(target_prefix) for f in source_dir.iterdir()
               if f.is_file() and f.suffix.lower() in MEDIA_SUFFIXES)


def already_sorted(stem: str) -> bool:
    """True if this stem appears anywhere under sorted/, including legacy locations.

    Checks, in order:
      1. <sorted>/<slug>/*/<SOURCE_NAME>/<SOURCE_NAME>_<stem>* (native path)
      2. For each configured migration: <sorted>/<slug>/*/<sorted_src>/<stem_prefix>_<stem>*
         (so a legacy `sorted/*/<src>/<prefix>_<stem>.*` counts as done.)
    """
    if not SORTED_DIR.exists():
        return False
    legacy: list[tuple[str, str]] = []
    for spec in _MIGRATIONS:
        if spec.sorted_dir is None or not spec.stem_prefix:
            continue
        # MigrationSpec.sorted_dir is the legacy *source* folder (e.g. an old
        # importer's name); we need its name to look up the same source under each category.
        legacy.append((spec.sorted_dir.name, spec.stem_prefix))

    for category in SORTED_DIR.iterdir():
        if not category.is_dir():
            continue
        if _match_in(category / SOURCE_NAME, SOURCE_NAME, stem):
            return True
        for sorted_src, stem_prefix in legacy:
            if _match_in(category / sorted_src, stem_prefix, stem):
                return True
    return False


def already_queued(stem: str) -> bool:
    qname = f"{SOURCE_NAME}_{stem}"
    target = QUEUE_DIR / SOURCE_NAME
    if not target.exists():
        return False
    return any((target / f"{qname}{suffix}").exists() for suffix in MEDIA_SUFFIXES)


def iter_sources() -> list[Path]:
    if SRC_DIR is None or not SRC_DIR.exists():
        return []
    # Scan for every media type the feeder can handle. The active-job media
    # policy filters at ingestion / classification time — but making the
    # feeder itself narrower means a user who drops an mp4 into a
    # default-image-only job never even sees a warning that their file was
    # ignored.
    images: list[Path] = []
    for suffix in MEDIA_SUFFIXES:
        images.extend(SRC_DIR.glob(f"*{suffix}"))
    return sorted(images)


def process_one(img_path: Path, seen: SeenStore) -> bool:
    stem = img_path.stem
    if stem in seen or already_sorted(stem) or already_queued(stem):
        seen.add(stem)
        return False

    txt_path = img_path.with_suffix(".txt")
    txt_exists = txt_path.exists()
    if not txt_exists and not prompt_optional():
        return False

    try:
        if img_path.stat().st_size < MIN_IMAGE_BYTES:
            seen.add(stem)
            return False
    except OSError:
        return False

    prompt = (
        txt_path.read_text(encoding="utf-8", errors="replace").strip()
        if txt_exists else ""
    )
    if len(prompt) < MIN_PROMPT_LENGTH and not prompt_optional():
        seen.add(stem)
        return False

    with tempfile.NamedTemporaryFile(suffix=img_path.suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        shutil.copy2(img_path, tmp_path)
        metadata = {
            "message_id": f"{SOURCE_NAME}_{stem}",
            "image_url": str(img_path),
            "source_channel": SOURCE_NAME,
            "source_guild": f"local:{SRC_DIR}",
            "author": "local",
            "timestamp": "",
            "queued_at": datetime.now(timezone.utc).isoformat(),
            "pipeline_topic": get_optional("PIPELINE_TOPIC"),
        }
        saved = save_to_queue(SOURCE_NAME, tmp_path, prompt, metadata)
        if saved is None:
            return False
        seen.add(stem)
        return True
    finally:
        tmp_path.unlink(missing_ok=True)


def main() -> None:
    if not ENABLED:
        logger.info("LOCAL_IMPORT_ENABLED=false - exiting")
        return
    if SRC_DIR is None:
        logger.info("LOCAL_IMPORT_DIR not set - exiting")
        return
    if not SRC_DIR.exists():
        logger.error("source directory missing: %s", SRC_DIR)
        return

    seen = SeenStore(SOURCE_NAME, slug=SLUG, autoflush_every=500, migrations=_MIGRATIONS)
    images = iter_sources()
    logger.info("scanning %s - found %d images (source=%s)", SRC_DIR, len(images), SOURCE_NAME)

    queued = 0
    for img in images:
        if process_one(img, seen):
            queued += 1

    seen.flush()
    logger.info("done - queued %d new items from %s", queued, SRC_DIR)


if __name__ == "__main__":
    main()
