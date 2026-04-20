"""
feed_zforfree_local.py - Mirror local ZforFree downloads into the vision queue.

Reads <ZFORFREE_LOCAL_SRC>/<n>.png + <n>.txt (or .jpg/.jpeg/.webp) and copies to
the source-based queue via queue_manager. Skips items already in the queue OR
already classified in <PIPELINE_SORTED>, so we never re-process something the
vision worker has already sorted.

All paths resolved from .env (WORKSPACE_ROOT/.env). No hardcoded paths.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from queue_manager import save_to_queue  # noqa: E402

logger = logging.getLogger("zff_local")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

SRC_DIR: Path = Path(os.environ.get("ZFORFREE_LOCAL_SRC", r"I:\AI\Scripts\zforfree\downloads"))
BASE_DIR: Path = Path(os.environ.get("PIPELINE_BASE_DIR", r"I:\AI\openclaw\workspace\prompt-library"))
SLUG: str = os.environ.get("PIPELINE_SLUG", "realistic_female_influencer")
QUEUE_DIR: Path = Path(os.environ.get("PIPELINE_QUEUE", str(BASE_DIR / "queue"))) / SLUG
SORTED_DIR: Path = Path(os.environ.get("PIPELINE_SORTED", str(BASE_DIR / "sorted"))) / SLUG
SEEN_FILE: Path = BASE_DIR / f"seen_zff_local_{SLUG}.json"

MIN_PROMPT_LENGTH: int = 40
MIN_IMAGE_BYTES: int = 5000  # queue_manager rejects anything smaller; skip upstream
IMAGE_SUFFIXES: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp")
ENABLED: bool = os.environ.get("ZFORFREE_LOCAL_ENABLED", "true").lower() == "true"


def load_seen() -> set[str]:
    if SEEN_FILE.exists():
        try:
            return set(json.loads(SEEN_FILE.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            logger.warning("seen file corrupt — starting fresh: %s", SEEN_FILE)
    return set()


def save_seen(seen: set[str]) -> None:
    SEEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    SEEN_FILE.write_text(json.dumps(sorted(seen)), encoding="utf-8")


def already_sorted(stem: str) -> bool:
    """True if this stem appears anywhere under <PIPELINE_SORTED>/<slug>/*/zforfree/."""
    qname = f"zff_{stem}"
    if not SORTED_DIR.exists():
        return False
    for category_dir in SORTED_DIR.iterdir():
        if not category_dir.is_dir():
            continue
        zff_dir = category_dir / "zforfree"
        if not zff_dir.exists():
            continue
        for suffix in IMAGE_SUFFIXES:
            if (zff_dir / f"{qname}{suffix}").exists():
                return True
    return False


def already_queued(stem: str) -> bool:
    qname = f"zff_{stem}"
    zff_queue = QUEUE_DIR / "zforfree"
    if not zff_queue.exists():
        return False
    return any((zff_queue / f"{qname}{suffix}").exists() for suffix in IMAGE_SUFFIXES)


def iter_sources() -> list[Path]:
    if not SRC_DIR.exists():
        logger.error("source directory missing: %s", SRC_DIR)
        return []
    images: list[Path] = []
    for suffix in IMAGE_SUFFIXES:
        images.extend(SRC_DIR.glob(f"*{suffix}"))
    return sorted(images)


def process_one(img_path: Path, seen: set[str]) -> bool:
    stem = img_path.stem
    if stem in seen:
        return False
    if already_sorted(stem):
        seen.add(stem)
        return False
    if already_queued(stem):
        seen.add(stem)
        return False

    txt_path = img_path.with_suffix(".txt")
    if not txt_path.exists():
        return False

    try:
        size = img_path.stat().st_size
    except OSError:
        return False
    if size < MIN_IMAGE_BYTES:
        logger.debug("skip %s — too small (%d bytes)", img_path.name, size)
        seen.add(stem)
        return False

    prompt = txt_path.read_text(encoding="utf-8", errors="replace").strip()
    if len(prompt) < MIN_PROMPT_LENGTH:
        seen.add(stem)
        return False

    with tempfile.NamedTemporaryFile(suffix=img_path.suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        shutil.copy2(img_path, tmp_path)
        metadata = {
            "message_id": f"zff_{stem}",
            "image_url": str(img_path),
            "source_channel": "zforfree-local",
            "source_guild": "zforfree.com",
            "author": "zforfree",
            "timestamp": "",
            "queued_at": datetime.utcnow().isoformat(),
            "pipeline_topic": os.environ.get("PIPELINE_TOPIC", ""),
        }
        saved = save_to_queue("zforfree", tmp_path, prompt, metadata)
        if saved is None:
            return False
        seen.add(stem)
        return True
    finally:
        tmp_path.unlink(missing_ok=True)


def main() -> None:
    if not ENABLED:
        logger.info("ZFORFREE_LOCAL_ENABLED=false — exiting")
        return

    seen = load_seen()
    images = iter_sources()
    logger.info("found %d images in %s", len(images), SRC_DIR)

    queued = 0
    for img_path in images:
        if process_one(img_path, seen):
            queued += 1
            if queued % 500 == 0:
                save_seen(seen)
                logger.info("queued %d so far...", queued)

    save_seen(seen)
    logger.info("done. queued %d items from local zforfree", queued)


if __name__ == "__main__":
    main()
