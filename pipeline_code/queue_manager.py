#!/usr/bin/env python3
"""Queue protocol + filesystem implementation.

The pipeline currently uses the local filesystem as its queue: every scraper
writes images into ``<base>/queue/<slug>/<source>/``, and vision workers pop
items round-robin across sources. The atomic-rename to ``<image>.processing``
is the lock — two workers racing for the same image cannot both win.

Why a Protocol if there's only one implementation?
  Two reasons. First, it documents the contract callers depend on. Second,
  swapping in a Redis or SQLite backend later becomes a one-line construction
  change at the call sites. The cost is one ``Protocol`` definition; the
  alternative was a tangle of module-level globals.

Module-level functions (``save_to_queue``, ``get_next_image_round_robin``,
``list_queue_sources``, ``get_queue_dir``) remain as thin shims around a
process-wide default ``FSQueue`` instance, so existing imports keep working.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Protocol

from dotenv import load_dotenv

from paths import base_dir
from pipeline_logging import get_logger

load_dotenv()

logger = get_logger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────────

BASE_DIR = Path(os.environ.get("PIPELINE_BASE_DIR", str(base_dir())))
TOPIC = os.environ.get("PIPELINE_TOPIC", "Realistic Female Influencer")
SLUG = os.environ.get("PIPELINE_SLUG", "realistic_female_influencer")
_RAW_QUEUE = Path(os.environ.get("PIPELINE_QUEUE", str(BASE_DIR / "queue")))
# PIPELINE_QUEUE may be either ".../queue" (root) or ".../queue/<slug>";
# normalise to include slug exactly once.
BASE_QUEUE_DIR = _RAW_QUEUE if _RAW_QUEUE.name == SLUG else _RAW_QUEUE / SLUG

SOURCES: dict[str, str] = {
    "civitai": "civitai",
    "civitai_red": "civitai_red",
    "discord_ud": "discord_ud",
    "discord_mj": "discord_mj",
    "reddit": "reddit",
    "twitter_x": "twitter_x",
    "nanobanana": "nanobanana",
    "unknown": "unknown",
}

_SAFE_SOURCE = re.compile(r"^[a-z0-9_]+$")
_QUEUE_IMAGE_GLOBS = ("*.jpg", "*.jpeg", "*.png", "*.webp")
# Video containers the round-robin pop will surface ONLY when the video-
# classification lane is enabled (VIDEO_CLASSIFY_ENABLED). Off by default so an
# image-only pipeline never starts popping clips it can't classify — behaviour
# stays byte-identical unless the flag is set. Mirrors video_frames.VIDEO_EXT /
# export_profiles.VIDEO_EXT.
_QUEUE_VIDEO_GLOBS = ("*.mp4", "*.mov", "*.webm", "*.mkv", "*.avi")
_DEFAULT_CACHE_TTL = 5.0  # seconds; see FSQueue docstring


def _video_classify_enabled() -> bool:
    """Whether VIDEO_CLASSIFY_ENABLED is truthy (read live so a dashboard toggle
    takes effect without restarting the worker)."""
    return os.environ.get("VIDEO_CLASSIFY_ENABLED", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _queue_globs() -> tuple[str, ...]:
    """The media globs the queue pops: images always, video clips only when the
    video-classification lane is enabled."""
    if _video_classify_enabled():
        return _QUEUE_IMAGE_GLOBS + _QUEUE_VIDEO_GLOBS
    return _QUEUE_IMAGE_GLOBS


# ── Protocol ─────────────────────────────────────────────────────────────────

class Queue(Protocol):
    """The minimum surface every queue backend must implement.

    Implementations must be safe to call from multiple processes / threads.
    """

    def save(
        self,
        source: str,
        image_path: Path,
        prompt_text: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> Path | None:
        """Move an image (with optional .txt + .meta.json siblings) into the
        queue under the given source label. Returns the destination path on
        success or None on validation failure."""

    def pop_next(self) -> tuple[str, Path] | tuple[None, None]:
        """Return the next ``(source, image_path)`` to process, round-robin
        across sources. Returns ``(None, None)`` when the queue is empty."""

    def list_sources(self) -> dict[str, int]:
        """Return ``{source_name: pending_count}`` for every non-empty source."""

    def source_dir(self, source: str) -> Path:
        """Return the per-source folder, creating it if missing."""


# ── Filesystem implementation ────────────────────────────────────────────────

class FSQueue:
    """Filesystem-backed queue with a small in-memory order cache.

    The architect's audit flagged that ``get_next_image_round_robin`` was
    O(sources × files-per-source) per call and ran for every dequeue. At 50k
    items × N parallel workers that turns into hundreds of thousands of stat
    calls per second. This impl caches the round-robin order for
    ``cache_ttl`` seconds; within the window pops are O(1). Files added in
    that window aren't visible until the next refresh, which is acceptable
    for a queue with a 1-2 second poll interval.

    Cross-worker race safety is unchanged: two workers can in theory hand
    out the same path if they both pop from the cache before either renames
    to ``.processing``, but the worker's own ``image_path.rename(processing_path)``
    is atomic. The loser gets ``FileNotFoundError`` and short-circuits with
    ``SKIPPED``, exactly as before.
    """

    def __init__(self, base_dir: Path, cache_ttl: float = _DEFAULT_CACHE_TTL) -> None:
        self.base_dir: Path = base_dir
        self.cache_ttl: float = cache_ttl
        self._cache: list[tuple[str, Path]] = []
        self._cache_ts: float = 0.0
        self._lock = threading.Lock()

    # ── Public API ───────────────────────────────────────────────────────

    def source_dir(self, source: str) -> Path:
        source_key = source.lower().replace(" ", "_").replace("-", "_")
        if source_key in SOURCES:
            dir_name = SOURCES[source_key]
        elif _SAFE_SOURCE.match(source_key):
            dir_name = source_key
        else:
            dir_name = "unknown"
        path = self.base_dir / dir_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save(
        self,
        source: str,
        image_path: Path,
        prompt_text: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> Path | None:
        queue_dir = self.source_dir(source)
        try:
            if not image_path.exists():
                logger.error("temp file does not exist: %s", image_path)
                return None

            size_before = image_path.stat().st_size
            if size_before < 5000:
                logger.warning("temp file too small (%d bytes): %s", size_before, image_path)
                image_path.unlink(missing_ok=True)
                return None

            dest_img = queue_dir / image_path.name
            try:
                shutil.move(str(image_path), str(dest_img))
                logger.info("moved %s -> %s/ (%d bytes)", image_path.name, source, size_before)
            except Exception as exc:
                logger.warning("move failed (%s); falling back to copy", exc)
                shutil.copy2(image_path, dest_img)
                image_path.unlink(missing_ok=True)
                logger.info("copied %s -> %s/ (%d bytes)", image_path.name, source, size_before)

            if not dest_img.exists():
                logger.error("file failed to reach destination: %s", dest_img)
                return None

            dest_size = dest_img.stat().st_size
            if dest_size < size_before * 0.95:
                logger.error(
                    "file corrupted in transfer (%d != %d): %s",
                    dest_size, size_before, dest_img,
                )
                dest_img.unlink(missing_ok=True)
                return None

            if prompt_text:
                dest_txt = queue_dir / f"{dest_img.stem}.txt"
                dest_txt.write_text(prompt_text, encoding="utf-8")
                logger.debug("saved prompt %s (%d chars)", dest_txt.name, len(prompt_text))

            if metadata:
                dest_meta = queue_dir / f"{dest_img.stem}.meta.json"
                dest_meta.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                logger.debug("saved metadata %s", dest_meta.name)

            logger.info("queued %s (%d bytes) -> %s/", dest_img.name, dest_size, source)
            # New write -> invalidate the order cache so workers see this item
            # promptly without waiting for the TTL.
            with self._lock:
                self._cache_ts = 0.0
                self._cache.clear()
            return dest_img
        except Exception:
            logger.exception("unexpected error saving to queue (%s)", source)
            image_path.unlink(missing_ok=True)
            return None

    def list_sources(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for src_dir in self._discover_source_dirs():
            imgs = self._images_in(src_dir)
            if imgs:
                counts[src_dir.name] = len(imgs)
        return counts

    def pop_next(self) -> tuple[str, Path] | tuple[None, None]:
        with self._lock:
            now = time.time()
            if (now - self._cache_ts) >= self.cache_ttl or not self._cache:
                self._cache = self._build_round_robin_order()
                self._cache_ts = now
            # Pop the oldest, skipping vanished files (another worker grabbed them).
            while self._cache:
                src, path = self._cache.pop(0)
                if path.exists():
                    return src, path
            return None, None

    # ── Internal helpers ─────────────────────────────────────────────────

    def _discover_source_dirs(self) -> list[Path]:
        if not self.base_dir.exists():
            return []
        return [
            child for child in self.base_dir.iterdir()
            if child.is_dir() and _SAFE_SOURCE.match(child.name)
        ]

    @staticmethod
    def _images_in(source_dir: Path) -> list[Path]:
        images: list[Path] = []
        for pattern in _queue_globs():
            images.extend(source_dir.glob(pattern))
        return sorted(images)

    def _build_round_robin_order(self) -> list[tuple[str, Path]]:
        source_images: dict[str, list[Path]] = {}
        for src_dir in self._discover_source_dirs():
            imgs = self._images_in(src_dir)
            if imgs:
                source_images[src_dir.name] = imgs
        if not source_images:
            return []
        order: list[tuple[str, Path]] = []
        max_len = max(len(v) for v in source_images.values())
        for i in range(max_len):
            for source in sorted(source_images.keys()):
                if i < len(source_images[source]):
                    order.append((source, source_images[source][i]))
        return order


# ── Default instance + module-level shims ─────────────────────────────────────

_default_queue: FSQueue = FSQueue(BASE_QUEUE_DIR)


def get_default_queue() -> FSQueue:
    """Return the process-wide default queue backend.

    Tests / dashboards can build their own ``FSQueue`` against a different
    ``base_dir`` if they need to inspect a non-active slug.
    """
    return _default_queue


def get_queue_dir(source: str) -> Path:
    return _default_queue.source_dir(source)


def save_to_queue(
    source: str,
    image_path: Path,
    prompt_text: str = "",
    metadata: dict[str, Any] | None = None,
) -> Path | None:
    return _default_queue.save(source, image_path, prompt_text, metadata)


def list_queue_sources() -> dict[str, int]:
    return _default_queue.list_sources()


def get_next_image_round_robin() -> tuple[str, Path] | tuple[None, None]:
    return _default_queue.pop_next()


__all__ = [
    "Queue",
    "FSQueue",
    "BASE_QUEUE_DIR",
    "SOURCES",
    "get_default_queue",
    "get_queue_dir",
    "save_to_queue",
    "list_queue_sources",
    "get_next_image_round_robin",
]
