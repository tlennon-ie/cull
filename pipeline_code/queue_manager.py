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
_QUEUE_IMAGE_EXTS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp", ".gif")
# Legacy glob-form aliases — tests + a couple of external callers still import
# these under the *_GLOBS names. Derived once so the two views can't drift.
_QUEUE_IMAGE_GLOBS: tuple[str, ...] = tuple(f"*{e}" for e in _QUEUE_IMAGE_EXTS)
# Video containers the round-robin pop will surface ONLY when the video-
# classification lane is enabled (VIDEO_CLASSIFY_ENABLED). Off by default so an
# image-only pipeline never starts popping clips it can't classify — behaviour
# stays byte-identical unless the flag is set. Mirrors video_frames.VIDEO_EXT /
# export_profiles.VIDEO_EXT.
_QUEUE_VIDEO_EXTS: tuple[str, ...] = (".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v")
_QUEUE_VIDEO_GLOBS: tuple[str, ...] = tuple(f"*{e}" for e in _QUEUE_VIDEO_EXTS)
_DEFAULT_CACHE_TTL = 5.0  # seconds; see FSQueue docstring


def _video_classify_enabled() -> bool:
    """Whether VIDEO_CLASSIFY_ENABLED is truthy (read live so a dashboard toggle
    takes effect without restarting the worker)."""
    return os.environ.get("VIDEO_CLASSIFY_ENABLED", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _queue_exts() -> frozenset[str]:
    """The media extensions the queue pops, derived from the active media policy.

    Returns a lowercase-suffix ``frozenset`` (``.jpg`` etc). One
    ``os.scandir`` + suffix-in-set check per source dir replaces the previous
    N-glob-per-source scan — measurable win on directories with 10k+ items.
    Defaults mirror the historical image/video set, so an unconfigured
    pipeline is byte-identical unless VIDEO_CLASSIFY_ENABLED is set (which
    always widens the set to include the standard video containers so the
    env gate alone is enough to surface clips)."""
    include_video = _video_classify_enabled()
    try:
        import media_policy
        exts: list[str] = []
        if media_policy.wants_image():
            exts.extend(media_policy.image_exts())
        if media_policy.wants_video() and include_video:
            exts.extend(media_policy.video_exts())
        if exts:
            lowered = frozenset(e.lower() for e in exts)
            # env-gate override: if video is on but policy didn't emit video
            # exts (image-only default policy), append the standard video set.
            if include_video and not (lowered & frozenset(_QUEUE_VIDEO_EXTS)):
                return frozenset(lowered | frozenset(_QUEUE_VIDEO_EXTS))
            return lowered
    except Exception:  # noqa: BLE001 - never let policy parsing break the queue
        pass
    # Fallback to the historical fixed set.
    if include_video:
        return frozenset(_QUEUE_IMAGE_EXTS + _QUEUE_VIDEO_EXTS)
    return frozenset(_QUEUE_IMAGE_EXTS)


def _queue_globs() -> tuple[str, ...]:
    """Legacy glob view of the queue extensions — used by tests and by
    callers that still consume the ``('*.jpg', ...)`` shape. Thin adapter
    over :func:`_queue_exts` that preserves the canonical source order
    (image globs then video globs) so callers can identity-compare against
    :data:`_QUEUE_IMAGE_GLOBS` / :data:`_QUEUE_VIDEO_GLOBS`."""
    exts = _queue_exts()
    # Preserve the canonical source order (images before videos) instead of
    # returning a sorted-alpha tuple that would break identity comparisons.
    ordered: list[str] = []
    for e in _QUEUE_IMAGE_EXTS + _QUEUE_VIDEO_EXTS:
        if e in exts:
            ordered.append(f"*{e}")
    # Any extra exts from a custom media_policy land at the end, sorted.
    known = set(_QUEUE_IMAGE_EXTS) | set(_QUEUE_VIDEO_EXTS)
    for e in sorted(exts - known):
        ordered.append(f"*{e}")
    return tuple(ordered)


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
        """One scandir pass per source dir; suffix-filter against a frozenset.

        Old code ran ``source_dir.glob(pattern)`` for each of 5-11 extension
        globs — for a source with 10k queued items that meant 5-11 full directory
        walks. ``os.scandir`` walks the dir once and returns ``DirEntry`` objects
        whose ``.name`` is a fast attribute; the ``in`` check against a
        ``frozenset`` is O(1). Sorted by name to keep the round-robin order
        deterministic (mirrors the prior sorted-glob behaviour)."""
        exts = _queue_exts()
        try:
            entries = os.scandir(source_dir)
        except (FileNotFoundError, NotADirectoryError, PermissionError):
            return []
        matches: list[Path] = []
        try:
            for entry in entries:
                # Fast-reject: suffix check on the raw name string (no Path alloc
                # yet). ``entry.is_file()`` may stat, but only for candidates
                # that already passed the suffix filter — the common case is
                # cheap.
                name = entry.name
                dot = name.rfind(".")
                if dot < 0 or name[dot:].lower() not in exts:
                    continue
                try:
                    if entry.is_file():
                        matches.append(Path(entry.path))
                except OSError:
                    continue
        finally:
            entries.close()
        matches.sort(key=lambda p: p.name)
        return matches

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


# ── Cross-slug (multi-active) round-robin ─────────────────────────────────────
#
# A shared vision worker consuming from multiple active jobs uses a WEIGHTED
# round-robin across the slugs' queue subdirs. Weights are per-job priorities
# (1-10) — weight ``w`` gives that slug ``w`` consecutive turns per cycle.
# Reuses the per-slug FSQueue caching so the heavy scan work isn't duplicated
# per call.
#
# The queue root (parent of the per-slug subdirs) defaults to the parent of
# BASE_QUEUE_DIR, so the multi-slug driver doesn't need to redo the .env
# parsing every call.

_QUEUE_ROOT: Path = BASE_QUEUE_DIR.parent

# Per-slug FSQueue cache. Constructed lazily; reused across pop calls so the
# in-memory round-robin order cache each FSQueue owns keeps paying off.
_MULTI_SLUG_QUEUES: dict[str, FSQueue] = {}
_MULTI_SLUG_LOCK = threading.Lock()


def _fsqueue_for_slug(slug: str, root: Path | None = None) -> FSQueue:
    root = root or _QUEUE_ROOT
    key = f"{root}::{slug}"
    with _MULTI_SLUG_LOCK:
        q = _MULTI_SLUG_QUEUES.get(key)
        if q is None:
            q = FSQueue(root / slug)
            _MULTI_SLUG_QUEUES[key] = q
        return q


def _weighted_round_robin_order(slugs: list[str], weights: dict[str, int]) -> list[str]:
    """Deterministic weighted round-robin schedule over ``slugs``.

    A slug with weight ``w`` gets ``w`` consecutive turns per cycle, so a
    priority-10 job pops ~10 images to a priority-1 job's one. Weights <=0
    are treated as 1 so a mis-typed value can never starve a slug.
    """
    order: list[str] = []
    for slug in slugs:
        w = max(1, int(weights.get(slug, 1) or 1))
        order.extend([slug] * w)
    return order


# Module-level round-robin cursor. The schedule ``([slugA]*wA + [slugB]*wB + …)``
# is stable per (slugs, weights); we walk it advance-by-one per pop so
# consecutive calls DISTRIBUTE the work across slugs by weight rather than
# always draining the head slug first.
_RR_STATE: dict[str, Any] = {"key": None, "schedule": [], "cursor": 0}
_RR_LOCK = threading.Lock()


def _rr_key(slugs: list[str], weights: dict[str, int]) -> str:
    """Stable key for the (slugs, weights) tuple so a config change resets."""
    parts = [f"{s}={int(weights.get(s, 1) or 1)}" for s in slugs]
    return "|".join(parts)


def pop_next_across_slugs(
    slugs: list[str],
    weights: dict[str, int] | None = None,
    *,
    queue_root: Path | None = None,
) -> tuple[str, str, Path] | tuple[None, None, None]:
    """Pop the next ``(slug, source, path)`` across multiple active slugs.

    Weighted round-robin — a slug with weight ``w`` gets ``w`` consecutive
    turns in the schedule, so a priority-10 job pops ~10 images per every
    ~1 pop a priority-1 job gets. Within a slug, the existing per-slug
    ``FSQueue`` round-robin over sources applies. The schedule cursor is
    process-wide state (thread-safe), so consecutive calls DISTRIBUTE work
    across slugs rather than always finding the head slug's item first and
    starving the tail.

    Returns ``(None, None, None)`` when every slug's queue is empty.
    """
    if not slugs:
        return None, None, None
    weights = weights or {}
    key = _rr_key(slugs, weights)
    with _RR_LOCK:
        if _RR_STATE["key"] != key:
            _RR_STATE["key"] = key
            _RR_STATE["schedule"] = _weighted_round_robin_order(slugs, weights)
            _RR_STATE["cursor"] = 0
        schedule: list[str] = _RR_STATE["schedule"]
        if not schedule:
            return None, None, None
        # Try each slot in the schedule, advancing the cursor. Wrap once so
        # every slot gets a chance before we declare the multi-slug queue empty.
        attempts = 0
        cursor = _RR_STATE["cursor"]
        result: tuple[str, str, Path] | tuple[None, None, None] = (None, None, None)
        while attempts < len(schedule):
            slug = schedule[cursor % len(schedule)]
            cursor += 1
            attempts += 1
            q = _fsqueue_for_slug(slug, queue_root)
            source, path = q.pop_next()
            if path is not None:
                result = (slug, source, path)
                break
        _RR_STATE["cursor"] = cursor % len(schedule) if schedule else 0
        return result


def _active_slugs_from_env() -> tuple[list[str], dict[str, int]] | None:
    """Parse the supervisor-projected multi-active blob (or return ``None``).

    Shape: JSON array of ``[slug, weight]`` pairs, e.g.
    ``[["job_a", 5], ["job_b", 3]]``. Empty / missing / malformed values yield
    ``None`` so the module-level pop shim falls back to the historic
    single-slug ``FSQueue`` behaviour (byte-identical to today).
    """
    raw = (os.environ.get("PIPELINE_ACTIVE_SLUGS_JSON", "") or "").strip()
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, list) or not data:
        return None
    slugs: list[str] = []
    weights: dict[str, int] = {}
    for entry in data:
        if isinstance(entry, list) and len(entry) >= 2 and isinstance(entry[0], str):
            slugs.append(entry[0])
            try:
                weights[entry[0]] = max(1, int(entry[1]))
            except (TypeError, ValueError):
                weights[entry[0]] = 1
        elif isinstance(entry, str):
            slugs.append(entry)
            weights[entry] = 1
    if not slugs:
        return None
    return slugs, weights


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
    """Return ``(source, path)`` for the next queued image.

    When the supervisor is running MULTIPLE active jobs it projects the active
    set + priority into ``PIPELINE_ACTIVE_SLUGS_JSON``; we detect that here
    and delegate to ``pop_next_across_slugs`` so a shared vision worker fair-
    shares across every active slug's queue. The slug is dropped from the
    return tuple for legacy caller compatibility — the popped path is under
    ``<queue_root>/<slug>/<source>/…`` so consumers that need the slug can
    read it from the path.

    In single-active (or standalone) mode the env is absent and we fall back
    to the historic per-process ``FSQueue`` — behaviour is byte-identical.
    """
    multi = _active_slugs_from_env()
    if multi is not None:
        slugs, weights = multi
        _slug, source, path = pop_next_across_slugs(slugs, weights)
        if path is None:
            return None, None
        return source, path
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
    "pop_next_across_slugs",
]
