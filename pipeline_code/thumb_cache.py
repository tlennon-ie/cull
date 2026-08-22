"""Disk-cached thumbnail generator.

Old behaviour: every ``GET /api/thumbnail?path=...&size=240`` re-opened the
source image with PIL, resized in memory, and served the bytes. With 60+
gallery thumbnails per page that's 60 PIL operations per F5.

New behaviour: write the resized JPEG to ``data/thumb_cache/`` once per
(image-content, size) tuple and serve the file with ``send_file`` which sets
sensible Last-Modified / ETag headers automatically. F5 gets 304s from the
browser cache; cache misses get the disk-cached file straight from the OS.

Key strategy:

  cache_path = THUMB_CACHE_ROOT / sha1[:2] / f"{sha1}_{size}.jpg"

  sha1 = SHA1(image_path + image_mtime + image_size)

The mtime+size in the hash means an image that gets overwritten in place
(unusual but possible) gets a fresh thumbnail without a cache invalidation
step. Old hashes orphan into the cache; the GC sweep at the bottom of this
module clears anything not touched in N days.
"""
from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from pathlib import Path

from PIL import Image

import paths

logger = logging.getLogger(__name__)

# Sizes the dashboard actually requests. Generating arbitrary sizes would
# explode the cache and let users DOS the disk via crafted URLs.
ALLOWED_SIZES: frozenset[int] = frozenset({120, 160, 240, 320, 480, 640, 960, 1280})

JPEG_QUALITY = 82

_root: Path | None = None
_root_lock = threading.Lock()

# Optional explicit allow-list of roots a source image may live under. When
# ``None`` the allowed roots default to the queue + sorted trees from ``paths``
# (exactly the roots the dashboard already validates a thumbnail request against
# with ``safe_inside``). The containment check below is the path-injection
# barrier: a request can only ever read/thumbnail a file inside these roots.
_allowed_source_roots: list[Path] | None = None


def configure(root: Path, allowed_source_roots: list[Path] | None = None) -> None:
    """Set the on-disk cache root. Call once at process start.

    ``allowed_source_roots`` optionally pins the directories a source image may
    live under; when omitted the queue + sorted roots from ``paths`` are used.
    """
    global _root, _allowed_source_roots
    _root = Path(root)
    _root.mkdir(parents=True, exist_ok=True)
    if allowed_source_roots is not None:
        _allowed_source_roots = [Path(p) for p in allowed_source_roots]


def _ensure_configured() -> Path:
    if _root is None:
        raise RuntimeError("thumb_cache.configure(root) must be called first")
    return _root


def _source_roots() -> list[Path]:
    """Resolve the directories a source image is allowed to live under."""
    if _allowed_source_roots is not None:
        roots = _allowed_source_roots
    else:
        roots = [paths.queue_root(), paths.sorted_root()]
    resolved: list[Path] = []
    for root in roots:
        try:
            resolved.append(Path(os.path.realpath(str(root))))
        except OSError:
            continue
    return resolved


def _validated_source(image_path: Path) -> Path:
    """Return the realpath of ``image_path`` iff it sits inside an allowed root.

    Resolves symlinks / ``..`` with ``os.path.realpath`` and confirms containment
    via ``os.path.commonpath`` — a crafted traversal path (``../../etc/passwd``)
    can never escape the queue / sorted roots. This is the path-injection barrier
    every thumbnail source flows through before it is stat-ed or opened.
    """
    real = Path(os.path.realpath(str(image_path)))
    for root in _source_roots():
        try:
            if os.path.commonpath([str(real), str(root)]) == str(root):
                return real
        except ValueError:
            # Different drives / un-comparable paths → not contained.
            continue
    raise ValueError(f"image path outside allowed roots: {image_path!r}")


def _hash_for(image_path: Path) -> str:
    # Containment barrier: only a file inside the allowed roots may be hashed
    # (and therefore stat-ed / opened). Rejects traversal before any disk touch.
    safe_path = _validated_source(image_path)
    try:
        stat = safe_path.stat()
    except OSError as exc:
        raise FileNotFoundError(str(image_path)) from exc
    digest = hashlib.sha1()
    digest.update(str(image_path).encode("utf-8", "replace"))
    digest.update(str(stat.st_mtime_ns).encode("ascii"))
    digest.update(str(stat.st_size).encode("ascii"))
    return digest.hexdigest()


def cache_path_for(image_path: Path, size: int) -> Path:
    """Return the on-disk cache location for a (path, size) pair without
    generating it. Guarantees the parent directory exists."""
    root = _ensure_configured()
    if size not in ALLOWED_SIZES:
        raise ValueError(f"size {size} not in ALLOWED_SIZES")
    digest = _hash_for(image_path)
    bucket = root / digest[:2]
    bucket.mkdir(parents=True, exist_ok=True)
    return bucket / f"{digest}_{size}.jpg"


def get_or_create(image_path: Path, size: int) -> Path:
    """Return a path to a cached JPEG of the requested size.

    Generates lazily. Idempotent — concurrent callers are race-tolerant
    (worst case: both render the same JPEG and one overwrites the other).
    """
    if size not in ALLOWED_SIZES:
        # Snap to the closest allowed size rather than 400ing the user.
        size = min(ALLOWED_SIZES, key=lambda s: abs(s - int(size)))
    cache_file = cache_path_for(image_path, size)
    if cache_file.exists():
        # Touch atime so the GC sweep keeps recently-served files.
        try:
            os.utime(cache_file, None)
        except OSError:
            pass  # atime bump is only a GC hint; the cached file still serves
        return cache_file
    # Render to a temp file then atomic-rename so concurrent readers never
    # see a half-written JPEG.
    tmp_file = cache_file.with_suffix(".tmp")
    try:
        with Image.open(image_path) as img:
            img.thumbnail((size, size))
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            img.save(tmp_file, format="JPEG", quality=JPEG_QUALITY, optimize=True)
        tmp_file.replace(cache_file)
    except Exception:
        # Best-effort cleanup; never let a thumbnail failure kill a request.
        try:
            tmp_file.unlink()
        except OSError:
            pass  # temp render never landed, or is already gone
        raise
    return cache_file


def gc_sweep(max_age_seconds: float = 30 * 24 * 60 * 60) -> int:
    """Delete cache files not accessed in `max_age_seconds`. Returns count.

    Default age = 30 days. Cheap to run periodically; the kernel's
    atime / mtime tracking is already on for these files.
    """
    root = _ensure_configured()
    cutoff = time.time() - max_age_seconds
    deleted = 0
    for cache_file in root.glob("**/*.jpg"):
        try:
            stat = cache_file.stat()
        except OSError:
            continue
        last_use = max(stat.st_atime, stat.st_mtime)
        if last_use < cutoff:
            try:
                cache_file.unlink()
                deleted += 1
            except OSError:
                continue
    return deleted


__all__ = [
    "ALLOWED_SIZES",
    "configure",
    "cache_path_for",
    "get_or_create",
    "gc_sweep",
]
