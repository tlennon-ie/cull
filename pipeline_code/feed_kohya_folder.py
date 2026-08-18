"""
feed_kohya_folder.py — Mirror a Kohya-style training-set folder into the queue.

Kohya training folders follow the ``<repeats>_<concept>`` subfolder convention::

    dataset_root/
      10_myconcept/
        img_001.jpg
        img_001.txt      <- caption sidecar
        img_002.png
        img_002.txt
      5_otherconcept/
        ...

This feeder walks ``KOHYA_IMPORT_DIR`` looking for subdirectories that match
``^\\d+_(.+)$``, ingests every image + ``.txt`` pair, and pushes them into the
queue via :func:`queue_manager.save_to_queue`. The concept name and repeat
count are preserved in the per-image metadata JSON (not used to duplicate the
image — repeats are informational only).

Danbooru-style flat folders (matching ``.txt`` sidecars at the root, no
``<repeats>_<concept>`` prefix) are also supported when
``KOHYA_ALLOW_FLAT=true``.

Env contract (mirrors ``feed_local_folder.py``):

    PIPELINE_SLUG          required — same slug used everywhere else.
    KOHYA_IMPORT_DIR       required — absolute path to the dataset root.
    KOHYA_IMPORT_NAME      optional — SeenStore name + queue source folder
                           (default ``"kohya"``).
    KOHYA_IMPORT_ENABLED   ``"true"`` to enable; anything else = skip.
    KOHYA_MOVE             ``"true"`` MOVES source files (destructive; a loud
                           warning goes to stderr on startup). Default false =
                           copy.
    KOHYA_ALLOW_FLAT       ``"true"`` also ingests images at the root of the
                           dataset dir. Default false.
    KOHYA_POLL_INTERVAL    seconds between rescans (default 30). ``0`` = run
                           once and exit.
    PIPELINE_SCRAPER_THROTTLE   respected between queue writes (see
                                ``_maybe_throttle`` in ``feed_local_folder.py``
                                pattern).

Design invariants:
  * Uses ``save_to_queue`` — never writes directly into ``data/queue/``.
  * Uses ``SeenStore("<name>", slug=SLUG)`` keyed by the absolute source path
    for dedup so a rerun is safe. Because Kohya folders reuse simple stems
    like ``img_001``, we cannot key on ``stem`` alone (two concepts might both
    have ``img_001.jpg``). The abspath is unique and stable.
  * Respects ``topic_filter.prompt_optional()`` — reject images without a
    ``.txt`` sidecar (or too-short prompt) unless ``REQUIRE_PROMPT=false``.
  * Path safety: every candidate image is resolved and rejected if it does not
    live under the declared ``KOHYA_IMPORT_DIR`` (guards symlink escape).
  * Structured logging goes to stdout/stderr with ``flush=True`` — this file
    is a subprocess worker so the supervisor labels its stdout cleanly.
"""
from __future__ import annotations

import os
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from queue_manager import save_to_queue
from seen_store import SeenStore
from topic_filter import prompt_optional

# ── Constants ────────────────────────────────────────────────────────────────

# Kohya's canonical concept-folder pattern: <repeats>_<concept>. Concept can
# contain any character other than the leading run of digits + underscore.
_KOHYA_SUBDIR_RE = re.compile(r"^(\d+)_(.+)$")

# Media suffixes match feed_local_folder.py — same defaults so behaviour is
# consistent across importers.
IMAGE_SUFFIXES: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp", ".gif")
VIDEO_SUFFIXES: tuple[str, ...] = (".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v")
MEDIA_SUFFIXES: tuple[str, ...] = IMAGE_SUFFIXES + VIDEO_SUFFIXES

MIN_IMAGE_BYTES: int = 5000


# ── Env-derived config (read once at import; supervisor respawns per job) ─────

def _bool_env(key: str, default: bool = False) -> bool:
    raw = (os.environ.get(key, "") or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _int_env(key: str, default: int) -> int:
    try:
        return int((os.environ.get(key, "") or "").strip() or default)
    except ValueError:
        return default


def _float_env(key: str, default: float) -> float:
    try:
        return float((os.environ.get(key, "") or "").strip() or default)
    except ValueError:
        return default


SLUG: str = (os.environ.get("PIPELINE_SLUG", "") or "default").strip() or "default"
ENABLED: bool = _bool_env("KOHYA_IMPORT_ENABLED", False)
SOURCE_NAME: str = (os.environ.get("KOHYA_IMPORT_NAME", "") or "kohya").strip() or "kohya"
MOVE_MODE: bool = _bool_env("KOHYA_MOVE", False)
ALLOW_FLAT: bool = _bool_env("KOHYA_ALLOW_FLAT", False)
POLL_INTERVAL: float = _float_env("KOHYA_POLL_INTERVAL", 30.0)
THROTTLE: float = _float_env("PIPELINE_SCRAPER_THROTTLE", 0.0)
MIN_PROMPT_LENGTH: int = _int_env("MIN_PROMPT_LENGTH", 40)


def _import_dir() -> Path | None:
    raw = (os.environ.get("KOHYA_IMPORT_DIR", "") or "").strip()
    return Path(raw) if raw else None


# ── Small helpers (side-effect free) ─────────────────────────────────────────

@dataclass(frozen=True)
class KohyaEntry:
    """One image the feeder found under the dataset root."""

    image_path: Path         # absolute, already resolved and inside root
    caption_path: Path       # sibling ``.txt`` (may not exist)
    concept: str             # "" for flat-folder ingests
    repeats: int             # 0 for flat-folder ingests
    relative: str            # slash-joined path relative to root (for logs / seen key)


def _log(msg: str, *, err: bool = False) -> None:
    """Emit a labelled line. Subprocess workers use ``print(..., flush=True)``
    so the supervisor can capture and prefix them (see pipeline_logging.py)."""
    stream = sys.stderr if err else sys.stdout
    print(f"[kohya:{SOURCE_NAME}] {msg}", file=stream, flush=True)


def _safe_inside(root: Path, candidate: Path) -> bool:
    """Return True iff ``candidate`` resolves to a path under ``root``.

    Walks symlinks via ``Path.resolve()`` so a symlink escape trips the check.
    Uses a string-prefix comparison for Python < 3.9 style compatibility
    even though this codebase targets 3.11+.
    """
    try:
        resolved = candidate.resolve()
        root_resolved = root.resolve()
    except OSError:
        return False
    try:
        return resolved.is_relative_to(root_resolved)
    except AttributeError:  # pragma: no cover - Python < 3.9
        try:
            resolved.relative_to(root_resolved)
            return True
        except ValueError:
            return False


def _parse_concept_dir(name: str) -> tuple[int, str] | None:
    """Return ``(repeats, concept)`` if ``name`` matches ``<n>_<concept>``."""
    m = _KOHYA_SUBDIR_RE.match(name)
    if not m:
        return None
    try:
        repeats = int(m.group(1))
    except ValueError:
        return None
    concept = m.group(2).strip()
    return (repeats, concept) if concept else None


def _iter_media_in(folder: Path) -> list[Path]:
    """Sorted list of media files directly in ``folder`` (non-recursive)."""
    if not folder.is_dir():
        return []
    out: list[Path] = []
    for child in folder.iterdir():
        if not child.is_file():
            continue
        if child.suffix.lower() in MEDIA_SUFFIXES:
            out.append(child)
    return sorted(out)


def discover_entries(root: Path) -> list[KohyaEntry]:
    """Walk ``root`` and yield every ingestible image.

    * Concept subdirs (``<repeats>_<concept>``) are always scanned.
    * Root-level images are scanned only when ``KOHYA_ALLOW_FLAT`` is true.
    """
    if not root.exists() or not root.is_dir():
        return []

    entries: list[KohyaEntry] = []

    # Concept-folder pass.
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        parsed = _parse_concept_dir(child.name)
        if parsed is None:
            continue
        repeats, concept = parsed
        for img in _iter_media_in(child):
            if not _safe_inside(root, img):
                _log(f"skip (escapes root): {img}", err=True)
                continue
            entries.append(KohyaEntry(
                image_path=img,
                caption_path=img.with_suffix(".txt"),
                concept=concept,
                repeats=repeats,
                relative=f"{child.name}/{img.name}",
            ))

    # Danbooru-style flat pass.
    if ALLOW_FLAT:
        for img in _iter_media_in(root):
            if not _safe_inside(root, img):
                _log(f"skip (escapes root): {img}", err=True)
                continue
            entries.append(KohyaEntry(
                image_path=img,
                caption_path=img.with_suffix(".txt"),
                concept="",
                repeats=0,
                relative=img.name,
            ))

    return entries


# ── Ingest one image ─────────────────────────────────────────────────────────

def _read_caption(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""


def _seen_key(root: Path, entry: KohyaEntry) -> str:
    """Stable dedup key. We use the absolute image path because Kohya folders
    happily reuse stems like ``img_001`` across concept subdirs — stems alone
    would collide across concepts."""
    try:
        return str(entry.image_path.resolve())
    except OSError:
        return str(entry.image_path)


def process_one(root: Path, entry: KohyaEntry, seen: SeenStore) -> bool:
    """Ingest one image into the queue. Returns True on a new successful queue push."""
    key = _seen_key(root, entry)
    if key in seen:
        return False

    if not entry.image_path.exists():
        seen.add(key)
        return False

    try:
        size = entry.image_path.stat().st_size
    except OSError:
        return False
    if size < MIN_IMAGE_BYTES:
        seen.add(key)
        return False

    caption_exists = entry.caption_path.exists()
    if not caption_exists and not prompt_optional():
        # Cannot ingest without a prompt AND prompts are required — drop silently.
        return False

    prompt = _read_caption(entry.caption_path) if caption_exists else ""
    if len(prompt) < MIN_PROMPT_LENGTH and not prompt_optional():
        seen.add(key)
        return False

    with tempfile.NamedTemporaryFile(suffix=entry.image_path.suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        try:
            shutil.copy2(entry.image_path, tmp_path)
        except OSError as exc:
            _log(f"copy failed for {entry.relative}: {exc}", err=True)
            return False

        metadata = {
            "message_id": f"{SOURCE_NAME}_{entry.relative}",
            "image_url": str(entry.image_path),
            "source_channel": SOURCE_NAME,
            "source_guild": f"kohya:{root}",
            "author": "kohya",
            "timestamp": "",
            "queued_at": datetime.now(timezone.utc).isoformat(),
            "pipeline_topic": os.environ.get("PIPELINE_TOPIC", ""),
            # Kohya-specific tags (recorded, not consumed by the pipeline yet).
            "kohya_concept": entry.concept,
            "kohya_repeats": entry.repeats,
            "kohya_relative_path": entry.relative,
        }

        saved = save_to_queue(SOURCE_NAME, tmp_path, prompt, metadata)
        if saved is None:
            return False

        seen.add(key)

        # Destructive path — only after the queue write succeeded. If the move
        # fails after a successful copy-into-queue, we log and continue: the
        # image is safely queued, we just did not delete the source.
        if MOVE_MODE:
            try:
                entry.image_path.unlink()
                if caption_exists:
                    entry.caption_path.unlink(missing_ok=True)
            except OSError as exc:
                _log(f"MOVE post-queue unlink failed for {entry.relative}: {exc}", err=True)

        return True
    finally:
        tmp_path.unlink(missing_ok=True)


# ── Main loop ────────────────────────────────────────────────────────────────

def _ingest_pass(root: Path, seen: SeenStore) -> int:
    entries = discover_entries(root)
    queued = 0
    for entry in entries:
        if process_one(root, entry, seen):
            queued += 1
            if THROTTLE > 0:
                time.sleep(THROTTLE)
    seen.flush()
    return queued


def main() -> int:
    if not ENABLED:
        _log("KOHYA_IMPORT_ENABLED=false — exiting")
        return 0

    root = _import_dir()
    if root is None:
        _log("KOHYA_IMPORT_DIR not set — exiting", err=True)
        return 0
    if not root.exists() or not root.is_dir():
        _log(f"dataset root missing: {root}", err=True)
        return 1

    if MOVE_MODE:
        # Loud, unmistakable warning so an operator who set this by mistake
        # notices before the first rescan removes their source files.
        _log(
            "!!! KOHYA_MOVE=true — SOURCE FILES WILL BE DELETED AFTER SUCCESSFUL QUEUE WRITE",
            err=True,
        )

    _log(
        f"starting: root={root} name={SOURCE_NAME} flat={ALLOW_FLAT} "
        f"poll={POLL_INTERVAL}s move={MOVE_MODE}"
    )

    seen = SeenStore(SOURCE_NAME, slug=SLUG, autoflush_every=500)

    # Single-shot mode when poll interval is zero (parity with tools/*).
    if POLL_INTERVAL <= 0:
        queued = _ingest_pass(root, seen)
        _log(f"done — queued {queued} new items from {root}")
        return 0

    total = 0
    try:
        while True:
            queued = _ingest_pass(root, seen)
            total += queued
            _log(f"pass complete — queued {queued} (total {total})")
            time.sleep(POLL_INTERVAL)
    except KeyboardInterrupt:
        _log("interrupted — flushing seen store")
        seen.flush()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
