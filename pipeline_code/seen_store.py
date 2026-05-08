"""Shared dedup store for scrapers + feeders.

Every scraper used to roll its own ``load_seen()`` / ``save_seen()`` / JSON
file dance, with subtle inconsistencies (some sorted the keys, some didn't;
some atomic-replaced on save, some didn't; only one supported migration
from legacy seen files). This module consolidates the contract so:

    >>> store = SeenStore("twitter_x", base_dir=base_dir())
    >>> if not store.contains(tweet_id):
    ...     download(tweet_id)
    ...     store.add(tweet_id)
    >>> store.flush()                       # call periodically; on shutdown

* JSON file lives at ``<base_dir>/seen_<source>_<slug>.json``.
* Loads are tolerant of corrupt / missing files (returns empty set + logs a
  warning instead of raising).
* Saves are atomic via ``Path.replace`` on a tempfile so a crash mid-write
  can't truncate the index.
* In-memory set is the source of truth for fast contains/add; flush()
  persists. autoflush_every controls how often add() triggers an implicit
  flush so callers don't have to remember.
* Migration: ``MigrationSpec`` lets a feeder pull in legacy seen files and
  scan a legacy sorted-folder for already-done IDs without writing back.
  The ``LOCAL_IMPORT_MIGRATE_FROM`` env var format
  ``<seen_prefix>:<stem_prefix>:<sorted_src>,<...>`` parses into specs.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from paths import base_dir as _default_base_dir
from pipeline_logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class MigrationSpec:
    """One legacy data source the new SeenStore should read on first load.

    * ``seen_file`` - read-only legacy seen JSON whose IDs we adopt.
    * ``sorted_dir`` - if set, every file under that folder whose stem
      starts with ``stem_prefix`` is treated as already-done (the feeder
      writes its own files with a different prefix).
    * ``stem_prefix`` - filename prefix that identifies legacy items in
      ``sorted_dir``.
    """
    seen_file: Path | None = None
    sorted_dir: Path | None = None
    stem_prefix: str = ""

    @classmethod
    def parse_env(cls, value: str, slug: str, base: Path, sorted_root: Path) -> list["MigrationSpec"]:
        """Parse ``<seen_prefix>:<stem_prefix>:<sorted_src>,<...>`` env strings."""
        out: list[MigrationSpec] = []
        for raw in (value or "").split(","):
            raw = raw.strip()
            if not raw:
                continue
            parts = raw.split(":")
            seen_prefix = parts[0] if len(parts) > 0 else ""
            stem_prefix = parts[1] if len(parts) > 1 else ""
            sorted_src = parts[2] if len(parts) > 2 else seen_prefix
            seen_file = base / f"seen_{seen_prefix}_{slug}.json" if seen_prefix else None
            # sorted_dir is the parent containing per-category subfolders;
            # callers pass the slug-level root so we can scan every category.
            out.append(cls(
                seen_file=seen_file,
                sorted_dir=(sorted_root / sorted_src) if sorted_src else None,
                stem_prefix=stem_prefix,
            ))
        return out


class SeenStore:
    """Per-source dedup index backed by a single JSON file."""

    def __init__(
        self,
        name: str,
        slug: str | None = None,
        base_dir: Path | None = None,
        autoflush_every: int = 25,
        migrations: Iterable[MigrationSpec] = (),
    ) -> None:
        self.name = name
        self.slug = slug or os.environ.get("PIPELINE_SLUG", "default")
        self.base_dir = base_dir or _default_base_dir()
        self.path: Path = self.base_dir / f"seen_{self.name}_{self.slug}.json"
        self.autoflush_every = max(1, autoflush_every)
        self._migrations = tuple(migrations)
        self._lock = threading.Lock()
        self._dirty: bool = False
        self._adds_since_flush: int = 0
        self._set: set[str] = set()
        self._load()

    # ── Public API ───────────────────────────────────────────────────────

    def contains(self, key: str) -> bool:
        return key in self._set

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self._set

    def __len__(self) -> int:
        return len(self._set)

    def add(self, key: str) -> bool:
        """Insert a key. Returns True if the key was new (not previously present)."""
        with self._lock:
            if key in self._set:
                return False
            self._set.add(key)
            self._dirty = True
            self._adds_since_flush += 1
            should_flush = self._adds_since_flush >= self.autoflush_every
        if should_flush:
            self.flush()
        return True

    def discard(self, key: str) -> bool:
        """Remove a key if present. Returns True if it was actually removed.

        Useful when a download fails after the ID was speculatively added so
        the next run can retry instead of treating the failure as permanent.
        """
        with self._lock:
            if key not in self._set:
                return False
            self._set.discard(key)
            self._dirty = True
        return True

    def update(self, keys: Iterable[str]) -> int:
        """Bulk insert. Returns the number of newly-added keys."""
        new = 0
        with self._lock:
            for k in keys:
                if k not in self._set:
                    self._set.add(k)
                    new += 1
            if new:
                self._dirty = True
        if new:
            self.flush()
        return new

    def flush(self) -> None:
        """Persist to disk if dirty. Atomic via tempfile + replace."""
        with self._lock:
            if not self._dirty:
                return
            payload = json.dumps(sorted(self._set))
            self._dirty = False
            self._adds_since_flush = 0
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(payload, encoding="utf-8")
            tmp.replace(self.path)
        except OSError as exc:
            logger.warning("seen store flush failed for %s: %s", self.name, exc)

    def snapshot(self) -> frozenset[str]:
        """Return a frozen copy of the current key set (for testing / inspection)."""
        with self._lock:
            return frozenset(self._set)

    # ── Internal ─────────────────────────────────────────────────────────

    def _load(self) -> None:
        for path in [self.path] + [m.seen_file for m in self._migrations if m.seen_file]:
            if path is None or not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(payload, list):
                    self._set.update(str(x) for x in payload)
                elif isinstance(payload, dict) and "seen" in payload:
                    self._set.update(str(x) for x in payload.get("seen", []))
            except (OSError, json.JSONDecodeError):
                logger.warning("seen file corrupt, skipping: %s", path)
        # Migrations from sorted-folder scans (one-way; we adopt the IDs).
        for spec in self._migrations:
            if spec.sorted_dir is None:
                continue
            self._set.update(self._adopt_from_sorted(spec))

    @staticmethod
    def _adopt_from_sorted(spec: MigrationSpec) -> set[str]:
        """Walk a legacy sorted folder and return every stem prefixed with
        ``spec.stem_prefix`` (with the prefix stripped)."""
        out: set[str] = set()
        if spec.sorted_dir is None or not spec.sorted_dir.exists():
            return out
        prefix = f"{spec.stem_prefix}_" if spec.stem_prefix else ""
        for path in spec.sorted_dir.rglob("*"):
            if not path.is_file():
                continue
            stem = path.stem
            if prefix and stem.startswith(prefix):
                out.add(stem[len(prefix):])
            elif not prefix:
                out.add(stem)
        return out


__all__ = ["SeenStore", "MigrationSpec"]
