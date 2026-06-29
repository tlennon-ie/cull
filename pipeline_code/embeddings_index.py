"""Persistent per-slug store of ``(key, embedding-vector)`` pairs.

This is the on-disk home for the CLIP-style vectors produced by
:func:`embeddings.embed_image`. It is **its own storage** — deliberately
independent of :mod:`index_store` (the SQLite image index). Embedding vectors are
large, append-heavy, and only needed by the embedding features (dedup /
balancing / semantic search), so co-mingling them with the perf-critical SQLite
index would bloat it for no benefit. Keeping them in a separate JSONL file per
slug means the SQLite index is never touched by this module.

Layout (under the paths base dir, i.e. ``data/`` by default):

    data/embeddings/<slug>.jsonl

Each line is a JSON object: ``{"key": "<image path>", "vec": [floats...]}``.
JSONL was chosen over ``.npz`` so the store works with **zero ML/numpy
dependencies** (matching ``embeddings.py``'s pure-Python vector layer), is
append-friendly, human-inspectable, and trivially crash-safe. The last line for
a given key wins, so re-embedding an image is a cheap append; ``flush()``
rewrites the file compacted (one line per key).

Contract (mirrors the conventions of :class:`seen_store.SeenStore`):

    >>> idx = EmbeddingIndex(slug="my_job")
    >>> idx.add("/data/sorted/my_job/Keep/civitai/img.png", embed_image(path))
    >>> idx.flush()                                  # atomic write
    >>> idx.search(query_vec, top_k=12)              # nearest by cosine
    >>> idx.get(path)                                # exact vector or None

* In-memory dict is the source of truth for fast get/add/search.
* Loads tolerate a missing / corrupt / partially-written file (returns empty +
  logs a warning, never raises) — same graceful-failure stance as
  ``index_store`` and ``seen_store``.
* Writes are atomic via a tempfile + ``Path.replace`` so a crash mid-write can't
  truncate the index.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Iterator, Sequence

from paths import base_dir as _default_base_dir
from paths import validate_slug as _validate_slug
from pipeline_logging import get_logger

import embeddings

logger = get_logger(__name__)

# Subdirectory under the base dir that holds the per-slug vector files. Its own
# namespace — never the SQLite DB used by index_store.
EMBEDDINGS_DIRNAME = "embeddings"


class EmbeddingIndex:
    """Per-slug persistent store of image-key → embedding vector."""

    def __init__(
        self,
        slug: str | None = None,
        base_dir: Path | None = None,
        autoflush_every: int = 64,
    ) -> None:
        # Sanitise the slug before it becomes part of the on-disk path. The
        # charset barrier rejects '/', '\\' and '.' so the per-slug filename can
        # never escape the embeddings dir (path-injection barrier for CodeQL).
        self.slug = _validate_slug(slug or os.environ.get("PIPELINE_SLUG", "default"))
        self.base_dir = base_dir or _default_base_dir()
        self.dir: Path = self.base_dir / EMBEDDINGS_DIRNAME
        self.path: Path = self.dir / f"{self.slug}.jsonl"
        self.autoflush_every = max(1, autoflush_every)
        self._lock = threading.Lock()
        self._dirty = False
        self._adds_since_flush = 0
        self._vectors: dict[str, list[float]] = {}
        self._load()

    # ── public API ────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._vectors)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self._vectors

    def add(self, key: str, vector: "Sequence[float]") -> None:
        """Insert or overwrite the vector for ``key``.

        The vector is copied into a plain ``list[float]`` so tensor / numpy types
        never leak into the store (and JSON serialisation always works).
        """
        vec = [float(x) for x in vector]
        with self._lock:
            self._vectors[key] = vec
            self._dirty = True
            self._adds_since_flush += 1
            should_flush = self._adds_since_flush >= self.autoflush_every
        if should_flush:
            self.flush()

    def get(self, key: str) -> list[float] | None:
        """Return a copy of the stored vector for ``key``, or ``None`` if absent."""
        with self._lock:
            vec = self._vectors.get(key)
            return list(vec) if vec is not None else None

    def discard(self, key: str) -> bool:
        """Remove ``key`` if present. Returns ``True`` if it was actually removed."""
        with self._lock:
            if key not in self._vectors:
                return False
            del self._vectors[key]
            self._dirty = True
            return True

    def iter(self) -> Iterator[tuple[str, list[float]]]:
        """Yield ``(key, vector)`` for every stored embedding.

        Iterates over a snapshot so callers can mutate the index concurrently
        (e.g. a backfill adding rows) without a ``RuntimeError``.
        """
        with self._lock:
            snapshot = list(self._vectors.items())
        for key, vec in snapshot:
            yield key, list(vec)

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._vectors.keys())

    def search(
        self, query_vec: "Sequence[float]", top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Return the ``top_k`` nearest stored vectors to ``query_vec``.

        Thin wrapper over :func:`embeddings.semantic_search` operating on this
        store's snapshot — ``(key, cosine_score)`` pairs, highest first.
        """
        return embeddings.semantic_search(query_vec, self.iter(), top_k=top_k)

    def flush(self) -> None:
        """Persist the index to disk if dirty. Atomic via tempfile + replace.

        Rewrites the whole file compacted (one line per key) rather than
        appending, so the on-disk file never accumulates superseded rows.
        """
        with self._lock:
            if not self._dirty:
                return
            lines = [
                json.dumps({"key": key, "vec": vec}, ensure_ascii=False)
                for key, vec in self._vectors.items()
            ]
            self._dirty = False
            self._adds_since_flush = 0
        payload = "\n".join(lines) + ("\n" if lines else "")
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(payload, encoding="utf-8")
            tmp.replace(self.path)
        except OSError as exc:
            logger.warning(
                "embedding index flush failed for slug %s: %s", self.slug, exc,
            )

    # ── internal ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Read the JSONL file into memory, tolerating missing / corrupt data.

        Parses line-by-line so a single malformed / truncated line (e.g. from a
        crash mid-append) only drops that row instead of failing the whole load.
        The last valid line for a key wins, matching ``add``'s overwrite
        semantics.
        """
        if not self.path.exists():
            return
        try:
            text = self.path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning(
                "embedding index unreadable for slug %s: %s", self.slug, exc,
            )
            return
        loaded = 0
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                key = obj["key"]
                vec = obj["vec"]
            except (json.JSONDecodeError, KeyError, TypeError):
                logger.debug("embedding index: skipping bad line in %s", self.path)
                continue
            if isinstance(key, str) and isinstance(vec, list):
                try:
                    self._vectors[key] = [float(x) for x in vec]
                    loaded += 1
                except (TypeError, ValueError):
                    continue
        if loaded:
            logger.debug(
                "loaded %d embeddings for slug %s from %s",
                loaded, self.slug, self.path,
            )


__all__ = ["EmbeddingIndex", "EMBEDDINGS_DIRNAME"]
