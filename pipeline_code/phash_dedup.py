"""Perceptual-hash near-duplicate detection for cull.

A self-contained difference-hash (dHash) built on Pillow + the standard library
only — no ``imagehash``/``numpy`` dependency. dHash is cheap, robust to small
crops / re-encodes / resizes, and (unlike average-hash) tolerant of gamma /
brightness shifts, which makes it a good fit for spotting re-uploads and
near-identical AI generations.

Public surface:

    compute_phash(image)            -> str | None   (16-char hex, 64-bit dHash)
    hamming_distance(a, b)          -> int
    is_duplicate(a, b, threshold)   -> bool
    find_near_duplicates(...)       -> list[list[str]]  (groups of index keys)

Design notes
------------
* **dHash algorithm.** Reduce to grayscale, resize to ``(HASH_SIZE+1) x HASH_SIZE``
  (default 9x8), then for each row compare each pixel to its right-hand
  neighbour: bit = 1 when the left pixel is brighter. That yields
  ``HASH_SIZE * HASH_SIZE`` = 64 bits, serialised as zero-padded hex.
* **Graceful failure.** Unreadable / corrupt / missing inputs return ``None``
  (and are skipped by the scan) rather than raising — a single bad file must
  never crash the indexer or a dashboard request.
* **Scanning.** ``find_near_duplicates`` reads stored hashes from
  ``index_store.iter_phashes`` and clusters them with union-find so that
  transitively-near images land in one group. It is an O(n^2) pairwise compare;
  fine for the tens-of-thousands range cull targets, and trivially cut down by
  the optional ``slug`` filter.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable

from PIL import Image

from pipeline_logging import get_logger

import index_store

logger = get_logger(__name__)

# dHash geometry. HASH_SIZE rows, each compared against HASH_SIZE+1 columns,
# producing HASH_SIZE*HASH_SIZE bits. 8 -> 64-bit hash -> 16 hex chars.
HASH_SIZE: int = 8
_HASH_BITS: int = HASH_SIZE * HASH_SIZE
_HEX_WIDTH: int = _HASH_BITS // 4
DEFAULT_THRESHOLD: int = 10

ImageInput = "Path | str | bytes | bytearray"


# ── hashing ───────────────────────────────────────────────────────────────────

def _open_image(image: "Path | str | bytes | bytearray") -> Image.Image | None:
    """Open ``image`` (a path, path-string, or raw bytes) as a PIL image.

    Returns ``None`` for anything unreadable — missing file, unsupported /
    corrupt data, truncated stream — so callers never have to wrap this in a
    try/except of their own.
    """
    try:
        if isinstance(image, (bytes, bytearray)):
            img = Image.open(io.BytesIO(bytes(image)))
        else:
            img = Image.open(Path(image))
        img.load()
        return img
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        logger.debug("phash: could not open image (%r): %s", _describe(image), exc)
        return None


def _describe(image: "Path | str | bytes | bytearray") -> str:
    if isinstance(image, (bytes, bytearray)):
        return f"<{len(image)} bytes>"
    return str(image)


def compute_phash(image: "Path | str | bytes | bytearray") -> str | None:
    """Compute the dHash of ``image`` as a 16-char hex string.

    ``image`` may be a filesystem path (``Path`` or ``str``) or raw image
    ``bytes``. Returns ``None`` if the image can't be read/decoded.
    """
    img = _open_image(image)
    if img is None:
        return None
    try:
        # Grayscale + downscale to (HASH_SIZE+1) wide, HASH_SIZE tall.
        small = img.convert("L").resize(
            (HASH_SIZE + 1, HASH_SIZE), Image.LANCZOS,
        )
        pixels = list(small.getdata())  # row-major, len == (HASH_SIZE+1)*HASH_SIZE
    except (OSError, ValueError) as exc:
        logger.debug("phash: failed to reduce image (%r): %s", _describe(image), exc)
        return None
    finally:
        img.close()

    row_stride = HASH_SIZE + 1
    bits = 0
    for row in range(HASH_SIZE):
        base = row * row_stride
        for col in range(HASH_SIZE):
            left = pixels[base + col]
            right = pixels[base + col + 1]
            bits = (bits << 1) | (1 if left > right else 0)
    return format(bits, f"0{_HEX_WIDTH}x")


# ── comparison ────────────────────────────────────────────────────────────────

def hamming_distance(a: str, b: str) -> int:
    """Number of differing bits between two hex-encoded hashes.

    Raises ``ValueError`` if the hashes are different lengths or not valid hex
    (comparing hashes of different bit-widths is meaningless).
    """
    if len(a) != len(b):
        raise ValueError(
            f"hash length mismatch: {len(a)} != {len(b)} (cannot compare)"
        )
    try:
        return bin(int(a, 16) ^ int(b, 16)).count("1")
    except ValueError as exc:
        raise ValueError(f"invalid hex hash: {a!r} / {b!r}") from exc


def is_duplicate(a: str, b: str, threshold: int = DEFAULT_THRESHOLD) -> bool:
    """True when ``a`` and ``b`` are within ``threshold`` bits of each other."""
    return hamming_distance(a, b) <= threshold


# ── scanning the index for near-duplicate groups ──────────────────────────────

class _UnionFind:
    """Minimal union-find over integer indices for clustering near-dupes."""

    def __init__(self, size: int) -> None:
        self._parent = list(range(size))

    def find(self, x: int) -> int:
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        # Path compression.
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra


def _valid_pairs(
    pairs: Iterable[tuple[str, str | None]],
) -> tuple[list[str], list[str]]:
    """Split ``(key, phash)`` pairs into parallel lists, dropping null hashes."""
    keys: list[str] = []
    hashes: list[str] = []
    for key, phash in pairs:
        if phash:
            keys.append(key)
            hashes.append(phash)
    return keys, hashes


def find_near_duplicates(
    threshold: int = DEFAULT_THRESHOLD,
    slug: str | None = None,
) -> list[list[str]]:
    """Group stored images whose perceptual hashes are within ``threshold``.

    Reads ``(key, phash)`` pairs from :mod:`index_store` (optionally filtered to
    a single ``slug``), skips rows without a computed hash, and returns a list of
    groups — each a list of index keys (image paths) — where every member is
    transitively near at least one other. Singletons are omitted; the result is
    empty when no near-duplicates exist.
    """
    keys, hashes = _valid_pairs(index_store.iter_phashes(slug=slug))
    n = len(keys)
    if n < 2:
        return []

    uf = _UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            # Hashes from the same index are uniform width; guard anyway so a
            # stray malformed value can't abort the whole scan.
            try:
                if hamming_distance(hashes[i], hashes[j]) <= threshold:
                    uf.union(i, j)
            except ValueError:
                continue

    clusters: dict[int, list[str]] = {}
    for idx in range(n):
        clusters.setdefault(uf.find(idx), []).append(keys[idx])

    return [members for members in clusters.values() if len(members) > 1]


__all__ = [
    "HASH_SIZE",
    "DEFAULT_THRESHOLD",
    "compute_phash",
    "hamming_distance",
    "is_duplicate",
    "find_near_duplicates",
]
