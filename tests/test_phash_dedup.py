"""Tests for perceptual-hash near-duplicate detection (``phash_dedup``).

``phash_dedup`` implements a self-contained dHash (difference hash) on top of
Pillow + the standard library only — no ``imagehash``/``numpy``. The companion
``index_store`` change adds a nullable ``phash`` column so stored hashes can be
scanned for near-duplicate groups.

Runnable:
    pytest tests/test_phash_dedup.py -q
    python tests/test_phash_dedup.py
"""
from __future__ import annotations

import importlib
import io
import sys
from pathlib import Path

import pytest
from PIL import Image

PIPELINE_CODE = Path(__file__).resolve().parent.parent / "pipeline_code"
if str(PIPELINE_CODE) not in sys.path:
    sys.path.insert(0, str(PIPELINE_CODE))


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def phash():
    import phash_dedup
    return importlib.reload(phash_dedup)


@pytest.fixture()
def index(tmp_path, monkeypatch):
    """A freshly-configured index_store pointed at a temp SQLite DB."""
    monkeypatch.setenv("PIPELINE_BASE_DIR", str(tmp_path))
    import index_store
    importlib.reload(index_store)
    index_store.configure(tmp_path / index_store.DB_FILENAME)
    return index_store


# ── tiny image builders (in-memory) ───────────────────────────────────────────

def _gradient(size: int = 64, shift: int = 0) -> Image.Image:
    """A smooth horizontal grayscale gradient, optionally shifted by `shift`px."""
    img = Image.new("L", (size, size))
    px = img.load()
    for y in range(size):
        for x in range(size):
            px[x, y] = (x + shift) % 256
    return img


def _checkerboard(size: int = 64, cell: int = 8) -> Image.Image:
    """A high-frequency checkerboard — visually very different to a gradient."""
    img = Image.new("L", (size, size))
    px = img.load()
    for y in range(size):
        for x in range(size):
            px[x, y] = 255 if ((x // cell) + (y // cell)) % 2 == 0 else 0
    return img


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── compute_phash ─────────────────────────────────────────────────────────────

def test_phash_is_16_char_hex(phash):
    h = phash.compute_phash(_png_bytes(_gradient()))
    assert isinstance(h, str)
    assert len(h) == 16
    int(h, 16)  # raises ValueError if not valid hex


def test_identical_images_hash_equal(phash):
    a = phash.compute_phash(_png_bytes(_gradient()))
    b = phash.compute_phash(_png_bytes(_gradient()))
    assert a == b
    assert phash.hamming_distance(a, b) == 0


def test_accepts_path_and_bytes_equivalently(phash, tmp_path):
    img = _gradient()
    p = tmp_path / "g.png"
    img.save(p)
    from_bytes = phash.compute_phash(_png_bytes(img))
    from_path = phash.compute_phash(p)
    from_str = phash.compute_phash(str(p))
    assert from_bytes == from_path == from_str


# ── hamming_distance + thresholds ─────────────────────────────────────────────

def test_near_identical_small_distance(phash):
    base = phash.compute_phash(_png_bytes(_gradient()))
    shifted = phash.compute_phash(_png_bytes(_gradient(shift=1)))
    dist = phash.hamming_distance(base, shifted)
    assert dist == 0 or dist <= 10  # a 1px gradient shift barely moves the hash


def test_different_images_large_distance(phash):
    grad = phash.compute_phash(_png_bytes(_gradient()))
    check = phash.compute_phash(_png_bytes(_checkerboard()))
    assert phash.hamming_distance(grad, check) > 10


def test_is_duplicate_respects_threshold(phash):
    grad = phash.compute_phash(_png_bytes(_gradient()))
    shifted = phash.compute_phash(_png_bytes(_gradient(shift=1)))
    check = phash.compute_phash(_png_bytes(_checkerboard()))
    assert phash.is_duplicate(grad, shifted) is True
    assert phash.is_duplicate(grad, check) is False
    # threshold of 0 demands exactness
    assert phash.is_duplicate(grad, grad, threshold=0) is True


def test_hamming_distance_rejects_mismatched_lengths(phash):
    with pytest.raises(ValueError):
        phash.hamming_distance("ffff", "ffffffffffffffff")


# ── graceful handling of unreadable input ─────────────────────────────────────

def test_corrupt_bytes_returns_none(phash):
    assert phash.compute_phash(b"not an image") is None


def test_missing_path_returns_none(phash, tmp_path):
    assert phash.compute_phash(tmp_path / "does_not_exist.png") is None


# ── index_store phash column round-trip ───────────────────────────────────────

def _seed_row(index_store, path: str, phash_val: str | None) -> None:
    index_store.upsert({
        "path": path,
        "status": "sorted",
        "topic_slug": "demo",
        "source": "civitai",
        "category": "Keep",
        "mtime": 1.0,
        "size": 10,
        "width": 64,
        "height": 64,
        "ovr": None,
        "rel": None,
        "quality": None,
        "nsfw": 0,
        "prompt": "",
        "vision_json": None,
    })
    if phash_val is not None:
        index_store.set_phash(path, phash_val)


def test_phash_column_exists_and_defaults_null(index):
    _seed_row(index, "/data/a.png", None)
    pairs = dict(index.iter_phashes())
    assert pairs == {"/data/a.png": None}


def test_set_phash_round_trips(index):
    _seed_row(index, "/data/a.png", "0123456789abcdef")
    pairs = dict(index.iter_phashes())
    assert pairs["/data/a.png"] == "0123456789abcdef"


def test_upsert_preserves_existing_phash(index):
    """Re-indexing an image (mtime/size change) must not wipe its phash."""
    _seed_row(index, "/data/a.png", "0123456789abcdef")
    # Simulate a re-scan upsert that doesn't carry phash.
    _seed_row(index, "/data/a.png", None)
    pairs = dict(index.iter_phashes())
    assert pairs["/data/a.png"] == "0123456789abcdef"


def test_iter_phashes_filters_by_slug(index):
    index.upsert({
        "path": "/data/b.png", "status": "sorted", "topic_slug": "other",
        "source": "x", "category": "Keep", "mtime": 1.0, "size": 1,
        "width": None, "height": None, "ovr": None, "rel": None,
        "quality": None, "nsfw": 0, "prompt": "", "vision_json": None,
    })
    _seed_row(index, "/data/a.png", "0123456789abcdef")
    index.set_phash("/data/b.png", "fedcba9876543210")
    demo = dict(index.iter_phashes(slug="demo"))
    assert set(demo) == {"/data/a.png"}


# ── find_near_duplicates (end-to-end against the index) ───────────────────────

def test_find_near_duplicates_groups_dupes(phash, index):
    grad = _gradient()
    grad_shift = _gradient(shift=1)
    check = _checkerboard()

    rows = {
        "/data/grad1.png": phash.compute_phash(_png_bytes(grad)),
        "/data/grad2.png": phash.compute_phash(_png_bytes(grad_shift)),
        "/data/check.png": phash.compute_phash(_png_bytes(check)),
    }
    for path, h in rows.items():
        _seed_row(index, path, h)

    groups = phash.find_near_duplicates(threshold=10)
    # Exactly one group, containing the two gradients (and not the checkerboard).
    assert len(groups) == 1
    group = sorted(groups[0])
    assert group == ["/data/grad1.png", "/data/grad2.png"]


def test_find_near_duplicates_ignores_null_phash(phash, index):
    _seed_row(index, "/data/a.png", None)
    _seed_row(index, "/data/b.png", None)
    assert phash.find_near_duplicates(threshold=10) == []


def test_find_near_duplicates_no_dupes_returns_empty(phash, index):
    _seed_row(index, "/data/grad.png", phash.compute_phash(_png_bytes(_gradient())))
    _seed_row(index, "/data/check.png", phash.compute_phash(_png_bytes(_checkerboard())))
    assert phash.find_near_duplicates(threshold=10) == []


def test_find_near_duplicates_scoped_by_slug(phash, index):
    g1 = phash.compute_phash(_png_bytes(_gradient()))
    g2 = phash.compute_phash(_png_bytes(_gradient(shift=1)))
    _seed_row(index, "/data/demo_a.png", g1)  # slug=demo
    index.upsert({
        "path": "/data/other_b.png", "status": "sorted", "topic_slug": "other",
        "source": "x", "category": "Keep", "mtime": 1.0, "size": 1,
        "width": None, "height": None, "ovr": None, "rel": None,
        "quality": None, "nsfw": 0, "prompt": "", "vision_json": None,
    })
    index.set_phash("/data/other_b.png", g2)
    # Scoped to demo, the cross-slug near-dup must not be grouped.
    assert phash.find_near_duplicates(threshold=10, slug="demo") == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
