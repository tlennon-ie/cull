"""Tests for queue_manager multi-slug pop.

Runnable two ways:
    pytest tests/test_queue_manager_multi.py
    python tests/test_queue_manager_multi.py

Covers ``pop_next_across_slugs`` (weighted round-robin) and the module-level
``get_next_image_round_robin`` shim's delegation when ``PIPELINE_ACTIVE_SLUGS_JSON``
is set — the same shim vision workers call every poll tick.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

PIPELINE_CODE = Path(__file__).resolve().parent.parent / "pipeline_code"
if str(PIPELINE_CODE) not in sys.path:
    sys.path.insert(0, str(PIPELINE_CODE))


@pytest.fixture()
def qm(tmp_path, monkeypatch):
    """Reload queue_manager against a temp base dir so its module-level state
    (BASE_QUEUE_DIR, _QUEUE_ROOT) points at the temp store."""
    monkeypatch.setenv("PIPELINE_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("PIPELINE_QUEUE", str(tmp_path / "queue"))
    monkeypatch.setenv("PIPELINE_SORTED", str(tmp_path / "sorted"))
    monkeypatch.delenv("PIPELINE_ACTIVE_SLUGS_JSON", raising=False)
    import queue_manager
    importlib.reload(queue_manager)
    return queue_manager, tmp_path


def _seed_queue_image(root: Path, slug: str, source: str, name: str) -> Path:
    """Create a >5KB image file so FSQueue.save-style consumers see it."""
    d = root / "queue" / slug / source
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_bytes(b"x" * 6000)
    return p


def test_pop_across_slugs_empty_returns_none(qm):
    q, _ = qm
    assert q.pop_next_across_slugs(["ghost"]) == (None, None, None)
    assert q.pop_next_across_slugs([]) == (None, None, None)


def test_pop_across_slugs_returns_slug_source_path(qm):
    q, tmp = qm
    _seed_queue_image(tmp, "job_a", "civitai", "a1.jpg")
    slug, source, path = q.pop_next_across_slugs(["job_a", "job_b"])
    assert slug == "job_a"
    assert source == "civitai"
    assert path.name == "a1.jpg"


def test_pop_across_slugs_weighted_round_robin(qm):
    """Weight ``w`` gives that slug w consecutive turns per cycle. With
    heavy=5 and light=1, five consecutive heavy pops precede a light pop."""
    q, tmp = qm
    for i in range(6):
        _seed_queue_image(tmp, "heavy", "civitai", f"h{i}.jpg")
    for i in range(6):
        _seed_queue_image(tmp, "light", "civitai", f"l{i}.jpg")
    slugs = ["heavy", "light"]
    weights = {"heavy": 5, "light": 1}
    popped = []
    for _ in range(6):
        slug, _source, _path = q.pop_next_across_slugs(slugs, weights)
        popped.append(slug)
    # First 5 pops should all be from heavy, then 1 light.
    assert popped[:5] == ["heavy"] * 5
    assert popped[5] == "light"


def test_pop_across_slugs_skips_exhausted_slug(qm):
    """Empty slugs must be silently skipped — the schedule falls through to
    the next slug with items."""
    q, tmp = qm
    _seed_queue_image(tmp, "has_items", "civitai", "x.jpg")
    # empty_a has no queue dir; empty_b has a dir but no files.
    (tmp / "queue" / "empty_b" / "civitai").mkdir(parents=True)
    slug, _source, path = q.pop_next_across_slugs(
        ["empty_a", "empty_b", "has_items"], {"has_items": 1}
    )
    assert slug == "has_items"
    assert path is not None


def test_shim_delegates_when_active_slugs_env_set(qm, monkeypatch):
    """The module-level ``get_next_image_round_robin`` reads
    PIPELINE_ACTIVE_SLUGS_JSON and delegates to pop_next_across_slugs; the
    return tuple drops the slug for legacy caller compatibility."""
    q, tmp = qm
    _seed_queue_image(tmp, "job_a", "civitai", "a.jpg")
    _seed_queue_image(tmp, "job_b", "civitai", "b.jpg")
    monkeypatch.setenv("PIPELINE_ACTIVE_SLUGS_JSON",
                       json.dumps([["job_a", 2], ["job_b", 1]]))
    source, path = q.get_next_image_round_robin()
    assert source == "civitai"
    assert path.name in ("a.jpg", "b.jpg")
    # Path is under the slug's queue dir — vision worker can derive the slug.
    assert "job_a" in path.parts or "job_b" in path.parts


def test_shim_falls_back_to_default_queue_when_env_absent(qm, monkeypatch):
    """No PIPELINE_ACTIVE_SLUGS_JSON → historic FSQueue path (single slug from
    the PIPELINE_QUEUE env). Behaviour must be byte-identical to pre-multi-active."""
    q, tmp = qm
    monkeypatch.delenv("PIPELINE_ACTIVE_SLUGS_JSON", raising=False)
    # Force the default queue at (tmp/queue/default) so the shim's shim has
    # somewhere to look. Absent items → (None, None).
    source, path = q.get_next_image_round_robin()
    assert (source, path) == (None, None)


def test_shim_handles_malformed_active_slugs_json(qm, monkeypatch):
    q, _ = qm
    for blob in ("not json", "{}", "[]", '["only_a_string"]'):
        monkeypatch.setenv("PIPELINE_ACTIVE_SLUGS_JSON", blob)
        # Never raises. Empty / malformed just returns (None, None) via the
        # underlying fallback / empty-pop path.
        _ = q.get_next_image_round_robin()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
