"""Path-injection barrier tests for the cull backend.

cull is a single-user localhost tool whose per-job filesystem layout is keyed by
a *slug*. Slugs are constrained to ``^[a-z0-9_]+$`` (no path separators, no
dots), which makes them safe to interpolate into a path. CodeQL's
``py/path-injection`` query cannot see that constraint, so each module that turns
a slug (or another user/model-derived component) into a filesystem path now runs
it through a shared, CodeQL-recognised barrier before use:

  * :func:`paths.validate_slug` — the slug charset barrier.
  * ``job_config`` reuses its own ``JOB_SLUG_RE.fullmatch`` for the same effect.
  * :func:`thumb_cache` confines a thumbnail source to the queue / sorted roots
    with a ``realpath`` + ``commonpath`` containment check.
  * :func:`vision_worker_base._safe_component` reduces a model/queue-derived
    category / source name to a single safe path component.

These tests assert the barriers accept every real slug and reject traversal —
without changing behaviour for legitimate inputs.

Runnable:
    pytest tests/test_path_safety.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_CODE = REPO_ROOT / "pipeline_code"
TOOLS = REPO_ROOT / "tools"
for _p in (PIPELINE_CODE, TOOLS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import paths  # noqa: E402


# Slugs the constraint must always accept (real, in-the-wild job slugs).
VALID_SLUGS = ("default", "my_job_2", "demo", "a", "x_y_z", "job123", "0")

# Strings the barrier must reject — traversal, separators, dots, empties.
TRAVERSAL_SLUGS = (
    "../x",
    "..",
    "a/b",
    "a\\b",
    "a.b",
    "",
    "/etc/passwd",
    "../../secret",
    "a/../b",
    ".",
    "Has Space",
    "UPPER",
    "dash-dash",
)


# ── paths.validate_slug ───────────────────────────────────────────────────────


@pytest.mark.parametrize("slug", VALID_SLUGS)
def test_validate_slug_accepts_real_slugs(slug):
    # Arrange / Act
    result = paths.validate_slug(slug)
    # Assert — returns the slug unchanged (identity for valid input).
    assert result == slug


@pytest.mark.parametrize("slug", TRAVERSAL_SLUGS)
def test_validate_slug_rejects_traversal(slug):
    with pytest.raises(ValueError):
        paths.validate_slug(slug)


def test_validate_slug_rejects_none():
    with pytest.raises(ValueError):
        paths.validate_slug(None)  # type: ignore[arg-type]


def test_validate_slug_exported():
    assert "validate_slug" in paths.__all__


# ── job_config reuses its own JOB_SLUG_RE.fullmatch barrier ───────────────────


def test_job_config_path_builder_rejects_traversal():
    import job_config

    for bad in ("../x", "a/b", "a.b", ""):
        with pytest.raises(ValueError):
            job_config._job_path(bad)


def test_job_config_get_job_rejects_traversal_returns_none(tmp_path, monkeypatch):
    import job_config

    # ``job_config`` resolves its base dir lazily (per call), so set the env and
    # call directly — no module reload needed (reloading would rebind the shared
    # module object and pollute later dashboard tests).
    monkeypatch.setenv("PIPELINE_BASE_DIR", str(tmp_path / "data"))
    # A traversal slug must never resolve to a file read — it returns None
    # rather than building an escaping path.
    assert job_config.get_job("../../etc/passwd") is None
    assert job_config.get_job("a/b") is None


# ── EmbeddingIndex constrains the slug at construction ────────────────────────


def test_embedding_index_rejects_traversal_slug(tmp_path, monkeypatch):
    import embeddings_index

    # base_dir is resolved per-construction, so pass it explicitly and avoid a
    # module reload (which would pollute later tests sharing the module).
    monkeypatch.setenv("PIPELINE_BASE_DIR", str(tmp_path / "data"))

    # A real slug builds a path inside the embeddings dir...
    idx = embeddings_index.EmbeddingIndex(slug="good_job", base_dir=tmp_path / "data")
    assert idx.path.name == "good_job.jsonl"
    assert idx.dir in idx.path.parents

    # ...a traversal slug raises rather than escaping the embeddings dir.
    for bad in ("../evil", "a/b", "a.b"):
        with pytest.raises(ValueError):
            embeddings_index.EmbeddingIndex(slug=bad)


# ── active_learning constrains the slug at the path boundary ──────────────────


def test_active_learning_store_path_rejects_traversal(tmp_path, monkeypatch):
    import active_learning

    # ``_store_dir`` resolves the base dir lazily, so set the env and call
    # directly — no reload (reloading would rebind the shared module object).
    monkeypatch.setenv("PIPELINE_BASE_DIR", str(tmp_path / "data"))

    # Real slug → a file directly inside the active_learning dir.
    good = active_learning._store_path("good_job")
    assert good.name == "good_job.json"
    assert good.parent.name == "active_learning"

    for bad in ("../evil", "a/b", "a.b", ""):
        with pytest.raises(ValueError):
            active_learning._store_path(bad)


# ── thumb_cache containment (realpath + commonpath) ───────────────────────────


def _write_png(path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), "red").save(path)


@pytest.fixture()
def thumb_cache_isolated():
    """Yield ``thumb_cache`` with its module globals saved/restored.

    We deliberately do NOT reload the module — other (dashboard) tests share the
    same module object, so we snapshot ``_root`` / ``_allowed_source_roots`` and
    restore them afterwards rather than rebinding the module.
    """
    import thumb_cache

    saved_root = thumb_cache._root
    saved_roots = thumb_cache._allowed_source_roots
    try:
        yield thumb_cache
    finally:
        thumb_cache._root = saved_root
        thumb_cache._allowed_source_roots = saved_roots


def test_thumb_cache_rejects_source_outside_roots(tmp_path, thumb_cache_isolated):
    thumb_cache = thumb_cache_isolated

    queue = tmp_path / "queue"
    sorted_ = tmp_path / "sorted"
    cache = tmp_path / "thumb_cache"
    thumb_cache.configure(cache, allowed_source_roots=[queue, sorted_])

    # An image inside an allowed root thumbnails fine.
    inside = queue / "civitai" / "a.png"
    _write_png(inside)
    digest = thumb_cache._hash_for(inside)
    assert isinstance(digest, str) and len(digest) == 40
    cache_file = thumb_cache.get_or_create(inside, 240)
    assert cache_file.exists()

    # An image OUTSIDE the allowed roots is rejected before any disk read.
    outside = tmp_path / "outside" / "secret.png"
    _write_png(outside)
    with pytest.raises(ValueError):
        thumb_cache._hash_for(outside)
    with pytest.raises(ValueError):
        thumb_cache.get_or_create(outside, 240)


def test_thumb_cache_blocks_dotdot_traversal(tmp_path, thumb_cache_isolated):
    thumb_cache = thumb_cache_isolated

    queue = tmp_path / "queue"
    thumb_cache.configure(tmp_path / "thumb_cache", allowed_source_roots=[queue])
    queue.mkdir(parents=True, exist_ok=True)

    # A crafted ``..`` path that escapes the root resolves outside → rejected.
    escaping = queue / ".." / "outside.png"
    _write_png(tmp_path / "outside.png")
    with pytest.raises(ValueError):
        thumb_cache._hash_for(escaping)


# ── vision_worker_base._safe_component ────────────────────────────────────────


def test_safe_component_passes_real_names():
    import vision_worker_base as vwb

    # Real category / source names survive unchanged.
    for name in ("Keep", "Borderline", "DISCARD", "civitai", "x", "reddit_2"):
        assert vwb._safe_component(name, "Unknown") == name


def test_safe_component_strips_traversal():
    import vision_worker_base as vwb

    assert vwb._safe_component("../../etc", "Unknown") == "etc"
    assert vwb._safe_component("a/b/c", "Unknown") == "c"
    assert vwb._safe_component("a\\b\\c", "Unknown") == "c"
    # Pure-traversal / empty inputs fall back to the supplied default.
    assert vwb._safe_component("..", "Unknown") == "Unknown"
    assert vwb._safe_component("", "Unknown") == "Unknown"
    assert vwb._safe_component("   ", "src") == "src"
    assert vwb._safe_component("/", "Unknown") == "Unknown"


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-q"]))
