"""Tests for the tools/embed_sorted.py backfill CLI.

``embed_sorted`` walks a job's ``data/sorted/<slug>/`` tree and embeds every
*kept* image into the per-slug :class:`embeddings_index.EmbeddingIndex`, skipping
the terminal DISCARD / CORRUPT buckets. The heavy CLIP model is import-guarded, so
here we mock it: ``embeddings.embed_image`` returns a tiny deterministic vector and
``embeddings.open_clip_available`` reports ``True``. That keeps the tests pure and
green in the base (no-ML) environment.

Runnable:
    pytest tests/test_embed_sorted.py -q
    python tests/test_embed_sorted.py
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PIPELINE_CODE = REPO_ROOT / "pipeline_code"
TOOLS = REPO_ROOT / "tools"
for _p in (PIPELINE_CODE, TOOLS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


SLUG = "demo_job"
KEPT_CATEGORY = "Keep"          # a keep bucket present in the default taxonomy
TERMINAL_CATEGORY = "DISCARD"   # terminal bucket — must never be embedded
SOURCE = "civitai"
FAKE_VECTOR = [0.1, 0.2, 0.3, 0.4]


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def tool(tmp_path, monkeypatch):
    """Reload embeddings / embeddings_index / embed_sorted against a temp data dir.

    Pointing ``PIPELINE_BASE_DIR`` at ``tmp_path`` isolates both the sorted tree
    we build and the JSONL index the tool writes. ``embed_image`` and
    ``open_clip_available`` are mocked so no real ML stack is needed.
    """
    monkeypatch.setenv("PIPELINE_BASE_DIR", str(tmp_path))

    import embeddings
    importlib.reload(embeddings)
    import embeddings_index
    importlib.reload(embeddings_index)
    import embed_sorted
    importlib.reload(embed_sorted)

    # Mock the heavy bits on the embeddings module the tool calls through.
    calls: list[str] = []

    def _fake_embed(image, **_kwargs):
        calls.append(str(image))
        return list(FAKE_VECTOR)

    monkeypatch.setattr(embeddings, "embed_image", _fake_embed)
    monkeypatch.setattr(embeddings, "open_clip_available", lambda: True)

    # Expose the call log for assertions.
    embed_sorted._test_calls = calls  # type: ignore[attr-defined]
    return embed_sorted


def _make_image(base_dir: Path, category: str, name: str) -> Path:
    """Create a stub image (+ sidecars) under sorted/<slug>/<category>/<source>/."""
    folder = base_dir / "sorted" / SLUG / category / SOURCE
    folder.mkdir(parents=True, exist_ok=True)
    img = folder / f"{name}.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n stub image bytes")
    # Sidecars the tool must ignore.
    (folder / f"{name}.txt").write_text("a caption", encoding="utf-8")
    (folder / f"{name}.vision.json").write_text("{}", encoding="utf-8")
    return img


@pytest.fixture()
def sorted_tree(tmp_path):
    """A tiny sorted tree: 2 kept images + 1 discarded image (+ sidecars)."""
    keep_a = _make_image(tmp_path, KEPT_CATEGORY, "keep_a")
    keep_b = _make_image(tmp_path, KEPT_CATEGORY, "keep_b")
    discard = _make_image(tmp_path, TERMINAL_CATEGORY, "trash")
    return {"keep": [keep_a, keep_b], "discard": discard}


# ── kept-only walking ──────────────────────────────────────────────────────────

def test_embeds_only_kept_categories(tool, sorted_tree):
    from embeddings_index import EmbeddingIndex

    embedded = tool.embed_slug(SLUG)

    assert embedded == 2  # the two Keep images, not the DISCARD one
    index = EmbeddingIndex(slug=SLUG)
    keys = set(index.keys())
    for kept in sorted_tree["keep"]:
        assert str(kept) in keys
        assert index.get(str(kept)) == pytest.approx(FAKE_VECTOR)
    # The discarded image must never be embedded.
    assert str(sorted_tree["discard"]) not in keys


def test_sidecars_are_not_embedded(tool, sorted_tree):
    tool.embed_slug(SLUG)
    # embed_image was only ever called on .png files, never .txt / .vision.json.
    for called in tool._test_calls:
        assert called.endswith(".png")
    assert len(tool._test_calls) == 2


def test_skips_already_indexed_without_force(tool, sorted_tree):
    # First pass embeds both kept images.
    assert tool.embed_slug(SLUG) == 2
    tool._test_calls.clear()

    # Second pass: everything is already indexed, so nothing new is embedded.
    assert tool.embed_slug(SLUG) == 0
    assert tool._test_calls == []


def test_force_reembeds(tool, sorted_tree):
    assert tool.embed_slug(SLUG) == 2
    tool._test_calls.clear()

    # --force re-embeds the kept images even though they're already present.
    assert tool.embed_slug(SLUG, force=True) == 2
    assert len(tool._test_calls) == 2


def test_limit_caps_embeddings(tool, sorted_tree):
    embedded = tool.embed_slug(SLUG, limit=1)
    assert embedded == 1
    assert len(tool._test_calls) == 1


def test_index_persisted_to_disk(tool, sorted_tree):
    from embeddings_index import EmbeddingIndex

    tool.embed_slug(SLUG)
    # A fresh instance (re-reads JSONL from disk) must see the backfilled vectors,
    # proving the tool flushed the index.
    reopened = EmbeddingIndex(slug=SLUG)
    assert len(reopened) == 2


# ── early-out when the ML extra is absent ───────────────────────────────────────

def test_main_early_outs_nonzero_when_open_clip_absent(tool, sorted_tree, monkeypatch, capsys):
    import embeddings

    # Simulate a base install without the [ml] extra.
    monkeypatch.setattr(embeddings, "open_clip_available", lambda: False)

    rc = tool.main(["--slug", SLUG])
    assert rc != 0

    err = capsys.readouterr().err
    assert "ml" in err.lower()  # actionable: mentions the ML extra
    assert "pip install" in err.lower()

    # Nothing was embedded because we bailed before walking.
    assert tool._test_calls == []


def test_main_happy_path_returns_zero(tool, sorted_tree):
    rc = tool.main(["--slug", SLUG])
    assert rc == 0
    assert len(tool._test_calls) == 2


def test_main_rejects_negative_limit(tool, sorted_tree):
    rc = tool.main(["--slug", SLUG, "--limit", "-5"])
    assert rc == 2
    assert tool._test_calls == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
