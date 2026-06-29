"""Tests for CLIP-style embedding utilities (``embeddings`` + ``embeddings_index``).

The heavy model dependency (``torch`` / ``open_clip_torch``) is import-guarded so
the module imports without them; only ``embed_image`` actually needs the model and
must raise a clear "install cull[ml]" error when it's absent. Every *vector* helper
(cosine, clustering, balancing, semantic search) is pure-Python and works on plain
``list[float]`` vectors with no torch / numpy — these tests run green in the base
(no-ML) environment.

Runnable:
    pytest tests/test_embeddings.py -q
    python tests/test_embeddings.py
"""
from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

import pytest

PIPELINE_CODE = Path(__file__).resolve().parent.parent / "pipeline_code"
if str(PIPELINE_CODE) not in sys.path:
    sys.path.insert(0, str(PIPELINE_CODE))


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def emb():
    import embeddings
    return importlib.reload(embeddings)


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """A freshly-configured embeddings_index pointed at a temp data dir."""
    monkeypatch.setenv("PIPELINE_BASE_DIR", str(tmp_path))
    import embeddings_index
    return importlib.reload(embeddings_index)


# ── module imports without torch / open_clip ──────────────────────────────────

def test_module_imports_without_ml_deps(emb):
    # The point of the import-guard: nothing in module scope may hard-require
    # torch / open_clip. If this fixture loaded, the guard worked.
    assert hasattr(emb, "embed_image")
    assert hasattr(emb, "cosine_similarity")
    assert hasattr(emb, "cluster_near_duplicates")
    assert hasattr(emb, "balance_dataset")
    assert hasattr(emb, "semantic_search")


def test_embed_image_raises_clear_error_when_open_clip_absent(emb):
    # open_clip is not installed in the base/test env, so embed_image must
    # raise a clear, actionable error (mentioning the cull[ml] extra) rather
    # than an opaque ImportError / AttributeError.
    if emb.open_clip_available():
        pytest.skip("open_clip is installed in this environment")
    with pytest.raises(RuntimeError) as exc:
        emb.embed_image(b"\x89PNG\r\n\x1a\n fake bytes")
    msg = str(exc.value).lower()
    assert "cull[ml]" in msg or "open_clip" in msg or "open-clip" in msg


# ── cosine_similarity ─────────────────────────────────────────────────────────

def test_cosine_identical_is_one(emb):
    v = [1.0, 2.0, 3.0, 4.0]
    assert emb.cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_orthogonal_is_zero(emb):
    assert emb.cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_opposite_is_minus_one(emb):
    assert emb.cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_is_scale_invariant(emb):
    a = [1.0, 2.0, 2.0]
    b = [10.0, 20.0, 20.0]
    assert emb.cosine_similarity(a, b) == pytest.approx(1.0)


def test_cosine_ranks_correctly(emb):
    query = [1.0, 0.0, 0.0]
    near = [0.9, 0.1, 0.0]
    far = [0.0, 1.0, 0.0]
    assert emb.cosine_similarity(query, near) > emb.cosine_similarity(query, far)


def test_cosine_zero_vector_is_zero_not_nan(emb):
    # A zero vector has no direction; define similarity as 0.0 rather than NaN
    # so downstream ranking / clustering never has to special-case NaN.
    sim = emb.cosine_similarity([0.0, 0.0, 0.0], [1.0, 2.0, 3.0])
    assert sim == 0.0
    assert not math.isnan(sim)


def test_cosine_length_mismatch_raises(emb):
    with pytest.raises(ValueError):
        emb.cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])


# ── cluster_near_duplicates ────────────────────────────────────────────────────

def test_cluster_groups_near_identical_vectors(emb):
    vectors = [
        [1.0, 0.0, 0.0],     # 0  ─┐ near-identical cluster
        [0.99, 0.01, 0.0],   # 1  ─┘
        [0.0, 1.0, 0.0],     # 2  ─┐ second cluster
        [0.0, 0.98, 0.02],   # 3  ─┘
        [0.0, 0.0, 1.0],     # 4   singleton
    ]
    groups = emb.cluster_near_duplicates(vectors, threshold=0.95)
    normalized = sorted(sorted(g) for g in groups)
    assert [0, 1] in normalized
    assert [2, 3] in normalized
    # The orthogonal singleton must not be grouped with anything.
    assert all(4 not in g for g in groups)


def test_cluster_is_transitive(emb):
    # A~B and B~C but A not directly ~C should still land in one group via
    # union-find transitivity.
    vectors = [
        [1.0, 0.0],
        [0.96, 0.28],   # close to 0
        [0.88, 0.47],   # close to 1, further from 0
    ]
    groups = emb.cluster_near_duplicates(vectors, threshold=0.9)
    assert len(groups) == 1
    assert sorted(groups[0]) == [0, 1, 2]


def test_cluster_no_dupes_returns_empty(emb):
    vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    assert emb.cluster_near_duplicates(vectors, threshold=0.95) == []


def test_cluster_empty_input(emb):
    assert emb.cluster_near_duplicates([], threshold=0.9) == []
    assert emb.cluster_near_duplicates([[1.0, 0.0]], threshold=0.9) == []


# ── balance_dataset ────────────────────────────────────────────────────────────

def test_balance_caps_per_cluster(emb):
    keys = ["a", "b", "c", "d", "e"]
    vectors = [
        [1.0, 0.0, 0.0],     # a ─┐
        [0.99, 0.01, 0.0],   # b  │ cluster of 3
        [0.98, 0.02, 0.0],   # c ─┘
        [0.0, 1.0, 0.0],     # d ─┐ cluster of 2
        [0.0, 0.99, 0.01],   # e ─┘
    ]
    kept = emb.balance_dataset(keys, vectors, max_per_cluster=1, threshold=0.95)
    # One representative from each of the two clusters => 2 kept.
    assert len(kept) == 2
    assert len(set(kept)) == len(kept)            # no duplicate keys
    assert all(k in keys for k in kept)
    # Exactly one of the a/b/c cluster and one of the d/e cluster.
    assert len({"a", "b", "c"} & set(kept)) == 1
    assert len({"d", "e"} & set(kept)) == 1


def test_balance_keeps_all_when_under_cap(emb):
    keys = ["a", "b"]
    vectors = [[1.0, 0.0], [0.99, 0.01]]
    kept = emb.balance_dataset(keys, vectors, max_per_cluster=5, threshold=0.95)
    assert set(kept) == {"a", "b"}


def test_balance_singletons_all_kept(emb):
    keys = ["a", "b", "c"]
    vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    kept = emb.balance_dataset(keys, vectors, max_per_cluster=1, threshold=0.95)
    assert set(kept) == {"a", "b", "c"}


def test_balance_rejects_mismatched_lengths(emb):
    with pytest.raises(ValueError):
        emb.balance_dataset(["a"], [[1.0], [2.0]], max_per_cluster=1)


def test_balance_empty(emb):
    assert emb.balance_dataset([], [], max_per_cluster=1) == []


# ── semantic_search ────────────────────────────────────────────────────────────

def test_semantic_search_returns_nearest_first(emb):
    index = [
        ("far", [0.0, 1.0, 0.0]),
        ("near", [0.95, 0.05, 0.0]),
        ("mid", [0.7, 0.7, 0.0]),
    ]
    ranked = emb.semantic_search([1.0, 0.0, 0.0], index, top_k=3)
    keys = [k for k, _score in ranked]
    assert keys[0] == "near"
    assert keys[-1] == "far"


def test_semantic_search_respects_top_k(emb):
    index = [(str(i), [float(i), 1.0]) for i in range(10)]
    ranked = emb.semantic_search([1.0, 0.0], index, top_k=3)
    assert len(ranked) == 3


def test_semantic_search_scores_descending(emb):
    index = [
        ("a", [1.0, 0.0]),
        ("b", [0.5, 0.5]),
        ("c", [0.0, 1.0]),
    ]
    ranked = emb.semantic_search([1.0, 0.0], index, top_k=3)
    scores = [s for _k, s in ranked]
    assert scores == sorted(scores, reverse=True)


def test_semantic_search_empty_index(emb):
    assert emb.semantic_search([1.0, 0.0], [], top_k=5) == []


# ── embeddings_index persistence (round-trip) ──────────────────────────────────

def test_index_add_and_get_round_trip(store):
    idx = store.EmbeddingIndex(slug="demo")
    idx.add("/data/a.png", [1.0, 2.0, 3.0])
    idx.add("/data/b.png", [4.0, 5.0, 6.0])
    assert idx.get("/data/a.png") == pytest.approx([1.0, 2.0, 3.0])
    assert idx.get("/data/b.png") == pytest.approx([4.0, 5.0, 6.0])
    assert idx.get("/data/missing.png") is None


def test_index_persists_across_instances(store):
    idx = store.EmbeddingIndex(slug="demo")
    idx.add("/data/a.png", [0.1, 0.2, 0.3])
    idx.flush()
    # A fresh instance pointed at the same slug must read the persisted vector.
    reopened = store.EmbeddingIndex(slug="demo")
    assert reopened.get("/data/a.png") == pytest.approx([0.1, 0.2, 0.3])
    assert len(reopened) == 1


def test_index_iter_yields_all(store):
    idx = store.EmbeddingIndex(slug="demo")
    idx.add("/data/a.png", [1.0, 0.0])
    idx.add("/data/b.png", [0.0, 1.0])
    items = dict(idx.iter())
    assert set(items) == {"/data/a.png", "/data/b.png"}
    assert items["/data/a.png"] == pytest.approx([1.0, 0.0])


def test_index_add_overwrites_existing_key(store):
    idx = store.EmbeddingIndex(slug="demo")
    idx.add("/data/a.png", [1.0, 0.0])
    idx.add("/data/a.png", [0.0, 1.0])  # re-embed same key
    assert idx.get("/data/a.png") == pytest.approx([0.0, 1.0])
    assert len(idx) == 1


def test_index_search_returns_nearest(store):
    idx = store.EmbeddingIndex(slug="demo")
    idx.add("/data/near.png", [0.95, 0.05, 0.0])
    idx.add("/data/far.png", [0.0, 1.0, 0.0])
    idx.add("/data/mid.png", [0.7, 0.7, 0.0])
    ranked = idx.search([1.0, 0.0, 0.0], top_k=2)
    assert len(ranked) == 2
    assert ranked[0][0] == "/data/near.png"


def test_index_slugs_are_isolated(store):
    a = store.EmbeddingIndex(slug="alpha")
    a.add("/data/a.png", [1.0, 0.0])
    a.flush()
    b = store.EmbeddingIndex(slug="beta")
    # beta's store must not see alpha's vectors.
    assert b.get("/data/a.png") is None
    assert len(b) == 0


def test_index_path_under_base_dir(store, tmp_path):
    idx = store.EmbeddingIndex(slug="demo")
    idx.add("/data/a.png", [1.0, 0.0])
    idx.flush()
    # The persistence file must live under the paths base dir (data/embeddings/),
    # never touching index_store's SQLite DB.
    assert tmp_path in idx.path.parents
    assert "embeddings" in idx.path.parts
    assert idx.path.exists()


def test_index_tolerates_corrupt_file(store):
    idx = store.EmbeddingIndex(slug="demo")
    idx.add("/data/a.png", [1.0, 0.0])
    idx.flush()
    # Corrupt the persisted file; a fresh instance must not raise, just start empty
    # (graceful failure, mirroring SeenStore / index_store behaviour).
    idx.path.write_text("{ this is not valid jsonl ][", encoding="utf-8")
    reopened = store.EmbeddingIndex(slug="demo")
    assert len(reopened) == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
