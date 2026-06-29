"""CLIP-style image embeddings + pure-Python vector utilities for cull.

This module has two layers that are deliberately decoupled:

1. **The model layer** (``embed_image``) turns an image into a dense vector
   using `open_clip <https://github.com/mlfoundations/open_clip>`_ + ``torch``.
   Those are heavy, GPU-flavoured dependencies, so they live behind the project's
   optional ``[ml]`` extra (``pip install -e .[ml]`` → ``torch``,
   ``open_clip_torch``). They are **import-guarded**: this module imports cleanly
   on a base install, and only ``embed_image`` raises — with a clear, actionable
   "install cull[ml]" message — when the model deps are missing.

2. **The vector layer** (``cosine_similarity``, ``cluster_near_duplicates``,
   ``balance_dataset``, ``semantic_search``) is **pure standard library**. It
   operates on plain ``list[float]`` vectors and needs neither ``torch`` nor
   ``numpy``, so dedup / balancing / semantic-search logic is fully testable and
   usable on any install. Cosine similarity is computed in plain Python; for the
   ~tens-of-thousands of vectors cull targets this is more than fast enough, and
   it keeps the base footprint tiny.

Design echoes :mod:`phash_dedup`: graceful failure, union-find clustering for
transitive near-duplicate grouping, and a small, explicit ``__all__``.

Why CLIP embeddings (semantic) when cull already has perceptual hashing?
``phash_dedup`` catches *pixel-level* near-duplicates (re-encodes, small crops).
CLIP embeddings capture *semantic* similarity — "another photo of the same
subject / style" — which powers "find more like this", dataset balancing across
visual concepts, and natural-language-free semantic search.
"""
from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Iterable, Sequence

from pipeline_logging import get_logger

logger = get_logger(__name__)

# Type aliases (kept as strings in older style for readability of signatures).
Vector = "Sequence[float]"
ImageInput = "Path | str | bytes | bytearray"

# Default model. ViT-B/32 with the OpenAI weights is the classic CLIP baseline:
# small enough to run on CPU, 512-dim output, and ubiquitous in open_clip.
DEFAULT_MODEL_NAME: str = "ViT-B-32"
DEFAULT_PRETRAINED: str = "openai"

# Default cosine threshold for treating two embeddings as near-duplicates.
# CLIP cosine similarities for "basically the same image" sit very high
# (~0.95+); distinct-but-related images land lower. Tunable per call.
DEFAULT_SIMILARITY_THRESHOLD: float = 0.92

# Actionable message when the optional model stack is absent. Surfaced verbatim
# so a user (or the dashboard) knows exactly how to enable embeddings.
_ML_MISSING_MSG = (
    "Image embeddings require the optional ML stack (torch + open_clip_torch). "
    "Install it with:  pip install -e .[ml]   (or:  pip install torch open_clip_torch). "
    "The pure-Python vector helpers (cosine_similarity, cluster_near_duplicates, "
    "balance_dataset, semantic_search) work without it."
)


# ── optional model stack (import-guarded) ─────────────────────────────────────
#
# We probe for the deps lazily and cache the loaded model so the heavy import +
# weight load happens at most once per process, and never at module import time.

try:  # pragma: no cover - exercised only on an [ml] install
    import open_clip as _open_clip  # type: ignore
    import torch as _torch  # type: ignore

    _OPEN_CLIP_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # ImportError, or a broken partial install
    _open_clip = None  # type: ignore
    _torch = None  # type: ignore
    _OPEN_CLIP_IMPORT_ERROR = exc


def open_clip_available() -> bool:
    """True when both ``open_clip`` and ``torch`` imported successfully.

    Callers (e.g. a backfill task or the dashboard) can branch on this to decide
    whether to offer embedding features without triggering the raise in
    :func:`embed_image`.
    """
    return _open_clip is not None and _torch is not None


# Lazily-initialised singletons: (model, preprocess) and the resolved device.
_model = None
_preprocess = None
_device: str | None = None


def _require_model(
    model_name: str = DEFAULT_MODEL_NAME,
    pretrained: str = DEFAULT_PRETRAINED,
):
    """Load (once) and return ``(model, preprocess, device)``.

    Raises :class:`RuntimeError` with :data:`_ML_MISSING_MSG` if the optional ML
    stack isn't installed — never an opaque ``ImportError``/``AttributeError``.
    """
    if not open_clip_available():
        raise RuntimeError(_ML_MISSING_MSG) from _OPEN_CLIP_IMPORT_ERROR

    global _model, _preprocess, _device
    if _model is None or _preprocess is None:
        _device = "cuda" if _torch.cuda.is_available() else "cpu"
        logger.info(
            "loading open_clip model %s/%s on %s", model_name, pretrained, _device,
        )
        model, _, preprocess = _open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained,
        )
        model = model.to(_device)
        model.eval()
        _model, _preprocess = model, preprocess
    return _model, _preprocess, _device


def _open_image(image: "Path | str | bytes | bytearray"):
    """Open ``image`` (path / path-string / raw bytes) as an RGB PIL image.

    Raises ``ValueError`` for unreadable input. PIL is a base dependency, so this
    works regardless of the ML extra.
    """
    from PIL import Image  # local import: base dep, keeps module-load side-effect-free

    try:
        if isinstance(image, (bytes, bytearray)):
            img = Image.open(io.BytesIO(bytes(image)))
        else:
            img = Image.open(Path(image))
        return img.convert("RGB")
    except (OSError, ValueError) as exc:
        raise ValueError(f"could not read image for embedding: {exc}") from exc


def embed_image(
    image: "Path | str | bytes | bytearray",
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    pretrained: str = DEFAULT_PRETRAINED,
    normalize: bool = True,
) -> list[float]:
    """Embed an image into a dense float vector via open_clip.

    ``image`` may be a filesystem path (``Path`` / ``str``) or raw image
    ``bytes``. Returns a plain ``list[float]`` (length = the model's embed dim,
    512 for the default ViT-B/32) so it round-trips through JSON and the pure-
    Python helpers without any tensor types leaking out.

    By default the vector is L2-normalised, which makes cosine similarity a plain
    dot product and keeps stored vectors comparable across batches.

    Raises:
        RuntimeError: if the optional ML stack (torch + open_clip_torch) is not
            installed — the message names the ``cull[ml]`` extra.
        ValueError: if the image can't be decoded.
    """
    model, preprocess, device = _require_model(model_name, pretrained)
    img = _open_image(image)
    tensor = preprocess(img).unsqueeze(0).to(device)
    with _torch.no_grad():
        features = model.encode_image(tensor)
        if normalize:
            features = features / features.norm(dim=-1, keepdim=True)
    return [float(x) for x in features[0].cpu().tolist()]


# ── pure-Python vector helpers (no torch / numpy) ─────────────────────────────

def _check_same_length(a: "Sequence[float]", b: "Sequence[float]") -> None:
    if len(a) != len(b):
        raise ValueError(
            f"vector length mismatch: {len(a)} != {len(b)} (cannot compare)"
        )


def cosine_similarity(a: "Sequence[float]", b: "Sequence[float]") -> float:
    """Cosine similarity of two equal-length vectors, in ``[-1.0, 1.0]``.

    Implemented in pure Python (no numpy). A zero-magnitude vector has no
    direction, so similarity is defined as ``0.0`` rather than ``NaN`` — this
    keeps ranking and clustering total (no NaN special-casing downstream).

    Raises ``ValueError`` if the vectors differ in length.
    """
    _check_same_length(a, b)
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    sim = dot / (math.sqrt(norm_a) * math.sqrt(norm_b))
    # Clamp tiny floating-point overshoot so callers can trust the [-1, 1] range.
    if sim > 1.0:
        return 1.0
    if sim < -1.0:
        return -1.0
    return sim


class _UnionFind:
    """Minimal union-find over integer indices for clustering near-dupes.

    Mirrors :class:`phash_dedup._UnionFind` so the two dedup paths cluster
    identically (path-compressed find + naive union).
    """

    def __init__(self, size: int) -> None:
        self._parent = list(range(size))

    def find(self, x: int) -> int:
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra


def _cluster_indices(
    vectors: "Sequence[Sequence[float]]", threshold: float,
) -> list[list[int]]:
    """Union-find clustering over vector indices by cosine ≥ ``threshold``.

    Returns one list of indices per connected component (including singletons),
    in ascending order of the component's smallest index — a stable, testable
    ordering. O(n²) pairwise cosine; fine for the tens-of-thousands range.
    """
    n = len(vectors)
    if n == 0:
        return []
    uf = _UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            try:
                if cosine_similarity(vectors[i], vectors[j]) >= threshold:
                    uf.union(i, j)
            except ValueError:
                # Defensive: a ragged vector shouldn't abort the whole scan.
                continue
    clusters: dict[int, list[int]] = {}
    for idx in range(n):
        clusters.setdefault(uf.find(idx), []).append(idx)
    # Sort members within a cluster, and clusters by their first member, so the
    # output is deterministic regardless of dict insertion order.
    return sorted((sorted(members) for members in clusters.values()),
                  key=lambda members: members[0])


def cluster_near_duplicates(
    vectors: "Sequence[Sequence[float]]",
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> list[list[int]]:
    """Group vectors whose pairwise cosine similarity is ``>= threshold``.

    Returns a list of groups, each a list of **indices into ``vectors``**, where
    every member is transitively similar to at least one other (union-find).
    Singletons are omitted; the result is empty when nothing is near-duplicate.

    This is the semantic analogue of
    :func:`phash_dedup.find_near_duplicates` — same shape, different distance.
    """
    return [g for g in _cluster_indices(vectors, threshold) if len(g) > 1]


def balance_dataset(
    keys: "Sequence[str]",
    vectors: "Sequence[Sequence[float]]",
    max_per_cluster: int,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> list[str]:
    """Cap how many near-duplicate images survive per semantic cluster.

    Clusters ``keys``/``vectors`` by cosine similarity, then keeps at most
    ``max_per_cluster`` representatives from each cluster (the first ``N`` by the
    input order, which is deterministic). Singletons are always kept. Returns the
    surviving ``keys`` in their original order — handy for de-biasing a training
    set that's over-represented in a few visual concepts.

    Raises ``ValueError`` if ``keys`` and ``vectors`` differ in length.
    """
    if len(keys) != len(vectors):
        raise ValueError(
            f"keys/vectors length mismatch: {len(keys)} != {len(vectors)}"
        )
    if not keys:
        return []
    cap = max(0, int(max_per_cluster))

    kept_indices: set[int] = set()
    for group in _cluster_indices(vectors, threshold):
        for idx in group[:cap]:  # group is already sorted ascending
            kept_indices.add(idx)
    # Preserve original input order in the returned keys.
    return [keys[i] for i in range(len(keys)) if i in kept_indices]


def semantic_search(
    query_vec: "Sequence[float]",
    index: "Iterable[tuple[str, Sequence[float]]]",
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """Rank indexed vectors by cosine similarity to ``query_vec`` (desc).

    ``index`` is any iterable of ``(key, vector)`` pairs (e.g.
    :meth:`embeddings_index.EmbeddingIndex.iter`). Returns the top ``top_k``
    ``(key, score)`` pairs, highest similarity first. Vectors whose length
    doesn't match the query are skipped rather than raising, so a stray bad row
    can't break a search.
    """
    scored: list[tuple[str, float]] = []
    for key, vec in index:
        try:
            score = cosine_similarity(query_vec, vec)
        except ValueError:
            continue
        scored.append((key, score))
    scored.sort(key=lambda kv: kv[1], reverse=True)
    if top_k is not None and top_k >= 0:
        return scored[:top_k]
    return scored


__all__ = [
    "DEFAULT_MODEL_NAME",
    "DEFAULT_PRETRAINED",
    "DEFAULT_SIMILARITY_THRESHOLD",
    "open_clip_available",
    "embed_image",
    "cosine_similarity",
    "cluster_near_duplicates",
    "balance_dataset",
    "semantic_search",
]
