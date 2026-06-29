"""Dataset export profiles — pack the sorted library into training-ready layouts.

cull sorts curated images into ``data/sorted/<slug>/<category>/<source>/`` with a
matching ``<name>.txt`` caption and ``<name>.vision.json`` audit record per image.
This module turns that KEPT set into the layouts trainers actually consume:

  * ``kohya``      — flat (or repeat-count) folders of ``image + .txt`` pairs
                     following the kohya/diffusers convention. With ``repeats``
                     the concept folder is named ``"<repeats>_<concept>"``.
  * ``webdataset`` — ``.tar`` shards (stdlib :mod:`tarfile`). Each sample is an
                     ``image + .txt + .json`` triple sharing one basename; shards
                     are split at ``max_samples_per_shard`` (default 1000) into
                     ``shard-000000.tar``, ``shard-000001.tar`` …
  * ``folders``    — a cleaned per-category copy of the sorted set.
  * ``clip_caption`` — clip + ``.txt`` caption pairs for VIDEO-model trainers
                     (LTX-Video, Wan, Hunyuan-Video, Mochi, CogVideoX). With
                     ``video_bucketing`` (``"duration"`` / ``"fps"`` /
                     ``"resolution"``) clips are grouped into per-bucket subdirs
                     probed from their container metadata.

Samples are media files — images (:data:`IMAGE_EXT`) and video clips
(:data:`VIDEO_EXT`). Every profile copies source bytes verbatim, so the image
profiles pack video too. Clip metadata is read via a *gated* prober
(:func:`probe_video` — ``av`` / ``ffmpeg-python`` if importable, else ``None``);
an unreadable clip buckets as ``"unknown"`` and never crashes the export.

Cross-cutting options apply to every profile:

  * ``categories``           — restrict to these KEPT categories (default: all
                               KEPT). Terminal buckets (DISCARD / CORRUPT) are
                               *never* exported, even if explicitly requested.
  * ``trigger_word``         — inject a token at the front of every caption.
  * ``train_val_split``      — e.g. ``0.1`` → deterministically partition into
                               ``train/`` and ``val/`` subdirs.
  * ``resolution_bucketing`` — group samples into ``<w>x<h>`` aspect/size buckets
                               (dimensions read with Pillow).

Hard guarantees:

  * EXPORT IS COPY-ONLY. Source files under ``data/sorted`` are never moved,
    deleted, or modified — only read and written into ``out_dir``.
  * Only KEPT categories (``categories.get_categories()``) are exported. The
    terminal buckets from ``categories.SYSTEM_TERMINAL`` are filtered out at the
    iteration boundary so no downstream profile can leak them.
  * Iteration is deterministic (sorted) so callers and tests are stable.

Public API::

    export_dataset(slug, profile, out_dir, **opts) -> dict   # summary
    iter_samples(slug, categories=None) -> Iterator[Sample]

This module has no Flask dependency; the dashboard wires a thin endpoint on top.
"""
from __future__ import annotations

import hashlib
import io
import json
import shutil
import tarfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from PIL import Image

from categories import SYSTEM_TERMINAL, get_categories
from paths import sorted_dir
from paths import validate_slug as _validate_slug
from pipeline_logging import get_logger

logger = get_logger(__name__)

# Image extensions the sorter is known to produce (mirrors requeue_sorted).
IMAGE_EXT: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")

# Video containers the sorter / scrapers (gallery-dl, yt-dlp) can land. Clips
# are exported as first-class samples alongside images; the prober reads their
# resolution / fps / duration for bucketing.
VIDEO_EXT: tuple[str, ...] = (".mp4", ".mov", ".webm", ".mkv", ".avi")

# Everything iter_samples treats as an exportable sample.
MEDIA_EXT: tuple[str, ...] = IMAGE_EXT + VIDEO_EXT

VISION_SUFFIX: str = ".vision.json"

PROFILES: tuple[str, ...] = ("kohya", "webdataset", "folders", "clip_caption")

# Video-bucketing modes for the clip_caption profile.
VIDEO_BUCKET_MODES: tuple[str, ...] = ("duration", "fps", "resolution")

# Defaults for cross-cutting / profile options.
_DEFAULT_SHARD_SIZE: int = 1000
_DEFAULT_CONCEPT: str = "concept"
_TRAIN_DIR: str = "train"
_VAL_DIR: str = "val"
_UNKNOWN_BUCKET: str = "unknown"

# Coarse duration buckets (seconds) for grouping clips by length. Edges are
# inclusive-low / exclusive-high; anything past the last edge is "16s+".
_DURATION_EDGES: tuple[float, ...] = (2.0, 4.0, 8.0, 16.0)


# ── Sample model ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Sample:
    """One exportable unit: a media file plus its caption and optional audit meta.

    ``image_path`` always points at the original file under ``data/sorted`` —
    callers must copy, never move it. The field keeps the historical name even
    though it may now reference a video clip; :meth:`is_video` distinguishes.
    """

    image_path: Path
    caption: str
    meta: dict[str, Any] | None
    category: str
    source: str
    stem: str

    @property
    def is_video(self) -> bool:
        """True when the underlying media is a video container, not a still."""
        return self.image_path.suffix.lower() in VIDEO_EXT


# ── Video metadata probing (gated) ───────────────────────────────────────────


@dataclass(frozen=True)
class VideoMeta:
    """Probed container metadata for one clip. ``fps`` / ``duration`` are
    floats (seconds, frames-per-second); ``width`` / ``height`` are pixels.
    Any field the prober couldn't determine is left at 0."""

    width: int = 0
    height: int = 0
    fps: float = 0.0
    duration: float = 0.0


def _probe_with_av(path: Path) -> VideoMeta | None:
    """Probe via PyAV (``av``) if it's importable. Returns None otherwise."""
    try:
        import av  # type: ignore
    except Exception:  # noqa: BLE001 — optional dep, any import error => skip
        return None
    try:
        with av.open(str(path)) as container:
            stream = next(
                (s for s in container.streams if s.type == "video"), None
            )
            if stream is None:
                return None
            width = int(getattr(stream, "width", 0) or 0)
            height = int(getattr(stream, "height", 0) or 0)
            rate = stream.average_rate or stream.base_rate
            fps = float(rate) if rate else 0.0
            duration = 0.0
            if stream.duration is not None and stream.time_base:
                duration = float(stream.duration * stream.time_base)
            elif container.duration:
                duration = float(container.duration) / 1_000_000.0  # microseconds
        return VideoMeta(width=width, height=height, fps=fps, duration=duration)
    except Exception as exc:  # noqa: BLE001 — corrupt clip / codec gap
        logger.warning("av probe failed for %s: %s", path, exc)
        return None


def _probe_with_ffmpeg(path: Path) -> VideoMeta | None:
    """Probe via the ``ffmpeg-python`` package (which invokes ffprobe). Returns
    None if the package is missing or the probe fails."""
    try:
        import ffmpeg  # type: ignore

        probe_json = ffmpeg.probe(str(path))
    except Exception:  # noqa: BLE001 — package missing or probe error => skip
        return None
    try:
        streams = probe_json.get("streams") or []
        video = next(
            (s for s in streams if s.get("codec_type") == "video"), None
        )
        if video is None:
            return None
        width = int(video.get("width") or 0)
        height = int(video.get("height") or 0)
        fps = _parse_fraction(video.get("avg_frame_rate") or video.get("r_frame_rate"))
        duration = 0.0
        for source in (video, probe_json.get("format") or {}):
            raw = source.get("duration")
            if raw:
                try:
                    duration = float(raw)
                    break
                except (TypeError, ValueError):
                    continue
        return VideoMeta(width=width, height=height, fps=fps, duration=duration)
    except Exception as exc:  # noqa: BLE001
        logger.warning("ffmpeg probe parse failed for %s: %s", path, exc)
        return None


def _parse_fraction(value: Any) -> float:
    """Parse an ffprobe frame-rate string like ``"24000/1001"`` -> float."""
    if not value:
        return 0.0
    text = str(value).strip()
    try:
        if "/" in text:
            num, den = text.split("/", 1)
            den_f = float(den)
            return float(num) / den_f if den_f else 0.0
        return float(text)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def probe_video(path: Path) -> VideoMeta | None:
    """Best-effort container probe for a clip — gated behind optional deps.

    Tries PyAV first, then ffmpeg-python. Returns a :class:`VideoMeta`, or
    ``None`` when no prober is installed or the probe fails. NEVER raises —
    callers fall back to the ``"unknown"`` bucket. This is the single seam tests
    patch to exercise bucketing without ffmpeg.
    """
    try:
        meta = _probe_with_av(path)
        if meta is not None:
            return meta
        return _probe_with_ffmpeg(path)
    except Exception as exc:  # noqa: BLE001 — belt-and-braces; never crash export
        logger.warning("video probe unexpectedly failed for %s: %s", path, exc)
        return None


# ── Category resolution ──────────────────────────────────────────────────────


def _kept_categories() -> set[str]:
    """The KEPT bucket names from the active taxonomy (never terminal)."""
    return set(get_categories())


def _resolve_categories(categories: Iterable[str] | None) -> set[str]:
    """Intersect a requested category list with the KEPT set.

    Terminal buckets (DISCARD / CORRUPT) are stripped here so no profile can
    ever emit them, regardless of what a caller passes.
    """
    kept = _kept_categories()
    terminal = {c.upper() for c in SYSTEM_TERMINAL}
    if categories is None:
        return {c for c in kept if c.upper() not in terminal}
    requested = {str(c) for c in categories}
    return {c for c in requested if c in kept and c.upper() not in terminal}


# ── Sample iteration ─────────────────────────────────────────────────────────


def _read_caption(folder: Path, stem: str, image_name: str) -> str:
    """Read the sibling ``.txt`` caption, tolerating both ``<stem>.txt`` and the
    ``<image_name>.txt`` fallback the vision worker's gather step allows."""
    candidates = (folder / f"{stem}.txt", folder / f"{image_name}.txt")
    for cand in candidates:
        if cand.exists():
            try:
                return cand.read_text(encoding="utf-8", errors="replace").strip()
            except OSError as exc:
                logger.warning("failed reading caption %s: %s", cand, exc)
                return ""
    return ""


def _read_meta(folder: Path, stem: str) -> dict[str, Any] | None:
    """Read the sibling ``<stem>.vision.json`` audit record if present."""
    meta_path = folder / f"{stem}{VISION_SUFFIX}"
    if not meta_path.exists():
        return None
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("failed reading meta %s: %s", meta_path, exc)
        return None
    return payload if isinstance(payload, dict) else None


def iter_samples(
    slug: str,
    categories: Iterable[str] | None = None,
) -> Iterator[Sample]:
    """Yield ``Sample`` tuples for the KEPT categories under ``data/sorted/<slug>``.

    Deterministic: categories, sources, and media files are walked in sorted
    order. Terminal buckets are never yielded. Hidden dirs (``.quarantine``
    etc.) and the ``.vision.json``/``.txt`` sidecars are skipped — only media
    files (images :data:`IMAGE_EXT` + video clips :data:`VIDEO_EXT`) seed a
    sample.
    """
    # Sanitise the slug before it selects the sorted-tree root to walk.
    root = sorted_dir(_validate_slug(slug))
    if not root.exists():
        return

    wanted = _resolve_categories(categories)
    if not wanted:
        return

    for category in sorted(wanted):
        cat_dir = root / category
        if not cat_dir.is_dir():
            continue
        for source_dir in sorted(p for p in cat_dir.iterdir() if p.is_dir()):
            if source_dir.name.startswith("."):
                continue
            source = source_dir.name
            images = sorted(
                p for p in source_dir.iterdir()
                if p.is_file() and p.suffix.lower() in MEDIA_EXT
            )
            for img in images:
                stem = img.stem
                caption = _read_caption(source_dir, stem, img.name)
                meta = _read_meta(source_dir, stem)
                yield Sample(
                    image_path=img,
                    caption=caption,
                    meta=meta,
                    category=category,
                    source=source,
                    stem=stem,
                )


# ── Cross-cutting transforms ─────────────────────────────────────────────────


def _apply_trigger_word(caption: str, trigger_word: str | None) -> str:
    """Prepend ``trigger_word`` to a caption (idempotent, comma-joined)."""
    if not trigger_word:
        return caption
    trigger = trigger_word.strip()
    if not trigger:
        return caption
    body = caption.strip()
    if not body:
        return trigger
    # Avoid double-prefixing if the token is already at the front.
    if body == trigger or body.startswith(f"{trigger},") or body.startswith(f"{trigger} "):
        return body
    return f"{trigger}, {body}"


def _split_bucket(stem: str, train_val_split: float | None) -> str | None:
    """Return ``"val"``/``"train"`` for a sample, or ``None`` when no split.

    Deterministic: a stable hash of the stem maps to ``[0, 1)``; samples below
    ``train_val_split`` go to validation. The same stem always lands the same
    way regardless of run order.
    """
    if not train_val_split or train_val_split <= 0:
        return None
    digest = hashlib.sha1(stem.encode("utf-8")).hexdigest()
    fraction = int(digest[:8], 16) / 0xFFFFFFFF
    return _VAL_DIR if fraction < float(train_val_split) else _TRAIN_DIR


def _resolution_bucket(image_path: Path) -> str:
    """``"<width>x<height>"`` read via Pillow; ``"unknown"`` if unreadable."""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
        return f"{width}x{height}"
    except (OSError, ValueError) as exc:
        logger.warning("could not read dimensions for %s: %s", image_path, exc)
        return "unknown"


def _duration_label(duration: float) -> str:
    """Coarse, human-readable length bucket from a clip's duration (seconds)."""
    if duration <= 0:
        return _UNKNOWN_BUCKET
    lo = 0.0
    for edge in _DURATION_EDGES:
        if duration < edge:
            return f"dur_{int(lo)}-{int(edge)}s"
        lo = edge
    return f"dur_{int(_DURATION_EDGES[-1])}s+"


def _meta_to_bucket(meta: "VideoMeta | None", mode: str) -> str:
    """Map probed :class:`VideoMeta` to a bucket label for ``mode``.

    Returns :data:`_UNKNOWN_BUCKET` when ``meta`` is missing or the relevant
    dimension is unknown (zero) — the graceful no-prober / probe-fail path.
    """
    if meta is None:
        return _UNKNOWN_BUCKET
    if mode == "duration":
        return _duration_label(meta.duration)
    if mode == "fps":
        return f"fps_{int(round(meta.fps))}" if meta.fps > 0 else _UNKNOWN_BUCKET
    if mode == "resolution" and meta.width > 0 and meta.height > 0:
        return f"{meta.width}x{meta.height}"
    return _UNKNOWN_BUCKET


def _video_bucket(sample: Sample, mode: str | None) -> str | None:
    """Bucket label for one sample's clip, or ``None`` when bucketing is off.

    Probes via :func:`probe_video` and never propagates an exception: a prober
    that crashes (corrupt clip, ffmpeg fault) is treated as ``"unknown"`` so the
    export always completes.
    """
    if not mode:
        return None
    try:
        meta = probe_video(sample.image_path)
    except Exception as exc:  # noqa: BLE001 — defensive; prober promised not to
        logger.warning("probe_video raised for %s: %s", sample.image_path, exc)
        meta = None
    return _meta_to_bucket(meta, mode)


# ── Path assembly ────────────────────────────────────────────────────────────


def _segments(
    *,
    split: str | None,
    bucket: str | None,
    leaf: str | None,
) -> tuple[str, ...]:
    """Ordered, optional sub-directory segments shared by every profile.

    Order: ``[split]/[leaf]/[bucket]`` — split is the outermost partition,
    then a profile-specific leaf (category or concept folder), then the
    resolution bucket nearest the files.
    """
    parts: list[str] = []
    if split:
        parts.append(split)
    if leaf:
        parts.append(leaf)
    if bucket:
        parts.append(bucket)
    return tuple(parts)


def _unique_stem(used: set[str], desired: str) -> str:
    """Disambiguate basenames so two folders can't collide in one flat dir."""
    candidate = desired
    attempt = 0
    while candidate in used:
        attempt += 1
        candidate = f"{desired}_{attempt}"
    used.add(candidate)
    return candidate


# ── Profile writers ──────────────────────────────────────────────────────────


def _copy_pair(
    sample: Sample,
    dest_dir: Path,
    stem: str,
    caption: str,
    *,
    write_json: bool,
) -> None:
    """Copy one image + caption (and optional JSON) into ``dest_dir``."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    ext = sample.image_path.suffix
    shutil.copy2(sample.image_path, dest_dir / f"{stem}{ext}")
    (dest_dir / f"{stem}.txt").write_text(caption, encoding="utf-8")
    if write_json:
        meta = sample.meta if sample.meta is not None else {}
        (dest_dir / f"{stem}.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _export_kohya(
    samples: list[Sample],
    out_dir: Path,
    opts: "_Options",
) -> dict[str, Any]:
    """Flat / repeat-count folders of ``image + .txt`` pairs."""
    concept = opts.concept or _DEFAULT_CONCEPT
    leaf = f"{opts.repeats}_{concept}" if opts.repeats and opts.repeats > 0 else None
    used_by_dir: dict[Path, set[str]] = defaultdict(set)
    splits: Counter[str] = Counter()
    buckets: Counter[str] = Counter()

    for sample in samples:
        split = _split_bucket(sample.stem, opts.train_val_split)
        bucket = _resolution_bucket(sample.image_path) if opts.resolution_bucketing else None
        if split:
            splits[split] += 1
        if bucket:
            buckets[bucket] += 1
        dest_dir = out_dir.joinpath(*_segments(split=split, bucket=bucket, leaf=leaf))
        stem = _unique_stem(used_by_dir[dest_dir], sample.stem)
        caption = _apply_trigger_word(sample.caption, opts.trigger_word)
        _copy_pair(sample, dest_dir, stem, caption, write_json=False)

    summary: dict[str, Any] = {"concept": concept}
    if leaf:
        summary["repeats"] = opts.repeats
    if splits:
        summary["splits"] = dict(splits)
    if buckets:
        summary["buckets"] = dict(buckets)
    return summary


def _export_folders(
    samples: list[Sample],
    out_dir: Path,
    opts: "_Options",
) -> dict[str, Any]:
    """Plain per-category copy of the sorted set (image + .txt [+ .json])."""
    used_by_dir: dict[Path, set[str]] = defaultdict(set)
    splits: Counter[str] = Counter()
    buckets: Counter[str] = Counter()

    for sample in samples:
        split = _split_bucket(sample.stem, opts.train_val_split)
        bucket = _resolution_bucket(sample.image_path) if opts.resolution_bucketing else None
        if split:
            splits[split] += 1
        if bucket:
            buckets[bucket] += 1
        dest_dir = out_dir.joinpath(
            *_segments(split=split, bucket=bucket, leaf=sample.category)
        )
        stem = _unique_stem(used_by_dir[dest_dir], sample.stem)
        caption = _apply_trigger_word(sample.caption, opts.trigger_word)
        _copy_pair(sample, dest_dir, stem, caption, write_json=True)

    summary: dict[str, Any] = {}
    if splits:
        summary["splits"] = dict(splits)
    if buckets:
        summary["buckets"] = dict(buckets)
    return summary


def _export_webdataset(
    samples: list[Sample],
    out_dir: Path,
    opts: "_Options",
) -> dict[str, Any]:
    """``.tar`` shards; each sample = image + .txt + .json sharing a basename.

    Shards are grouped first by the optional split/bucket sub-directory so a
    consumer can point at ``train/`` or a bucket folder; within each group the
    samples are chunked by ``max_samples_per_shard``.
    """
    shard_size = opts.max_samples_per_shard or _DEFAULT_SHARD_SIZE
    if shard_size <= 0:
        shard_size = _DEFAULT_SHARD_SIZE

    splits: Counter[str] = Counter()
    buckets: Counter[str] = Counter()
    # Group samples by their target sub-directory so shard numbering is local.
    groups: dict[tuple[str, ...], list[tuple[Sample, str]]] = defaultdict(list)
    used_by_group: dict[tuple[str, ...], set[str]] = defaultdict(set)

    for sample in samples:
        split = _split_bucket(sample.stem, opts.train_val_split)
        bucket = _resolution_bucket(sample.image_path) if opts.resolution_bucketing else None
        if split:
            splits[split] += 1
        if bucket:
            buckets[bucket] += 1
        seg = _segments(split=split, bucket=bucket, leaf=None)
        stem = _unique_stem(used_by_group[seg], sample.stem)
        groups[seg].append((sample, stem))

    shard_count = 0
    for seg in sorted(groups):
        group_dir = out_dir.joinpath(*seg)
        group_dir.mkdir(parents=True, exist_ok=True)
        members = groups[seg]
        for shard_index, start in enumerate(range(0, len(members), shard_size)):
            chunk = members[start:start + shard_size]
            shard_path = group_dir / f"shard-{shard_index:06d}.tar"
            with tarfile.open(shard_path, mode="w") as tar:
                for sample, stem in chunk:
                    _add_webdataset_sample(tar, sample, stem, opts.trigger_word)
            shard_count += 1

    summary: dict[str, Any] = {
        "shard_count": shard_count,
        "max_samples_per_shard": shard_size,
    }
    if splits:
        summary["splits"] = dict(splits)
    if buckets:
        summary["buckets"] = dict(buckets)
    return summary


def _add_webdataset_sample(
    tar: tarfile.TarFile,
    sample: Sample,
    stem: str,
    trigger_word: str | None,
) -> None:
    """Add the image, caption, and json members for one webdataset sample."""
    ext = sample.image_path.suffix
    # Image — copy bytes verbatim (no re-encode), source untouched.
    tar.add(str(sample.image_path), arcname=f"{stem}{ext}")

    caption = _apply_trigger_word(sample.caption, trigger_word).encode("utf-8")
    _add_bytes(tar, f"{stem}.txt", caption)

    meta = sample.meta if sample.meta is not None else {}
    payload = json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8")
    _add_bytes(tar, f"{stem}.json", payload)


def _add_bytes(tar: tarfile.TarFile, arcname: str, data: bytes) -> None:
    """Append in-memory ``data`` to a tar as ``arcname``."""
    info = tarfile.TarInfo(name=arcname)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _export_clip_caption(
    samples: list[Sample],
    out_dir: Path,
    opts: "_Options",
) -> dict[str, Any]:
    """Clip + ``.txt`` caption pairs for video-model trainers.

    The trainer convention is one media file beside one same-stem caption file
    (``clip.mp4`` + ``clip.txt``). With ``opts.video_bucketing`` set, clips are
    grouped into per-bucket subdirs (``dur_2-4s`` / ``fps_24`` / ``1920x1080``)
    probed from container metadata, after an optional ``train/val`` split. Image
    samples pair the same way and bucket as ``"unknown"`` under a video mode.
    """
    used_by_dir: dict[Path, set[str]] = defaultdict(set)
    splits: Counter[str] = Counter()
    buckets: Counter[str] = Counter()

    for sample in samples:
        split = _split_bucket(sample.stem, opts.train_val_split)
        bucket = _video_bucket(sample, opts.video_bucketing)
        if split:
            splits[split] += 1
        if bucket:
            buckets[bucket] += 1
        dest_dir = out_dir.joinpath(*_segments(split=split, bucket=bucket, leaf=None))
        stem = _unique_stem(used_by_dir[dest_dir], sample.stem)
        caption = _apply_trigger_word(sample.caption, opts.trigger_word)
        _copy_pair(sample, dest_dir, stem, caption, write_json=False)

    summary: dict[str, Any] = {}
    if opts.video_bucketing:
        summary["video_bucketing"] = opts.video_bucketing
    if splits:
        summary["splits"] = dict(splits)
    if buckets:
        summary["buckets"] = dict(buckets)
    return summary


_WRITERS = {
    "kohya": _export_kohya,
    "webdataset": _export_webdataset,
    "folders": _export_folders,
    "clip_caption": _export_clip_caption,
}


# ── Options ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Options:
    """Validated, normalised export options (parsed once from ``**opts``)."""

    categories: tuple[str, ...] | None
    trigger_word: str | None
    train_val_split: float | None
    resolution_bucketing: bool
    repeats: int | None
    concept: str | None
    max_samples_per_shard: int | None
    video_bucketing: str | None


def _parse_options(opts: dict[str, Any]) -> _Options:
    """Coerce and validate raw keyword options; fail fast on bad input."""
    categories = opts.get("categories")
    if categories is not None:
        categories = tuple(str(c) for c in categories)

    trigger_word = opts.get("trigger_word")
    if trigger_word is not None:
        trigger_word = str(trigger_word).strip() or None

    split_raw = opts.get("train_val_split")
    train_val_split: float | None = None
    if split_raw is not None:
        train_val_split = float(split_raw)
        if not 0.0 <= train_val_split < 1.0:
            raise ValueError("train_val_split must be in [0.0, 1.0)")

    repeats_raw = opts.get("repeats")
    repeats: int | None = None
    if repeats_raw is not None:
        repeats = int(repeats_raw)
        if repeats < 0:
            raise ValueError("repeats must be >= 0")

    shard_raw = opts.get("max_samples_per_shard")
    max_samples_per_shard: int | None = None
    if shard_raw is not None:
        max_samples_per_shard = int(shard_raw)
        if max_samples_per_shard <= 0:
            raise ValueError("max_samples_per_shard must be > 0")

    concept = opts.get("concept")
    if concept is not None:
        concept = str(concept).strip() or None

    video_bucketing = opts.get("video_bucketing")
    if video_bucketing is not None:
        video_bucketing = str(video_bucketing).strip().lower() or None
        if video_bucketing is not None and video_bucketing not in VIDEO_BUCKET_MODES:
            raise ValueError(
                f"video_bucketing must be one of {VIDEO_BUCKET_MODES}"
            )

    return _Options(
        categories=categories,
        trigger_word=trigger_word,
        train_val_split=train_val_split,
        resolution_bucketing=bool(opts.get("resolution_bucketing", False)),
        repeats=repeats,
        concept=concept,
        max_samples_per_shard=max_samples_per_shard,
        video_bucketing=video_bucketing,
    )


# ── Public entrypoint ────────────────────────────────────────────────────────


def export_dataset(
    slug: str,
    profile: str,
    out_dir: str | Path,
    **opts: Any,
) -> dict[str, Any]:
    """Export ``data/sorted/<slug>`` (KEPT categories only) into ``out_dir``.

    Args:
        slug: Job slug whose sorted library to export.
        profile: One of :data:`PROFILES` (``kohya`` / ``webdataset`` /
            ``folders`` / ``clip_caption``).
        out_dir: Destination directory (created if absent). Existing files there
            are not cleaned — pass a fresh dir for a clean export.
        **opts: Cross-cutting + profile options. See module docstring.
            ``categories``, ``trigger_word``, ``train_val_split``,
            ``resolution_bucketing``, ``repeats``, ``concept``,
            ``max_samples_per_shard``, ``video_bucketing`` (``"duration"`` /
            ``"fps"`` / ``"resolution"``, used by ``clip_caption``).

    Returns:
        A summary dict: ``slug``, ``profile``, ``out_dir``, ``sample_count``,
        sorted ``categories``, plus profile-specific keys (``splits``,
        ``buckets``, ``shard_count``, ``repeats``, ``concept`` …).

    Raises:
        ValueError: unknown ``profile`` or invalid option values.
    """
    if profile not in _WRITERS:
        raise ValueError(
            f"unknown export profile {profile!r}; choose one of {PROFILES}"
        )

    options = _parse_options(opts)
    # Sanitise the slug (it selects the sorted source tree below) and resolve the
    # caller-supplied destination to one canonical absolute root. Every profile
    # writer joins this resolved ``out_path`` with computed segments, so all
    # writes stay anchored under the normalised export directory.
    slug = _validate_slug(slug)
    out_path = Path(out_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    samples = list(iter_samples(slug, categories=options.categories))

    profile_summary = _WRITERS[profile](samples, out_path, options)

    summary: dict[str, Any] = {
        "slug": slug,
        "profile": profile,
        "out_dir": str(out_path),
        "sample_count": len(samples),
        "categories": sorted({s.category for s in samples}),
    }
    summary.update(profile_summary)
    logger.info(
        "exported %d sample(s) for slug=%s profile=%s -> %s",
        len(samples), slug, profile, out_path,
    )
    return summary


__all__ = [
    "Sample",
    "VideoMeta",
    "PROFILES",
    "IMAGE_EXT",
    "VIDEO_EXT",
    "MEDIA_EXT",
    "VIDEO_BUCKET_MODES",
    "iter_samples",
    "probe_video",
    "export_dataset",
]
