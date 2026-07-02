"""media_policy.py — single source of truth for which media a job accepts.

Driven by three env vars (projected by ``job_config.resolve_env`` from the active
job's ``media`` block, merged over ``.env``):

    MEDIA_TYPES        comma list, subset of {image, video}. Default "image".
    MEDIA_IMAGE_EXTS   comma list of image extensions. Default jpg,jpeg,png,webp.
    MEDIA_VIDEO_EXTS   comma list of video extensions. Default mp4,mov,webm,mkv,avi.

Every scraper consults this instead of hardcoding its own suffix set, and the
queue derives its poppable globs from it — so a job set to "video" or "both"
flows one consistent media policy through fetch → queue → classify, in parallel.
Read LIVE from ``os.environ`` so a dashboard change applies on the next scraper
spawn without a code edit (the supervisor merges the job's projection over .env).

NOTE: the vision worker must be able to OPEN whatever you accept — PIL for images
(jpg/png/webp/gif/…), the ``.[video]`` frame backend (ffmpeg) for clips. The
defaults are the formats cull already handles end-to-end; add exotic extensions
only if your Pillow / ffmpeg install can decode them.
"""
from __future__ import annotations

import os

# Defaults deliberately mirror the queue's poppable globs + export_profiles so
# the out-of-the-box behaviour is byte-identical to the pre-media_policy pipeline.
_DEFAULT_IMAGE_EXTS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp", ".gif")
_DEFAULT_VIDEO_EXTS: tuple[str, ...] = (".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v")
_DEFAULT_TYPES: tuple[str, ...] = ("image",)
_VALID_TYPES: frozenset[str] = frozenset({"image", "video"})


def _norm_ext(raw: str) -> str:
    """Lower-case, dot-prefixed extension (``JPG`` / ``.JPG`` / ` jpg ` -> ``.jpg``)."""
    e = (raw or "").strip().lower()
    if not e:
        return ""
    return e if e.startswith(".") else "." + e


def _parse_exts(env_key: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = (os.environ.get(env_key, "") or "").strip()
    if not raw:
        return default
    out: list[str] = []
    for part in raw.replace(";", ",").split(","):
        e = _norm_ext(part)
        if e and e not in out:
            out.append(e)
    return tuple(out) or default


def media_types() -> tuple[str, ...]:
    """Ordered subset of {image, video}. Empty / invalid env -> ("image",)."""
    raw = (os.environ.get("MEDIA_TYPES", "") or "").strip().lower()
    if not raw:
        return _DEFAULT_TYPES
    out: list[str] = []
    for part in raw.replace(";", ",").split(","):
        p = part.strip()
        if p in _VALID_TYPES and p not in out:
            out.append(p)
    return tuple(out) or _DEFAULT_TYPES


def wants_image() -> bool:
    return "image" in media_types()


def wants_video() -> bool:
    return "video" in media_types()


def image_exts() -> tuple[str, ...]:
    return _parse_exts("MEDIA_IMAGE_EXTS", _DEFAULT_IMAGE_EXTS)


def video_exts() -> tuple[str, ...]:
    return _parse_exts("MEDIA_VIDEO_EXTS", _DEFAULT_VIDEO_EXTS)


def accepted_exts() -> tuple[str, ...]:
    """Every extension this job accepts, per its media_types. Never empty (a
    misconfigured job falls back to images so it can't silently drop everything)."""
    exts: tuple[str, ...] = ()
    if wants_image():
        exts += image_exts()
    if wants_video():
        exts += video_exts()
    return exts or image_exts()


def accepts(suffix: str) -> bool:
    """Whether a file with this extension should be ingested under the policy."""
    return _norm_ext(suffix) in accepted_exts()


def is_video_ext(suffix: str) -> bool:
    return _norm_ext(suffix) in video_exts()


def is_image_ext(suffix: str) -> bool:
    return _norm_ext(suffix) in image_exts()


__all__ = [
    "media_types", "wants_image", "wants_video",
    "image_exts", "video_exts", "accepted_exts",
    "accepts", "is_video_ext", "is_image_ext",
]
