"""Representative-frame extraction for the optional video lane.

cull classifies stills. To let a vision worker judge a *video clip*, we pull one
(or a few) representative frame(s) and run the existing image-classification path
on the frame, while the CLIP itself is what gets sorted. This module owns that
extraction step and nothing else.

Public surface::

    extract_keyframes(video_path, max_frames=1) -> list[Path]

It returns paths to freshly-written JPEG frames in a temp directory (the caller
reads their bytes, then is responsible for cleanup — see :func:`cleanup_frames`).

Design rules (load-bearing):

* **Gated, optional deps.** Frame extraction needs a backend — ``scenedetect``
  or ``ffmpeg`` (via the ``ffmpeg-python`` package, which shells out to the
  ffmpeg binary). Both are imported *lazily inside* the extraction functions so
  importing this module never drags them in, and the base install stays lean.
* **Never crash.** If no backend is installed, or extraction fails for any
  reason (corrupt container, missing binary, codec gap), we LOG and return an
  empty list. A bad clip must never crash the vision worker or the supervisor —
  the worker treats ``[]`` as "skip this clip".
* **Frames land in a tempdir.** We never write next to the source clip or into
  the queue/sorted trees; extraction is a scratch operation.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from pipeline_logging import get_logger

logger = get_logger(__name__)

# Containers we attempt to extract from. Mirrors export_profiles.VIDEO_EXT and
# the queue-manager video globs so the three stay in lockstep.
VIDEO_EXT: tuple[str, ...] = (".mp4", ".mov", ".webm", ".mkv", ".avi")

# Where extracted frames go. A dedicated subdir of the system tempdir keeps them
# off the queue/sorted trees and easy to sweep.
_FRAME_TMP_PREFIX = "cull_frames_"


def is_video(path: Path) -> bool:
    """True when ``path`` has a known video container extension."""
    return path.suffix.lower() in VIDEO_EXT


def is_enabled() -> bool:
    """Whether the video-classification lane is enabled via ``VIDEO_CLASSIFY_ENABLED``.

    Off by default — turning videos into classifiable frames is an explicit
    opt-in so existing image-only pipelines behave byte-identically on upgrade.
    """
    return os.environ.get("VIDEO_CLASSIFY_ENABLED", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


# ── backends (each gated; each returns [] on any failure) ─────────────────────

def _extract_with_ffmpeg(video_path: Path, max_frames: int, out_dir: Path) -> list[Path]:
    """Extract up to ``max_frames`` evenly-spaced JPEG frames via ffmpeg-python.

    Uses the ``ffmpeg-python`` wrapper (which invokes the ffmpeg binary). Returns
    the frames written, or ``[]`` if the package/binary is missing or the probe /
    extraction fails. Never raises.
    """
    try:
        import ffmpeg  # type: ignore
    except Exception:  # noqa: BLE001 - optional dep absent
        return []

    # Probe duration so we can seek to representative offsets. A probe failure is
    # non-fatal — we fall back to grabbing from the very start (offset 0).
    duration = 0.0
    try:
        info = ffmpeg.probe(str(video_path))
        raw = (info.get("format") or {}).get("duration")
        if raw:
            duration = float(raw)
    except Exception as exc:  # noqa: BLE001 - probe is best-effort
        logger.debug("video_frames: ffmpeg probe failed for %s: %s", video_path, exc)

    offsets = _frame_offsets(duration, max_frames)
    frames: list[Path] = []
    for idx, offset in enumerate(offsets):
        out_path = out_dir / f"frame_{idx:03d}.jpg"
        try:
            (
                ffmpeg
                .input(str(video_path), ss=offset)
                .output(str(out_path), vframes=1, format="image2", vcodec="mjpeg")
                .global_args("-loglevel", "error")
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
        except Exception as exc:  # noqa: BLE001 - per-frame failure shouldn't abort the rest
            logger.debug("video_frames: ffmpeg frame at %.2fs failed for %s: %s",
                         offset, video_path, exc)
            continue
        if out_path.exists() and out_path.stat().st_size > 0:
            frames.append(out_path)
    return frames


def _extract_with_scenedetect(video_path: Path, max_frames: int, out_dir: Path) -> list[Path]:
    """Extract representative frames via PySceneDetect's image-saving helper.

    Returns the frames written, or ``[]`` if ``scenedetect`` (and its OpenCV
    backend) is missing or detection/saving fails. Never raises.
    """
    try:
        from scenedetect import SceneManager, open_video  # type: ignore
        from scenedetect.detectors import ContentDetector  # type: ignore
        from scenedetect.scene_manager import save_images  # type: ignore
    except Exception:  # noqa: BLE001 - optional dep absent
        return []

    try:
        video = open_video(str(video_path))
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector())
        scene_manager.detect_scenes(video)
        scene_list = scene_manager.get_scene_list()
        if not scene_list:
            # No cuts detected (e.g. a short single-shot clip): synthesise one
            # scene spanning the whole video so we still get a frame.
            from scenedetect.frame_timecode import FrameTimecode  # type: ignore
            start = FrameTimecode(0, video.frame_rate)
            scene_list = [(start, video.duration)]
        images = save_images(
            scene_list=scene_list[:max_frames],
            video=video,
            num_images=1,
            output_dir=str(out_dir),
            image_extension="jpg",
        )
    except Exception as exc:  # noqa: BLE001 - detection/saving failure is non-fatal
        logger.debug("video_frames: scenedetect failed for %s: %s", video_path, exc)
        return []

    # save_images returns {scene_index: [filename, ...]} of names relative to
    # out_dir; resolve them to existing paths, capped at max_frames.
    frames: list[Path] = []
    try:
        for names in images.values():
            for name in names:
                candidate = out_dir / name
                if candidate.exists() and candidate.stat().st_size > 0:
                    frames.append(candidate)
                if len(frames) >= max_frames:
                    return frames
    except Exception as exc:  # noqa: BLE001 - defensive over the return shape
        logger.debug("video_frames: scenedetect result parse failed for %s: %s",
                     video_path, exc)
    return frames[:max_frames]


def _frame_offsets(duration: float, max_frames: int) -> list[float]:
    """Evenly-spaced seek offsets (seconds) for ``max_frames`` representative
    frames across a clip of length ``duration``.

    With one frame we grab the midpoint (a fair representative); with more we
    spread them across the interior, avoiding the very first/last frames which
    are often black/transitional. A zero/unknown duration yields offset(s) at 0.
    """
    n = max(1, int(max_frames))
    if duration <= 0:
        return [0.0] * n
    if n == 1:
        return [duration / 2.0]
    step = duration / (n + 1)
    return [step * (i + 1) for i in range(n)]


# ── public entrypoint ─────────────────────────────────────────────────────────

def extract_keyframes(video_path: "Path | str", max_frames: int = 1) -> list[Path]:
    """Extract up to ``max_frames`` representative JPEG frames from a video clip.

    Tries ffmpeg first (lighter, more reliable for a single representative grab),
    then scenedetect. Frames are written to a fresh temp directory and their
    paths returned in order.

    Returns ``[]`` when no extraction backend is installed, the path isn't a
    readable video, or extraction otherwise fails — NEVER raises, so the caller
    can simply skip a clip that yields no frame. Callers should pass the returned
    paths to :func:`cleanup_frames` (or remove their parent dir) when done.
    """
    path = Path(video_path)
    if not path.is_file():
        logger.debug("video_frames: not a file: %s", path)
        return []
    n = max(1, int(max_frames))

    out_dir = Path(tempfile.mkdtemp(prefix=_FRAME_TMP_PREFIX))
    frames = _extract_with_ffmpeg(path, n, out_dir)
    if not frames:
        frames = _extract_with_scenedetect(path, n, out_dir)

    if not frames:
        # Nothing worked (no backend installed, or both failed). Clean up the
        # empty scratch dir and tell the caller to skip the clip.
        _rmdir_quietly(out_dir)
        logger.info(
            "video_frames: no frame extracted from %s (no backend installed or "
            "extraction failed); skipping clip", path.name,
        )
        return []
    return frames


def cleanup_frames(frames: list[Path]) -> None:
    """Best-effort removal of extracted frames and their temp parent dir.

    Never raises — a failed cleanup of scratch files must not affect the caller.
    """
    parents: set[Path] = set()
    for frame in frames:
        try:
            parent = frame.parent
            if parent.name.startswith(_FRAME_TMP_PREFIX):
                parents.add(parent)
            frame.unlink(missing_ok=True)
        except OSError:
            pass
    for parent in parents:
        _rmdir_quietly(parent)


def _rmdir_quietly(directory: Path) -> None:
    try:
        import shutil
        shutil.rmtree(directory, ignore_errors=True)
    except Exception as exc:  # noqa: BLE001 - cleanup is best-effort
        logger.debug("video_frames: temp cleanup failed for %s: %s", directory, exc)


__all__ = [
    "VIDEO_EXT",
    "is_video",
    "is_enabled",
    "extract_keyframes",
    "cleanup_frames",
]
