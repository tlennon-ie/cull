"""scraper_yt_dlp.py - Video scraping lane backed by yt-dlp.

Sibling to ``scraper_gallery_dl.py``. Where gallery-dl pulls *images* from 340+
sites, this pulls *videos* from the ~1000 sites yt-dlp supports (YouTube,
TikTok, Twitter/X, Reddit, Vimeo, Instagram, Twitch clips, …) by wrapping the
yt-dlp Python API. The downloaded ``.mp4``/``.webm``/… lands in the queue right
next to scraped images, with the same ``.txt`` prompt + ``.meta.json`` sidecar
contract, so the rest of the pipeline treats a video exactly like an image.

Configuration (all read from .env, all editable from the dashboard):

    YT_DLP_ENABLED      on/off toggle.
    YT_DLP_URLS         newline OR comma separated URLs (or playlist URLs).
    YT_DLP_LIMIT        max items per URL (default 25). Playlists are capped to
                        this via yt-dlp's ``playlistend``.
    YT_DLP_COOKIES      optional Netscape cookies.txt path. Required for sites
                        that gate content behind login (some TikTok / Instagram
                        / age-restricted YouTube).

Per-item caption extraction reads yt-dlp's info dict and takes the first
non-empty value in:

    description, title, tags

(joined for tags). That mirrors the gallery-dl backend's precedence list,
adapted to yt-dlp's metadata. The captured text is written as the item's
``.txt`` prompt — so by the time the vision worker runs, yt-dlp-sourced videos
look exactly like every other queued item.

Dedup is two-layered, exactly like gallery-dl:

  * yt-dlp's own *download archive* (a flat text file of ``<extractor> <id>``
    lines) at ``<base>/yt_dlp_archive_<slug>.txt`` — it understands per-site IDs
    and stops re-downloading across runs.
  * a ``SeenStore("yt_dlp", slug=SLUG)`` at the cull layer so analytics +
    counters stay consistent with the other scrapers.

The REQUIRE_PROMPT gate (``topic_filter.prompt_optional``) is honoured before
any prompt-length floor: with ``REQUIRE_PROMPT=true`` a captionless item is
dropped; with ``false`` it is queued and the vision worker's auto-caption step
fills the ``.txt`` afterwards.

Run as ``python scraper_yt_dlp.py``; the supervisor (``run_pipeline.py``) spawns
it as the ``YT-DLP`` agent when ``YT_DLP_ENABLED=true`` and ``YT_DLP_URLS`` is
non-empty (gated identically to the gallery-dl agent).

yt-dlp is declared in ``requirements.txt`` — it is imported lazily inside the
download path so this module imports cleanly even when yt-dlp is absent.
"""
from __future__ import annotations

import html
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# Allow direct invocation: `python scraper_yt_dlp.py`.
sys.path.insert(0, str(Path(__file__).parent))

from credentials import get_optional  # noqa: E402
from paths import base_dir, queue_dir as _queue_dir  # noqa: E402
from pipeline_logging import get_logger  # noqa: E402
from queue_manager import save_to_queue  # noqa: E402
from seen_store import SeenStore  # noqa: E402
from topic_filter import prompt_optional  # noqa: E402
import media_policy  # noqa: E402

logger = get_logger(__name__)

SOURCE_NAME: str = "yt_dlp"
SLUG: str = (os.environ.get("PIPELINE_SLUG") or "default").strip() or "default"

# Caption-bearing fields we read out of yt-dlp's info dict, in precedence order:
# prose description first, then the title, then a tag list as a last resort.
_CAPTION_FIELDS: tuple[str, ...] = ("description", "title")
# Video container suffixes yt-dlp may produce. We ingest any of these.
_VIDEO_SUFFIXES: tuple[str, ...] = (
    ".mp4", ".mkv", ".webm", ".mov", ".avi", ".flv", ".m4v", ".ts",
)
# Sidecar files yt-dlp drops next to the media that we must NOT mistake for a
# downloaded video when walking the work dir.
_SIDECAR_SUFFIXES: tuple[str, ...] = (
    ".json", ".txt", ".description", ".part", ".ytdl", ".vtt", ".srt",
    ".jpg", ".jpeg", ".png", ".webp",
)
_DEFAULT_LIMIT: int = 25


def _split_urls(raw: str | None) -> list[str]:
    """Tolerate both newline and comma separators in YT_DLP_URLS."""
    if not raw:
        return []
    candidates: list[str] = []
    for line in raw.replace("\r", "").split("\n"):
        for chunk in line.split(","):
            url = chunk.strip()
            if url and not url.startswith("#"):
                candidates.append(url)
    # Preserve order while de-duplicating.
    seen: set[str] = set()
    deduped: list[str] = []
    for url in candidates:
        if url not in seen:
            seen.add(url)
            deduped.append(url)
    return deduped


def _caption_from_info(info: dict[str, Any]) -> str:
    """Pull a usable caption out of a yt-dlp info dict.

    Precedence: ``description`` -> ``title`` -> ``tags`` (joined). The text is
    HTML-unescaped and stripped. Returns ``""`` when nothing usable is present.
    """
    for key in _CAPTION_FIELDS:
        value = info.get(key)
        if isinstance(value, str) and value.strip():
            return html.unescape(value).strip()
    tags = info.get("tags")
    if isinstance(tags, (list, tuple)) and tags:
        return ", ".join(str(t).strip() for t in tags if str(t).strip())
    if isinstance(tags, str) and tags.strip():
        return html.unescape(tags).strip()
    return ""


def _item_id(info: dict[str, Any], fallback: str) -> str:
    """Stable analytics id for the SeenStore: ``<extractor>_<id>``."""
    extractor = str(info.get("extractor") or info.get("extractor_key") or "yt_dlp")
    item = str(info.get("id") or fallback)
    return f"{extractor}_{item}"


def _safe_int(raw: str | None, default: int) -> int:
    try:
        return max(1, int(str(raw)))
    except (TypeError, ValueError):
        return default


def _build_opts(work_dir: Path, archive_path: Path, limit: int) -> dict[str, Any]:
    """Construct the ``ydl_opts`` dict for a download run.

    A flat output template inside ``work_dir`` (no per-site subfolders), the
    shared download archive for cross-run dedup, ``playlistend`` to cap playlist
    size, and ``writeinfojson`` so each item leaves a ``*.info.json`` we mine for
    the caption + metadata.
    """
    opts: dict[str, Any] = {
        "paths": {"home": str(work_dir)},
        "outtmpl": {"default": "%(extractor)s_%(id)s.%(ext)s"},
        "download_archive": str(archive_path),
        "writeinfojson": True,
        "playlistend": max(1, limit),
        "max_downloads": max(1, limit),
        "ignoreerrors": True,
        "noprogress": True,
        "quiet": True,
        "no_warnings": True,
        # Polite default: a single, reasonable-quality file rather than the
        # full bestvideo+bestaudio merge (no ffmpeg dependency assumption).
        "format": "best[ext=mp4]/best",
        "retries": 3,
        "socket_timeout": 30,
    }
    cookies = get_optional("YT_DLP_COOKIES")
    if cookies:
        opts["cookiefile"] = cookies
    return opts


def _info_json_for(video: Path) -> dict[str, Any]:
    """Read the ``*.info.json`` yt-dlp wrote next to ``video`` (best effort).

    yt-dlp names it ``<stem>.info.json``; we also try ``<name>.info.json`` for
    safety. Returns ``{}`` if absent or malformed.
    """
    import json

    for candidate in (
        video.with_suffix(".info.json"),
        video.with_name(video.name + ".info.json"),
    ):
        if candidate.exists():
            try:
                return json.loads(candidate.read_text(encoding="utf-8", errors="replace"))
            except json.JSONDecodeError:
                logger.debug("malformed info json %s", candidate.name)
            return {}
    return {}


def _ingest_one(
    video: Path,
    info: dict[str, Any],
    seen: SeenStore,
    source_url: str,
) -> int:
    """Route a single downloaded video into the queue.

    Mirrors ``scraper_gallery_dl._ingest_tmp_dir`` per-item: mine the caption,
    apply the REQUIRE_PROMPT gate, dedup against the SeenStore, then hand a
    fresh temp copy to ``save_to_queue`` (whose atomic-move + size validation
    we reuse). Returns 1 if queued, else 0.
    """
    if not video.is_file():
        return 0

    item_id = _item_id(info, video.stem)
    if item_id in seen:
        video.unlink(missing_ok=True)
        return 0

    caption = _caption_from_info(info)
    if not caption and not prompt_optional():
        logger.info("skip %s - no caption and REQUIRE_PROMPT=true", video.name)
        video.unlink(missing_ok=True)
        return 0

    suffix = video.suffix.lower() or ".mp4"
    with tempfile.NamedTemporaryFile(
        suffix=suffix, delete=False, dir=str(video.parent),
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        shutil.copy2(video, tmp_path)
        metadata: dict[str, Any] = {
            "message_id": video.stem,
            "image_url": info.get("webpage_url") or source_url,
            "source_channel": f"yt_dlp:{info.get('extractor', 'unknown')}",
            "source_guild": info.get("uploader")
            or info.get("channel")
            or info.get("extractor", "yt_dlp"),
            "source_url": source_url,
            "extractor": info.get("extractor"),
            "post_id": info.get("id"),
            "duration": info.get("duration"),
            "tags": info.get("tags"),
            "media_kind": "video",
            "pipeline_topic": os.environ.get("PIPELINE_TOPIC", ""),
        }
        saved = save_to_queue(SOURCE_NAME, tmp_path, caption, metadata)
    finally:
        tmp_path.unlink(missing_ok=True)
        video.unlink(missing_ok=True)

    if saved is None:
        return 0

    seen.add(item_id)
    preview = caption[:80].replace("\n", " ") if caption else "(no caption)"
    logger.info("queued %s via yt_dlp: %s", saved.name, preview)
    return 1


def _iter_downloaded(work_dir: Path) -> list[Path]:
    """Every video file yt-dlp left in ``work_dir`` (sidecars excluded)."""
    if not work_dir.exists():
        return []
    videos: list[Path] = []
    for path in sorted(work_dir.iterdir()):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in _VIDEO_SUFFIXES or media_policy.is_video_ext(suffix):
            videos.append(path)
        elif suffix not in _SIDECAR_SUFFIXES:
            # Unknown extension that isn't a known sidecar — treat as media so
            # we don't silently drop a download with an unusual container.
            videos.append(path)
    return videos


def _ingest_work_dir(work_dir: Path, seen: SeenStore, source_url: str) -> int:
    """Ingest every downloaded video in ``work_dir``. Returns count queued."""
    moved = 0
    for video in _iter_downloaded(work_dir):
        info = _info_json_for(video)
        moved += _ingest_one(video, info, seen, source_url)
    return moved


def run_one(url: str, work_dir: Path, seen: SeenStore) -> int:
    """Download + ingest a single URL. Returns count of queued videos.

    yt-dlp is imported here (not at module top) so the module stays importable
    when the dependency is absent — the import only fails when an actual run is
    attempted.
    """
    import yt_dlp
    from yt_dlp.utils import DownloadError

    # Fresh work dir per URL so leftovers from a previous URL aren't re-ingested.
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    archive_path = base_dir() / f"yt_dlp_archive_{SLUG}.txt"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    limit = _safe_int(get_optional("YT_DLP_LIMIT", str(_DEFAULT_LIMIT)), _DEFAULT_LIMIT)
    opts = _build_opts(work_dir, archive_path, limit)

    info: dict[str, Any] | None = None
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except DownloadError as exc:
        # MaxDownloadsReached is signalled via DownloadError; treat as success
        # (we got our cap of items) rather than a hard failure.
        if "MaxDownloads" in str(exc) or "max-downloads" in str(exc).lower():
            logger.info("hit download cap (%d) for %s", limit, url)
        else:
            logger.warning("yt-dlp download error on %s: %s", url, exc)
    except Exception:  # noqa: BLE001
        logger.exception("unexpected yt-dlp failure on %s", url)

    _ = info  # info is per-extraction; we ingest from the work dir on disk.
    return _ingest_work_dir(work_dir, seen, url)


def main() -> int:
    if get_optional("YT_DLP_ENABLED", "false").lower() != "true":
        logger.info("YT_DLP_ENABLED is not 'true' - skipping run")
        return 0

    # yt-dlp only produces video. If the job's media policy excludes video the
    # queue won't pop the clips anyway, so don't fetch them.
    if not media_policy.wants_video():
        logger.info("media policy excludes video - skipping yt-dlp run")
        return 0

    urls = _split_urls(get_optional("YT_DLP_URLS"))
    if not urls:
        logger.info("YT_DLP_URLS is empty - nothing to fetch")
        return 0

    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        raise SystemExit(
            "yt-dlp is not installed. Install pipeline dependencies first:\n"
            "    pip install -r requirements.txt\n"
            "(or `pip install yt-dlp` for just this scraper)."
        )

    base = base_dir()
    work_dir = base / "tmp" / "yt_dlp" / SLUG
    archive_path = base / f"yt_dlp_archive_{SLUG}.txt"
    queue_dest = _queue_dir(SLUG) / SOURCE_NAME
    queue_dest.mkdir(parents=True, exist_ok=True)
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    limit = _safe_int(get_optional("YT_DLP_LIMIT", str(_DEFAULT_LIMIT)), _DEFAULT_LIMIT)
    seen = SeenStore(SOURCE_NAME, slug=SLUG)
    print(f"=== yt-dlp scraper ({len(urls)} URL(s), limit {limit}/URL) ===", flush=True)
    print(f"Archive: {archive_path}", flush=True)
    print(f"Cookies: {get_optional('YT_DLP_COOKIES') or '(none)'}", flush=True)

    total = 0
    try:
        for url in urls:
            print(f"  -> {url}", flush=True)
            total += run_one(url, work_dir, seen)
            seen.flush()
    finally:
        seen.flush()
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)

    print(f"=== yt-dlp done. Queued {total} video(s). ===", flush=True)
    return total


if __name__ == "__main__":
    main()
