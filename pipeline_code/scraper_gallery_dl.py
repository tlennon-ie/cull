"""scraper_gallery_dl.py - Generic URL scraper backed by gallery-dl.

Wraps the `gallery-dl` library (https://codeberg.org/mikf/gallery-dl) so cull
can pull images from any of its 340+ supported sites (Pixiv, DeviantArt, the
booru family, ArtStation, Tumblr, Newgrounds, FurAffinity / e621, Imgur,
Flickr, Reddit, X, etc.) without writing per-site code.

Pinned to a specific tag in `requirements.txt` (currently v1.32.1). gallery-dl
is maintained by Mike Fährmann (@mikf) — see README's Acknowledgements section.

Configuration (all read from .env, all editable from the dashboard):

    GALLERY_DL_ENABLED          on/off toggle.
    GALLERY_DL_URLS             newline OR comma separated URLs to fetch.
    GALLERY_DL_LIMIT_PER_URL    images per URL (default 50). gallery-dl's
                                `image-range` / `file-range` setting.
    GALLERY_DL_COOKIES_FILE     optional Netscape cookies.txt path. Required
                                for sites that gate galleries behind login
                                (Pixiv, X, Patreon, Tumblr in some cases).
    GALLERY_DL_CONFIG_PATH      optional path to a gallery-dl JSON config.
                                Advanced override for power users; loaded on
                                top of the in-process defaults below.

Per-image caption extraction reads the metadata JSON gallery-dl emits next
to each image and looks for the first non-empty value in:

    description, caption, selftext, content, title

That covers Pixiv (`description`), Tumblr (`caption`), Reddit (`selftext`),
X (`content`), DeviantArt (`description`), and most booru sites (`tags` are
joined separately). The captured text is HTML-unescaped, stripped, and
written as the image's `.txt` prompt — so by the time the vision worker
runs, gallery-dl-sourced images look exactly like civitai or X images.

Dedup uses gallery-dl's own SQLite archive (it understands per-site IDs the
SeenStore can't), keyed at `<base>/gallery_dl_archive_<slug>.sqlite3`. The
SeenStore is still used at the cull layer so analytics + counters stay
consistent across scrapers.

Run as `python scraper_gallery_dl.py`; the supervisor (`run_pipeline.py`)
spawns it as the `Gallery-DL` agent when `GALLERY_DL_ENABLED=true` and
`GALLERY_DL_URLS` is non-empty.
"""
from __future__ import annotations

import html
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# Allow direct invocation: `python scraper_gallery_dl.py`.
sys.path.insert(0, str(Path(__file__).parent))

from credentials import get_optional  # noqa: E402
from paths import base_dir, queue_dir as _queue_dir  # noqa: E402
from pipeline_logging import get_logger  # noqa: E402
from queue_manager import save_to_queue  # noqa: E402
from seen_store import SeenStore  # noqa: E402
from topic_filter import prompt_optional  # noqa: E402
import media_policy  # noqa: E402

logger = get_logger(__name__)

SOURCE_NAME: str = "gallery_dl"
SLUG: str = (os.environ.get("PIPELINE_SLUG") or "default").strip() or "default"

# Caption-bearing fields we read out of gallery-dl's metadata JSON, in
# precedence order. Pixiv prefers `description`, Tumblr `caption`, Reddit
# `selftext`, X `content`, DeviantArt `description`, generic fallback `title`.
_CAPTION_FIELDS: tuple[str, ...] = (
    "description", "caption", "selftext", "content", "title",
)
# Media suffixes are governed by the active media policy (MEDIA_TYPES + ext
# lists) — gallery-dl already downloads whatever the extractor yields (images
# AND video), so we simply stop discarding the kinds the job accepts.


def _split_urls(raw: str) -> list[str]:
    """Tolerate both newline and comma separators in GALLERY_DL_URLS."""
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


def _caption_from_meta(meta: dict[str, Any]) -> str:
    """Pull a usable caption out of a gallery-dl metadata JSON.

    The JSON is per-site; we walk a precedence list of fields until we find
    one with non-empty content. Tags get appended when no prose caption
    exists, so booru-only sites still produce a useful `.txt`.
    """
    for key in _CAPTION_FIELDS:
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return html.unescape(value).strip()
    # Booru-style fallback: join tags into a comma-separated string.
    for tag_key in ("tags", "tag_string", "tag_string_general"):
        tags = meta.get(tag_key)
        if isinstance(tags, list) and tags:
            return ", ".join(str(t).strip() for t in tags if t)
        if isinstance(tags, str) and tags.strip():
            return tags.replace(" ", ", ").strip()
    return ""


def _apply_inline_config(config, raw_json: str, label: str) -> None:
    """Merge an inline gallery-dl config (JSON object string) into the module
    config by writing it to a temp file and using gallery-dl's own loader (which
    deep-merges). No-op on empty / invalid JSON — a bad custom arg can never
    break the run, it's just logged and skipped."""
    raw = (raw_json or "").strip()
    if not raw:
        return
    try:
        data = json.loads(raw)
    except (ValueError, TypeError) as exc:
        logger.warning("ignoring %s (not valid JSON): %s", label, exc)
        return
    if not isinstance(data, dict):
        logger.warning("ignoring %s: expected a JSON object", label)
        return
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                         encoding="utf-8") as fh:
            json.dump(data, fh)
            tmp_path = fh.name
        config.load([tmp_path])          # gallery-dl reads immediately + merges
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not apply %s: %s", label, exc)
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _configure_gallery_dl(tmp_dir: Path, archive_path: Path, limit: int) -> None:
    """Push our defaults into gallery-dl's module-level config."""
    from gallery_dl import config

    config.clear()
    config.set((), "extractor", {})
    # Flat output directory: empty list = no per-site subfolders.
    config.set(("extractor",), "base-directory", str(tmp_dir))
    config.set(("extractor",), "directory", [])
    # IMPORTANT: do NOT override `filename` either. Each extractor's native
    # filename_fmt is designed to be unique per file; our `{category}_{num}`
    # template collapsed every post in a listing to the same name (`reddit_0.jpg`,
    # `redgifs_0.m4v`) because `{num}` is only the per-POST file index — so all but
    # the first "already exist" and the run churns the same handful of files.
    # Native names are flat (no path separators) so they stay in one directory.
    config.set(("extractor",), "archive", str(archive_path))
    # IMPORTANT: do NOT override `archive-format`. Each extractor ships a native
    # `archive_fmt` purpose-built to uniquely identify a file for that site. The
    # previous `{category}_{id}_{num}` override broke dedup on every site without
    # a numeric `id`: pornpics emits id=None, so all galleries collapsed to the
    # same keys (`pornpics_None_1..N`) and gallery-dl skipped every gallery after
    # the first one — i.e. the scraper silently downloaded nothing.
    range_value = f"1-{max(1, limit)}"
    config.set(("extractor",), "image-range", range_value)
    config.set(("extractor",), "file-range", range_value)
    # Polite request pacing so we don't trip rate-limiters.
    config.set(("extractor",), "sleep-request", [1.0, 3.0])
    # Write per-image metadata JSON next to the image. This is what we mine
    # for the caption / .txt later.
    config.set(("extractor",), "postprocessors", [{
        "name": "metadata",
        "mode": "json",
        "event": "after",
    }])

    cookies_path = get_optional("GALLERY_DL_COOKIES_FILE")
    if cookies_path:
        config.set(("extractor",), "cookies", cookies_path)

    extra_config = get_optional("GALLERY_DL_CONFIG_PATH")
    if extra_config:
        try:
            config.load([extra_config])
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not load extra gallery-dl config %s: %s",
                           extra_config, exc)

    # Custom gallery-dl arguments (inline JSON) — any option from the gallery-dl
    # docs. Global first, then the per-job block on top, so a job can extend or
    # override the global. Both merge into gallery-dl's config.
    _apply_inline_config(config, get_optional("GALLERY_DL_CONFIG_JSON"),
                         "global GALLERY_DL_CONFIG_JSON")
    _apply_inline_config(config, get_optional("GALLERY_DL_CONFIG_JSON_JOB"),
                         "per-job gallery-dl config")


def _safe_int(raw: str, default: int) -> int:
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def _ingest_tmp_dir(tmp_dir: Path, seen: SeenStore, source_url: str) -> int:
    """Move every downloaded image from `tmp_dir` into the cull queue.

    Reads the sibling `<image>.json` (gallery-dl's metadata postprocessor
    output), extracts a caption, and routes the image through `save_to_queue`
    so the existing dedup + size validation + path normalisation kicks in.
    """
    moved = 0
    for img_path in sorted(tmp_dir.iterdir()):
        if not img_path.is_file():
            continue
        suffix = img_path.suffix.lower()
        if not media_policy.accepts(suffix):
            continue

        # gallery-dl writes `<image>.<ext>.json` (extension `json`, suffix
        # appended). Older versions used `<stem>.json` — try both.
        meta: dict[str, Any] = {}
        for candidate in (
            img_path.with_suffix(img_path.suffix + ".json"),
            img_path.with_suffix(".json"),
        ):
            if candidate.exists():
                try:
                    meta = json.loads(candidate.read_text(encoding="utf-8", errors="replace"))
                except json.JSONDecodeError:
                    logger.debug("malformed metadata json %s", candidate.name)
                    meta = {}
                candidate.unlink(missing_ok=True)
                break

        caption = _caption_from_meta(meta)
        if not caption and not prompt_optional():
            logger.info("skip %s — no caption and REQUIRE_PROMPT=true", img_path.name)
            img_path.unlink(missing_ok=True)
            continue

        # Build a stable, GLOBALLY-unique id for the SeenStore. Prefer the source
        # image URL (unique per image across every gallery). Fall back to
        # category+id+num — but use ``or`` (not dict.get's default) because many
        # extractors emit ``id`` as an explicit null: ``meta.get('id', stem)``
        # returns None there, which collapsed every gallery to ``<cat>_None_<n>``
        # and made the cull-layer dedup skip images across unrelated galleries.
        category_id = meta.get("url") or (
            f"{meta.get('category', 'unknown')}_"
            f"{meta.get('id') or img_path.stem}_"
            f"{meta.get('num', 0)}"
        )
        if category_id in seen:
            img_path.unlink(missing_ok=True)
            continue

        # Move into a real tempfile so save_to_queue's atomic move semantics
        # apply (it expects a writable, deletable temp source).
        with tempfile.NamedTemporaryFile(
            suffix=suffix, delete=False, dir=str(tmp_dir),
        ) as tmp:
            tmp_path = Path(tmp.name)
        try:
            shutil.copy2(img_path, tmp_path)
            metadata: dict[str, Any] = {
                "message_id": img_path.stem,
                "image_url": meta.get("url", source_url),
                "source_channel": f"gallery_dl:{meta.get('category', 'unknown')}",
                "source_guild": meta.get("subcategory") or meta.get("category", "gallery_dl"),
                "source_url": source_url,
                "extractor": meta.get("extractor"),
                "post_id": meta.get("id"),
                "post_num": meta.get("num"),
                "tags": meta.get("tags"),
                "pipeline_topic": os.environ.get("PIPELINE_TOPIC", ""),
            }
            saved = save_to_queue(SOURCE_NAME, tmp_path, caption, metadata)
        finally:
            tmp_path.unlink(missing_ok=True)
            img_path.unlink(missing_ok=True)

        if saved is not None:
            seen.add(category_id)
            moved += 1
            preview = caption[:80].replace("\n", " ") if caption else "(no caption)"
            logger.info("queued %s via gallery_dl: %s", saved.name, preview)
    return moved


def run_one(url: str, tmp_dir: Path, seen: SeenStore) -> int:
    """Download + ingest a single URL. Returns count of queued images."""
    from gallery_dl import job
    from gallery_dl import exception as gex

    # Fresh tmp dir per URL so leftover files from a previous URL don't get
    # ingested twice.
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        status = job.DownloadJob(url).run()
    except gex.NoExtractorError:
        logger.warning("unsupported URL (no gallery-dl extractor): %s", url)
        return 0
    except gex.AuthorizationError:
        logger.warning("authorization required (login / cookies): %s", url)
        return 0
    except gex.AuthenticationError:
        logger.warning("authentication failed for %s", url)
        return 0
    except gex.NotFoundError:
        logger.info("404 / not found: %s", url)
        return 0
    except gex.HttpError as exc:
        logger.warning("http error on %s: %s", url, exc)
        return 0
    except gex.GalleryDLException as exc:
        logger.error("gallery-dl error on %s: %s", url, exc)
        return 0
    except Exception:  # noqa: BLE001
        logger.exception("unexpected gallery-dl failure on %s", url)
        return 0

    if status:
        logger.warning("gallery-dl returned non-zero status %s for %s", status, url)

    return _ingest_tmp_dir(tmp_dir, seen, url)


def main() -> int:
    if get_optional("GALLERY_DL_ENABLED", "false").lower() != "true":
        logger.info("GALLERY_DL_ENABLED is not 'true' — skipping run")
        return 0

    urls = _split_urls(get_optional("GALLERY_DL_URLS"))
    if not urls:
        logger.info("GALLERY_DL_URLS is empty — nothing to fetch")
        return 0

    try:
        import gallery_dl  # noqa: F401
    except ImportError:
        raise SystemExit(
            "gallery-dl is not installed. Install pipeline dependencies first:\n"
            "    pip install -r requirements.txt\n"
            "(or `pip install gallery-dl` for just this scraper)."
        )

    base = base_dir()
    tmp_dir = base / "tmp" / "gallery_dl" / SLUG
    archive_path = base / f"gallery_dl_archive_{SLUG}.sqlite3"
    queue_dest = _queue_dir(SLUG) / SOURCE_NAME
    queue_dest.mkdir(parents=True, exist_ok=True)
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    limit = _safe_int(get_optional("GALLERY_DL_LIMIT_PER_URL", "50"), 50)
    _configure_gallery_dl(tmp_dir, archive_path, limit)

    seen = SeenStore(SOURCE_NAME, slug=SLUG)
    print(f"=== gallery-dl scraper ({len(urls)} URL(s), limit {limit}/URL) ===", flush=True)
    print(f"Archive: {archive_path}", flush=True)
    print(f"Cookies: {get_optional('GALLERY_DL_COOKIES_FILE') or '(none)'}", flush=True)

    total = 0
    try:
        for url in urls:
            print(f"  → {url}", flush=True)
            total += run_one(url, tmp_dir, seen)
            seen.flush()
    finally:
        seen.flush()
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"=== gallery-dl done. Queued {total} image(s). ===", flush=True)
    return total


if __name__ == "__main__":
    main()
