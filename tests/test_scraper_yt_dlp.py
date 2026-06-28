"""tests/test_scraper_yt_dlp.py - pytest suite for pipeline_code/scraper_yt_dlp.py.

TDD: written before the implementation. yt_dlp and every network call are
mocked; nothing here touches the real internet or the real yt-dlp binary.

Run with:
    .venv/Scripts/python.exe -m pytest tests/test_scraper_yt_dlp.py -q

What we assert about scraper_yt_dlp:
  * captions are mined from the yt-dlp info dict (description > title > tags);
  * items are deduped via the SeenStore (a second ingest of the same id is a
    no-op and never re-queues);
  * downloaded videos are routed through queue_manager.save_to_queue (we never
    iterate the queue filesystem directly);
  * the REQUIRE_PROMPT gate (topic_filter.prompt_optional) is honoured — with
    REQUIRE_PROMPT=true a video that yields no caption is dropped, with
    REQUIRE_PROMPT=false it is kept.

Plus one test for feed_local_folder: it now accepts a .mp4 video alongside the
existing image extensions.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import pytest

# Make pipeline_code importable as a flat namespace (mirrors the other suites).
PIPELINE_CODE = Path(__file__).resolve().parent.parent / "pipeline_code"
if str(PIPELINE_CODE) not in sys.path:
    sys.path.insert(0, str(PIPELINE_CODE))


# ---------------------------------------------------------------------------
# Import helpers — reload with a clean env each time so module-level SLUG /
# toggles pick up the monkeypatched environment.
# ---------------------------------------------------------------------------

def _import_fresh(mod_name: str):
    sys.modules.pop(mod_name, None)
    spec = importlib.util.spec_from_file_location(
        mod_name, PIPELINE_CODE / f"{mod_name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def yt(monkeypatch, tmp_path):
    """Return a freshly-imported scraper_yt_dlp module with an isolated base dir."""
    monkeypatch.setenv("PIPELINE_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("PIPELINE_SLUG", "default")
    monkeypatch.delenv("YT_DLP_COOKIES", raising=False)
    monkeypatch.delenv("REQUIRE_PROMPT", raising=False)  # default => true
    return _import_fresh("scraper_yt_dlp")


def _make_seen(yt, tmp_path):
    """Build a real SeenStore against the temp base dir for assertions."""
    from seen_store import SeenStore
    return SeenStore(yt.SOURCE_NAME, slug="default", base_dir=tmp_path)


def _write_fake_video(path: Path, size: int = 20000) -> None:
    path.write_bytes(b"\x00" * size)


# ---------------------------------------------------------------------------
# Caption mining
# ---------------------------------------------------------------------------

def test_caption_prefers_description(yt):
    info = {"description": "  a sunny beach  ", "title": "clip", "tags": ["x"]}
    assert yt._caption_from_info(info) == "a sunny beach"


def test_caption_falls_back_to_title(yt):
    info = {"description": "", "title": "My Cool Title", "tags": []}
    assert yt._caption_from_info(info) == "My Cool Title"


def test_caption_falls_back_to_tags(yt):
    info = {"description": "", "title": "", "tags": ["sunset", "ocean", "calm"]}
    assert yt._caption_from_info(info) == "sunset, ocean, calm"


def test_caption_empty_when_nothing_present(yt):
    assert yt._caption_from_info({"id": "abc"}) == ""


# ---------------------------------------------------------------------------
# URL splitting (comma + whitespace tolerant, like gallery-dl's)
# ---------------------------------------------------------------------------

def test_split_urls_handles_commas_and_newlines(yt):
    raw = "https://a/1, https://b/2\nhttps://c/3"
    assert yt._split_urls(raw) == ["https://a/1", "https://b/2", "https://c/3"]


def test_split_urls_dedupes_preserving_order(yt):
    raw = "https://a/1, https://a/1, https://b/2"
    assert yt._split_urls(raw) == ["https://a/1", "https://b/2"]


# ---------------------------------------------------------------------------
# Ingest: dedup + queue + caption + prompt gate
# ---------------------------------------------------------------------------

def test_ingest_saves_to_queue_and_mines_caption(yt, tmp_path, monkeypatch):
    """A downloaded video with a description is queued via save_to_queue, and
    the caption mined from the info dict is the one passed to the queue."""
    tmp_dir = tmp_path / "work"
    tmp_dir.mkdir()
    video = tmp_dir / "vid_clip1.mp4"
    _write_fake_video(video)
    info = {
        "id": "clip1",
        "extractor": "youtube",
        "title": "ignored title",
        "description": "a neon city at night",
        "webpage_url": "https://yt/clip1",
        "ext": "mp4",
    }

    calls: list[dict[str, Any]] = []

    def fake_save(source, image_path, prompt_text="", metadata=None):
        calls.append({
            "source": source,
            "name": Path(image_path).name,
            "prompt": prompt_text,
            "metadata": metadata,
        })
        return tmp_path / "queue" / source / Path(image_path).name

    monkeypatch.setattr(yt, "save_to_queue", fake_save)

    seen = _make_seen(yt, tmp_path)
    moved = yt._ingest_one(video, info, seen, "https://yt/clip1")

    assert moved == 1
    assert len(calls) == 1
    assert calls[0]["source"] == yt.SOURCE_NAME
    assert calls[0]["prompt"] == "a neon city at night"
    # save_to_queue must have been handed a real, existing temp file.
    assert calls[0]["name"].endswith(".mp4")
    # The id is now recorded in the dedup store.
    assert "youtube_clip1" in seen


def test_ingest_dedupes_via_seen_store(yt, tmp_path, monkeypatch):
    """A video whose id is already in the SeenStore is never re-queued."""
    tmp_dir = tmp_path / "work"
    tmp_dir.mkdir()
    video = tmp_dir / "vid_dupe.mp4"
    _write_fake_video(video)
    info = {"id": "dupe", "extractor": "youtube",
            "description": "something", "ext": "mp4"}

    saved: list[str] = []
    monkeypatch.setattr(
        yt, "save_to_queue",
        lambda *a, **k: saved.append("x") or (tmp_path / "q"),
    )

    seen = _make_seen(yt, tmp_path)
    seen.add("youtube_dupe")  # pre-seed: already processed

    moved = yt._ingest_one(video, info, seen, "https://yt/dupe")

    assert moved == 0
    assert saved == []  # save_to_queue never called for a known id


def test_ingest_drops_when_no_caption_and_prompt_required(yt, tmp_path, monkeypatch):
    """REQUIRE_PROMPT=true (default): a captionless video is dropped, not queued."""
    monkeypatch.setenv("REQUIRE_PROMPT", "true")
    tmp_dir = tmp_path / "work"
    tmp_dir.mkdir()
    video = tmp_dir / "vid_nocap.mp4"
    _write_fake_video(video)
    info = {"id": "nocap", "extractor": "youtube", "ext": "mp4"}  # no caption

    saved: list[str] = []
    monkeypatch.setattr(
        yt, "save_to_queue",
        lambda *a, **k: saved.append("x") or (tmp_path / "q"),
    )

    seen = _make_seen(yt, tmp_path)
    moved = yt._ingest_one(video, info, seen, "https://yt/nocap")

    assert moved == 0
    assert saved == []


def test_ingest_keeps_captionless_when_prompt_optional(yt, tmp_path, monkeypatch):
    """REQUIRE_PROMPT=false: a captionless video is still queued (auto-caption
    fills the .txt downstream)."""
    monkeypatch.setenv("REQUIRE_PROMPT", "false")
    tmp_dir = tmp_path / "work"
    tmp_dir.mkdir()
    video = tmp_dir / "vid_optional.mp4"
    _write_fake_video(video)
    info = {"id": "opt", "extractor": "youtube", "ext": "mp4"}  # no caption

    saved: list[dict[str, Any]] = []

    def fake_save(source, image_path, prompt_text="", metadata=None):
        saved.append({"prompt": prompt_text})
        return tmp_path / "queue" / source / Path(image_path).name

    monkeypatch.setattr(yt, "save_to_queue", fake_save)

    seen = _make_seen(yt, tmp_path)
    moved = yt._ingest_one(video, info, seen, "https://yt/opt")

    assert moved == 1
    assert len(saved) == 1
    assert saved[0]["prompt"] == ""  # empty caption is allowed when optional


# ---------------------------------------------------------------------------
# run_one: yt_dlp is mocked; verify it drives extract_info + ingest.
# ---------------------------------------------------------------------------

def test_run_one_uses_mocked_yt_dlp_and_queues(yt, tmp_path, monkeypatch):
    """End-to-end-ish: a fake YoutubeDL writes a video + info into the work dir,
    run_one ingests it through save_to_queue. No real yt-dlp / network."""
    work = tmp_path / "ytwork"

    captured_prompts: list[str] = []

    def fake_save(source, image_path, prompt_text="", metadata=None):
        captured_prompts.append(prompt_text)
        return tmp_path / "queue" / source / Path(image_path).name

    monkeypatch.setattr(yt, "save_to_queue", fake_save)

    class FakeYDL:
        """Minimal stand-in for yt_dlp.YoutubeDL used as a context manager."""

        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def extract_info(self, url, download=True):
            # Simulate yt-dlp dropping a file into the configured paths/home,
            # plus the *.info.json sidecar it writes when writeinfojson=True.
            import json as _json

            target = Path(self.opts["paths"]["home"])
            target.mkdir(parents=True, exist_ok=True)
            vid = target / "youtube_movieid.mp4"
            _write_fake_video(vid)
            info = {
                "id": "movieid",
                "extractor": "youtube",
                "title": "t",
                "description": "a quiet forest stream",
                "ext": "mp4",
                "webpage_url": url,
                "requested_downloads": [{"filepath": str(vid)}],
            }
            vid.with_suffix(".info.json").write_text(
                _json.dumps(info), encoding="utf-8"
            )
            return info

    # Inject a fake yt_dlp module so `import yt_dlp` inside the scraper resolves.
    fake_mod = type(sys)("yt_dlp")
    fake_mod.YoutubeDL = FakeYDL

    class _DLErr(Exception):
        pass

    fake_mod.utils = type(sys)("yt_dlp.utils")
    fake_mod.utils.DownloadError = _DLErr
    monkeypatch.setitem(sys.modules, "yt_dlp", fake_mod)
    monkeypatch.setitem(sys.modules, "yt_dlp.utils", fake_mod.utils)

    seen = _make_seen(yt, tmp_path)
    moved = yt.run_one("https://yt/movieid", work, seen)

    assert moved == 1
    assert captured_prompts == ["a quiet forest stream"]
    assert "youtube_movieid" in seen


# ---------------------------------------------------------------------------
# feed_local_folder now accepts a .mp4 video.
# ---------------------------------------------------------------------------

def test_feed_local_folder_accepts_mp4(monkeypatch, tmp_path):
    """feed_local_folder's accepted-extension set now includes video files, so a
    .mp4 dropped in a watched folder is recognised and queued."""
    src = tmp_path / "incoming"
    src.mkdir()
    monkeypatch.setenv("PIPELINE_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("PIPELINE_SLUG", "default")
    monkeypatch.setenv("LOCAL_IMPORT_DIR", str(src))
    monkeypatch.setenv("LOCAL_IMPORT_NAME", "local")
    monkeypatch.setenv("LOCAL_IMPORT_ENABLED", "true")
    monkeypatch.setenv("REQUIRE_PROMPT", "false")  # video may have no .txt

    feed = _import_fresh("feed_local_folder")

    # .mp4 is now an accepted suffix.
    assert ".mp4" in feed.MEDIA_SUFFIXES

    video = src / "myclip.mp4"
    _write_fake_video(video)

    saved: list[str] = []

    def fake_save(source, image_path, prompt_text="", metadata=None):
        saved.append(Path(image_path).name)
        return tmp_path / "queue" / source / Path(image_path).name

    monkeypatch.setattr(feed, "save_to_queue", fake_save)

    # iter_sources must surface the .mp4 file.
    sources = feed.iter_sources()
    assert any(p.suffix == ".mp4" for p in sources)

    seen = _make_seen_for_feed(feed, tmp_path)
    assert feed.process_one(video, seen) is True
    assert len(saved) == 1
    assert saved[0].endswith(".mp4")


def _make_seen_for_feed(feed, tmp_path):
    from seen_store import SeenStore
    return SeenStore(feed.SOURCE_NAME, slug="default", base_dir=tmp_path)
