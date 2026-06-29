"""Integration tests for the Wave-2 core wiring (feat/roadmap-wave-1).

Covers the five seams added by the integration slice, every one of which is
gated behind an env flag that defaults OFF (so with no new env set the pipeline
behaves byte-identically):

  1. pHASH AT FINALIZE  — ``vision_worker_base._finalise`` perceptual-hashes the
     just-sorted image and stores it via ``index_store.set_phash`` (best-effort).
  2. PER-JOB yt-dlp      — ``job_config.resolve_env`` projects a ``scrapers.yt_dlp``
     block to ``YT_DLP_*`` env, mirroring the gallery-dl mapping.
  3. SCHEDULER TICK      — ``run_pipeline._scheduler_tick`` ticks due schedules via
     the injected runner ONLY when ``SCHEDULER_ENABLED`` is truthy.
  4. VIDEO QUEUE GLOBS   — ``queue_manager`` only surfaces video clips in the
     round-robin pop when ``VIDEO_CLASSIFY_ENABLED`` is set.
  5. PREFILTER PRE-STAGE — ``vision_worker_base._prefilter_reject`` routes a
     failing image to DISCARD when ``PREFILTER_ENABLED`` is set; a no-op (fall
     through to the VLM) when unset.

Runnable:
    pytest tests/test_core_integration.py -q
    python tests/test_core_integration.py
"""
from __future__ import annotations

import importlib
import io
import json
import os
import sys
import types
from pathlib import Path

import pytest

PIPELINE_CODE = Path(__file__).resolve().parent.parent / "pipeline_code"
if str(PIPELINE_CODE) not in sys.path:
    sys.path.insert(0, str(PIPELINE_CODE))


@pytest.fixture(autouse=True)
def _restore_environ():
    """Snapshot/restore os.environ around every test so module-level
    ``load_dotenv()`` (run when run_pipeline / queue_manager import) can't leak
    real .env values into sibling suites."""
    snapshot = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(snapshot)


# ── shared helpers ─────────────────────────────────────────────────────────────

def _write_jpeg(path: Path, size: tuple[int, int] = (64, 64), color=(120, 90, 200)) -> None:
    """Write a small real JPEG so PIL-based code paths (resize, phash) work."""
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path, format="JPEG")


def _jpeg_bytes(size: tuple[int, int] = (64, 64), color=(10, 10, 10)) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return buf.getvalue()


def _make_worker(monkeypatch, tmp_path: Path):
    """Construct a minimal concrete BaseVisionWorker pointed at a temp sorted dir.

    ``__init__`` mkdirs the active taxonomy's folders under ``sorted_dir()``; we
    pin ``PIPELINE_BASE_DIR`` so that lands in the temp tree, not the repo.
    """
    monkeypatch.setenv("PIPELINE_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("PIPELINE_SORTED", str(tmp_path / "sorted"))
    import paths
    importlib.reload(paths)
    import vision_worker_base
    importlib.reload(vision_worker_base)

    class _Worker(vision_worker_base.BaseVisionWorker):
        name = "test-worker"

        def classify_image_bytes(self, b64_jpeg, prompt_instruction):  # pragma: no cover
            return None

    return vision_worker_base, _Worker()


def _make_ctx(vwb_mod, image_path: Path, processing_path: Path, source: str = "civitai"):
    return vwb_mod.ClassifyContext(
        image_path=image_path,
        processing_path=processing_path,
        source_name=source,
        prompt_text="",
        metadata={},
    )


# ── 1. pHASH AT FINALIZE ───────────────────────────────────────────────────────

def test_finalise_calls_set_phash_on_sorted_image(monkeypatch, tmp_path):
    """_finalise must perceptual-hash the FINAL sorted image and store it via
    index_store.set_phash(<final path>, <hash>)."""
    vwb, worker = _make_worker(monkeypatch, tmp_path)

    # Stub the two lazily-imported modules so no real DB / hashing is required.
    calls: dict[str, object] = {}
    fake_phash = types.ModuleType("phash_dedup")
    fake_phash.compute_phash = lambda p: "deadbeefdeadbeef"  # type: ignore[attr-defined]
    fake_index = types.ModuleType("index_store")

    def _set_phash(key, value):
        calls["key"] = key
        calls["value"] = value

    fake_index.set_phash = _set_phash  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "phash_dedup", fake_phash)
    monkeypatch.setitem(sys.modules, "index_store", fake_index)

    # A claimed .processing image ready to be moved into sorted/.
    queue_dir = tmp_path / "queue" / "civitai"
    img = queue_dir / "pic.jpg"
    _write_jpeg(img)
    processing = Path(str(img) + ".processing")
    img.rename(processing)
    ctx = _make_ctx(vwb, img, processing)

    result = {"category": "DISCARD", "OVR_Quality_Score": 5, "REL_Quality_Score": 4}
    final_category = worker._finalise(ctx, result)

    assert final_category == "DISCARD"
    # set_phash was called with the FINAL sorted path (not the queue path) + hash.
    assert calls.get("value") == "deadbeefdeadbeef"
    final_key = calls.get("key")
    assert final_key is not None
    assert "sorted" in str(final_key)
    assert Path(str(final_key)).exists()  # the moved image really is there


def test_finalise_phash_failure_never_breaks_finalize(monkeypatch, tmp_path):
    """A phash/index error must be swallowed — finalize still returns the category
    and the image is still sorted."""
    vwb, worker = _make_worker(monkeypatch, tmp_path)

    fake_phash = types.ModuleType("phash_dedup")

    def _boom(_p):
        raise RuntimeError("phash exploded")

    fake_phash.compute_phash = _boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "phash_dedup", fake_phash)
    # index_store deliberately left unstubbed; the try/except must absorb anything.

    queue_dir = tmp_path / "queue" / "civitai"
    img = queue_dir / "pic.jpg"
    _write_jpeg(img)
    processing = Path(str(img) + ".processing")
    img.rename(processing)
    ctx = _make_ctx(vwb, img, processing)

    final_category = worker._finalise(ctx, {"category": "DISCARD"})
    assert final_category == "DISCARD"


# ── 2. PER-JOB yt-dlp PROJECTION ───────────────────────────────────────────────

@pytest.fixture()
def jobs_isolated(tmp_path, monkeypatch):
    """Point job_config storage at a temp dir (mirrors test_job_config.isolated)."""
    monkeypatch.setenv("PIPELINE_BASE_DIR", str(tmp_path))
    import categories
    monkeypatch.setattr(categories, "ACTIVE_PATH", tmp_path / "cull_categories.json")
    monkeypatch.setattr(categories, "_cache", None, raising=False)
    monkeypatch.setattr(categories, "_cache_mtime", 0.0, raising=False)
    import job_config
    importlib.reload(job_config)
    return job_config, tmp_path


def test_resolve_env_emits_yt_dlp_defaults(jobs_isolated):
    """A job with no yt_dlp override still gets the (disabled) YT_DLP_* keys —
    every key is always emitted so a job fully overrides stale .env values."""
    jc, _ = jobs_isolated
    job = jc.create_job("V", subject="S")
    env = jc.resolve_env(job)
    assert env["YT_DLP_ENABLED"] == "false"
    assert env["YT_DLP_URLS"] == ""
    assert env["YT_DLP_LIMIT"] == "200"
    assert env["YT_DLP_COOKIES"] == ""


def test_resolve_env_emits_yt_dlp_block(jobs_isolated):
    """An overridden scrapers.yt_dlp block projects to YT_DLP_* exactly like
    the gallery-dl mapping (urls newline-joined, limit stringified)."""
    jc, _ = jobs_isolated
    job = jc.create_job("V", subject="S")
    job = jc.set_override(job, "scrapers.yt_dlp", {
        "enabled": True,
        "urls": ["https://youtu.be/a", "https://tiktok.com/@x/video/1"],
        "limit": 50,
        "cookies": "/tmp/cookies.txt",
    })
    env = jc.resolve_env(job)
    assert env["YT_DLP_ENABLED"] == "true"
    assert env["YT_DLP_URLS"] == "https://youtu.be/a\nhttps://tiktok.com/@x/video/1"
    assert env["YT_DLP_LIMIT"] == "50"
    assert env["YT_DLP_COOKIES"] == "/tmp/cookies.txt"
    # All env values remain strings (the resolve_env contract).
    assert all(isinstance(v, str) for v in env.values())


def test_yt_dlp_override_flows_through_effective_config(jobs_isolated):
    """A scrapers.yt_dlp override flows through effective_config into resolve_env
    (the projection layer keeps yt_dlp in lockstep with gallery_dl). NOTE: the
    block is intentionally NOT persisted into _default_preset_cfg — config_io's
    strict preset validator (an out-of-scope file for this slice) rejects unknown
    scrapers keys, so yt_dlp lives at the projection layer and as a per-job
    override; resolve_env supplies disabled defaults when absent. The dashboard /
    config_io agent should add yt_dlp to the config_io allowlist + Scrapers UI to
    make it a first-class persisted preset field."""
    jc, _ = jobs_isolated
    job = jc.create_job("V", subject="S")
    job = jc.set_override(job, "scrapers.yt_dlp",
                          {"enabled": True, "urls": ["u"], "limit": 9, "cookies": "c"})
    eff = jc.effective_config(job)
    assert eff["scrapers"]["yt_dlp"] == {
        "enabled": True, "urls": ["u"], "limit": 9, "cookies": "c"}


# ── 3. SCHEDULER TICK ──────────────────────────────────────────────────────────

def _install_fake_scheduler(monkeypatch):
    """Install a fake ``scheduler`` module that records run_due_now invocations."""
    captured: dict[str, object] = {"calls": 0, "runner": None}
    fake = types.ModuleType("scheduler")

    def _run_due_now(runner):
        captured["calls"] = int(captured["calls"]) + 1  # type: ignore[arg-type]
        captured["runner"] = runner
        return []

    fake.run_due_now = _run_due_now  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "scheduler", fake)
    return captured


def test_scheduler_tick_noop_when_flag_unset(monkeypatch):
    import run_pipeline
    importlib.reload(run_pipeline)
    captured = _install_fake_scheduler(monkeypatch)
    monkeypatch.delenv("SCHEDULER_ENABLED", raising=False)

    run_pipeline._scheduler_tick()
    assert captured["calls"] == 0  # scheduler never consulted when disabled


def test_scheduler_tick_runs_when_enabled(monkeypatch):
    import run_pipeline
    importlib.reload(run_pipeline)
    captured = _install_fake_scheduler(monkeypatch)
    monkeypatch.setenv("SCHEDULER_ENABLED", "true")

    run_pipeline._scheduler_tick()
    assert captured["calls"] == 1
    # The supervisor passes its own runner (the schedule→primitive mapper).
    assert captured["runner"] is run_pipeline._schedule_runner


def test_scheduler_tick_swallows_runner_errors(monkeypatch):
    """A scheduler that raises must never propagate out of the tick (a schedule
    failure can't be allowed to stop the supervisor)."""
    import run_pipeline
    importlib.reload(run_pipeline)
    fake = types.ModuleType("scheduler")

    def _boom(_runner):
        raise RuntimeError("scheduler blew up")

    fake.run_due_now = _boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "scheduler", fake)
    monkeypatch.setenv("SCHEDULER_ENABLED", "true")

    run_pipeline._scheduler_tick()  # must not raise


def test_schedule_runner_scrape_activates_job(monkeypatch):
    """The runner maps scrape/curate onto job_config.activate (an existing
    supervisor primitive)."""
    import run_pipeline
    importlib.reload(run_pipeline)
    activated: list[str] = []
    monkeypatch.setattr(run_pipeline.job_config, "activate",
                        lambda slug: activated.append(slug))

    run_pipeline._schedule_runner("myjob", "scrape")
    run_pipeline._schedule_runner("myjob", "curate")
    assert activated == ["myjob", "myjob"]


def test_schedule_runner_export_is_best_effort(monkeypatch):
    """An export action delegates to export_profiles; a failure there is logged
    and swallowed (never raises out of the runner)."""
    import run_pipeline
    importlib.reload(run_pipeline)
    fake_export = types.ModuleType("export_profiles")

    def _boom(*a, **k):
        raise RuntimeError("export failed")

    fake_export.export_dataset = _boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "export_profiles", fake_export)

    run_pipeline._schedule_runner("myjob", "export")  # must not raise


# ── 4. VIDEO QUEUE GLOBS ───────────────────────────────────────────────────────

def test_queue_globs_images_only_by_default(monkeypatch):
    import queue_manager
    importlib.reload(queue_manager)
    monkeypatch.delenv("VIDEO_CLASSIFY_ENABLED", raising=False)
    globs = queue_manager._queue_globs()
    assert globs == queue_manager._QUEUE_IMAGE_GLOBS
    assert "*.mp4" not in globs


def test_queue_globs_include_video_when_enabled(monkeypatch):
    import queue_manager
    importlib.reload(queue_manager)
    monkeypatch.setenv("VIDEO_CLASSIFY_ENABLED", "true")
    globs = queue_manager._queue_globs()
    for ext in ("*.mp4", "*.mov", "*.webm", "*.mkv", "*.avi"):
        assert ext in globs
    # Images are still present too.
    assert "*.jpg" in globs


def test_queue_popping_surfaces_video_only_when_enabled(monkeypatch, tmp_path):
    """End-to-end through FSQueue: a clip in a source dir is popped only when the
    video lane is enabled; with it off the queue reports empty."""
    import queue_manager
    importlib.reload(queue_manager)
    src = tmp_path / "civitai"
    src.mkdir(parents=True)
    clip = src / "clip.mp4"
    clip.write_bytes(b"\x00\x00\x00\x18ftypmp42fake-clip-bytes")
    q = queue_manager.FSQueue(tmp_path)

    monkeypatch.delenv("VIDEO_CLASSIFY_ENABLED", raising=False)
    assert q.list_sources() == {}  # clip invisible to an image-only queue

    monkeypatch.setenv("VIDEO_CLASSIFY_ENABLED", "true")
    q2 = queue_manager.FSQueue(tmp_path)  # fresh instance → no stale cache
    source, path = q2.pop_next()
    assert source == "civitai"
    assert path is not None and path.name == "clip.mp4"


# ── 5. PREFILTER PRE-STAGE ─────────────────────────────────────────────────────

def test_prefilter_reject_noop_when_flag_unset(monkeypatch, tmp_path):
    """With PREFILTER_ENABLED unset, _prefilter_reject returns None (the image
    proceeds to the VLM) and assess is never consulted."""
    vwb, worker = _make_worker(monkeypatch, tmp_path)
    fake = types.ModuleType("prefilter_aesthetic")
    fake.is_enabled = lambda: False  # type: ignore[attr-defined]
    consulted = {"assess": 0}

    def _assess(*a, **k):  # pragma: no cover - must not be called
        consulted["assess"] += 1
        return {"passed": False, "reasons": ["x"]}

    fake.assess = _assess  # type: ignore[attr-defined]
    fake.env_min_score = lambda: 35  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "prefilter_aesthetic", fake)

    assert worker._prefilter_reject(_jpeg_bytes()) is None
    assert consulted["assess"] == 0


def test_prefilter_reject_returns_reason_when_failing(monkeypatch, tmp_path):
    """When enabled and assess() fails the image, a DISCARD reason is returned."""
    vwb, worker = _make_worker(monkeypatch, tmp_path)
    fake = types.ModuleType("prefilter_aesthetic")
    fake.is_enabled = lambda: True  # type: ignore[attr-defined]
    fake.env_min_score = lambda: 80  # type: ignore[attr-defined]
    fake.assess = lambda *a, **k: {  # type: ignore[attr-defined]
        "passed": False, "reasons": ["too blurry / low sharpness"],
    }
    monkeypatch.setitem(sys.modules, "prefilter_aesthetic", fake)

    reason = worker._prefilter_reject(_jpeg_bytes())
    assert reason is not None
    assert "too blurry" in reason


def test_prefilter_reject_passes_through_when_image_ok(monkeypatch, tmp_path):
    """A passing assess() yields None so the image continues to classification."""
    vwb, worker = _make_worker(monkeypatch, tmp_path)
    fake = types.ModuleType("prefilter_aesthetic")
    fake.is_enabled = lambda: True  # type: ignore[attr-defined]
    fake.env_min_score = lambda: 10  # type: ignore[attr-defined]
    fake.assess = lambda *a, **k: {"passed": True, "reasons": []}  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "prefilter_aesthetic", fake)

    assert worker._prefilter_reject(_jpeg_bytes()) is None


def test_prefilter_reject_fails_open_on_error(monkeypatch, tmp_path):
    """If the prefilter raises, _prefilter_reject returns None (fail-open) so the
    image is NOT wrongly dropped — it falls through to normal classification."""
    vwb, worker = _make_worker(monkeypatch, tmp_path)
    fake = types.ModuleType("prefilter_aesthetic")
    fake.is_enabled = lambda: True  # type: ignore[attr-defined]
    fake.env_min_score = lambda: 50  # type: ignore[attr-defined]

    def _boom(*a, **k):
        raise RuntimeError("prefilter crashed")

    fake.assess = _boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "prefilter_aesthetic", fake)

    assert worker._prefilter_reject(_jpeg_bytes()) is None


def test_process_image_routes_prefilter_failure_to_discard(monkeypatch, tmp_path):
    """End-to-end: with the prefilter enabled and failing, _process_image routes
    the claimed image to a terminal DISCARD WITHOUT calling the VLM."""
    vwb, worker = _make_worker(monkeypatch, tmp_path)

    fake = types.ModuleType("prefilter_aesthetic")
    fake.is_enabled = lambda: True  # type: ignore[attr-defined]
    fake.env_min_score = lambda: 90  # type: ignore[attr-defined]
    fake.assess = lambda *a, **k: {  # type: ignore[attr-defined]
        "passed": False, "reasons": ["near solid-colour / low detail"],
    }
    monkeypatch.setitem(sys.modules, "prefilter_aesthetic", fake)

    # The worker's classify must never be reached on a prefilter reject.
    def _must_not_call(*a, **k):  # pragma: no cover
        raise AssertionError("VLM classify_image_bytes called despite prefilter reject")

    monkeypatch.setattr(worker, "classify_image_bytes", _must_not_call)

    captured_reason: dict[str, str] = {}
    real_finalise_discard = worker._finalise_discard

    def _spy_discard(ctx, reason):
        captured_reason["reason"] = reason
        return real_finalise_discard(ctx, reason)

    monkeypatch.setattr(worker, "_finalise_discard", _spy_discard)

    queue_dir = tmp_path / "queue" / "civitai"
    img = queue_dir / "pic.jpg"
    _write_jpeg(img)
    # Sibling .txt so _gather_context is happy with a prompt file present.
    (queue_dir / "pic.txt").write_text("a prompt", encoding="utf-8")

    outcome = worker._process_image(img, "civitai")
    assert outcome == "DISCARD"
    assert "near solid-colour" in captured_reason.get("reason", "")


# ── 5b. VIDEO FRAME CLASSIFICATION ─────────────────────────────────────────────

def test_extract_keyframes_returns_empty_without_backend(monkeypatch, tmp_path):
    """With neither ffmpeg-python nor scenedetect importable, extract_keyframes
    returns [] and never raises (the caller then skips the clip)."""
    import video_frames
    importlib.reload(video_frames)
    # Force both backends to look absent.
    monkeypatch.setitem(sys.modules, "ffmpeg", None)
    monkeypatch.setitem(sys.modules, "scenedetect", None)

    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"\x00\x00\x00\x18ftypmp42not-a-real-clip")
    assert video_frames.extract_keyframes(clip, max_frames=1) == []


def test_extract_keyframes_missing_file_returns_empty(monkeypatch, tmp_path):
    import video_frames
    importlib.reload(video_frames)
    assert video_frames.extract_keyframes(tmp_path / "nope.mp4") == []


def test_frame_offsets_distribution():
    import video_frames
    importlib.reload(video_frames)
    # One frame from a known duration → the midpoint.
    assert video_frames._frame_offsets(10.0, 1) == [5.0]
    # Unknown duration → safe zero offsets.
    assert video_frames._frame_offsets(0.0, 3) == [0.0, 0.0, 0.0]
    # Several frames spread across the interior, none at 0 or the end.
    offs = video_frames._frame_offsets(12.0, 3)
    assert offs == [3.0, 6.0, 9.0]


def test_acquire_bytes_reads_still_when_video_disabled(monkeypatch, tmp_path):
    """A still image always reads its own bytes regardless of the video flag."""
    vwb, worker = _make_worker(monkeypatch, tmp_path)
    monkeypatch.delenv("VIDEO_CLASSIFY_ENABLED", raising=False)
    img = tmp_path / "queue" / "civitai" / "pic.jpg"
    _write_jpeg(img)
    processing = Path(str(img) + ".processing")
    img.rename(processing)
    ctx = _make_ctx(vwb, img, processing)

    data = worker._acquire_classify_bytes(ctx, processing)
    assert data == processing.read_bytes()


def test_acquire_bytes_extracts_frame_for_video_when_enabled(monkeypatch, tmp_path):
    """When the video lane is on and a clip is popped, _acquire_classify_bytes
    returns the extracted FRAME's bytes (not the clip bytes)."""
    vwb, worker = _make_worker(monkeypatch, tmp_path)
    monkeypatch.setenv("VIDEO_CLASSIFY_ENABLED", "true")

    frame_bytes = _jpeg_bytes(color=(200, 50, 50))
    fake_vf = types.ModuleType("video_frames")
    fake_vf.VIDEO_EXT = (".mp4", ".mov", ".webm", ".mkv", ".avi")  # type: ignore[attr-defined]
    fake_vf.is_video = lambda p: Path(p).suffix.lower() in fake_vf.VIDEO_EXT  # type: ignore[attr-defined]
    fake_vf.is_enabled = lambda: True  # type: ignore[attr-defined]
    cleaned: list[object] = []

    def _extract(path, max_frames=1):
        frame = tmp_path / "cull_frames_x" / "frame_000.jpg"
        frame.parent.mkdir(parents=True, exist_ok=True)
        frame.write_bytes(frame_bytes)
        return [frame]

    fake_vf.extract_keyframes = _extract  # type: ignore[attr-defined]
    fake_vf.cleanup_frames = lambda frames: cleaned.append(frames)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "video_frames", fake_vf)

    clip = tmp_path / "queue" / "civitai" / "clip.mp4"
    clip.parent.mkdir(parents=True, exist_ok=True)
    clip.write_bytes(b"\x00\x00\x00\x18ftypmp42clip")
    processing = Path(str(clip) + ".processing")
    clip.rename(processing)
    ctx = _make_ctx(vwb, clip, processing)

    data = worker._acquire_classify_bytes(ctx, processing)
    assert data == frame_bytes              # classified the FRAME, not the clip
    assert cleaned                           # extracted frames were cleaned up


def test_acquire_bytes_skips_clip_when_no_frame(monkeypatch, tmp_path):
    """A clip that yields no frame returns None → caller skips gracefully."""
    vwb, worker = _make_worker(monkeypatch, tmp_path)
    monkeypatch.setenv("VIDEO_CLASSIFY_ENABLED", "true")

    fake_vf = types.ModuleType("video_frames")
    fake_vf.VIDEO_EXT = (".mp4",)  # type: ignore[attr-defined]
    fake_vf.is_video = lambda p: Path(p).suffix.lower() in fake_vf.VIDEO_EXT  # type: ignore[attr-defined]
    fake_vf.is_enabled = lambda: True  # type: ignore[attr-defined]
    fake_vf.extract_keyframes = lambda path, max_frames=1: []  # type: ignore[attr-defined]
    fake_vf.cleanup_frames = lambda frames: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "video_frames", fake_vf)

    clip = tmp_path / "queue" / "civitai" / "clip.mp4"
    clip.parent.mkdir(parents=True, exist_ok=True)
    clip.write_bytes(b"clip")
    processing = Path(str(clip) + ".processing")
    clip.rename(processing)
    ctx = _make_ctx(vwb, clip, processing)

    assert worker._acquire_classify_bytes(ctx, processing) is None


def test_process_image_sorts_clip_via_frame_classification(monkeypatch, tmp_path):
    """End-to-end: a popped video clip is classified via its frame and the CLIP
    itself (the .mp4) is what lands in sorted/<category>/<source>/."""
    vwb, worker = _make_worker(monkeypatch, tmp_path)
    monkeypatch.setenv("VIDEO_CLASSIFY_ENABLED", "true")

    # Avoid real phash/index work at finalize.
    fake_phash = types.ModuleType("phash_dedup")
    fake_phash.compute_phash = lambda p: None  # type: ignore[attr-defined]
    fake_index = types.ModuleType("index_store")
    fake_index.set_phash = lambda k, v: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "phash_dedup", fake_phash)
    monkeypatch.setitem(sys.modules, "index_store", fake_index)

    frame_bytes = _jpeg_bytes(color=(30, 120, 30))
    fake_vf = types.ModuleType("video_frames")
    fake_vf.VIDEO_EXT = (".mp4",)  # type: ignore[attr-defined]
    fake_vf.is_video = lambda p: Path(p).suffix.lower() in fake_vf.VIDEO_EXT  # type: ignore[attr-defined]
    fake_vf.is_enabled = lambda: True  # type: ignore[attr-defined]

    def _extract(path, max_frames=1):
        frame = tmp_path / "cull_frames_y" / "f.jpg"
        frame.parent.mkdir(parents=True, exist_ok=True)
        frame.write_bytes(frame_bytes)
        return [frame]

    fake_vf.extract_keyframes = _extract  # type: ignore[attr-defined]
    fake_vf.cleanup_frames = lambda frames: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "video_frames", fake_vf)

    # The model classifies the frame and returns a kept verdict.
    monkeypatch.setattr(
        worker, "classify_image_bytes",
        lambda b64, instr: {"category": "DISCARD", "caption": ""},
    )

    clip = tmp_path / "queue" / "civitai" / "clip.mp4"
    clip.parent.mkdir(parents=True, exist_ok=True)
    clip.write_bytes(b"\x00\x00\x00\x18ftypmp42the-real-clip-bytes")
    (clip.parent / "clip.txt").write_text("a clip prompt", encoding="utf-8")

    outcome = worker._process_image(clip, "civitai")
    assert outcome == "DISCARD"

    # The CLIP (.mp4) must be what landed in sorted/, not a frame.
    sorted_clips = list((tmp_path / "sorted").glob("**/*.mp4"))
    assert len(sorted_clips) == 1
    assert sorted_clips[0].read_bytes() == b"\x00\x00\x00\x18ftypmp42the-real-clip-bytes"
    # No .jpg frame leaked into the sorted tree.
    assert not list((tmp_path / "sorted").glob("**/*.jpg"))


def test_process_image_routes_undecodable_clip_to_discard(monkeypatch, tmp_path):
    """End-to-end: with the video lane on but extraction yielding NO frame, the
    clip must land in terminal DISCARD — never orphaned as a stray .processing
    file and never re-queued (re-queuing would hot-loop on an undecodable clip).
    The VLM must not be consulted (there's no frame to classify)."""
    vwb, worker = _make_worker(monkeypatch, tmp_path)
    monkeypatch.setenv("VIDEO_CLASSIFY_ENABLED", "true")

    # Avoid real phash/index work at finalize.
    fake_phash = types.ModuleType("phash_dedup")
    fake_phash.compute_phash = lambda p: None  # type: ignore[attr-defined]
    fake_index = types.ModuleType("index_store")
    fake_index.set_phash = lambda k, v: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "phash_dedup", fake_phash)
    monkeypatch.setitem(sys.modules, "index_store", fake_index)

    # Extraction backend present but yields no frame (e.g. undecodable clip).
    fake_vf = types.ModuleType("video_frames")
    fake_vf.VIDEO_EXT = (".mp4",)  # type: ignore[attr-defined]
    fake_vf.is_video = lambda p: Path(p).suffix.lower() in fake_vf.VIDEO_EXT  # type: ignore[attr-defined]
    fake_vf.is_enabled = lambda: True  # type: ignore[attr-defined]
    fake_vf.extract_keyframes = lambda path, max_frames=1: []  # type: ignore[attr-defined]
    fake_vf.cleanup_frames = lambda frames: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "video_frames", fake_vf)

    # The VLM must never be reached — there is no frame to classify.
    def _must_not_call(*a, **k):  # pragma: no cover
        raise AssertionError("classify_image_bytes called despite no extracted frame")

    monkeypatch.setattr(worker, "classify_image_bytes", _must_not_call)

    clip = tmp_path / "queue" / "civitai" / "clip.mp4"
    clip.parent.mkdir(parents=True, exist_ok=True)
    clip.write_bytes(b"\x00\x00\x00\x18ftypmp42undecodable")
    (clip.parent / "clip.txt").write_text("a clip prompt", encoding="utf-8")

    outcome = worker._process_image(clip, "civitai")
    assert outcome == "DISCARD"

    # The CLIP itself lands in sorted/DISCARD/<source>/ — not orphaned, not requeued.
    sorted_clips = list((tmp_path / "sorted").glob("**/*.mp4"))
    assert len(sorted_clips) == 1
    assert "DISCARD" in str(sorted_clips[0])
    assert sorted_clips[0].read_bytes() == b"\x00\x00\x00\x18ftypmp42undecodable"
    # No stray .processing left behind, and the clip didn't bounce back to the queue.
    assert not list((tmp_path / "queue").glob("**/*.processing"))
    assert not (tmp_path / "queue" / "civitai" / "clip.mp4").exists()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
