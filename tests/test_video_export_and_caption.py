"""Tests for video-aware dataset export + motion-aware captioning.

Two Wave-2 roadmap slices live here:

  * §2c video export — ``export_profiles`` must treat video files
    (``.mp4`` / ``.mov`` / ``.webm`` / ``.mkv`` / ``.avi``) as first-class
    samples, bucket them by duration / fps / resolution via a *gated* prober
    (ffmpeg-python / av if importable, else a graceful ``"unknown"`` bucket),
    and offer a clip+caption pairing profile (``clip_caption``) for video
    trainers (LTX-Video, Wan, Hunyuan-Video, Mochi, CogVideoX).

  * §3d motion captioning — ``vision_prompt`` must expose a ``"motion"``
    caption style whose instruction text steers the model toward camera
    movement, subject action, shot type, and temporal description (the format
    video-model trainers consume), while leaving the existing image styles
    (sd_prompt / booru_tags / natural_language) untouched.

The export tests build a hermetic fake sorted tree under a temp
``PIPELINE_BASE_DIR`` and *mock the video prober* so they never shell out to
ffmpeg — the prober contract is exercised both present (patched) and absent
(forced ``None``) so the fallback bucket is proven.

Runnable:
    pytest tests/test_video_export_and_caption.py -q
"""
from __future__ import annotations

import json
import sys
import tarfile
from pathlib import Path
from typing import Any

import pytest

PIPELINE_CODE = Path(__file__).resolve().parent.parent / "pipeline_code"
if str(PIPELINE_CODE) not in sys.path:
    sys.path.insert(0, str(PIPELINE_CODE))


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop a developer's real ``.env`` leaking into these hermetic tests."""
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)


def _write_video(path: Path, payload: bytes = b"\x00FAKE-VIDEO\x00") -> None:
    """Write a placeholder video file. Bytes are never decoded — the prober is
    mocked, so only the extension and the verbatim copy matter."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _vision_record(category: str, caption: str = "") -> dict[str, Any]:
    return {
        "description": "a deterministic test clip",
        "primary_subject": "a moving subject",
        "category": category,
        "OVR_Quality_Score": 70,
        "REL_Quality_Score": 60,
        "quality_score": 7,
        "caption": caption,
    }


def _make_video_sample(
    sorted_root: Path,
    category: str,
    source: str,
    stem: str,
    *,
    caption: str,
    ext: str = ".mp4",
    with_vision: bool = True,
) -> Path:
    """Create one (video + .txt [+ .vision.json]) sample. Returns the clip path."""
    folder = sorted_root / category / source
    clip = folder / f"{stem}{ext}"
    _write_video(clip)
    (folder / f"{stem}.txt").write_text(caption, encoding="utf-8")
    if with_vision:
        (folder / f"{stem}.vision.json").write_text(
            json.dumps(_vision_record(category, caption), ensure_ascii=False),
            encoding="utf-8",
        )
    return clip


@pytest.fixture()
def fake_video_sorted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Build a fake sorted tree of *video* clips under a temp PIPELINE_BASE_DIR.

    Two KEPT categories (Keep, Borderline) holding clips of differing
    extensions, plus a terminal DISCARD bucket that must never be exported.
    """
    base = tmp_path / "data"
    monkeypatch.setenv("PIPELINE_BASE_DIR", str(base))
    monkeypatch.setenv("PIPELINE_SLUG", "clips")

    slug = "clips"
    sorted_root = base / "sorted" / slug

    _make_video_sample(sorted_root, "Keep", "civitai", "clip_a_1700000000_001",
                        caption="a slow pan over a city", ext=".mp4")
    _make_video_sample(sorted_root, "Keep", "x", "clip_b_1700000001_002",
                        caption="a tracking shot of a runner", ext=".webm")
    _make_video_sample(sorted_root, "Borderline", "reddit", "clip_c_1700000002_003",
                        caption="a static interior", ext=".mov", with_vision=False)
    # TERMINAL: must never be exported.
    _make_video_sample(sorted_root, "DISCARD", "civitai", "clip_d_1700000003_004",
                        caption="garbage clip", ext=".mp4")

    return {
        "slug": slug,
        "base": base,
        "sorted_root": sorted_root,
        "out_dir": tmp_path / "export",
    }


def _mock_probe(
    monkeypatch: pytest.MonkeyPatch,
    table: dict[str, Any] | None = None,
    *,
    default: Any = None,
) -> None:
    """Patch ``export_profiles.probe_video`` with a deterministic stub.

    ``table`` maps a clip's *stem* to a ``VideoMeta`` (or dict); anything not in
    the table returns ``default`` (``None`` => prober "absent / failed", which
    must drive the graceful ``"unknown"`` bucket).
    """
    import export_profiles
    table = table or {}

    def _fake(path: Path) -> Any:
        return table.get(Path(path).stem, default)

    monkeypatch.setattr(export_profiles, "probe_video", _fake)


def _meta(export_profiles, **kw: Any):
    """Build a VideoMeta if the module exposes one, else fall back to a dict."""
    factory = getattr(export_profiles, "VideoMeta", None)
    if factory is not None:
        return factory(**kw)
    return dict(**kw)


# ── Import smoke ─────────────────────────────────────────────────────────────

def test_module_exposes_video_surface() -> None:
    import export_profiles

    assert hasattr(export_profiles, "VIDEO_EXT")
    assert hasattr(export_profiles, "probe_video")
    # The clip+caption pairing profile is registered.
    assert "clip_caption" in export_profiles.PROFILES


def test_video_extensions_cover_common_containers() -> None:
    import export_profiles

    exts = {e.lower() for e in export_profiles.VIDEO_EXT}
    assert {".mp4", ".mov", ".webm", ".mkv", ".avi"} <= exts


# ── Sample iteration includes video ──────────────────────────────────────────

def test_iter_samples_includes_video_clips(fake_video_sorted: dict[str, Any]) -> None:
    import export_profiles

    samples = list(export_profiles.iter_samples(fake_video_sorted["slug"]))
    # 3 KEPT clips (2 Keep + 1 Borderline); the DISCARD clip is excluded.
    assert len(samples) == 3
    suffixes = sorted(s.image_path.suffix.lower() for s in samples)
    assert suffixes == [".mov", ".mp4", ".webm"]
    cats = {s.category for s in samples}
    assert cats == {"Keep", "Borderline"}


def test_iter_samples_still_excludes_terminal_for_video(
    fake_video_sorted: dict[str, Any]
) -> None:
    import export_profiles

    samples = list(
        export_profiles.iter_samples(
            fake_video_sorted["slug"], categories=["Keep", "DISCARD"]
        )
    )
    assert {s.category for s in samples} == {"Keep"}
    assert all("garbage" not in s.caption for s in samples)


# ── Video bucketing: duration / fps / resolution (prober mocked) ─────────────

def test_video_buckets_by_duration_with_prober(
    fake_video_sorted: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    import export_profiles

    table = {
        "clip_a_1700000000_001": _meta(export_profiles, width=1920, height=1080,
                                       fps=24.0, duration=2.0),
        "clip_b_1700000001_002": _meta(export_profiles, width=1920, height=1080,
                                       fps=24.0, duration=9.0),
        "clip_c_1700000002_003": _meta(export_profiles, width=1280, height=720,
                                       fps=30.0, duration=2.0),
    }
    _mock_probe(monkeypatch, table)

    out = fake_video_sorted["out_dir"]
    summary = export_profiles.export_dataset(
        fake_video_sorted["slug"], "clip_caption", out, video_bucketing="duration",
    )
    assert summary["sample_count"] == 3
    assert "buckets" in summary
    # Two short clips (2.0s) share a bucket; the 9.0s clip lands in another.
    bucket_names = set(summary["buckets"])
    assert len(bucket_names) == 2
    total = sum(summary["buckets"].values())
    assert total == 3
    # The bucket holding two clips must reflect the two short clips.
    assert max(summary["buckets"].values()) == 2
    # Files actually landed inside per-bucket subdirs.
    clips = sorted(p for p in out.rglob("*") if p.suffix.lower() in {".mp4", ".webm", ".mov"})
    assert len(clips) == 3
    for clip in clips:
        assert (clip.parent / f"{clip.stem}.txt").exists()


def test_video_buckets_by_fps_with_prober(
    fake_video_sorted: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    import export_profiles

    table = {
        "clip_a_1700000000_001": _meta(export_profiles, width=1920, height=1080,
                                       fps=24.0, duration=4.0),
        "clip_b_1700000001_002": _meta(export_profiles, width=1920, height=1080,
                                       fps=24.0, duration=4.0),
        "clip_c_1700000002_003": _meta(export_profiles, width=1280, height=720,
                                       fps=30.0, duration=4.0),
    }
    _mock_probe(monkeypatch, table)

    out = fake_video_sorted["out_dir"]
    summary = export_profiles.export_dataset(
        fake_video_sorted["slug"], "clip_caption", out, video_bucketing="fps",
    )
    # 24fps x2, 30fps x1 -> two buckets.
    assert len(summary["buckets"]) == 2
    assert max(summary["buckets"].values()) == 2
    # The fps value should be reflected in at least one bucket label.
    assert any("24" in str(k) for k in summary["buckets"])


def test_video_buckets_by_resolution_with_prober(
    fake_video_sorted: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    import export_profiles

    table = {
        "clip_a_1700000000_001": _meta(export_profiles, width=1920, height=1080,
                                       fps=24.0, duration=4.0),
        "clip_b_1700000001_002": _meta(export_profiles, width=1920, height=1080,
                                       fps=24.0, duration=4.0),
        "clip_c_1700000002_003": _meta(export_profiles, width=1280, height=720,
                                       fps=30.0, duration=4.0),
    }
    _mock_probe(monkeypatch, table)

    out = fake_video_sorted["out_dir"]
    summary = export_profiles.export_dataset(
        fake_video_sorted["slug"], "clip_caption", out, video_bucketing="resolution",
    )
    bucket_names = set(summary["buckets"])
    assert any("1920x1080" in str(k) for k in bucket_names)
    assert any("1280x720" in str(k) for k in bucket_names)


def test_video_bucketing_falls_back_to_unknown_when_prober_absent(
    fake_video_sorted: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the prober returns None for everything (ffmpeg/av not installed, or a
    probe failure), bucketing must NOT crash — every clip lands in 'unknown'."""
    import export_profiles

    _mock_probe(monkeypatch, {}, default=None)  # prober "absent": always None

    out = fake_video_sorted["out_dir"]
    summary = export_profiles.export_dataset(
        fake_video_sorted["slug"], "clip_caption", out, video_bucketing="duration",
    )
    assert summary["sample_count"] == 3
    assert summary["buckets"] == {"unknown": 3}
    # All three clips physically live under an 'unknown' subdir.
    assert (out / "unknown").is_dir()
    unknown_clips = [p for p in (out / "unknown").rglob("*")
                     if p.suffix.lower() in {".mp4", ".webm", ".mov"}]
    assert len(unknown_clips) == 3


def test_video_bucketing_never_raises_on_probe_exception(
    fake_video_sorted: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A prober that *raises* (corrupt clip, ffmpeg crash) must be swallowed and
    treated as 'unknown' — export must complete."""
    import export_profiles

    def _boom(path: Path) -> Any:
        raise RuntimeError("ffmpeg exploded")

    monkeypatch.setattr(export_profiles, "probe_video", _boom)

    out = fake_video_sorted["out_dir"]
    summary = export_profiles.export_dataset(
        fake_video_sorted["slug"], "clip_caption", out, video_bucketing="duration",
    )
    assert summary["sample_count"] == 3
    assert summary["buckets"] == {"unknown": 3}


# ── clip_caption profile pairing ─────────────────────────────────────────────

def test_clip_caption_pairs_each_clip_with_caption(
    fake_video_sorted: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    import export_profiles

    _mock_probe(monkeypatch, {}, default=None)
    out = fake_video_sorted["out_dir"]
    summary = export_profiles.export_dataset(
        fake_video_sorted["slug"], "clip_caption", out,
    )
    assert summary["profile"] == "clip_caption"
    assert summary["sample_count"] == 3

    clips = sorted(p for p in out.rglob("*")
                   if p.suffix.lower() in {".mp4", ".webm", ".mov"})
    txts = sorted(out.rglob("*.txt"))
    assert len(clips) == 3
    assert len(txts) == 3
    for clip in clips:
        sidecar = clip.parent / f"{clip.stem}.txt"
        assert sidecar.exists()
        assert sidecar.read_text(encoding="utf-8").strip()  # caption non-empty


def test_clip_caption_trigger_word_injected(
    fake_video_sorted: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    import export_profiles

    _mock_probe(monkeypatch, {}, default=None)
    out = fake_video_sorted["out_dir"]
    export_profiles.export_dataset(
        fake_video_sorted["slug"], "clip_caption", out, trigger_word="vtok",
    )
    txts = sorted(out.rglob("*.txt"))
    assert txts
    for txt in txts:
        body = txt.read_text(encoding="utf-8")
        assert body.startswith("vtok")


def test_clip_caption_copies_clip_bytes_verbatim(
    fake_video_sorted: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    import export_profiles

    _mock_probe(monkeypatch, {}, default=None)
    out = fake_video_sorted["out_dir"]
    export_profiles.export_dataset(fake_video_sorted["slug"], "clip_caption", out)
    # Source bytes are the placeholder payload; copies must match.
    exported = [p for p in out.rglob("*") if p.suffix.lower() in {".mp4", ".webm", ".mov"}]
    assert exported
    for clip in exported:
        assert clip.read_bytes() == b"\x00FAKE-VIDEO\x00"


def test_clip_caption_excludes_terminal(
    fake_video_sorted: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    import export_profiles

    _mock_probe(monkeypatch, {}, default=None)
    out = fake_video_sorted["out_dir"]
    export_profiles.export_dataset(fake_video_sorted["slug"], "clip_caption", out)
    for txt in out.rglob("*.txt"):
        assert "garbage" not in txt.read_text(encoding="utf-8")
    assert not any("discard" in str(p).lower() for p in out.rglob("*"))


# ── Existing image profiles still accept video samples ───────────────────────

def test_kohya_profile_handles_video_samples(
    fake_video_sorted: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The existing kohya profile must pack video clips too (image+txt logic is
    extension-agnostic for the copy step)."""
    import export_profiles

    _mock_probe(monkeypatch, {}, default=None)
    out = fake_video_sorted["out_dir"]
    summary = export_profiles.export_dataset(
        fake_video_sorted["slug"], "kohya", out, concept="clip",
    )
    assert summary["sample_count"] == 3
    clips = [p for p in out.rglob("*") if p.suffix.lower() in {".mp4", ".webm", ".mov"}]
    assert len(clips) == 3
    for clip in clips:
        assert (clip.parent / f"{clip.stem}.txt").exists()


def test_webdataset_profile_handles_video_samples(
    fake_video_sorted: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    import export_profiles

    _mock_probe(monkeypatch, {}, default=None)
    out = fake_video_sorted["out_dir"]
    export_profiles.export_dataset(fake_video_sorted["slug"], "webdataset", out)
    members: list[str] = []
    for tar_path in out.rglob("*.tar"):
        with tarfile.open(tar_path, mode="r") as tf:
            members.extend(tf.getnames())
    vids = [m for m in members if m.rsplit(".", 1)[-1].lower() in {"mp4", "webm", "mov"}]
    assert len(vids) == 3


# ── Source immutability with video clips ─────────────────────────────────────

def _snapshot_tree(root: Path) -> dict[str, bytes]:
    snap: dict[str, bytes] = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            snap[str(p.relative_to(root))] = p.read_bytes()
    return snap


def test_video_source_files_untouched_after_export(
    fake_video_sorted: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    import export_profiles

    _mock_probe(monkeypatch, {}, default=None)
    sorted_root = fake_video_sorted["sorted_root"]
    before = _snapshot_tree(sorted_root)
    for profile in ("clip_caption", "kohya", "webdataset"):
        export_profiles.export_dataset(
            fake_video_sorted["slug"], profile, fake_video_sorted["out_dir"] / profile,
            trigger_word="t", video_bucketing="duration",
        )
    after = _snapshot_tree(sorted_root)
    assert before == after, "export must not move, delete, or modify source clips"


# ══ §3d motion-aware captioning (vision_prompt) ══════════════════════════════

def test_motion_caption_style_registered() -> None:
    import vision_prompt

    assert "motion" in vision_prompt.CAPTION_STYLES
    # Existing styles remain available.
    for style in ("sd_prompt", "booru_tags", "natural_language"):
        assert style in vision_prompt.CAPTION_STYLES


def test_motion_caption_instruction_is_motion_oriented() -> None:
    import vision_prompt

    text = vision_prompt.caption_style_instruction("motion").lower()
    assert text.strip()
    # Mentions camera movement, subject action, shot type, temporal cues.
    assert "camera" in text
    assert any(w in text for w in ("motion", "movement", "moving"))
    assert any(w in text for w in ("action", "subject"))
    assert any(w in text for w in ("shot", "angle", "framing"))
    assert any(w in text for w in ("temporal", "over time", "sequence",
                                   "throughout", "duration", "as the"))


def test_motion_style_distinct_from_image_styles() -> None:
    import vision_prompt

    motion = vision_prompt.caption_style_instruction("motion")
    sd = vision_prompt.caption_style_instruction("sd_prompt")
    booru = vision_prompt.caption_style_instruction("booru_tags")
    nat = vision_prompt.caption_style_instruction("natural_language")
    assert motion not in (sd, booru, nat)


def test_existing_caption_instructions_unchanged() -> None:
    """Adding 'motion' must not perturb the existing style instructions."""
    import vision_prompt

    sd = vision_prompt.caption_style_instruction("sd_prompt")
    booru = vision_prompt.caption_style_instruction("booru_tags")
    nat = vision_prompt.caption_style_instruction("natural_language")
    assert "Stable Diffusion" in sd or "stable diffusion" in sd.lower()
    assert "booru" in booru.lower()
    assert "natural-language" in nat.lower() or "natural language" in nat.lower()


def test_unknown_caption_style_still_falls_back_to_sd_prompt() -> None:
    import vision_prompt

    assert (vision_prompt.caption_style_instruction("does_not_exist")
            == vision_prompt.caption_style_instruction("sd_prompt"))


def test_build_prompt_with_motion_style_includes_motion_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vision_prompt

    cfg = vision_prompt.ScoreConfig(topic="cinematic city b-roll",
                                    ovr_min=0, rel_min=0, notes="")
    caption_cfg = vision_prompt.CaptionConfig(enabled=True, style="motion",
                                              overwrite=False)
    prompt = vision_prompt.build_classification_prompt(cfg, caption_cfg)
    assert "motion" in prompt.lower()
    assert "camera" in prompt.lower()
    # The AUTO-CAPTION block names the active style.
    assert "Style: motion" in prompt


def test_build_prompt_motion_style_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    import vision_prompt

    monkeypatch.setenv("AUTO_CAPTION_ENABLED", "true")
    monkeypatch.setenv("AUTO_CAPTION_STYLE", "motion")
    caption_cfg = vision_prompt.CaptionConfig.from_env()
    assert caption_cfg.enabled is True
    assert caption_cfg.style == "motion"


def test_build_prompt_image_styles_unaffected_by_motion_addition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-motion build must not accidentally inject motion instructions."""
    import vision_prompt

    cfg = vision_prompt.ScoreConfig(topic="x", ovr_min=0, rel_min=0, notes="")
    caption_cfg = vision_prompt.CaptionConfig(enabled=True, style="sd_prompt",
                                              overwrite=False)
    prompt = vision_prompt.build_classification_prompt(cfg, caption_cfg)
    assert "Style: sd_prompt" in prompt
    # The motion-specific phrase must NOT bleed into the sd_prompt build.
    assert "camera movement" not in prompt.lower()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
