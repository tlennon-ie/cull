"""Tests for the shipped VIDEO starter presets (builtin_presets.py).

cull's image presets (default / aerial / underwater / wildlife / product /
anime / portrait / quality) seed image-dataset curation. This module adds a
parallel set of *video* presets aimed at video-model dataset trainers
(LTX-Video, Wan, Hunyuan-Video, Mochi, CogVideoX). The video presets MUST mirror
the image presets' dict shape EXACTLY so the same `job_config` projection +
content-signature baseline reconciliation keeps working unchanged.

These tests pin the load-bearing contract:
  * each `video_*` preset exists in the builtin library,
  * it carries the full inheritable key set (same keys as an image preset),
  * banned_keywords carries the shared spam baseline forward (`onlyfans`),
  * generation_hints is non-empty AND contains a motion descriptor,
  * categories reuse the SAME vocabulary and keep the terminal DISCARD/CORRUPT
    reachable (DISCARD/CORRUPT are never invented as keep buckets).

Runnable:
    pytest tests/test_video_presets.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PIPELINE_CODE = Path(__file__).resolve().parent.parent / "pipeline_code"
if str(PIPELINE_CODE) not in sys.path:
    sys.path.insert(0, str(PIPELINE_CODE))

import builtin_presets as bp  # noqa: E402

# The video presets shipped alongside the image set.
EXPECTED_VIDEO_PRESETS = {
    "video_default",
    "video_cinematic",
    "video_anime",
    "video_product",
    "video_nature",
}

# Keys every preset (image or video) must carry — the inheritable taste blocks.
REQUIRED_PRESET_KEYS = {
    "topic_filters", "scrapers", "categories", "category_rules",
    "scoring", "captioning",
}

# Motion/temporal descriptors — at least one must appear in generation_hints so
# the require-a-generation-marker gate selects for video-shaped prompts.
_MOTION_MARKERS = (
    "camera", "pan", "tracking", "dolly", "tilt", "zoom", "motion",
    "slow motion", "b-roll", "establishing", "handheld", "cinematic",
    "footage", "clip", "moving", "orbit", "timelapse", "drone",
)


def _video_presets() -> dict[str, dict]:
    return bp.builtin_library()["presets"]


def _has_motion_marker(hints: list[str]) -> bool:
    blob = " ".join(hints).lower()
    return any(marker in blob for marker in _MOTION_MARKERS)


# ── presence + shape ─────────────────────────────────────────────────────────

def test_all_video_presets_exist():
    names = set(_video_presets())
    missing = EXPECTED_VIDEO_PRESETS - names
    assert not missing, f"missing video presets: {sorted(missing)}"


def test_video_default_pointer_is_unchanged():
    # Adding video presets must NOT hijack the shipped default pointer.
    assert bp.builtin_library()["default"] == "default"
    assert bp.DEFAULT_PRESET == "default"


def test_video_presets_in_preset_names_tuple():
    for name in EXPECTED_VIDEO_PRESETS:
        assert name in bp.PRESET_NAMES, name


def test_video_presets_have_full_key_set():
    presets = _video_presets()
    # Use an image preset's keys as the canonical reference set.
    image_keys = set(presets["default"])
    for name in EXPECTED_VIDEO_PRESETS:
        cfg = presets[name]
        assert REQUIRED_PRESET_KEYS <= set(cfg), name
        assert set(cfg) == image_keys, f"{name} key set diverges from image presets"


def test_video_presets_inheritable_blocks_well_formed():
    presets = _video_presets()
    for name in EXPECTED_VIDEO_PRESETS:
        cfg = presets[name]
        assert isinstance(cfg["categories"], list) and cfg["categories"], name
        for cat in cfg["categories"]:
            assert set(cat) >= {"name", "hint"}, name
            assert cat["name"] and isinstance(cat["name"], str), name
        assert isinstance(cfg["category_rules"], str) and cfg["category_rules"].strip(), name
        sc = cfg["scoring"]
        assert {"ovr_min", "rel_min", "notes"} <= set(sc), name
        cap = cfg["captioning"]
        assert {"enabled", "style", "overwrite"} <= set(cap), name
        assert cap["enabled"] is False, name          # never auto-spend caption tokens
        assert cap["style"] in {"sd_prompt", "booru_tags", "natural_language"}, name


# ── topic-filter contract: spam baseline + motion hints ──────────────────────

def test_video_presets_carry_spam_baseline():
    for name, cfg in _video_presets().items():
        if name not in EXPECTED_VIDEO_PRESETS:
            continue
        banned = cfg["topic_filters"]["banned_keywords"]
        assert banned, name
        # the shared _SPAM_BANNED markers are carried forward, not redefined
        assert "onlyfans" in banned, name
        assert "link in bio" in banned, name


def test_video_presets_reuse_shared_spam_tuple():
    # Every _SPAM_BANNED entry must be present (carried forward), proving the
    # video presets reuse the shared tuple instead of inventing their own.
    for name, cfg in _video_presets().items():
        if name not in EXPECTED_VIDEO_PRESETS:
            continue
        banned = set(cfg["topic_filters"]["banned_keywords"])
        assert set(bp._SPAM_BANNED) <= banned, name


def test_video_presets_have_motion_generation_hints():
    for name, cfg in _video_presets().items():
        if name not in EXPECTED_VIDEO_PRESETS:
            continue
        hints = cfg["topic_filters"]["generation_hints"]
        assert hints, f"{name} has no generation_hints"
        assert _has_motion_marker(hints), f"{name} hints lack a motion descriptor: {hints}"


def test_video_presets_have_motion_keywords_extra():
    for name, cfg in _video_presets().items():
        if name not in EXPECTED_VIDEO_PRESETS:
            continue
        kw = cfg["topic_filters"]["keywords_extra"]
        assert kw, f"{name} has no keywords_extra"
        assert _has_motion_marker(kw), f"{name} keywords lack a motion descriptor: {kw}"


def test_video_presets_have_nonzero_ovr_floor():
    for name, cfg in _video_presets().items():
        if name not in EXPECTED_VIDEO_PRESETS:
            continue
        assert cfg["scoring"]["ovr_min"] > 0, name


def test_video_presets_have_min_prompt_length_set():
    # video prompts describe motion; a sensible floor weeds out one-word tags.
    for name, cfg in _video_presets().items():
        if name not in EXPECTED_VIDEO_PRESETS:
            continue
        assert cfg["topic_filters"]["min_prompt_length"] > 0, name


# ── categories: reuse vocabulary, keep terminal reachable ────────────────────

def test_video_presets_keep_terminal_buckets_reachable():
    # DISCARD/CORRUPT are system terminals — never invented as keep buckets, and
    # the rules must drive composites/corruption toward DISCARD.
    for name, cfg in _video_presets().items():
        if name not in EXPECTED_VIDEO_PRESETS:
            continue
        names = {c["name"] for c in cfg["categories"]}
        assert "CORRUPT" not in names, f"{name} must not list CORRUPT as a keep bucket"
        assert "DISCARD" not in names, f"{name} must not list DISCARD as a keep bucket"
        assert "DISCARD" in cfg["category_rules"], f"{name} rules must route to DISCARD"


def test_video_presets_do_not_invent_woman_gate():
    # Mirror the image-preset policy: no photoreal woman gate in video presets.
    for name, cfg in _video_presets().items():
        if name not in EXPECTED_VIDEO_PRESETS:
            continue
        assert "woman_present" not in cfg["category_rules"], name
        assert "InstagramInfluencer" not in json.dumps(cfg), name


def test_video_presets_have_a_keep_and_offtopic_bucket():
    # Same triage spine as the image presets: a strong-keep bucket + an off-topic.
    for name, cfg in _video_presets().items():
        if name not in EXPECTED_VIDEO_PRESETS:
            continue
        names = {c["name"] for c in cfg["categories"]}
        assert "Keep" in names, name
        assert "OffTopic" in names, name


# ── civitai domains mirror the image presets (both hosts) ────────────────────

def test_video_presets_seed_both_civitai_domains():
    for name, cfg in _video_presets().items():
        if name not in EXPECTED_VIDEO_PRESETS:
            continue
        assert cfg["scrapers"]["civitai_domains"] == ["civitai.com", "civitai.red"], name


# ── library integrity: fresh copy + stable serialisation ─────────────────────

def test_builtin_library_returns_fresh_copy_with_video_presets():
    a = bp.builtin_library()
    a["presets"]["video_default"]["scoring"]["ovr_min"] = 999
    b = bp.builtin_library()
    assert b["presets"]["video_default"]["scoring"]["ovr_min"] != 999


def test_video_presets_serialise_cleanly_for_signatures():
    # The baseline reconcile hashes json.dumps(...); a video preset that can't
    # round-trip through JSON would break _merge_builtin_presets.
    for name, cfg in _video_presets().items():
        if name not in EXPECTED_VIDEO_PRESETS:
            continue
        blob = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
        assert json.loads(blob) == cfg, name


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
