"""Tests for the last-mile wiring of wave 3:

  1. yt-dlp is a FIRST-CLASS per-job scraper block (mirrors gallery_dl):
     * the default preset carries a ``scrapers.yt_dlp`` block;
     * ``is_overridden("scrapers.yt_dlp.enabled")`` works;
     * config_io export→import of a job carrying ``yt_dlp`` round-trips;
     * ``resolve_env`` projects ``YT_DLP_*`` from the (now-defaulted) block.

  2. Active-learning exemplars are projected into the classification prompt:
     * ``resolve_env`` emits ``ACTIVE_LEARNING_EXEMPLARS_JSON`` (the active job's
       keep + terminal exemplars) — populated when corrections exist, ``"{}"``
       otherwise — an internal projection, never user-set;
     * ``vision_prompt.build_classification_prompt`` injects a few-shot block
       when that env var is non-empty and is byte-identical to today when absent.

Runnable:
    pytest tests/test_yt_dlp_firstclass.py
    .venv/Scripts/python.exe -m pytest tests/test_yt_dlp_firstclass.py -q
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

PIPELINE_CODE = Path(__file__).resolve().parent.parent / "pipeline_code"
if str(PIPELINE_CODE) not in sys.path:
    sys.path.insert(0, str(PIPELINE_CODE))


@pytest.fixture()
def isolated(tmp_path, monkeypatch):
    """Reload job_config + config_io against an isolated data dir (mirrors the
    job_config/config_io fixtures so preset/job CRUD writes under tmp_path)."""
    monkeypatch.setenv("PIPELINE_BASE_DIR", str(tmp_path))
    monkeypatch.delenv("ACTIVE_LEARNING_EXEMPLARS_JSON", raising=False)
    import categories
    monkeypatch.setattr(categories, "ACTIVE_PATH", tmp_path / "cull_categories.json")
    monkeypatch.setattr(categories, "_cache", None, raising=False)
    monkeypatch.setattr(categories, "_cache_mtime", 0.0, raising=False)
    import job_config
    importlib.reload(job_config)
    import config_io
    importlib.reload(config_io)
    import active_learning
    importlib.reload(active_learning)
    return job_config, config_io, active_learning, tmp_path


# ── TASK 1: yt-dlp first-class per-job block ─────────────────────────────────

def test_default_preset_has_yt_dlp_block(isolated):
    jc, _cio, _al, _ = isolated
    yt = jc.get_preset("default")["scrapers"]["yt_dlp"]
    # limit default mirrors the resolve_env fallback / gallery_dl semantics (200);
    # the committed test_core_integration yt_dlp contract pins this.
    assert yt == {"enabled": False, "urls": [], "limit": 200, "cookies": ""}


def test_default_preset_yt_dlp_disabled_and_empty(isolated):
    # Gate is OFF / empty by default — no behaviour change out of the box.
    jc, _cio, _al, _ = isolated
    yt = jc.get_preset("default")["scrapers"]["yt_dlp"]
    assert yt["enabled"] is False
    assert yt["urls"] == []


def test_is_overridden_on_yt_dlp_override(isolated):
    jc, _cio, _al, _ = isolated
    job = jc.create_job("Clips", subject="drone clips")
    assert not jc.is_overridden(job, "scrapers.yt_dlp.enabled")
    job = jc.set_override(job, "scrapers.yt_dlp.enabled", True)
    assert jc.is_overridden(job, "scrapers.yt_dlp.enabled")
    # the override flows through to the effective config
    assert jc.effective_config(job)["scrapers"]["yt_dlp"]["enabled"] is True


def test_reset_yt_dlp_override_returns_to_inherited(isolated):
    jc, _cio, _al, _ = isolated
    job = jc.create_job("Clips", subject="s")
    job = jc.set_override(job, "scrapers.yt_dlp.enabled", True)
    job = jc.reset_override(job, "scrapers.yt_dlp.enabled")
    assert not jc.is_overridden(job, "scrapers.yt_dlp.enabled")
    assert jc.effective_config(job)["scrapers"]["yt_dlp"]["enabled"] is False


def test_resolve_env_projects_yt_dlp_from_default_block(isolated):
    jc, _cio, _al, _ = isolated
    job = jc.create_job("Clips", subject="s")
    env = jc.resolve_env(job)
    assert env["YT_DLP_ENABLED"] == "false"
    assert env["YT_DLP_URLS"] == ""
    assert env["YT_DLP_LIMIT"] == "200"         # default block limit (gallery_dl-parity)
    assert env["YT_DLP_COOKIES"] == ""


def test_resolve_env_projects_yt_dlp_override(isolated):
    jc, _cio, _al, _ = isolated
    job = jc.create_job("Clips", subject="s")
    job = jc.set_override(job, "scrapers.yt_dlp", {
        "enabled": True,
        "urls": ["https://yt/a", "https://yt/b"],
        "limit": 7,
        "cookies": "/c/cookies.txt",
    })
    env = jc.resolve_env(job)
    assert env["YT_DLP_ENABLED"] == "true"
    assert env["YT_DLP_URLS"] == "https://yt/a\nhttps://yt/b"
    assert env["YT_DLP_LIMIT"] == "7"
    assert env["YT_DLP_COOKIES"] == "/c/cookies.txt"


def test_config_io_round_trips_job_with_yt_dlp(isolated):
    jc, cio, _al, _ = isolated
    job = jc.create_job("Clips", subject="drone clips", preset="default")
    job = jc.set_override(job, "scrapers.yt_dlp", {
        "enabled": True, "urls": ["https://yt/x"], "limit": 12, "cookies": ""})
    job = jc.save_job(job)

    env = cio.export_job(job.slug)
    assert env["job"]["overrides"]["scrapers"]["yt_dlp"]["enabled"] is True

    restored = cio.import_job(env)
    yt = restored.overrides["scrapers"]["yt_dlp"]
    assert yt["enabled"] is True
    assert yt["urls"] == ["https://yt/x"]
    assert yt["limit"] == 12


def test_config_io_rejects_unknown_yt_dlp_key(isolated):
    jc, cio, _al, _ = isolated
    payload = {
        "cull_preset_version": cio.PRESET_VERSION, "kind": "cull.job",
        "job": {"slug": "x", "name": "X", "subject": "s", "preset": "default",
                "overrides": {"scrapers": {"yt_dlp": {"bogus_key": 1}}}},
    }
    with pytest.raises(cio.ValidationError):
        cio.import_job(payload)


def test_config_io_clamps_yt_dlp_limit(isolated):
    jc, cio, _al, _ = isolated
    cleaned = cio.validate_inheritable_cfg(
        {"scrapers": {"yt_dlp": {"limit": 10_000_000}}}, partial=True)
    assert cleaned["scrapers"]["yt_dlp"]["limit"] == cio._MAX_GALLERY_LIMIT


def test_config_io_preset_body_accepts_yt_dlp(isolated):
    # A full preset body (partial=False) must validate the default yt_dlp block.
    jc, cio, _al, _ = isolated
    body = jc.get_preset("default")
    cleaned = cio.validate_inheritable_cfg(body, partial=False)
    assert cleaned["scrapers"]["yt_dlp"]["limit"] == 200


# ── TASK 2: active-learning exemplars projection + prompt injection ──────────

def test_resolve_env_emits_empty_exemplars_by_default(isolated):
    jc, _cio, _al, _ = isolated
    job = jc.create_job("J", subject="s")
    env = jc.resolve_env(job)
    assert env["ACTIVE_LEARNING_EXEMPLARS_JSON"] == "{}"
    assert isinstance(env["ACTIVE_LEARNING_EXEMPLARS_JSON"], str)


def test_resolve_env_emits_exemplars_when_corrections_exist(isolated):
    jc, _cio, al, _ = isolated
    job = jc.create_job("J", subject="s")
    # A keep bucket the default preset ships with, plus a terminal negative.
    keep_cat = jc.effective_config(job)["categories"][0]["name"]
    al.record_correction("j", "img_keep_1", "OffTopic", keep_cat,
                         {"OVR_Quality_Score": 80, "REL_Quality_Score": 70})
    al.record_correction("j", "img_drop_1", keep_cat, "DISCARD",
                         {"OVR_Quality_Score": 20, "REL_Quality_Score": 15})

    env = jc.resolve_env(job)
    data = json.loads(env["ACTIVE_LEARNING_EXEMPLARS_JSON"])
    assert keep_cat in data            # positive (keep bucket)
    assert "DISCARD" in data           # negative (terminal)
    assert data[keep_cat][0]["image_key"] == "img_keep_1"


def test_resolve_env_exemplars_best_effort_never_raises(isolated, monkeypatch):
    # If active_learning blows up, projection still yields an empty map.
    jc, _cio, al, _ = isolated
    job = jc.create_job("J", subject="s")

    def boom(*a, **k):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(al, "exemplars", boom)
    env = jc.resolve_env(job)                       # must not raise
    assert env["ACTIVE_LEARNING_EXEMPLARS_JSON"] == "{}"


def test_build_prompt_injects_exemplar_block_when_env_set(isolated, monkeypatch):
    jc, _cio, _al, _ = isolated
    import vision_prompt
    importlib.reload(vision_prompt)
    payload = json.dumps({
        "Keep": [{"image_key": "k1", "caption": "a calm lake at dawn",
                  "note": "user moved OffTopic -> Keep"}],
        "DISCARD": [{"image_key": "d1", "note": "user moved Keep -> DISCARD"}],
    })
    monkeypatch.setenv("ACTIVE_LEARNING_EXEMPLARS_JSON", payload)
    prompt = vision_prompt.build_classification_prompt()
    assert "past corrections" in prompt
    assert "a calm lake at dawn" in prompt
    assert "user moved Keep -> DISCARD" in prompt


def test_build_prompt_byte_identical_when_env_absent(isolated, monkeypatch):
    """With no exemplar env var the prompt is byte-for-byte what it was before
    this wiring landed (the empty-string concatenation is a no-op)."""
    jc, _cio, _al, _ = isolated
    import vision_prompt
    importlib.reload(vision_prompt)
    monkeypatch.delenv("ACTIVE_LEARNING_EXEMPLARS_JSON", raising=False)
    baseline = vision_prompt.build_classification_prompt()
    # Setting it to an empty JSON object also yields no block.
    monkeypatch.setenv("ACTIVE_LEARNING_EXEMPLARS_JSON", "{}")
    assert vision_prompt.build_classification_prompt() == baseline
    # …and so does whitespace / malformed JSON (tolerant by design).
    monkeypatch.setenv("ACTIVE_LEARNING_EXEMPLARS_JSON", "   ")
    assert vision_prompt.build_classification_prompt() == baseline
    monkeypatch.setenv("ACTIVE_LEARNING_EXEMPLARS_JSON", "{not json")
    assert vision_prompt.build_classification_prompt() == baseline


def test_exemplar_block_renderer_tolerates_junk(isolated):
    jc, _cio, _al, _ = isolated
    import vision_prompt
    importlib.reload(vision_prompt)
    fn = vision_prompt._active_learning_exemplars_block
    assert fn(None) == ""
    assert fn("") == ""
    assert fn("[]") == ""              # not a dict
    assert fn("{}") == ""             # empty dict
    assert fn(json.dumps({"Keep": []})) == ""     # no usable exemplars


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
