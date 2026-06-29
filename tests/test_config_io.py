"""Tests for config_io — share a preset/job as a portable file.

Runnable:
    pytest tests/test_config_io.py
    python tests/test_config_io.py

config_io is the reusable CORE behind the dashboard's export/import endpoints:
a versioned ``{"cull_preset_version": N, "name", "preset": {...}}`` envelope for
presets (and a parallel job envelope carrying subject + preset ref + overrides),
with strict validation (reject unknown top-level keys / bad block shapes /
oversized payloads / wrong-version) raising a clear ``ValidationError``.

The tests are written first (TDD): round-trip export->import equals the original,
malformed/oversized/unknown-key/wrong-version payloads are rejected with clear
errors, and every shipped community preset import-validates cleanly.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PIPELINE_CODE = Path(__file__).resolve().parent.parent / "pipeline_code"
if str(PIPELINE_CODE) not in sys.path:
    sys.path.insert(0, str(PIPELINE_CODE))

COMMUNITY_DIR = Path(__file__).resolve().parent.parent / "presets" / "community"


@pytest.fixture()
def isolated(tmp_path, monkeypatch):
    """Reload job_config + config_io against an isolated data dir (mirrors the
    job_config test fixture so preset/job CRUD writes under tmp_path)."""
    monkeypatch.setenv("PIPELINE_BASE_DIR", str(tmp_path))
    import categories
    monkeypatch.setattr(categories, "ACTIVE_PATH", tmp_path / "cull_categories.json")
    monkeypatch.setattr(categories, "_cache", None, raising=False)
    monkeypatch.setattr(categories, "_cache_mtime", 0.0, raising=False)
    import importlib
    import job_config
    importlib.reload(job_config)
    import config_io
    importlib.reload(config_io)
    return config_io, job_config, tmp_path


# ── envelope basics ───────────────────────────────────────────────────────────

def test_export_preset_returns_versioned_envelope(isolated):
    cio, jc, _ = isolated
    env = cio.export_preset("default")
    assert env["cull_preset_version"] == cio.PRESET_VERSION
    assert env["kind"] == "cull.preset"
    assert env["name"] == "default"
    assert isinstance(env["preset"], dict)
    # carries the real inheritable shape
    assert "categories" in env["preset"] and "topic_filters" in env["preset"]


def test_export_preset_unknown_name_raises(isolated):
    cio, jc, _ = isolated
    with pytest.raises(cio.ValidationError):
        cio.export_preset("does_not_exist_at_all")


def test_export_bytes_is_valid_utf8_json(isolated):
    cio, jc, _ = isolated
    raw = cio.export_preset_bytes("default")
    assert isinstance(raw, (bytes, bytearray))
    env = json.loads(raw.decode("utf-8"))
    assert env["name"] == "default"


# ── round-trip: export -> import equals original ──────────────────────────────

def test_preset_round_trip_equals_original(isolated):
    cio, jc, _ = isolated
    original = jc.get_preset("default")
    env = cio.export_preset("default")
    name, cfg = cio.import_preset(env)
    assert name == "default"
    # import returns a validated, projection-ready preset equal to the source
    merged = jc._deep_merge(jc._default_preset_cfg(), cfg)
    assert merged == original


def test_import_preset_from_bytes(isolated):
    cio, jc, _ = isolated
    raw = cio.export_preset_bytes("anime_illustration")
    name, cfg = cio.import_preset(raw)
    assert name == "anime_illustration"
    assert any(c["name"] == "Photoreal" for c in cfg["categories"])


def test_import_preset_accepts_raw_string(isolated):
    cio, jc, _ = isolated
    env = cio.export_preset("default")
    name, cfg = cio.import_preset(json.dumps(env))
    assert name == "default"


# ── job envelope: subject + preset ref + overrides ────────────────────────────

def test_job_round_trip_carries_subject_preset_overrides(isolated):
    cio, jc, _ = isolated
    job = jc.create_job("My Cars", subject="red sports cars", preset="default")
    job = jc.set_override(job, "scoring.ovr_min", 71)
    job = jc.save_job(job)

    env = cio.export_job(job.slug)
    assert env["cull_preset_version"] == cio.PRESET_VERSION
    assert env["kind"] == "cull.job"
    assert env["job"]["subject"] == "red sports cars"
    assert env["job"]["preset"] == "default"
    assert env["job"]["overrides"]["scoring"]["ovr_min"] == 71

    restored = cio.import_job(env)
    assert restored.subject == "red sports cars"
    assert restored.preset == "default"
    assert restored.overrides["scoring"]["ovr_min"] == 71
    assert jc.JOB_SLUG_RE.match(restored.slug)


def test_export_job_unknown_slug_raises(isolated):
    cio, jc, _ = isolated
    with pytest.raises(cio.ValidationError):
        cio.export_job("no_such_job")


def test_import_job_rejects_bad_overrides(isolated):
    cio, jc, _ = isolated
    payload = {
        "cull_preset_version": cio.PRESET_VERSION, "kind": "cull.job",
        "job": {"slug": "x", "name": "X", "subject": "s", "preset": "default",
                "overrides": {"scoring": {"ovr_min": 9999}}},
    }
    with pytest.raises(cio.ValidationError):
        cio.import_job(payload)


def test_import_job_rejects_bad_slug(isolated):
    cio, jc, _ = isolated
    payload = {
        "cull_preset_version": cio.PRESET_VERSION, "kind": "cull.job",
        "job": {"slug": "Bad Slug!", "name": "X", "subject": "s", "preset": "default",
                "overrides": {}},
    }
    with pytest.raises(cio.ValidationError):
        cio.import_job(payload)


# ── rejection: unknown top-level keys ─────────────────────────────────────────

def test_import_rejects_unknown_top_level_key(isolated):
    cio, jc, _ = isolated
    env = cio.export_preset("default")
    env["evil"] = {"rm": "-rf"}
    with pytest.raises(cio.ValidationError) as exc:
        cio.import_preset(env)
    assert "unknown" in str(exc.value).lower()


def test_import_rejects_unknown_preset_block_key(isolated):
    cio, jc, _ = isolated
    env = cio.export_preset("default")
    env["preset"]["totally_unknown_block"] = 1
    with pytest.raises(cio.ValidationError) as exc:
        cio.import_preset(env)
    assert "unknown" in str(exc.value).lower()


# ── rejection: wrong / missing version ────────────────────────────────────────

def test_import_rejects_future_version(isolated):
    cio, jc, _ = isolated
    env = cio.export_preset("default")
    env["cull_preset_version"] = cio.PRESET_VERSION + 99
    with pytest.raises(cio.ValidationError) as exc:
        cio.import_preset(env)
    assert "version" in str(exc.value).lower()


def test_import_rejects_empty_categories(isolated):
    cio, jc, _ = isolated
    with pytest.raises(cio.ValidationError) as exc:
        cio.import_preset({"cull_preset_version": cio.PRESET_VERSION,
                           "name": "x", "preset": {"categories": []}})
    assert "categor" in str(exc.value).lower()


def test_import_accepts_missing_kind_when_otherwise_valid(isolated):
    # A kind-less {name, preset} envelope is tolerated (forward-compat), as long
    # as the body is valid. Only a WRONG kind is rejected.
    cio, jc, _ = isolated
    env = cio.export_preset("default")
    env.pop("kind")
    name, cfg = cio.import_preset(env)
    assert name == "default"


def test_import_rejects_wrong_kind(isolated):
    cio, jc, _ = isolated
    env = cio.export_preset("default")
    env["kind"] = "cull.job"
    with pytest.raises(cio.ValidationError) as exc:
        cio.import_preset(env)
    assert "kind" in str(exc.value).lower()


def test_import_rejects_non_object(isolated):
    cio, jc, _ = isolated
    for junk in ("[]", "42", '"hi"', "null"):
        with pytest.raises(cio.ValidationError):
            cio.import_preset(junk)


def test_import_rejects_invalid_json_text(isolated):
    cio, jc, _ = isolated
    with pytest.raises(cio.ValidationError):
        cio.import_preset("{not valid json")


# ── tolerance: older / legacy envelopes migrate forward ───────────────────────

def test_import_tolerates_legacy_version_field(isolated):
    cio, jc, _ = isolated
    # The dashboard's commit-9fcf2ca envelope used {"version": 1} + "cfg".
    legacy = {"kind": "cull.preset", "version": 1, "name": "legacy_demo",
              "cfg": jc.get_preset("default")}
    name, cfg = cio.import_preset(legacy)
    assert name == "legacy_demo"
    assert "categories" in cfg


def test_import_tolerates_bare_name_cfg_envelope(isolated):
    cio, jc, _ = isolated
    # Oldest shape: just {name, cfg} with no kind/version at all.
    bare = {"name": "bare_demo", "cfg": jc.get_preset("default")}
    name, cfg = cio.import_preset(bare)
    assert name == "bare_demo"


# ── rejection: oversized payload ──────────────────────────────────────────────

def test_import_rejects_oversized_payload(isolated):
    cio, jc, _ = isolated
    env = cio.export_preset("default")
    env["preset"]["category_rules"] = "x" * (cio.MAX_ENVELOPE_BYTES + 10)
    with pytest.raises(cio.ValidationError) as exc:
        cio.import_preset(env)
    assert "large" in str(exc.value).lower() or "size" in str(exc.value).lower()


def test_import_rejects_bad_preset_name(isolated):
    cio, jc, _ = isolated
    env = cio.export_preset("default")
    env["name"] = "no/slashes/allowed"
    with pytest.raises(cio.ValidationError):
        cio.import_preset(env)


# ── community gallery: every shipped preset imports cleanly ───────────────────

def test_community_dir_exists_with_readme():
    assert COMMUNITY_DIR.is_dir(), "presets/community/ must exist (the gallery)"
    assert (COMMUNITY_DIR / "README.md").is_file(), "community README missing"


def test_community_presets_present():
    files = sorted(COMMUNITY_DIR.glob("*.json"))
    assert len(files) >= 2, "expected at least 2 example community presets"


def test_shipped_community_presets_all_import_validate(isolated):
    cio, jc, _ = isolated
    files = sorted(COMMUNITY_DIR.glob("*.json"))
    assert files, "no community preset files found"
    for path in files:
        raw = path.read_bytes()
        name, cfg = cio.import_preset(raw)  # must not raise
        assert name, f"{path.name}: empty name"
        # a real, projection-ready preset: required categories drive the schema
        assert cfg.get("categories"), f"{path.name}: missing categories"
        merged = jc._deep_merge(jc._default_preset_cfg(), cfg)
        # round-trips through save + projection without error
        jc.save_preset("imported_check", merged)
        assert jc.get_preset("imported_check")["categories"]


def test_save_imported_preset_persists(isolated):
    cio, jc, _ = isolated
    files = sorted(COMMUNITY_DIR.glob("*.json"))
    name, cfg = cio.import_preset(files[0].read_bytes())
    merged = jc._deep_merge(jc._default_preset_cfg(), cfg)
    jc.save_preset(name, merged)
    assert name in jc.list_presets()["presets"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
