"""Tests for the dynamic vision-worker fleet (Jobs v2 ``vision.workers`` block).

The local vision workers are a user-defined list of instances
(``{id, name, provider, base_url, model, api_key, enabled}``) inherited from the
preset and overridable per job, projected to ``VISION_WORKERS_JSON`` for the
supervisor to fan out into one worker subprocess per instance.

Runnable:
    pytest tests/test_vision_fleet.py
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
    monkeypatch.setenv("PIPELINE_BASE_DIR", str(tmp_path))
    # Clear legacy vision env so migration tests start from a clean slate.
    for k in ("LMSTUDIO_PRIMARY_URL", "LMSTUDIO_SECONDARY_URL", "OLLAMA_URL",
              "OPENAI_COMPAT_URL", "LMSTUDIO_PRIMARY_MODEL", "OPENAI_COMPAT_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    import categories
    monkeypatch.setattr(categories, "ACTIVE_PATH", tmp_path / "cull_categories.json")
    monkeypatch.setattr(categories, "_cache", None, raising=False)
    monkeypatch.setattr(categories, "_cache_mtime", 0.0, raising=False)
    import job_config
    importlib.reload(job_config)
    return job_config, tmp_path


# ── shipped default fleet ─────────────────────────────────────────────────────

def test_default_preset_ships_one_lmstudio_worker(isolated):
    jc, _ = isolated
    workers = jc.get_preset("default")["vision"]["workers"]
    assert len(workers) == 1
    w = workers[0]
    assert w["provider"] == "lmstudio"
    assert w["base_url"].startswith("http://127.0.0.1")
    assert w["enabled"] is True


def test_effective_config_exposes_vision_block(isolated):
    jc, _ = isolated
    job = jc.create_job("Set", preset="default")
    eff = jc.effective_config(job)
    assert "vision" in eff and isinstance(eff["vision"].get("workers"), list)


# ── clean_vision_fleet (the sanitizer) ───────────────────────────────────────

def test_clean_vision_fleet_filters_invalid_and_disabled(isolated):
    jc, _ = isolated
    fleet = jc.clean_vision_fleet([
        {"id": "a", "provider": "lmstudio", "base_url": "http://h:1234", "enabled": True},
        {"id": "b", "provider": "ollama", "base_url": "http://h:11434", "enabled": False},  # off
        {"id": "c", "provider": "lmstudio", "base_url": "", "enabled": True},               # no url
        {"id": "d", "provider": "bogus", "base_url": "http://h:1", "enabled": True},         # bad provider
    ])
    assert [w["id"] for w in fleet] == ["a"]
    assert fleet[0]["model"] == "" and fleet[0]["api_key"] == ""


def test_clean_vision_fleet_normalises_provider_case_and_fills_name(isolated):
    jc, _ = isolated
    fleet = jc.clean_vision_fleet([
        {"id": "w1", "provider": "LMStudio", "base_url": " http://h:1234 "}])
    assert fleet[0]["provider"] == "lmstudio"
    assert fleet[0]["base_url"] == "http://h:1234"
    assert fleet[0]["name"] == "w1"            # falls back to id


# ── resolve_env projection ───────────────────────────────────────────────────

def test_resolve_env_projects_default_fleet(isolated):
    jc, _ = isolated
    job = jc.create_job("Set", preset="default")
    fleet = json.loads(jc.resolve_env(job)["VISION_WORKERS_JSON"])
    assert len(fleet) == 1 and fleet[0]["provider"] == "lmstudio"


def test_resolve_env_reflects_per_job_fleet_override(isolated):
    jc, _ = isolated
    job = jc.create_job("Set", preset="default")
    job = jc.set_override(job, "vision.workers", [
        {"id": "gpu1", "name": "RunPod A", "provider": "ollama",
         "base_url": "http://10.0.0.5:11434", "model": "llama3.2-vision",
         "api_key": "", "enabled": True},
        {"id": "gpu2", "name": "RunPod B", "provider": "llamacpp",
         "base_url": "http://10.0.0.6:8000", "model": "", "api_key": "sk-x",
         "enabled": True},
    ])
    fleet = json.loads(jc.resolve_env(job)["VISION_WORKERS_JSON"])
    assert {w["provider"] for w in fleet} == {"ollama", "llamacpp"}
    assert fleet[1]["api_key"] == "sk-x"


# ── legacy migration ─────────────────────────────────────────────────────────

def test_migrate_legacy_vision_builds_fleet_from_env(isolated, monkeypatch):
    jc, _ = isolated
    jc.list_presets()
    monkeypatch.setenv("LMSTUDIO_PRIMARY_URL", "http://169.254.83.107:1234")
    monkeypatch.setenv("LMSTUDIO_PRIMARY_MODEL", "qwen3-vl")
    monkeypatch.setenv("OLLAMA_URL", "http://100.95.148.26:11434")
    assert jc.migrate_legacy_vision_to_fleet() is True
    workers = jc.get_preset("default")["vision"]["workers"]
    providers = {w["provider"]: w["base_url"] for w in workers}
    assert providers["lmstudio"] == "http://169.254.83.107:1234"
    assert providers["ollama"] == "http://100.95.148.26:11434"
    # idempotent: a second call is a no-op (fleet already customised)
    assert jc.migrate_legacy_vision_to_fleet() is False


def test_migrate_legacy_vision_noop_without_env(isolated):
    jc, _ = isolated
    jc.list_presets()
    assert jc.migrate_legacy_vision_to_fleet() is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
