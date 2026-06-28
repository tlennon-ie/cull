"""Flask test-client tests for the dashboard's Wave-2 integration surface.

These cover the endpoints wired in this slice on top of the existing dashboard:

  * /api/vision/test  — cloud-provider model dropdown (req #1, the headline)
  * /api/duplicates (+ /api/duplicates/delete) — perceptual-hash near-dupes
  * /api/export/dataset — training-ready dataset profiles
  * /api/jobs/<slug>/export/huggingface — push to a HF dataset repo
  * /api/schedules — GET/POST/DELETE per-job schedules
  * /api/vision/health — fleet telemetry probe
  * the new SETTINGS_KEYS / caption-style sync

Network + real work is mocked (vision_model_catalog.list_models,
phash_dedup.find_near_duplicates, export_profiles.export_dataset,
hf_export.push_to_hf, fleet_health.probe_fleet) so nothing leaves the box and
the suite stays fast. The fixture mirrors test_dashboard_jobs.py: it sets
PIPELINE_BASE_DIR first, then reloads the storage modules + the dashboard so all
state lands under a temp dir.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

PIPELINE_CODE = Path(__file__).resolve().parent.parent / "pipeline_code"
if str(PIPELINE_CODE) not in sys.path:
    sys.path.insert(0, str(PIPELINE_CODE))

# Neutralise dotenv so the developer's real .env can't leak into the temp store
# (same reasoning as test_dashboard_jobs.py).
try:
    import dotenv as _dotenv
    _dotenv.load_dotenv = lambda *a, **k: False  # type: ignore[assignment]
except Exception:  # pragma: no cover
    pass

_DOTENV_SOURCED_KEYS = (
    "PIPELINE_SLUG", "PIPELINE_TOPIC", "SCRAPER_DISABLED", "X_ACCOUNTS",
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY",
    "OPENROUTER_API_KEY", "HF_TOKEN", "HUGGINGFACE_TOKEN",
)
for _k in _DOTENV_SOURCED_KEYS:
    os.environ.pop(_k, None)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """A Flask test client isolated under tmp_path; returns (client, dashboard, tmp_path)."""
    monkeypatch.setenv("PIPELINE_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("PIPELINE_QUEUE", str(tmp_path / "queue"))
    monkeypatch.setenv("PIPELINE_SORTED", str(tmp_path / "sorted"))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    for leaked in _DOTENV_SOURCED_KEYS:
        monkeypatch.delenv(leaked, raising=False)
    (tmp_path / ".env").write_text("BLUR_NSFW_THUMBS=true\n", encoding="utf-8")
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))

    import categories
    monkeypatch.setattr(categories, "ACTIVE_PATH", tmp_path / "cull_categories.json")
    monkeypatch.setattr(categories, "_cache", None, raising=False)
    monkeypatch.setattr(categories, "_cache_mtime", 0.0, raising=False)

    import paths
    importlib.reload(paths)
    import job_config
    importlib.reload(job_config)
    import index_store
    importlib.reload(index_store)
    import thumb_cache
    importlib.reload(thumb_cache)

    import dashboard_enhanced
    dashboard = importlib.reload(dashboard_enhanced)
    dashboard.app.config.update(TESTING=True)

    jobs_dir = tmp_path / "jobs"
    if jobs_dir.is_dir():
        for f in jobs_dir.glob("*"):
            f.unlink()
    return dashboard.app.test_client(), dashboard, tmp_path


# ── /api/vision/test : cloud-provider model dropdown (req #1) ─────────────────

def test_vision_test_openai_returns_models_with_env_key(client, monkeypatch):
    c, dash, _ = client
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    captured = {}

    def fake_list(provider, base_url=None, api_key=None, timeout=10):
        captured["provider"] = provider
        captured["api_key"] = api_key
        return ["gpt-4o", "gpt-4o-mini"]

    monkeypatch.setattr(dash.vision_model_catalog, "list_models", fake_list)
    j = c.post("/api/vision/test", json={"provider": "openai"}).get_json()
    assert j["ok"] is True
    assert j["models"] == ["gpt-4o", "gpt-4o-mini"]
    assert captured["provider"] == "openai"
    assert captured["api_key"] == "sk-test"          # resolved from env


def test_vision_test_openai_body_key_overrides_env(client, monkeypatch):
    c, dash, _ = client
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    seen = {}

    def fake_list(provider, base_url=None, api_key=None, timeout=10):
        seen["api_key"] = api_key
        return ["gpt-4o"]

    monkeypatch.setattr(dash.vision_model_catalog, "list_models", fake_list)
    j = c.post("/api/vision/test", json={"provider": "openai", "api_key": "sk-body"}).get_json()
    assert j["ok"] is True
    assert seen["api_key"] == "sk-body"              # body wins over env


def test_vision_test_cloud_without_key_is_400(client, monkeypatch):
    c, dash, _ = client
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # Should never even call list_models when no key is available.
    monkeypatch.setattr(dash.vision_model_catalog, "list_models",
                        lambda *a, **k: pytest.fail("should not be called without a key"))
    r = c.post("/api/vision/test", json={"provider": "anthropic"})
    assert r.status_code == 400
    assert r.get_json()["ok"] is False


def test_vision_test_gemini_uses_google_api_key_fallback(client, monkeypatch):
    c, dash, _ = client
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "g-key")
    seen = {}

    def fake_list(provider, base_url=None, api_key=None, timeout=10):
        seen["api_key"] = api_key
        return ["gemini-2.0-flash"]

    monkeypatch.setattr(dash.vision_model_catalog, "list_models", fake_list)
    j = c.post("/api/vision/test", json={"provider": "gemini"}).get_json()
    assert j["ok"] is True and j["models"] == ["gemini-2.0-flash"]
    assert seen["api_key"] == "g-key"


def test_vision_test_cloud_empty_models_is_not_ok(client, monkeypatch):
    c, dash, _ = client
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or")
    monkeypatch.setattr(dash.vision_model_catalog, "list_models", lambda *a, **k: [])
    j = c.post("/api/vision/test", json={"provider": "openrouter"}).get_json()
    assert j["ok"] is False
    assert j["models"] == []


# ── /api/duplicates ───────────────────────────────────────────────────────────

def test_duplicates_envelope_and_threshold(client, monkeypatch):
    c, dash, _ = client
    dash.job_config.create_job("Dupes", subject="x")
    seen = {}

    def fake_find(threshold=10, slug=None):
        seen["threshold"] = threshold
        seen["slug"] = slug
        return []

    monkeypatch.setattr(dash.phash_dedup, "find_near_duplicates", fake_find)
    body = c.get("/api/duplicates?threshold=7&job=dupes").get_json()
    assert body["success"] is True
    assert body["data"]["groups"] == []
    assert body["data"]["threshold"] == 7
    assert seen["threshold"] == 7 and seen["slug"] == "dupes"


def test_duplicates_maps_groups_guarded_by_safe_inside(client, monkeypatch):
    c, dash, tmp = client
    dash.job_config.create_job("Dg", subject="x")
    # One path inside sorted (kept), one outside (must be dropped by safe_inside).
    inside = dash.PIPELINE_SORTED / "Keepers" / "civitai" / "a.png"
    inside.parent.mkdir(parents=True, exist_ok=True)
    inside.write_bytes(b"x")
    inside2 = dash.PIPELINE_SORTED / "Keepers" / "civitai" / "b.png"
    inside2.write_bytes(b"x")
    outside = tmp / "evil" / "c.png"
    outside.parent.mkdir(parents=True, exist_ok=True)
    outside.write_bytes(b"x")

    monkeypatch.setattr(dash.phash_dedup, "find_near_duplicates",
                        lambda threshold=10, slug=None: [[str(inside), str(inside2), str(outside)]])
    body = c.get("/api/duplicates?job=dg").get_json()
    assert body["success"] is True
    groups = body["data"]["groups"]
    assert len(groups) == 1
    paths = {m["path"] for m in groups[0]}
    assert str(inside) in paths and str(inside2) in paths
    assert str(outside) not in paths             # traversal-guarded out
    assert all(m["thumb_url"].startswith("/api/thumbnail?path=") for m in groups[0])


def test_duplicates_delete_guards_path(client):
    c, dash, _ = client
    # A path outside the pipeline roots is refused.
    r = c.post("/api/duplicates/delete", json={"path": "/etc/passwd"})
    assert r.status_code == 400
    assert r.get_json()["success"] is False


def test_duplicates_delete_removes_file_and_sidecars(client):
    c, dash, _ = client
    img = dash.PIPELINE_SORTED / "Keepers" / "civitai" / "d.png"
    img.parent.mkdir(parents=True, exist_ok=True)
    img.write_bytes(b"x")
    (img.with_suffix(".txt")).write_text("caption", encoding="utf-8")
    (img.with_name("d.vision.json")).write_text("{}", encoding="utf-8")
    r = c.post("/api/duplicates/delete", json={"path": str(img)})
    assert r.status_code == 200 and r.get_json()["success"] is True
    assert not img.exists()
    assert not img.with_suffix(".txt").exists()


# ── /api/export/dataset ───────────────────────────────────────────────────────

def test_export_dataset_invokes_module_and_returns_summary(client, monkeypatch):
    c, dash, _ = client
    dash.job_config.create_job("Exp", subject="x")
    dash.job_config.activate("exp")
    seen = {}

    def fake_export(slug, profile, out_dir, **opts):
        seen["slug"] = slug
        seen["profile"] = profile
        seen["opts"] = opts
        seen["out_dir"] = str(out_dir)
        return {"slug": slug, "profile": profile, "sample_count": 3, "out_dir": str(out_dir)}

    monkeypatch.setattr(dash.export_profiles, "export_dataset", fake_export)
    r = c.post("/api/export/dataset", json={
        "profile": "kohya", "trigger_word": "tok", "repeats": 5,
        "resolution_bucketing": True,
    })
    assert r.status_code == 200
    body = r.get_json()
    assert body["success"] is True
    assert body["data"]["sample_count"] == 3
    assert seen["slug"] == "exp" and seen["profile"] == "kohya"
    assert seen["opts"]["trigger_word"] == "tok"
    assert seen["opts"]["repeats"] == 5
    assert "exports" in seen["out_dir"] and "exp" in seen["out_dir"]


def test_export_dataset_rejects_unknown_profile(client, monkeypatch):
    c, dash, _ = client
    dash.job_config.create_job("Exp2", subject="x")
    dash.job_config.activate("exp2")
    monkeypatch.setattr(dash.export_profiles, "export_dataset",
                        lambda *a, **k: pytest.fail("should not export an unknown profile"))
    r = c.post("/api/export/dataset", json={"profile": "totally-bogus"})
    assert r.status_code == 400
    assert r.get_json()["success"] is False


def test_export_dataset_without_active_job_is_409(client):
    c, dash, _ = client
    # No active job and no ?job= → 409.
    r = c.post("/api/export/dataset", json={"profile": "kohya"})
    assert r.status_code == 409


def test_export_dataset_rejects_malformed_option_type(client, monkeypatch):
    """A dict where a scalar is expected must be a clean 400, not a 500."""
    c, dash, _ = client
    dash.job_config.create_job("Exp3", subject="x")
    dash.job_config.activate("exp3")
    monkeypatch.setattr(dash.export_profiles, "export_dataset",
                        lambda *a, **k: pytest.fail("should not export with a bad option type"))
    r = c.post("/api/export/dataset", json={"profile": "kohya", "repeats": {"bad": 1}})
    assert r.status_code == 400
    assert r.get_json()["success"] is False


def test_duplicates_thumb_url_is_url_encoded(client, monkeypatch):
    """A sorted path with a space must come back percent-encoded in thumb_url."""
    c, dash, _ = client
    dash.job_config.create_job("Enc", subject="x")
    spaced = dash.PIPELINE_SORTED / "Keepers" / "civitai" / "a b.png"
    spaced.parent.mkdir(parents=True, exist_ok=True)
    spaced.write_bytes(b"x")
    other = dash.PIPELINE_SORTED / "Keepers" / "civitai" / "c.png"
    other.write_bytes(b"x")
    monkeypatch.setattr(dash.phash_dedup, "find_near_duplicates",
                        lambda threshold=10, slug=None: [[str(spaced), str(other)]])
    body = c.get("/api/duplicates?job=enc").get_json()
    urls = [m["thumb_url"] for m in body["data"]["groups"][0]]
    # The space must be encoded (no raw ' ' in the query string).
    spaced_url = next(u for u in urls if "a%20b" in u or "a+b" in u)
    assert " " not in spaced_url.split("path=", 1)[1]


# ── /api/jobs/<slug>/export/huggingface ───────────────────────────────────────

def test_hf_push_invokes_module(client, monkeypatch):
    c, dash, _ = client
    dash.job_config.create_job("Hf", subject="x")
    seen = {}

    def fake_push(slug, repo_id, private=True, include_video=False, categories=None):
        seen.update(slug=slug, repo_id=repo_id, private=private, include_video=include_video)
        return {"repo_id": repo_id, "uploaded": 12, "private": private}

    monkeypatch.setattr(dash.hf_export, "push_to_hf", fake_push)
    r = c.post("/api/jobs/hf/export/huggingface",
               json={"repo_id": "me/ds", "private": True, "include_video": False})
    assert r.status_code == 200
    body = r.get_json()
    assert body["success"] is True and body["data"]["uploaded"] == 12
    assert seen["slug"] == "hf" and seen["repo_id"] == "me/ds"


def test_hf_push_rejects_bad_repo_id(client, monkeypatch):
    c, dash, _ = client
    dash.job_config.create_job("Hf2", subject="x")
    monkeypatch.setattr(dash.hf_export, "push_to_hf",
                        lambda *a, **k: pytest.fail("should not push with a bad repo_id"))
    r = c.post("/api/jobs/hf2/export/huggingface", json={"repo_id": "no-slash"})
    assert r.status_code == 400
    assert r.get_json()["success"] is False


def test_hf_push_unknown_job_is_404(client):
    c, dash, _ = client
    r = c.post("/api/jobs/ghost/export/huggingface", json={"repo_id": "me/ds"})
    assert r.status_code == 404


def test_hf_push_missing_token_message(client, monkeypatch):
    c, dash, _ = client
    dash.job_config.create_job("Hf3", subject="x")

    def boom(*a, **k):
        raise dash.credentials.MissingCredentialError("HF_TOKEN", scraper="hf_export")

    monkeypatch.setattr(dash.hf_export, "push_to_hf", boom)
    r = c.post("/api/jobs/hf3/export/huggingface", json={"repo_id": "me/ds"})
    assert r.status_code == 400
    assert "HF_TOKEN" in r.get_json()["error"]


def test_hf_push_runtime_error_maps_to_install_hint(client, monkeypatch):
    c, dash, _ = client
    dash.job_config.create_job("Hf4", subject="x")

    def boom(*a, **k):
        raise RuntimeError("huggingface_hub is required")

    monkeypatch.setattr(dash.hf_export, "push_to_hf", boom)
    r = c.post("/api/jobs/hf4/export/huggingface", json={"repo_id": "me/ds"})
    assert r.status_code == 400
    assert "huggingface_hub" in r.get_json()["error"]


# ── /api/schedules ────────────────────────────────────────────────────────────

def test_schedules_crud_round_trip(client):
    c, dash, _ = client
    dash.job_config.create_job("Sched", subject="x")

    # Empty initially, with the action vocabulary exposed for the UI.
    body = c.get("/api/schedules").get_json()
    assert body["success"] is True
    assert body["data"]["schedules"] == []
    assert "scrape" in body["data"]["actions"]

    # Add one.
    r = c.post("/api/schedules", json={"slug": "sched", "cadence": "@daily", "action": "scrape"})
    assert r.status_code == 200 and r.get_json()["success"] is True
    rows = c.get("/api/schedules").get_json()["data"]["schedules"]
    assert len(rows) == 1 and rows[0]["slug"] == "sched"

    # Delete it.
    r = c.delete("/api/schedules?slug=sched&action=scrape")
    assert r.status_code == 200 and r.get_json()["data"]["removed"] is True
    assert c.get("/api/schedules").get_json()["data"]["schedules"] == []


def test_schedules_add_rejects_unknown_job(client):
    c, dash, _ = client
    r = c.post("/api/schedules", json={"slug": "nope", "cadence": "@daily", "action": "scrape"})
    assert r.status_code == 400


def test_schedules_add_rejects_bad_cadence(client):
    c, dash, _ = client
    dash.job_config.create_job("Sb", subject="x")
    r = c.post("/api/schedules", json={"slug": "sb", "cadence": "every blue moon", "action": "scrape"})
    assert r.status_code == 400


def test_schedules_add_rejects_bad_action(client):
    c, dash, _ = client
    dash.job_config.create_job("Sa", subject="x")
    r = c.post("/api/schedules", json={"slug": "sa", "cadence": "@daily", "action": "frobnicate"})
    assert r.status_code == 400


def test_schedules_add_disk_error_is_clean_500(client, monkeypatch):
    """An OSError from the persist layer returns a clean envelope, not a raw 500."""
    c, dash, _ = client
    dash.job_config.create_job("Sd", subject="x")

    def boom(_sched):
        raise OSError("disk full")

    monkeypatch.setattr(dash.scheduler, "add_schedule", boom)
    r = c.post("/api/schedules", json={"slug": "sd", "cadence": "@daily", "action": "scrape"})
    assert r.status_code == 500
    body = r.get_json()
    assert body["success"] is False and "could not persist" in body["error"]


# ── /api/vision/health ────────────────────────────────────────────────────────

def test_vision_health_probes_clean_fleet(client, monkeypatch):
    c, dash, _ = client
    dash.job_config.create_job("Vh", subject="x")
    dash.job_config.activate("vh")
    captured = {}

    def fake_probe(fleet, timeout=None):
        captured["fleet"] = fleet
        return {ep.get("id"): {"ok": True, "latency_ms": 5, "error": None} for ep in fleet}

    monkeypatch.setattr(dash.fleet_health, "probe_fleet", fake_probe)
    body = c.get("/api/vision/health?job=vh").get_json()
    assert body["success"] is True
    assert "probes" in body["data"] and "fleet" in body["data"]
    # The default preset ships one LM Studio worker, so the fleet is non-empty.
    assert isinstance(captured["fleet"], list)


# ── new SETTINGS_KEYS + caption-style sync ────────────────────────────────────

def test_new_settings_keys_present(client):
    c, dash, _ = client
    body = c.get("/api/settings").get_json()
    for key in ("YT_DLP_ENABLED", "YT_DLP_URLS", "PREFILTER_ENABLED",
                "WEBHOOK_URL", "DESKTOP_NOTIFICATIONS", "EMBEDDINGS_ENABLED",
                "SCHEDULER_ENABLED", "VIDEO_CLASSIFY_ENABLED",
                "OPENAI_API_KEY", "OPENAI_VISION_MODEL"):
        assert key in body, key


def test_settings_post_accepts_new_toggle(client):
    c, dash, _ = client
    r = c.post("/api/settings", json={"YT_DLP_ENABLED": "true"})
    assert r.status_code == 200
    assert r.get_json()["success"] is True


def test_settings_openai_key_masked_on_get(client, monkeypatch):
    c, dash, _ = client
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    body = c.get("/api/settings").get_json()
    # Secret keys are masked, never echoed in the clear.
    assert body["OPENAI_API_KEY"] == dash.SECRET_MASK


def test_caption_style_motion_now_accepted(client):
    c, dash, _ = client
    dash.job_config.create_job("Cap", subject="x")
    # "motion" must round-trip now that the gate is sourced from vision_prompt.
    r = c.put("/api/jobs/cap", json={"overrides": {"captioning": {"style": "motion"}}})
    assert r.status_code == 200
    body = c.get("/api/jobs/cap").get_json()
    assert body["effective"]["captioning"]["style"] == "motion"


def test_caption_style_still_rejects_garbage(client):
    c, dash, _ = client
    dash.job_config.create_job("Cap2", subject="x")
    r = c.put("/api/jobs/cap2", json={"overrides": {"captioning": {"style": "haiku"}}})
    assert r.status_code == 400


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
