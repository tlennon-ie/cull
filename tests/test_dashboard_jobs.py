"""Flask test-client tests for the dashboard's jobs model surface (v2).

Runnable two ways:
    pytest tests/test_dashboard_jobs.py
    python tests/test_dashboard_jobs.py        # falls back to pytest.main

v2 model: a Job is {slug, name, status, subject, preset, overrides}; the
effective config is preset ⊕ overrides with subject as topic.topic. These
tests cover: jobs CRUD with the v2 GET shape, PUT overrides round-trip,
preset CRUD/clone/default, scraper toggles via override, the multi-folder
local_imports list, the shrunk global settings, the pipeline-start gate, and
?job= scoping.

The dashboard resolves PIPELINE_QUEUE / PIPELINE_SORTED / the SQLite index path
at IMPORT time, so the fixture sets PIPELINE_BASE_DIR first, then reloads the
storage modules and the dashboard so everything lands under a temp dir.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

import pytest

PIPELINE_CODE = Path(__file__).resolve().parent.parent / "pipeline_code"
if str(PIPELINE_CODE) not in sys.path:
    sys.path.insert(0, str(PIPELINE_CODE))

# The dashboard calls dotenv.load_dotenv() at module import, which reads the
# repo's real .env into os.environ. That would (a) leak the developer's config
# into our temp store and (b) pollute the process environment for sibling test
# files. Neutralise it for the whole session and strip any .env-sourced keys a
# prior import may already have set. Tests configure what they need explicitly.
try:
    import dotenv as _dotenv
    _dotenv.load_dotenv = lambda *a, **k: False  # type: ignore[assignment]
except Exception:  # pragma: no cover - dotenv always present in this project
    pass

_DOTENV_SOURCED_KEYS = (
    "PIPELINE_SLUG", "PIPELINE_TOPIC", "TOPIC_KEYWORDS_EXTRA",
    "TOPIC_BANNED_KEYWORDS", "TOPIC_GENERATION_HINTS", "SCRAPER_DISABLED",
    "X_ACCOUNTS", "REDDIT_SUBREDDITS", "GALLERY_DL_URLS", "GALLERY_DL_ENABLED",
    "VISION_OVR_MIN_SCORE", "VISION_REL_MIN_SCORE", "AUTO_CAPTION_ENABLED",
    "MIN_PROMPT_LENGTH", "REQUIRE_PROMPT", "LOCAL_IMPORT_DIR",
)
for _k in _DOTENV_SOURCED_KEYS:
    os.environ.pop(_k, None)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """A Flask test client whose storage is isolated under tmp_path.

    Hands back (test_client, job_config_module, tmp_path). The fixture wipes the
    auto-migrated default job so each test starts from a blank store.
    """
    monkeypatch.setenv("PIPELINE_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("PIPELINE_QUEUE", str(tmp_path / "queue"))
    monkeypatch.setenv("PIPELINE_SORTED", str(tmp_path / "sorted"))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    for leaked in _DOTENV_SOURCED_KEYS:
        monkeypatch.delenv(leaked, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("BLUR_NSFW_THUMBS=true\n", encoding="utf-8")
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

    # Wipe the auto-migrated default job + presets so tests start blank.
    jobs_dir = tmp_path / "jobs"
    if jobs_dir.is_dir():
        for f in jobs_dir.glob("*"):
            f.unlink()
    return dashboard.app.test_client(), job_config, tmp_path


# ── presets CRUD ─────────────────────────────────────────────────────────────

def test_presets_list_seeds_default(client):
    c, _jc, _tmp = client
    body = c.get("/api/presets").get_json()
    assert body["default"] == "default"
    assert "default" in body["presets"]


def test_create_preset_from_default(client):
    c, _jc, _tmp = client
    r = c.post("/api/presets", json={"name": "Cars", "base_on": "default"})
    assert r.status_code == 201
    assert r.get_json()["name"] == "Cars"
    assert "Cars" in c.get("/api/presets").get_json()["presets"]


def test_create_preset_duplicate_is_400(client):
    c, _jc, _tmp = client
    c.post("/api/presets", json={"name": "Dup"})
    assert c.post("/api/presets", json={"name": "Dup"}).status_code == 400


def test_create_preset_invalid_name_is_400(client):
    c, _jc, _tmp = client
    assert c.post("/api/presets", json={"name": "bad/name!"}).status_code == 400


def test_preset_get_update_round_trip(client):
    c, _jc, _tmp = client
    c.post("/api/presets", json={"name": "Edit"})
    cfg = c.get("/api/presets/Edit").get_json()["cfg"]
    cfg["scoring"] = {"ovr_min": 72, "rel_min": 61, "notes": "n"}
    cfg["categories"] = [{"name": "Keepers", "hint": ""}]
    r = c.put("/api/presets/Edit", json={"cfg": cfg})
    assert r.status_code == 200
    reread = c.get("/api/presets/Edit").get_json()["cfg"]
    assert reread["scoring"]["ovr_min"] == 72
    assert [cat["name"] for cat in reread["categories"]] == ["Keepers"]


def test_preset_update_rejects_bad_category(client):
    c, _jc, _tmp = client
    c.post("/api/presets", json={"name": "Vuln"})
    cfg = c.get("/api/presets/Vuln").get_json()["cfg"]
    cfg["categories"] = [{"name": "../../etc", "hint": ""}]
    assert c.put("/api/presets/Vuln", json={"cfg": cfg}).status_code == 400


def test_preset_update_rejects_out_of_range_scoring(client):
    c, _jc, _tmp = client
    c.post("/api/presets", json={"name": "Score"})
    cfg = c.get("/api/presets/Score").get_json()["cfg"]
    cfg["scoring"] = {"ovr_min": 500, "rel_min": 0, "notes": ""}
    assert c.put("/api/presets/Score", json={"cfg": cfg}).status_code == 400


def test_preset_clone(client):
    c, _jc, _tmp = client
    c.post("/api/presets", json={"name": "Src"})
    r = c.post("/api/presets/Src/clone", json={"name": "SrcCopy"})
    assert r.status_code == 201
    assert "SrcCopy" in c.get("/api/presets").get_json()["presets"]


def test_set_default_preset(client):
    c, jc, _tmp = client
    c.post("/api/presets", json={"name": "NewDefault"})
    r = c.post("/api/presets/default", json={"name": "NewDefault"})
    assert r.status_code == 200
    assert jc.default_preset_name() == "NewDefault"


def test_delete_default_preset_is_409(client):
    c, _jc, _tmp = client
    assert c.delete("/api/presets/default").status_code == 409


def test_delete_referenced_preset_is_409(client):
    c, _jc, _tmp = client
    c.post("/api/presets", json={"name": "Used"})
    c.post("/api/jobs", json={"name": "Job On Used", "preset": "Used"})
    assert c.delete("/api/presets/Used").status_code == 409


def test_delete_unreferenced_preset_ok(client):
    c, _jc, _tmp = client
    c.post("/api/presets", json={"name": "Orphan"})
    assert c.delete("/api/presets/Orphan").status_code == 200


# ── jobs v2 CRUD + override round-trip ───────────────────────────────────────

def test_create_job_with_subject_and_preset(client):
    c, _jc, _tmp = client
    c.post("/api/presets", json={"name": "P1"})
    r = c.post("/api/jobs", json={"name": "Sedans", "subject": "Red Sedans", "preset": "P1"})
    assert r.status_code == 201
    detail = r.get_json()["job"]
    assert detail["job"]["slug"] == "sedans"
    assert detail["job"]["subject"] == "Red Sedans"
    assert detail["job"]["preset"] == "P1"


def test_create_job_unknown_preset_is_400(client):
    c, _jc, _tmp = client
    assert c.post("/api/jobs", json={"name": "X", "preset": "ghost"}).status_code == 400


def test_job_get_v2_shape(client):
    c, _jc, _tmp = client
    c.post("/api/jobs", json={"name": "Shape", "subject": "A Subject"})
    body = c.get("/api/jobs/shape").get_json()
    assert set(body.keys()) >= {"job", "effective", "overrides", "presets", "default_preset"}
    assert body["job"]["subject"] == "A Subject"
    assert body["effective"]["topic"]["topic"] == "A Subject"   # subject → topic.topic
    assert body["overrides"] == {}
    assert "default" in body["presets"]


def test_job_put_override_round_trip(client):
    c, jc, _tmp = client
    c.post("/api/jobs", json={"name": "Ov"})
    r = c.put("/api/jobs/ov", json={"overrides": {
        "scoring": {"ovr_min": 88, "rel_min": 50, "notes": ""},
        "scrapers": {"x_accounts": ["acct1"]},
    }})
    assert r.status_code == 200
    body = c.get("/api/jobs/ov").get_json()
    assert body["effective"]["scoring"]["ovr_min"] == 88       # override wins over preset
    assert body["effective"]["scrapers"]["x_accounts"] == ["acct1"]
    assert jc.is_overridden(jc.get_job("ov"), "scoring.ovr_min")


def test_job_put_subject_and_preset(client):
    c, _jc, _tmp = client
    c.post("/api/presets", json={"name": "P2"})
    c.post("/api/jobs", json={"name": "Sw"})
    r = c.put("/api/jobs/sw", json={"subject": "New Subject", "preset": "P2"})
    assert r.status_code == 200
    body = c.get("/api/jobs/sw").get_json()
    assert body["job"]["subject"] == "New Subject"
    assert body["job"]["preset"] == "P2"
    assert body["effective"]["topic"]["topic"] == "New Subject"


def test_job_put_rejects_unknown_preset(client):
    c, _jc, _tmp = client
    c.post("/api/jobs", json={"name": "Pj"})
    assert c.put("/api/jobs/pj", json={"preset": "ghost"}).status_code == 400


def test_job_put_rejects_bad_override_category(client):
    c, jc, _tmp = client
    c.post("/api/jobs", json={"name": "Badcat"})
    r = c.put("/api/jobs/badcat", json={"overrides": {"categories": [{"name": "DISCARD"}]}})
    assert r.status_code == 400
    assert not jc.is_overridden(jc.get_job("badcat"), "categories")


def test_job_put_rejects_out_of_range_scoring(client):
    c, _jc, _tmp = client
    c.post("/api/jobs", json={"name": "Soj"})
    r = c.put("/api/jobs/soj", json={"overrides": {"scoring": {"ovr_min": 250}}})
    assert r.status_code == 400


def test_job_put_does_not_auto_activate(client):
    """Apply-on-leave model: PUT writes the file but must NOT bump _index.json
    (no re-project/restart of the active job on every keystroke)."""
    c, jc, _tmp = client
    c.post("/api/jobs", json={"name": "NoApply"})
    c.post("/api/jobs/noapply/activate")
    index_path = jc.jobs_dir() / "_index.json"
    before = index_path.stat().st_mtime_ns
    import time
    time.sleep(0.01)
    c.put("/api/jobs/noapply", json={"overrides": {"scoring": {"ovr_min": 10}}})
    after = index_path.stat().st_mtime_ns
    assert after == before, "PUT must not touch _index.json (apply happens on leave)"


def test_reset_override_via_empty_overrides(client):
    c, jc, _tmp = client
    c.post("/api/jobs", json={"name": "Reset"})
    c.put("/api/jobs/reset", json={"overrides": {"scoring": {"ovr_min": 99}}})
    assert jc.is_overridden(jc.get_job("reset"), "scoring.ovr_min")
    # The UI removes the override leaf and PUTs the trimmed map.
    c.put("/api/jobs/reset", json={"overrides": {}})
    assert not jc.is_overridden(jc.get_job("reset"), "scoring.ovr_min")


def test_clone_job_copies_preset_and_overrides(client):
    c, _jc, _tmp = client
    c.post("/api/presets", json={"name": "CP"})
    c.post("/api/jobs", json={"name": "Origin", "preset": "CP"})
    c.put("/api/jobs/origin", json={"overrides": {"scoring": {"ovr_min": 44}}})
    r = c.post("/api/jobs/origin/clone", json={"name": "Origin Copy"})
    assert r.status_code == 201
    body = c.get("/api/jobs/origin_copy").get_json()
    assert body["job"]["preset"] == "CP"
    assert body["effective"]["scoring"]["ovr_min"] == 44


def test_list_jobs_shape(client):
    c, _jc, _tmp = client
    c.post("/api/jobs", json={"name": "Alpha"})
    c.post("/api/jobs", json={"name": "Beta"})
    body = c.get("/api/jobs").get_json()
    assert {j["slug"] for j in body["jobs"]} == {"alpha", "beta"}
    card = body["jobs"][0]
    assert set(card.keys()) >= {"slug", "name", "status", "is_active", "queued", "sorted"}


def test_delete_refuses_active_409(client):
    c, jc, _tmp = client
    c.post("/api/jobs", json={"name": "Keep"})
    c.post("/api/jobs/keep/activate")
    assert c.delete("/api/jobs/keep").status_code == 409
    assert jc.get_job("keep") is not None


def test_create_reserved_slug_is_400(client):
    c, _jc, _tmp = client
    for name in ("Queue", "advance"):
        r = c.post("/api/jobs", json={"name": name})
        assert r.status_code == 400
        assert "reserved" in r.get_json()["error"]


# ── scraper toggles via override (v2) ────────────────────────────────────────

def test_scraper_toggle_writes_override(client):
    c, jc, _tmp = client
    c.post("/api/jobs", json={"name": "Scr"})
    c.post("/api/jobs/scr/activate")
    r = c.post("/api/scrapers/toggle?job=scr", json={"name": "Web", "enabled": False})
    assert r.status_code == 200
    job = jc.get_job("scr")
    assert jc.is_overridden(job, "scrapers.enabled")
    assert jc.effective_config(job)["scrapers"]["enabled"]["Web"] is False
    # GET reflects it.
    rows = {s["name"]: s["enabled"] for s in c.get("/api/scrapers?job=scr").get_json()}
    assert rows["Web"] is False


def test_scraper_bulk_disable_writes_override(client):
    c, jc, _tmp = client
    c.post("/api/jobs", json={"name": "Bulk"})
    c.post("/api/scrapers/bulk?job=bulk", json={"action": "disable_all"})
    job = jc.get_job("bulk")
    enabled = jc.effective_config(job)["scrapers"]["enabled"]
    assert all(v is False for v in enabled.values())


def test_scraper_toggle_rejects_local_row(client):
    c, _jc, _tmp = client
    c.post("/api/jobs", json={"name": "Loc"})
    c.post("/api/jobs/loc/activate")
    # Local-<name> rows are read-only; toggling one is rejected.
    r = c.post("/api/scrapers/toggle?job=loc", json={"name": "Local-foo", "enabled": True})
    assert r.status_code == 400


def test_scrapers_list_has_no_local_folder_entry(client):
    c, _jc, _tmp = client
    c.post("/api/jobs", json={"name": "NoLocal"})
    c.post("/api/jobs/nolocal/activate")
    names = [s["name"] for s in c.get("/api/scrapers?job=nolocal").get_json()]
    assert all("local" not in n.lower() for n in names)
    assert len([n for n in names if n in ("X.com", "Discord-1", "Civitai-Com",
                                          "Civitai-Red", "Web", "Gallery-DL")]) == 6


# ── multi-folder local imports ───────────────────────────────────────────────

def test_local_imports_list_via_put(client):
    c, jc, _tmp = client
    c.post("/api/jobs", json={"name": "Multi"})
    c.post("/api/jobs/multi/activate")
    r = c.put("/api/jobs/multi", json={"overrides": {"scrapers": {"local_imports": [
        {"name": "ds1", "dir": "/d1", "enabled": True, "migrate_from": ""},
        {"name": "ds2", "dir": "/d2", "enabled": False, "migrate_from": ""},
    ]}}})
    assert r.status_code == 200
    # Scrapers tab surfaces one read-only Local-<name> row per folder.
    rows = c.get("/api/scrapers?job=multi").get_json()
    names = [s["name"] for s in rows]
    assert "Local-ds1" in names and "Local-ds2" in names
    ds1 = next(s for s in rows if s["name"] == "Local-ds1")
    assert ds1["kind"] == "local" and ds1["enabled"] is True
    # resolve_env emits LOCAL_IMPORTS_JSON (enabled only).
    env = jc.resolve_env(jc.get_job("multi"))
    assert "LOCAL_IMPORTS_JSON" in env
    assert [f["name"] for f in json.loads(env["LOCAL_IMPORTS_JSON"])] == ["ds1"]


def test_local_import_name_rejects_path_component(client):
    """local_imports[].name becomes a Local-<name> queue subfolder + SeenStore
    key, so a name with a separator / traversal must be rejected."""
    c, jc, _tmp = client
    c.post("/api/jobs", json={"name": "Liname"})
    for bad in ("../evil", "a/b", "a\\b", "with space", "", "a" * 41):
        r = c.put("/api/jobs/liname", json={"overrides": {"scrapers": {"local_imports": [
            {"name": bad, "dir": "/d", "enabled": True, "migrate_from": ""}]}}})
        assert r.status_code == 400, bad
    assert not jc.is_overridden(jc.get_job("liname"), "scrapers.local_imports")
    # A clean slug name is accepted.
    ok = c.put("/api/jobs/liname", json={"overrides": {"scrapers": {"local_imports": [
        {"name": "my_set-1", "dir": "/d", "enabled": True, "migrate_from": ""}]}}})
    assert ok.status_code == 200


def test_preset_local_import_name_rejects_path_component(client):
    c, _jc, _tmp = client
    c.post("/api/presets", json={"name": "LiPreset"})
    cfg = c.get("/api/presets/LiPreset").get_json()["cfg"]
    cfg["scrapers"]["local_imports"] = [{"name": "../x", "dir": "/d", "enabled": True, "migrate_from": ""}]
    assert c.put("/api/presets/LiPreset", json={"cfg": cfg}).status_code == 400


def test_scoring_notes_length_capped(client):
    c, _jc, _tmp = client
    c.post("/api/jobs", json={"name": "Notes"})
    r = c.put("/api/jobs/notes", json={"overrides": {"scoring": {"notes": "x" * 2001}}})
    assert r.status_code == 400
    # 2000 is the limit and is accepted.
    ok = c.put("/api/jobs/notes", json={"overrides": {"scoring": {"notes": "x" * 2000}}})
    assert ok.status_code == 200


def test_categories_count_capped_server_side(client):
    """A direct PUT can't exceed the server category cap even though the client
    also limits it."""
    c, _jc, _tmp = client
    c.post("/api/jobs", json={"name": "Manycats"})
    many = [{"name": f"Cat{i}", "hint": ""} for i in range(13)]
    r = c.put("/api/jobs/manycats", json={"overrides": {"categories": many}})
    assert r.status_code == 400


# ── sparse scraper-enabled override + clobber-safety ─────────────────────────

def test_toggle_stores_only_toggled_key(client):
    """Toggling ONE scraper stores exactly that scraper's key under
    overrides.scrapers.enabled — the other five stay inherited (no full map)."""
    c, jc, _tmp = client
    c.post("/api/jobs", json={"name": "Sparse"})
    c.post("/api/jobs/sparse/activate")
    r = c.post("/api/scrapers/toggle?job=sparse", json={"name": "Web", "enabled": False})
    assert r.status_code == 200
    job = jc.get_job("sparse")
    enabled_override = job.overrides["scrapers"]["enabled"]
    assert enabled_override == {"Web": False}          # ONLY the toggled key
    # The other scrapers are NOT overridden (still inherit → 'global' chip).
    assert not jc.is_overridden(job, "scrapers.enabled.X.com")
    # The toggle response echoes the fresh sparse overrides for the client.
    assert r.get_json()["overrides"]["scrapers"]["enabled"] == {"Web": False}


def test_second_toggle_merges_into_existing_override(client):
    c, jc, _tmp = client
    c.post("/api/jobs", json={"name": "Sparse2"})
    c.post("/api/scrapers/toggle?job=sparse2", json={"name": "Web", "enabled": False})
    c.post("/api/scrapers/toggle?job=sparse2", json={"name": "X.com", "enabled": False})
    enabled_override = jc.get_job("sparse2").overrides["scrapers"]["enabled"]
    assert enabled_override == {"Web": False, "X.com": False}


def test_toggle_then_stale_put_does_not_lose_toggle(client):
    """Simulate the clobber race: a job-editor PUT carrying the PRE-toggle
    overrides arrives after a toggle. The merge model means the toggle survives
    only if the client resends merged overrides — the server stores exactly what
    the latest PUT sends, so the test asserts the documented client contract:
    the toggle response returns fresh overrides the client must carry forward."""
    c, jc, _tmp = client
    c.post("/api/jobs", json={"name": "Race"})
    # 1) Toggle Web off → sparse override {Web: False}; capture the fresh map.
    tog = c.post("/api/scrapers/toggle?job=race", json={"name": "Web", "enabled": False}).get_json()
    fresh = tog["overrides"]
    assert fresh["scrapers"]["enabled"] == {"Web": False}
    # 2) A subsequent job PUT that carries the fresh overrides (as the JS now
    #    does after resyncing je.overrides) keeps the toggle.
    fresh.setdefault("scoring", {})["ovr_min"] = 30
    c.put("/api/jobs/race", json={"overrides": fresh})
    job = jc.get_job("race")
    assert job.overrides["scrapers"]["enabled"] == {"Web": False}    # toggle preserved
    assert job.overrides["scoring"]["ovr_min"] == 30


# ── settings shrunk to global keys ───────────────────────────────────────────

def test_settings_get_excludes_per_job_keys(client):
    c, _jc, _tmp = client
    body = c.get("/api/settings").get_json()
    assert "GROQ_API_KEY" in body and "PIPELINE_BASE_DIR" in body
    for moved in ("PIPELINE_TOPIC", "TOPIC_KEYWORDS_EXTRA", "X_ACCOUNTS",
                  "REDDIT_SUBREDDITS", "GALLERY_DL_URLS", "AUTO_CAPTION_ENABLED",
                  "VISION_OVR_MIN_SCORE", "DISCORD_CHANNELS_JSON",
                  "LOCAL_IMPORT_DIR"):
        assert moved not in body


def test_settings_post_rejects_per_job_key(client):
    c, _jc, _tmp = client
    r = c.post("/api/settings", json={"PIPELINE_TOPIC": "nope"})
    assert r.status_code == 400


# ── pipeline start gate + ?job= scoping ──────────────────────────────────────

def test_pipeline_start_refused_without_active_job(client):
    c, jc, _tmp = client
    assert jc.get_active_slug() is None
    r = c.post("/api/pipeline/start")
    assert r.status_code == 409


def test_status_reports_active_and_scoped_job(client):
    c, _jc, _tmp = client
    c.post("/api/jobs", json={"name": "Scoped"})
    c.post("/api/jobs/scoped/activate")
    body = c.get("/api/status").get_json()
    assert body["active_job"] == "scoped"
    assert body["job"] == "scoped"
    other = c.get("/api/status?job=somethingelse").get_json()
    assert other["job"] == "somethingelse"
    assert other["queue"]["total"] == 0 and other["sorted"]["total"] == 0


def test_status_malformed_job_param_falls_back_to_active(client):
    c, _jc, _tmp = client
    c.post("/api/jobs", json={"name": "Fallback"})
    c.post("/api/jobs/fallback/activate")
    body = c.get("/api/status?job=Bad-Slug").get_json()
    assert body["job"] == "fallback"


# ── structural validation of inheritable config (defense in depth) ───────────

def test_put_rejects_traversal_local_dir(client):
    c, jc, _tmp = client
    c.post("/api/jobs", json={"name": "Litrav"})
    r = c.put("/api/jobs/litrav", json={"overrides": {"scrapers": {"local_imports": [
        {"name": "x", "dir": "../../etc", "enabled": True, "migrate_from": ""}]}}})
    assert r.status_code == 400
    assert not jc.is_overridden(jc.get_job("litrav"), "scrapers.local_imports")


def test_put_rejects_unknown_caption_style(client):
    c, _jc, _tmp = client
    c.post("/api/jobs", json={"name": "Capsty"})
    r = c.put("/api/jobs/capsty", json={"overrides": {"captioning": {"style": "haiku"}}})
    assert r.status_code == 400


def test_put_rejects_non_bool_scrapers_enabled(client):
    c, _jc, _tmp = client
    c.post("/api/jobs", json={"name": "Enbad"})
    r = c.put("/api/jobs/enbad", json={"overrides": {"scrapers": {"enabled": {"Web": "yes"}}}})
    assert r.status_code == 400


def test_put_rejects_unknown_override_key(client):
    c, _jc, _tmp = client
    c.post("/api/jobs", json={"name": "Unk"})
    r = c.put("/api/jobs/unk", json={"overrides": {"bogus_section": {"x": 1}}})
    assert r.status_code == 400


def test_preset_put_deep_merge_keeps_nested(client):
    """A partial preset PUT that touches scrapers.enabled must NOT wipe the
    sibling scrapers.local_imports / gallery_dl (deep merge, not shallow)."""
    c, _jc, _tmp = client
    c.post("/api/presets", json={"name": "Deep"})
    cfg = c.get("/api/presets/Deep").get_json()["cfg"]
    # Seed a local folder + categories (categories required for full PUT).
    cfg["scrapers"]["local_imports"] = [{"name": "keepme", "dir": "/k", "enabled": True, "migrate_from": ""}]
    c.put("/api/presets/Deep", json={"cfg": cfg})
    # Now PUT a body that includes scrapers.enabled (and categories) but NOT
    # local_imports — the seeded folder must survive.
    cur = c.get("/api/presets/Deep").get_json()["cfg"]
    body = {"categories": cur["categories"], "scrapers": {"enabled": {"Web": False,
            "X.com": True, "Discord-1": True, "Civitai-Com": True, "Civitai-Red": True, "Gallery-DL": False}}}
    r = c.put("/api/presets/Deep", json={"cfg": body})
    assert r.status_code == 200
    after = c.get("/api/presets/Deep").get_json()["cfg"]
    assert [f["name"] for f in after["scrapers"]["local_imports"]] == ["keepme"]
    assert after["scrapers"]["enabled"]["Web"] is False


# ── force delete (data dirs, guarded by safe_inside) ─────────────────────────

def test_force_delete_removes_data_dirs(client):
    c, jc, _tmp = client
    c.post("/api/jobs", json={"name": "Trash"})
    import paths
    qd = paths.queue_dir("trash")
    sd = paths.sorted_dir("trash")
    (qd / "civitai").mkdir(parents=True, exist_ok=True)
    (sd / "Keepers").mkdir(parents=True, exist_ok=True)
    r = c.delete("/api/jobs/trash?force=1")
    assert r.status_code == 200
    assert not qd.exists() and not sd.exists()
    assert jc.get_job("trash") is None


# ── scraper connectivity / auth test endpoint ────────────────────────────────

def test_scrapers_test_rejects_unsupported(client):
    c, _jc, _tmp = client
    r = c.post("/api/scrapers/test", json={"scraper": "Nope"})
    assert r.status_code == 400
    assert r.get_json()["ok"] is False


def test_scrapers_test_returns_module_result(client, monkeypatch):
    c, _jc, _tmp = client
    import scraper_test
    monkeypatch.setattr(scraper_test, "test_scraper",
                        lambda name, config=None, env=None: {
                            "ok": True, "message": "authenticated OK",
                            "latency_ms": 12, "detail": name})
    r = c.post("/api/scrapers/test", json={"scraper": "Civitai-Com"})
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True and body["detail"] == "Civitai-Com"


def test_scrapers_test_typed_value_used_mask_falls_back(client, monkeypatch):
    """A freshly typed cred is tested as-is; a masked secret falls back to the
    stored env value so an untouched field still tests the real credential."""
    c, _jc, _tmp = client
    import scraper_test
    seen = {}

    def _capture(name, config=None, env=None):
        seen["env"] = env
        return {"ok": True, "message": "ok", "latency_ms": 1, "detail": ""}

    monkeypatch.setattr(scraper_test, "test_scraper", _capture)
    monkeypatch.setenv("CIVITAI_API_KEY", "REAL_STORED_KEY")
    r = c.post("/api/scrapers/test", json={
        "scraper": "Civitai-Com",
        "settings": {"CIVITAI_API_RED_KEY": "TYPED_NEW", "CIVITAI_API_KEY": "********"},
    })
    assert r.status_code == 200
    env = seen["env"]
    assert env["CIVITAI_API_RED_KEY"] == "TYPED_NEW"       # typed value passed through
    assert env["CIVITAI_API_KEY"] == "REAL_STORED_KEY"     # mask fell back to stored


def test_scrapers_test_never_500s_on_bad_body(client):
    c, _jc, _tmp = client
    r = c.post("/api/scrapers/test", json={})
    assert r.status_code == 400


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
