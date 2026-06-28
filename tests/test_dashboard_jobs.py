"""Flask test-client tests for the dashboard's jobs model surface.

Runnable two ways:
    pytest tests/test_dashboard_jobs.py
    python tests/test_dashboard_jobs.py        # falls back to pytest.main

Covers the jobs CRUD routes added in Phase B2, plus the per-job scoping of the
scraper toggles, the per-job categories editor, the shrunk global settings,
the pipeline-start gate, and ?job= scoping defaulting to the active job.

The dashboard resolves PIPELINE_QUEUE / PIPELINE_SORTED / the SQLite index path
at IMPORT time, so the fixture sets PIPELINE_BASE_DIR first, then reloads the
storage modules and the dashboard so everything lands under a temp dir.
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

# The dashboard calls dotenv.load_dotenv() at module import, which reads the
# repo's real .env into os.environ (PIPELINE_SLUG, credentials, …). That would
# (a) leak the developer's config into our temp store and (b) pollute the
# process environment for sibling test files (e.g. test_job_config.py asserts
# migration produces a `default` slug, which only holds when PIPELINE_SLUG is
# absent). Neutralise it for the whole test session: make load_dotenv a no-op
# and strip any .env-sourced keys a prior import may already have set. Tests
# configure every var they need explicitly via monkeypatch.
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
    "MIN_PROMPT_LENGTH", "REQUIRE_PROMPT",
)
for _k in _DOTENV_SOURCED_KEYS:
    os.environ.pop(_k, None)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """A Flask test client whose storage is isolated under tmp_path.

    Hands back (test_client, job_config_module, tmp_path). The dashboard's
    import-time migration runs against the (empty) temp store, so no jobs exist
    until a test creates one — letting us assert the pre-migration fallbacks too
    where relevant.
    """
    monkeypatch.setenv("PIPELINE_BASE_DIR", str(tmp_path))
    # Pin queue/sorted/log roots under the temp dir; the dashboard reads these
    # env vars at import to build its module-level Path constants.
    monkeypatch.setenv("PIPELINE_QUEUE", str(tmp_path / "queue"))
    monkeypatch.setenv("PIPELINE_SORTED", str(tmp_path / "sorted"))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    # The repo .env (loaded via load_dotenv at first import) leaks per-job env
    # vars like PIPELINE_SLUG/PIPELINE_TOPIC into os.environ. Clear them so the
    # import-time migration can't adopt the developer's real slug into our temp
    # store — tests then start from a clean, predictable empty store.
    for leaked in ("PIPELINE_SLUG", "PIPELINE_TOPIC", "TOPIC_KEYWORDS_EXTRA",
                   "TOPIC_BANNED_KEYWORDS", "TOPIC_GENERATION_HINTS",
                   "SCRAPER_DISABLED", "X_ACCOUNTS", "REDDIT_SUBREDDITS",
                   "GALLERY_DL_URLS", "GALLERY_DL_ENABLED", "VISION_OVR_MIN_SCORE",
                   "VISION_REL_MIN_SCORE", "AUTO_CAPTION_ENABLED"):
        monkeypatch.delenv(leaked, raising=False)
    # A throwaway .env so update_env() has a file to write to (global settings).
    env_file = tmp_path / ".env"
    env_file.write_text("BLUR_NSFW_THUMBS=true\n", encoding="utf-8")
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))

    # categories.ACTIVE_PATH is computed at import from PIPELINE_BASE_DIR; redirect
    # it and bust its cache so taxonomy projection lands in the temp dir.
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

    # The dashboard's import-time migrate_env_to_default_job() seeds a `default`
    # job (and activates it) when the store is empty. For test determinism we
    # want a blank store: each test creates exactly the jobs it asserts on.
    # Wipe the jobs dir so we start from nothing. (Migration is idempotent and
    # only fires at import, so it won't re-run mid-test.)
    jobs_dir = tmp_path / "jobs"
    if jobs_dir.is_dir():
        for f in jobs_dir.glob("*"):
            f.unlink()

    return dashboard.app.test_client(), job_config, tmp_path


# ── jobs CRUD ────────────────────────────────────────────────────────────────

def test_create_job_returns_201_and_slug(client):
    c, _jc, _tmp = client
    r = c.post("/api/jobs", json={"name": "Female Influencer"})
    assert r.status_code == 201
    body = r.get_json()
    assert body["success"] is True
    assert body["job"]["slug"] == "female_influencer"
    assert body["job"]["name"] == "Female Influencer"


def test_create_duplicate_job_is_400(client):
    c, _jc, _tmp = client
    assert c.post("/api/jobs", json={"name": "Dup"}).status_code == 201
    r = c.post("/api/jobs", json={"name": "Dup"})
    assert r.status_code == 400
    assert "error" in r.get_json()


def test_create_job_empty_name_is_400(client):
    c, _jc, _tmp = client
    r = c.post("/api/jobs", json={"name": "   "})
    assert r.status_code == 400


def test_list_jobs_shape(client):
    c, _jc, _tmp = client
    c.post("/api/jobs", json={"name": "Alpha"})
    c.post("/api/jobs", json={"name": "Beta"})
    r = c.get("/api/jobs")
    assert r.status_code == 200
    body = r.get_json()
    assert set(body.keys()) >= {"active", "queue", "jobs", "pipeline_running"}
    slugs = {j["slug"] for j in body["jobs"]}
    assert slugs == {"alpha", "beta"}
    # Each card carries the contract fields used by the grid.
    card = body["jobs"][0]
    assert set(card.keys()) >= {
        "slug", "name", "status", "is_active", "queue_position", "queued", "sorted",
    }
    assert card["queued"] == 0 and card["sorted"] == 0


def test_get_and_update_round_trip(client):
    c, _jc, _tmp = client
    c.post("/api/jobs", json={"name": "Round Trip"})
    got = c.get("/api/jobs/round_trip")
    assert got.status_code == 200
    cfg = got.get_json()
    assert cfg["slug"] == "round_trip"

    # Patch a couple of fields; slug must remain immutable.
    cfg["name"] = "Renamed"
    cfg["scoring"] = {"ovr_min": 61, "rel_min": 50, "notes": "be picky"}
    cfg["slug"] = "hacked_slug"   # should be ignored
    put = c.put("/api/jobs/round_trip", json=cfg)
    assert put.status_code == 200
    saved = put.get_json()["job"]
    assert saved["slug"] == "round_trip"
    assert saved["name"] == "Renamed"
    assert saved["scoring"]["ovr_min"] == 61

    reread = c.get("/api/jobs/round_trip").get_json()
    assert reread["name"] == "Renamed"
    assert reread["scoring"]["ovr_min"] == 61


def test_get_missing_job_is_404(client):
    c, _jc, _tmp = client
    assert c.get("/api/jobs/does_not_exist").status_code == 404


def test_get_job_invalid_slug_is_400(client):
    c, _jc, _tmp = client
    # Flask routes won't match a slug with a slash, so use an otherwise-routable
    # but JOB_SLUG_RE-invalid slug.
    assert c.get("/api/jobs/Bad-Slug").status_code == 400


def test_activate_sets_active(client):
    c, jc, _tmp = client
    c.post("/api/jobs", json={"name": "To Activate"})
    r = c.post("/api/jobs/to_activate/activate")
    assert r.status_code == 200
    assert r.get_json()["active"] == "to_activate"
    assert jc.get_active_slug() == "to_activate"


def test_delete_refuses_active_with_409(client):
    c, jc, _tmp = client
    c.post("/api/jobs", json={"name": "Keep Me"})
    c.post("/api/jobs/keep_me/activate")
    r = c.delete("/api/jobs/keep_me")
    assert r.status_code == 409
    assert jc.get_job("keep_me") is not None


def test_delete_non_active_succeeds(client):
    c, jc, _tmp = client
    c.post("/api/jobs", json={"name": "Disposable"})
    r = c.delete("/api/jobs/disposable")
    assert r.status_code == 200
    assert jc.get_job("disposable") is None


def test_clone_creates_new_job_from_base(client):
    c, _jc, _tmp = client
    c.post("/api/jobs", json={"name": "Source"})
    # Give the source a distinctive scoring value, then clone it.
    src = c.get("/api/jobs/source").get_json()
    src["scoring"] = {"ovr_min": 77, "rel_min": 50, "notes": ""}
    c.put("/api/jobs/source", json=src)
    r = c.post("/api/jobs/source/clone", json={"name": "Source Copy"})
    assert r.status_code == 201
    clone = r.get_json()["job"]
    assert clone["slug"] == "source_copy"
    assert clone["scoring"]["ovr_min"] == 77


def test_queue_enqueue_dequeue_and_order(client):
    c, jc, _tmp = client
    for n in ("One", "Two", "Three"):
        c.post("/api/jobs", json={"name": n})
    assert c.post("/api/jobs/one/enqueue").status_code == 200
    assert c.post("/api/jobs/two/enqueue").status_code == 200
    assert c.post("/api/jobs/three/enqueue").status_code == 200
    assert jc.get_index()["queue"] == ["one", "two", "three"]
    assert c.post("/api/jobs/two/dequeue").status_code == 200
    assert jc.get_index()["queue"] == ["one", "three"]
    r = c.post("/api/jobs/queue", json={"order": ["three", "one"]})
    assert r.status_code == 200
    assert jc.get_index()["queue"] == ["three", "one"]


def test_set_queue_rejects_unknown_slug(client):
    c, _jc, _tmp = client
    c.post("/api/jobs", json={"name": "Real"})
    r = c.post("/api/jobs/queue", json={"order": ["real", "ghost"]})
    assert r.status_code == 400


def test_advance_promotes_next_queued(client):
    c, jc, _tmp = client
    c.post("/api/jobs", json={"name": "First"})
    c.post("/api/jobs", json={"name": "Second"})
    c.post("/api/jobs/first/activate")
    c.post("/api/jobs/second/enqueue")
    r = c.post("/api/jobs/advance")
    assert r.status_code == 200
    assert r.get_json()["active"] == "second"
    assert jc.get_active_slug() == "second"


# ── per-job scraper toggles ──────────────────────────────────────────────────

def test_scraper_toggle_persists_to_job(client):
    c, jc, _tmp = client
    c.post("/api/jobs", json={"name": "Scraper Job"})
    c.post("/api/jobs/scraper_job/activate")
    # Disable X.com for this job.
    r = c.post("/api/scrapers/toggle?job=scraper_job",
               json={"name": "X.com", "enabled": False})
    assert r.status_code == 200
    assert r.get_json()["enabled"] is False
    # The job file now records X.com disabled.
    job = jc.get_job("scraper_job")
    assert job.scrapers["enabled"]["X.com"] is False
    # And the GET reflects it.
    listed = c.get("/api/scrapers?job=scraper_job").get_json()
    by_name = {s["name"]: s["enabled"] for s in listed}
    assert by_name["X.com"] is False
    assert by_name["Web"] is True


def test_scraper_bulk_disable_all_writes_job(client):
    c, jc, _tmp = client
    c.post("/api/jobs", json={"name": "Bulk Job"})
    c.post("/api/jobs/bulk_job/activate")
    r = c.post("/api/scrapers/bulk?job=bulk_job", json={"action": "disable_all"})
    assert r.status_code == 200
    job = jc.get_job("bulk_job")
    assert all(v is False for v in job.scrapers["enabled"].values())


def test_scraper_toggle_defaults_to_active_job(client):
    c, jc, _tmp = client
    c.post("/api/jobs", json={"name": "Active One"})
    c.post("/api/jobs/active_one/activate")
    # No ?job= → defaults to active.
    c.post("/api/scrapers/toggle", json={"name": "Web", "enabled": False})
    assert jc.get_job("active_one").scrapers["enabled"]["Web"] is False


# ── per-job categories ───────────────────────────────────────────────────────

def test_categories_get_and_post_scope_to_job(client):
    c, jc, _tmp = client
    c.post("/api/jobs", json={"name": "Cat Job"})
    c.post("/api/jobs/cat_job/activate")
    payload = {
        "categories": [{"name": "Keepers", "hint": "good"}, {"name": "Maybe", "hint": ""}],
        "global_rules": "be strict",
    }
    r = c.post("/api/categories?job=cat_job", json=payload)
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert [cat["name"] for cat in body["active"]["categories"]] == ["Keepers", "Maybe"]
    # Persisted to the job.
    job = jc.get_job("cat_job")
    assert [cat["name"] for cat in job.categories] == ["Keepers", "Maybe"]
    assert job.category_rules == "be strict"
    # Active job → projected into the live taxonomy file.
    import categories
    assert categories.get_categories() == ("Keepers", "Maybe")


def test_active_category_save_bumps_index_mtime(client):
    """An active-job category edit must bump data/jobs/_index.json so the
    supervisor's _index.json mtime watch fires (activate() does this; a bare
    project_categories would not)."""
    c, jc, _tmp = client
    c.post("/api/jobs", json={"name": "Idx Job"})
    c.post("/api/jobs/idx_job/activate")
    index_path = jc.jobs_dir() / "_index.json"
    before = index_path.stat().st_mtime_ns
    # Make sure the mtime can actually differ.
    import time
    time.sleep(0.01)
    r = c.post("/api/categories?job=idx_job", json={
        "categories": [{"name": "Solo", "hint": ""}], "global_rules": "r"})
    assert r.status_code == 200
    after = index_path.stat().st_mtime_ns
    assert after > before, "category save on the active job must rewrite _index.json"


def test_inactive_category_warning_scoped_to_that_job(client, monkeypatch):
    """Removing a populated category on an INACTIVE job must count THAT job's
    sorted rows, never the active job's (#4)."""
    c, jc, _tmp = client
    c.post("/api/jobs", json={"name": "Active Cats"})
    c.post("/api/jobs", json={"name": "Idle Cats"})
    c.post("/api/jobs/active_cats/activate")
    # Give the INACTIVE job a category "Keepers" and seed a sorted row under it
    # via the index (scoped to its slug).
    cfg = c.get("/api/jobs/idle_cats").get_json()
    cfg["categories"] = [{"name": "Keepers", "hint": ""}]
    c.put("/api/jobs/idle_cats", json=cfg)
    import index_store
    index_store.upsert({
        "path": str(_tmp / "sorted" / "idle_cats" / "Keepers" / "x.jpg"),
        "status": "sorted", "topic_slug": "idle_cats", "source": "civitai",
        "category": "Keepers", "mtime": 1.0, "size": 10, "width": 1, "height": 1,
        "ovr": None, "rel": None, "quality": None, "nsfw": 0, "prompt": "", "vision_json": None,
    })
    # Now remove "Keepers" on the INACTIVE job without force → should 409 because
    # idle_cats has a sorted row, proving the count was scoped to idle_cats.
    r = c.post("/api/categories?job=idle_cats", json={
        "categories": [{"name": "Other", "hint": ""}], "global_rules": ""})
    assert r.status_code == 409
    assert r.get_json()["error"] == "removing_populated_categories"
    assert r.get_json()["legacy"].get("Keepers") == 1


# ── job PUT validation (categories + scoring) ────────────────────────────────

def test_put_rejects_traversal_category_name(client):
    c, jc, _tmp = client
    c.post("/api/jobs", json={"name": "Vuln Job"})
    cfg = c.get("/api/jobs/vuln_job").get_json()
    cfg["categories"] = [{"name": "../../etc", "hint": ""}]
    r = c.put("/api/jobs/vuln_job", json=cfg)
    assert r.status_code == 400
    assert "categories" in r.get_json()["error"]
    # Nothing persisted — the original categories remain valid.
    assert all("/" not in cat.get("name", "") for cat in jc.get_job("vuln_job").categories)


def test_put_rejects_reserved_category_name(client):
    c, _jc, _tmp = client
    c.post("/api/jobs", json={"name": "Reserved Cat"})
    cfg = c.get("/api/jobs/reserved_cat").get_json()
    cfg["categories"] = [{"name": "DISCARD", "hint": ""}]
    r = c.put("/api/jobs/reserved_cat", json=cfg)
    assert r.status_code == 400


def test_put_rejects_out_of_range_scoring(client):
    c, jc, _tmp = client
    c.post("/api/jobs", json={"name": "Score Job"})
    cfg = c.get("/api/jobs/score_job").get_json()
    cfg["scoring"] = {"ovr_min": 250, "rel_min": 10, "notes": ""}
    r = c.put("/api/jobs/score_job", json=cfg)
    assert r.status_code == 400
    assert "ovr_min" in r.get_json()["error"]
    # Unchanged on disk.
    assert jc.get_job("score_job").scoring.get("ovr_min") != 250


def test_put_accepts_in_range_scoring(client):
    c, jc, _tmp = client
    c.post("/api/jobs", json={"name": "Good Score"})
    cfg = c.get("/api/jobs/good_score").get_json()
    cfg["scoring"] = {"ovr_min": 80, "rel_min": 70, "notes": "tasteful"}
    r = c.put("/api/jobs/good_score", json=cfg)
    assert r.status_code == 200
    saved = jc.get_job("good_score")
    assert saved.scoring["ovr_min"] == 80 and saved.scoring["rel_min"] == 70


# ── settings shrunk to global keys ───────────────────────────────────────────

def test_settings_get_excludes_per_job_keys(client):
    c, _jc, _tmp = client
    body = c.get("/api/settings").get_json()
    # Global keys present.
    assert "GROQ_API_KEY" in body
    assert "BLUR_NSFW_THUMBS" in body
    assert "PIPELINE_BASE_DIR" in body
    # Per-job keys moved out of settings.
    for moved in ("PIPELINE_TOPIC", "TOPIC_KEYWORDS_EXTRA", "X_ACCOUNTS",
                  "REDDIT_SUBREDDITS", "GALLERY_DL_URLS", "AUTO_CAPTION_ENABLED",
                  "VISION_OVR_MIN_SCORE", "DISCORD_CHANNELS_JSON"):
        assert moved not in body


def test_settings_post_rejects_per_job_key(client):
    c, _jc, _tmp = client
    r = c.post("/api/settings", json={"PIPELINE_TOPIC": "nope"})
    assert r.status_code == 400
    assert "PIPELINE_TOPIC" in r.get_json()["errors"]


# ── pipeline start gate ──────────────────────────────────────────────────────

def test_pipeline_start_refused_without_active_job(client):
    c, jc, _tmp = client
    assert jc.get_active_slug() is None
    r = c.post("/api/pipeline/start")
    assert r.status_code == 409
    assert "Activate a job" in r.get_json()["error"]


# ── ?job= scoping defaults to active ─────────────────────────────────────────

def test_status_reports_active_and_scoped_job(client):
    c, _jc, _tmp = client
    c.post("/api/jobs", json={"name": "Scoped"})
    c.post("/api/jobs/scoped/activate")
    body = c.get("/api/status").get_json()
    assert body["active_job"] == "scoped"
    assert body["job"] == "scoped"           # default scope = active
    # Explicit ?job= overrides the default scope; a well-formed but empty slug
    # scopes to that slug (zero counts), never silently to the active job's data.
    other = c.get("/api/status?job=somethingelse").get_json()
    assert other["job"] == "somethingelse"
    assert other["active_job"] == "scoped"
    assert other["queue"]["total"] == 0
    assert other["sorted"]["total"] == 0


def test_status_malformed_job_param_falls_back_to_active(client):
    c, _jc, _tmp = client
    c.post("/api/jobs", json={"name": "Fallback"})
    c.post("/api/jobs/fallback/activate")
    # A malformed ?job= must fall back to the active slug, never widen to global.
    body = c.get("/api/status?job=Bad-Slug").get_json()
    assert body["job"] == "fallback"


def test_create_reserved_slug_is_400(client):
    c, _jc, _tmp = client
    # 'queue' and 'advance' collide with literal /api/jobs/ route segments.
    for name in ("Queue", "advance"):
        r = c.post("/api/jobs", json={"name": name})
        assert r.status_code == 400, name
        assert "reserved" in r.get_json()["error"]


def test_update_cannot_set_arbitrary_status(client):
    c, jc, _tmp = client
    c.post("/api/jobs", json={"name": "Status Job"})
    cfg = c.get("/api/jobs/status_job").get_json()
    cfg["status"] = "hacked"
    c.put("/api/jobs/status_job", json=cfg)
    # status is a runtime projection, not user-editable — the bogus value must
    # not be persisted to the job file.
    assert jc.get_job("status_job").status != "hacked"


# ── force delete (data dirs, guarded by safe_inside) ─────────────────────────

def test_force_delete_removes_data_dirs(client, tmp_path):
    c, jc, _tmp = client
    c.post("/api/jobs", json={"name": "Trash Me"})
    import paths
    qd = paths.queue_dir("trash_me")
    sd = paths.sorted_dir("trash_me")
    (qd / "civitai").mkdir(parents=True, exist_ok=True)
    (sd / "Keepers").mkdir(parents=True, exist_ok=True)
    assert qd.exists() and sd.exists()
    r = c.delete("/api/jobs/trash_me?force=1")
    assert r.status_code == 200
    assert not qd.exists() and not sd.exists()
    assert jc.get_job("trash_me") is None
    # The response reports exactly which dirs were removed (all passed safe_inside).
    assert len(r.get_json()["removed_dirs"]) == 2


def test_delete_without_force_keeps_data_dirs(client):
    c, jc, _tmp = client
    c.post("/api/jobs", json={"name": "Keep Data"})
    import paths
    qd = paths.queue_dir("keep_data")
    (qd / "civitai").mkdir(parents=True, exist_ok=True)
    r = c.delete("/api/jobs/keep_data")
    assert r.status_code == 200
    assert qd.exists()                        # data left on disk
    assert r.get_json()["removed_dirs"] == []


# ── categories projection only for the active job ────────────────────────────

def test_categories_post_to_inactive_job_does_not_project(client):
    c, _jc, _tmp = client
    # Two jobs; activate the first, edit categories on the SECOND.
    c.post("/api/jobs", json={"name": "Active Cats"})
    c.post("/api/jobs", json={"name": "Other Cats"})
    c.post("/api/jobs/active_cats/activate")
    import categories
    before = categories.get_categories()
    c.post("/api/categories?job=other_cats", json={
        "categories": [{"name": "OnlyHere", "hint": ""}], "global_rules": "x"})
    # The live taxonomy file (active job's projection) must be untouched.
    assert categories.get_categories() == before
    assert categories.get_categories() != ("OnlyHere",)


# ── global advance route ─────────────────────────────────────────────────────

def test_global_advance_route(client):
    c, jc, _tmp = client
    c.post("/api/jobs", json={"name": "A"})
    c.post("/api/jobs", json={"name": "B"})
    c.post("/api/jobs/a/activate")
    c.post("/api/jobs/b/enqueue")
    r = c.post("/api/jobs/advance")           # no slug — global form
    assert r.status_code == 200
    assert r.get_json()["active"] == "b"
    assert jc.get_active_slug() == "b"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
