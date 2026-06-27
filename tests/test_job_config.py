"""Tests for job_config — the per-job configuration keystone.

Runnable two ways:
    pytest tests/test_job_config.py
    python tests/test_job_config.py        # falls back to pytest.main

A Job is a JSON bundle under data/jobs/<slug>.json that, on activation,
projects down into the existing env-var + cull_categories.json contracts.
These tests pin that contract.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PIPELINE_CODE = Path(__file__).resolve().parent.parent / "pipeline_code"
if str(PIPELINE_CODE) not in sys.path:
    sys.path.insert(0, str(PIPELINE_CODE))


@pytest.fixture()
def isolated(tmp_path, monkeypatch):
    """Point all storage at a temp dir and hand back the module under test."""
    monkeypatch.setenv("PIPELINE_BASE_DIR", str(tmp_path))
    # categories.ACTIVE_PATH is computed at import time from PIPELINE_BASE_DIR;
    # redirect it (and bust its cache) so projection lands in the temp dir.
    import categories
    monkeypatch.setattr(categories, "ACTIVE_PATH", tmp_path / "cull_categories.json")
    monkeypatch.setattr(categories, "_cache", None, raising=False)
    monkeypatch.setattr(categories, "_cache_mtime", 0.0, raising=False)

    import importlib
    import job_config
    importlib.reload(job_config)
    return job_config, tmp_path


# ── slugify ──────────────────────────────────────────────────────────────────

def test_slugify_lowercases_and_underscores(isolated):
    job_config, _ = isolated
    assert job_config.slugify("Female Influencer") == "female_influencer"
    assert job_config.slugify("Car Ads!! 2026") == "car_ads_2026"
    assert job_config.slugify("  trim--me  ") == "trim_me"


# ── create / get / list round-trip ───────────────────────────────────────────

def test_create_job_writes_file_and_defaults(isolated):
    job_config, tmp = isolated
    job = job_config.create_job("Female Influencer")
    assert job.slug == "female_influencer"
    assert job.name == "Female Influencer"
    assert job.status == "idle"
    assert (tmp / "jobs" / "female_influencer.json").is_file()
    # fresh job carries a usable default taxonomy
    assert len(job.categories) >= 1


def test_get_job_round_trips(isolated):
    job_config, _ = isolated
    job_config.create_job("Car Ads")
    loaded = job_config.get_job("car_ads")
    assert loaded is not None
    assert loaded.name == "Car Ads"
    assert job_config.get_job("missing") is None


def test_list_jobs_returns_all_created(isolated):
    job_config, _ = isolated
    job_config.create_job("Alpha")
    job_config.create_job("Beta")
    slugs = {j.slug for j in job_config.list_jobs()}
    assert slugs == {"alpha", "beta"}


def test_create_duplicate_name_raises(isolated):
    job_config, _ = isolated
    job_config.create_job("Dup")
    with pytest.raises(ValueError):
        job_config.create_job("Dup")


def test_save_job_is_immutable_update(isolated):
    job_config, _ = isolated
    job = job_config.create_job("Edit Me")
    updated = job.with_updates(name="Renamed")
    # original object unchanged (frozen)
    assert job.name == "Edit Me"
    saved = job_config.save_job(updated)
    assert saved.name == "Renamed"
    assert job_config.get_job("edit_me").name == "Renamed"


def test_clone_via_base_on(isolated):
    job_config, _ = isolated
    src = job_config.create_job("Source")
    src = src.with_updates(scoring={"ovr_min": 77, "rel_min": 50, "notes": "x"})
    job_config.save_job(src)
    clone = job_config.create_job("Source Copy", base_on="source")
    assert clone.slug == "source_copy"
    assert clone.scoring["ovr_min"] == 77


# ── queue / active pointer / advance ─────────────────────────────────────────

def test_active_pointer_set_and_get(isolated):
    job_config, _ = isolated
    job_config.create_job("Job A")
    assert job_config.get_active_slug() is None
    job_config.set_active("job_a")
    assert job_config.get_active_slug() == "job_a"


def test_set_active_unknown_raises(isolated):
    job_config, _ = isolated
    with pytest.raises(ValueError):
        job_config.set_active("nope")


def test_enqueue_dequeue_and_order(isolated):
    job_config, _ = isolated
    for n in ("One", "Two", "Three"):
        job_config.create_job(n)
    job_config.enqueue("one")
    job_config.enqueue("two")
    job_config.enqueue("three")
    assert job_config.get_index()["queue"] == ["one", "two", "three"]
    job_config.dequeue("two")
    assert job_config.get_index()["queue"] == ["one", "three"]
    job_config.set_queue(["three", "one"])
    assert job_config.get_index()["queue"] == ["three", "one"]


def test_advance_promotes_head_of_queue(isolated):
    job_config, _ = isolated
    for n in ("First", "Second"):
        job_config.create_job(n)
    job_config.set_active("first")
    job_config.enqueue("second")
    new_active = job_config.advance()
    assert new_active == "second"
    assert job_config.get_active_slug() == "second"
    assert job_config.get_index()["queue"] == []
    # advancing with an empty queue clears active
    assert job_config.advance() is None
    assert job_config.get_active_slug() is None


def test_delete_refuses_active_job(isolated):
    job_config, _ = isolated
    job_config.create_job("Keep")
    job_config.set_active("keep")
    with pytest.raises(ValueError):
        job_config.delete_job("keep")
    job_config.set_active(None)
    job_config.delete_job("keep")
    assert job_config.get_job("keep") is None


def test_delete_job_removes_from_queue(isolated):
    job_config, _ = isolated
    job_config.create_job("Q1")
    job_config.create_job("Q2")
    job_config.enqueue("q1")
    job_config.enqueue("q2")
    job_config.delete_job("q1")
    assert job_config.get_index()["queue"] == ["q2"]
    assert job_config.get_job("q1") is None


def test_advance_skips_orphaned_slug(isolated):
    """A queued slug whose file was removed must be skipped, never promoted to
    active (a dangling active pointer would leave the supervisor with no config)."""
    job_config, _ = isolated
    for n in ("First", "Second", "Third"):
        job_config.create_job(n)
    job_config.set_active("first")
    job_config.enqueue("second")
    job_config.enqueue("third")
    (job_config.jobs_dir() / "second.json").unlink()   # orphan it
    assert job_config.advance() == "third"
    assert job_config.get_active_slug() == "third"


# ── slug safety ──────────────────────────────────────────────────────────────

def test_get_job_rejects_malformed_slug(isolated):
    job_config, _ = isolated
    assert job_config.get_job("../../etc/passwd") is None
    assert job_config.get_job("bad/slug") is None


def test_delete_job_rejects_traversal_slug(isolated):
    job_config, _ = isolated
    with pytest.raises(ValueError):
        job_config.delete_job("../../../boot")


def test_save_job_rejects_invalid_slug(isolated):
    job_config, _ = isolated
    bad = job_config._make_job("Bad Slug!", "Bad")
    with pytest.raises(ValueError):
        job_config.save_job(bad)


def test_from_dict_tolerates_empty(isolated):
    job_config, _ = isolated
    j = job_config.Job.from_dict({})
    assert j.status == "idle"
    assert isinstance(j.topic, dict) and "topic" in j.topic


# ── resolve_env (authoritative mapping) ──────────────────────────────────────

def _rich_job(job_config):
    job = job_config.create_job("Rich")
    return job.with_updates(
        topic={
            "topic": "Realistic Female Influencer",
            "keywords_extra": ["a", "b"],
            "banned_keywords": ["x"],
            "generation_hints": ["p"],
            "min_prompt_length": 30,
            "require_prompt": True,
        },
        scrapers={
            "enabled": {"X.com": False, "Civitai-Com": True, "Web": True},
            "x_accounts": ["acct1"],
            "reddit_subreddits": ["sub1"],
            "discord_channels_json": "[]",
            "civitai_domains": ["civitai.com"],
            "gallery_dl": {
                "enabled": True, "urls": ["u1", "u2"], "limit_per_url": 200,
                "cookies_file": "c", "config_path": "cfg",
            },
            "local_import": {"enabled": True, "dir": "/d", "name": "loc", "migrate_from": "m"},
            "zforfree": {"local_enabled": True, "web_enabled": False, "local_src": "/z"},
        },
        scoring={"ovr_min": 60, "rel_min": 55, "notes": "note"},
        captioning={"enabled": True, "style": "booru_tags", "overwrite": True},
    )


def test_resolve_env_maps_topic_and_scoring(isolated):
    job_config, _ = isolated
    env = job_config.resolve_env(_rich_job(job_config))
    assert env["PIPELINE_SLUG"] == "rich"
    assert env["PIPELINE_TOPIC"] == "Realistic Female Influencer"
    assert env["TOPIC_KEYWORDS_EXTRA"] == "a,b"
    assert env["TOPIC_BANNED_KEYWORDS"] == "x"
    assert env["TOPIC_GENERATION_HINTS"] == "p"
    assert env["MIN_PROMPT_LENGTH"] == "30"
    assert env["REQUIRE_PROMPT"] == "true"
    assert env["VISION_OVR_MIN_SCORE"] == "60"
    assert env["VISION_REL_MIN_SCORE"] == "55"
    assert env["VISION_SCORE_NOTES"] == "note"


def test_resolve_env_maps_scrapers(isolated):
    job_config, _ = isolated
    env = job_config.resolve_env(_rich_job(job_config))
    assert env["SCRAPER_DISABLED"] == "X.com"           # only the disabled name
    assert env["X_ACCOUNTS"] == "acct1"
    assert env["REDDIT_SUBREDDITS"] == "sub1"
    assert env["DISCORD_CHANNELS_JSON"] == "[]"
    assert env["CIVITAI_DOMAINS"] == "civitai.com"
    assert env["GALLERY_DL_ENABLED"] == "true"
    assert env["GALLERY_DL_URLS"] == "u1\nu2"            # newline-joined like today
    assert env["GALLERY_DL_LIMIT_PER_URL"] == "200"
    assert env["GALLERY_DL_COOKIES_FILE"] == "c"
    assert env["GALLERY_DL_CONFIG_PATH"] == "cfg"
    assert env["LOCAL_IMPORT_ENABLED"] == "true"
    assert env["LOCAL_IMPORT_DIR"] == "/d"
    assert env["LOCAL_IMPORT_NAME"] == "loc"
    assert env["LOCAL_IMPORT_MIGRATE_FROM"] == "m"
    assert env["ZFORFREE_LOCAL_ENABLED"] == "true"
    assert env["ZFORFREE_WEB_ENABLED"] == "false"
    assert env["ZFORFREE_LOCAL_SRC"] == "/z"


def test_resolve_env_maps_captioning(isolated):
    job_config, _ = isolated
    env = job_config.resolve_env(_rich_job(job_config))
    assert env["AUTO_CAPTION_ENABLED"] == "true"
    assert env["AUTO_CAPTION_STYLE"] == "booru_tags"
    assert env["AUTO_CAPTION_OVERWRITE"] == "true"


def test_resolve_env_all_values_are_strings(isolated):
    job_config, _ = isolated
    env = job_config.resolve_env(_rich_job(job_config))
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in env.items())


def test_resolve_env_no_disabled_when_all_enabled(isolated):
    job_config, _ = isolated
    job = job_config.create_job("Allon")
    job = job.with_updates(scrapers={**job.scrapers, "enabled": {"Web": True, "Civitai-Com": True}})
    env = job_config.resolve_env(job)
    assert env["SCRAPER_DISABLED"] == ""


# ── project_categories ───────────────────────────────────────────────────────

def test_project_categories_writes_active_taxonomy(isolated):
    job_config, tmp = isolated
    import categories
    job = job_config.create_job("Taxon")
    job = job.with_updates(
        categories=[{"name": "Keepers", "hint": "good"}, {"name": "Maybe", "hint": ""}],
        category_rules="be strict",
    )
    job_config.project_categories(job)
    payload = json.loads((tmp / "cull_categories.json").read_text(encoding="utf-8"))
    assert [c["name"] for c in payload["categories"]] == ["Keepers", "Maybe"]
    assert payload["global_rules"] == "be strict"
    # categories.py now reports the projected taxonomy
    assert categories.get_categories() == ("Keepers", "Maybe")


def test_activate_sets_active_and_projects(isolated):
    job_config, tmp = isolated
    import categories
    job = job_config.create_job("Act")
    job = job.with_updates(categories=[{"name": "OnlyCat", "hint": ""}], category_rules="r")
    job_config.save_job(job)
    job_config.activate("act")
    assert job_config.get_active_slug() == "act"
    assert categories.get_categories() == ("OnlyCat",)


# ── migration ────────────────────────────────────────────────────────────────

def test_migrate_env_to_default_job_builds_from_env(isolated, monkeypatch):
    job_config, _ = isolated
    monkeypatch.setenv("PIPELINE_TOPIC", "Legacy Topic")
    monkeypatch.setenv("TOPIC_KEYWORDS_EXTRA", "k1,k2")
    monkeypatch.setenv("SCRAPER_DISABLED", "X.com,Discord-1")
    monkeypatch.setenv("VISION_OVR_MIN_SCORE", "42")
    monkeypatch.setenv("AUTO_CAPTION_ENABLED", "true")
    monkeypatch.setenv("AUTO_CAPTION_STYLE", "sd_prompt")

    job = job_config.migrate_env_to_default_job()
    assert job is not None
    assert job.slug == "default"
    assert job.topic["topic"] == "Legacy Topic"
    assert job.topic["keywords_extra"] == ["k1", "k2"]
    assert job.scrapers["enabled"]["X.com"] is False
    assert job.scrapers["enabled"]["Discord-1"] is False
    assert job.scrapers["enabled"]["Web"] is True
    assert job.scoring["ovr_min"] == 42
    assert job.captioning["enabled"] is True
    assert job_config.get_active_slug() == "default"


def test_migrate_is_idempotent(isolated):
    job_config, _ = isolated
    first = job_config.migrate_env_to_default_job()
    assert first is not None
    second = job_config.migrate_env_to_default_job()
    assert second is None                       # already migrated → no-op
    assert len(job_config.list_jobs()) == 1


def test_migrate_adopts_existing_pipeline_slug(isolated, monkeypatch):
    """An upgrading user's existing data lives under data/<queue|sorted>/<slug>/.
    Migration MUST adopt that slug so the job inherits the existing queues."""
    job_config, _ = isolated
    monkeypatch.setenv("PIPELINE_SLUG", "realistic_female_influencer")
    monkeypatch.setenv("PIPELINE_TOPIC", "Realistic Female Influencer")
    job = job_config.migrate_env_to_default_job()
    assert job is not None
    assert job.slug == "realistic_female_influencer"   # NOT "default"
    assert job.name == "Realistic Female Influencer"
    assert job_config.get_active_slug() == "realistic_female_influencer"


# ── discovery of pre-existing on-disk data (multi-slug upgraders) ─────────────

def test_discover_data_slugs_finds_existing_queue_and_sorted(isolated):
    job_config, tmp = isolated
    (tmp / "queue" / "old_job_a" / "civitai").mkdir(parents=True)
    (tmp / "sorted" / "old_job_b" / "Keepers" / "civitai").mkdir(parents=True)
    (tmp / "queue" / "old_job_b" / "reddit").mkdir(parents=True)
    found = set(job_config.discover_data_slugs())
    assert found == {"old_job_a", "old_job_b"}


def test_discover_handles_slug_included_paths(isolated, monkeypatch):
    """Regression: when PIPELINE_SORTED/QUEUE already include the slug (as a real
    upgrader's .env does), discovery must return the SLUG, not its category
    children. The multi-slug parent is the folder literally named queue/sorted."""
    job_config, tmp = isolated
    (tmp / "sorted" / "myslug" / "Keepers" / "civitai").mkdir(parents=True)
    (tmp / "queue" / "myslug" / "civitai").mkdir(parents=True)
    monkeypatch.setenv("PIPELINE_SORTED", str(tmp / "sorted" / "myslug"))
    monkeypatch.setenv("PIPELINE_QUEUE", str(tmp / "queue" / "myslug"))
    monkeypatch.setenv("PIPELINE_SLUG", "myslug")
    found = set(job_config.discover_data_slugs())
    assert found == {"myslug"}                  # NOT {"Keepers", "civitai"}


def test_migrate_existing_data_creates_a_job_per_slug(isolated, monkeypatch):
    """The one-shot upgrader: adopt every slug already present on disk."""
    job_config, tmp = isolated
    monkeypatch.setenv("PIPELINE_SLUG", "primary_slug")
    monkeypatch.setenv("PIPELINE_TOPIC", "Primary Topic")
    (tmp / "queue" / "primary_slug" / "civitai").mkdir(parents=True)
    (tmp / "sorted" / "secondary_slug" / "Keepers").mkdir(parents=True)

    created = job_config.migrate_existing_data()
    slugs = {j.slug for j in job_config.list_jobs()}
    assert "primary_slug" in slugs            # the env/active one, full config
    assert "secondary_slug" in slugs          # discovered, adopted with defaults
    assert {j.slug for j in created} >= {"primary_slug", "secondary_slug"}
    assert job_config.get_active_slug() == "primary_slug"
    # idempotent: re-running adopts nothing new
    assert job_config.migrate_existing_data() == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
