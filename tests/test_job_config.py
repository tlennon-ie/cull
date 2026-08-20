"""Tests for job_config v2 — preset library + inherit-by-default overrides.

Runnable:
    pytest tests/test_job_config.py
    python tests/test_job_config.py

A Job v2 stores {slug, name, status, subject, preset, overrides}. The effective
config is `preset ⊕ overrides` with the job's subject injected as topic.topic.
Activating a job still projects down into env vars + cull_categories.json.
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
    monkeypatch.setenv("PIPELINE_BASE_DIR", str(tmp_path))
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
    jc, _ = isolated
    assert jc.slugify("Female Influencer") == "female_influencer"
    assert jc.slugify("Car Ads!! 2026") == "car_ads_2026"


# ── preset library ───────────────────────────────────────────────────────────

def test_default_preset_seeded(isolated):
    jc, _ = isolated
    lib = jc.list_presets()
    assert lib["default"] == "default"
    assert "default" in lib["presets"]
    # default preset has the inheritable shape
    cfg = jc.get_preset("default")
    assert "topic_filters" in cfg and "scrapers" in cfg and "categories" in cfg
    assert "local_imports" in cfg["scrapers"]


def test_save_and_get_preset(isolated):
    jc, _ = isolated
    cfg = jc.get_preset("default")
    cfg["scoring"] = {"ovr_min": 70, "rel_min": 60, "notes": "p"}
    jc.save_preset("fashion", cfg)
    assert jc.get_preset("fashion")["scoring"]["ovr_min"] == 70


def test_get_unknown_preset_falls_back_to_default(isolated):
    jc, _ = isolated
    assert jc.get_preset("nope") == jc.get_preset("default")


def test_delete_preset_refuses_default_and_referenced(isolated):
    jc, _ = isolated
    jc.save_preset("p2", jc.get_preset("default"))
    with pytest.raises(ValueError):
        jc.delete_preset("default")                 # is the default
    jc.create_job("Job", preset="p2")
    with pytest.raises(ValueError):
        jc.delete_preset("p2")                       # referenced by a job
    jc.save_preset("p3", jc.get_preset("default"))
    jc.delete_preset("p3")                           # free → ok
    assert "p3" not in jc.list_presets()["presets"]


def test_set_default_preset(isolated):
    jc, _ = isolated
    jc.save_preset("p2", jc.get_preset("default"))
    jc.set_default_preset("p2")
    assert jc.default_preset_name() == "p2"


# ── job create / inherit-by-default ──────────────────────────────────────────

def test_create_job_inherits_preset(isolated):
    jc, _ = isolated
    cfg = jc.get_preset("default")
    cfg["scrapers"]["x_accounts"] = ["preset_acct"]
    jc.save_preset("default", cfg)
    job = jc.create_job("Influencer", subject="Realistic Female Influencer")
    assert job.subject == "Realistic Female Influencer"
    assert job.preset == "default"
    assert job.overrides == {}
    eff = jc.effective_config(job)
    assert eff["scrapers"]["x_accounts"] == ["preset_acct"]      # inherited
    assert eff["topic"]["topic"] == "Realistic Female Influencer"  # subject injected


def test_create_job_defaults_subject_to_name(isolated):
    jc, _ = isolated
    job = jc.create_job("Car Ads")
    assert job.subject == "Car Ads"
    assert jc.effective_config(job)["topic"]["topic"] == "Car Ads"


def test_create_duplicate_raises(isolated):
    jc, _ = isolated
    jc.create_job("Dup")
    with pytest.raises(ValueError):
        jc.create_job("Dup")


# ── overrides: set / reset / is_overridden ───────────────────────────────────

def test_set_and_reset_override(isolated):
    jc, _ = isolated
    job = jc.create_job("J", subject="S")
    job = jc.set_override(job, "scrapers.x_accounts", ["job_acct"])
    assert jc.is_overridden(job, "scrapers.x_accounts")
    assert jc.effective_config(job)["scrapers"]["x_accounts"] == ["job_acct"]
    job = jc.reset_override(job, "scrapers.x_accounts")
    assert not jc.is_overridden(job, "scrapers.x_accounts")
    assert jc.effective_config(job)["scrapers"]["x_accounts"] == \
        jc.get_preset("default")["scrapers"]["x_accounts"]


def test_reset_prunes_empty_parents(isolated):
    jc, _ = isolated
    job = jc.create_job("J", subject="S")
    job = jc.set_override(job, "scoring.notes", "hi")
    assert job.overrides["scoring"]["notes"] == "hi"
    job = jc.reset_override(job, "scoring.notes")
    assert "scoring" not in job.overrides            # parent pruned when empty


def test_override_is_immutable_update(isolated):
    jc, _ = isolated
    job = jc.create_job("J", subject="S")
    job2 = jc.set_override(job, "scoring.ovr_min", 50)
    assert job.overrides == {}                       # original untouched (frozen)
    assert job2.overrides["scoring"]["ovr_min"] == 50


def test_override_leaf_does_not_clobber_sibling_preset_values(isolated):
    jc, _ = isolated
    cfg = jc.get_preset("default")
    cfg["scoring"] = {"ovr_min": 10, "rel_min": 20, "notes": "preset"}
    jc.save_preset("default", cfg)
    job = jc.create_job("J", subject="S")
    job = jc.set_override(job, "scoring.notes", "mine")
    eff = jc.effective_config(job)
    assert eff["scoring"]["notes"] == "mine"          # overridden leaf
    assert eff["scoring"]["ovr_min"] == 10            # sibling still inherited


# ── round-trip / persistence ─────────────────────────────────────────────────

def test_get_job_round_trips_v2(isolated):
    jc, _ = isolated
    job = jc.create_job("Round Trip", subject="Subj")
    job = jc.set_override(job, "scoring.ovr_min", 65)
    jc.save_job(job)
    loaded = jc.get_job("round_trip")
    assert loaded.subject == "Subj"
    assert loaded.overrides["scoring"]["ovr_min"] == 65


def test_from_dict_upgrades_v1_job_file(isolated, tmp_path):
    """Existing v1 job files (topic/scrapers/categories at top level, no
    overrides) must load as v2: subject from topic.topic, preset=default, and
    the v1 config captured as overrides so effective config is unchanged."""
    jc, _ = isolated
    v1 = {
        "slug": "legacy", "name": "Legacy", "status": "idle",
        "topic": {"topic": "Legacy Subject", "keywords_extra": ["k1"],
                  "banned_keywords": [], "generation_hints": [],
                  "min_prompt_length": 30, "require_prompt": True},
        "scrapers": {"enabled": {"X.com": False, "Web": True},
                     "x_accounts": ["acc"], "reddit_subreddits": [],
                     "discord_channels_json": "", "civitai_domains": [],
                     "gallery_dl": {"enabled": False, "urls": [], "limit_per_url": 200,
                                     "cookies_file": "", "config_path": ""},
                     "local_import": {"enabled": True, "dir": "/d", "name": "loc", "migrate_from": ""}},
        "categories": [{"name": "Keepers", "hint": "k"}],
        "category_rules": "rules", "scoring": {"ovr_min": 42, "rel_min": 0, "notes": ""},
        "captioning": {"enabled": True, "style": "booru_tags", "overwrite": False},
    }
    job = jc.Job.from_dict(v1)
    assert job.subject == "Legacy Subject"
    assert job.preset == "default"
    eff = jc.effective_config(job)
    assert eff["topic"]["keywords_extra"] == ["k1"]
    assert eff["scrapers"]["enabled"]["X.com"] is False
    assert eff["scoring"]["ovr_min"] == 42
    # the legacy single-folder local source folds into the local_imports list
    li = eff["scrapers"]["local_imports"]
    names = {f["name"] for f in li}
    assert "loc" in names and any(f["dir"] == "/d" for f in li)


# ── resolve_env via effective config ─────────────────────────────────────────

def test_resolve_env_uses_effective(isolated):
    jc, _ = isolated
    job = jc.create_job("Rich", subject="Realistic Female Influencer")
    job = jc.set_override(job, "topic_filters.keywords_extra", ["a", "b"])
    job = jc.set_override(job, "scoring.ovr_min", 60)
    job = jc.set_override(job, "captioning", {"enabled": True, "style": "booru_tags", "overwrite": True})
    env = jc.resolve_env(job)
    assert env["PIPELINE_SLUG"] == "rich"
    assert env["PIPELINE_TOPIC"] == "Realistic Female Influencer"
    assert env["TOPIC_KEYWORDS_EXTRA"] == "a,b"
    assert env["VISION_OVR_MIN_SCORE"] == "60"
    assert env["AUTO_CAPTION_ENABLED"] == "true"
    assert env["AUTO_CAPTION_STYLE"] == "booru_tags"


def test_resolve_env_scraper_disabled_from_enabled_map(isolated):
    jc, _ = isolated
    job = jc.create_job("S", subject="S")
    job = jc.set_override(job, "scrapers.enabled", {"X.com": False, "Web": True, "Civitai-Com": True})
    env = jc.resolve_env(job)
    assert env["SCRAPER_DISABLED"] == "X.com"


def test_resolve_env_local_imports_json_only_enabled(isolated):
    jc, _ = isolated
    job = jc.create_job("L", subject="S")
    job = jc.set_override(job, "scrapers.local_imports", [
        {"name": "selfies", "dir": "/a", "enabled": True, "migrate_from": ""},
        {"name": "refs", "dir": "/b", "enabled": False, "migrate_from": ""},
    ])
    env = jc.resolve_env(job)
    folders = json.loads(env["LOCAL_IMPORTS_JSON"])
    assert [f["name"] for f in folders] == ["selfies"]
    assert folders[0]["dir"] == "/a"


def test_resolve_env_values_are_all_strings(isolated):
    jc, _ = isolated
    job = jc.create_job("Z", subject="S")
    env = jc.resolve_env(job)
    assert all(isinstance(v, str) for v in env.values())


def test_scraper_names_has_no_local_folder_entry(isolated):
    jc, _ = isolated
    # Local folders are projected via LOCAL_IMPORTS_JSON, never as a scraper name.
    assert "Web" in jc.SCRAPER_NAMES
    assert all("local" not in n.lower() for n in jc.SCRAPER_NAMES)


def test_priority_names_extends_scraper_names_with_yt_dlp(isolated):
    """PRIORITY_NAMES is the reorderable set (SCRAPER_NAMES + YT-DLP). The extra
    name is intentionally OUT of SCRAPER_NAMES so the on/off contract stays
    exactly what scrapers/UI already read."""
    jc, _ = isolated
    assert set(jc.SCRAPER_NAMES).issubset(set(jc.PRIORITY_NAMES))
    assert "YT-DLP" in jc.PRIORITY_NAMES
    assert "YT-DLP" not in jc.SCRAPER_NAMES


def test_clean_scraper_priority_defaults_and_clamps(isolated):
    """The projection helper is defensive: unknown names dropped, missing
    weights back-filled with the default, out-of-range values clamped."""
    jc, _ = isolated
    # None → full-default block (deterministic PRIORITY_NAMES order, all default weight).
    p = jc.clean_scraper_priority(None)
    assert p["order"] == list(jc.PRIORITY_NAMES)
    assert all(w == jc.PRIORITY_WEIGHT_DEFAULT for w in p["weights"].values())
    # Partial user order → prefix wins, missing names appended in PRIORITY_NAMES order.
    p = jc.clean_scraper_priority({"order": ["Web", "X.com", "not-a-scraper"],
                                   "weights": {"Web": 99, "X.com": -3, "junk": 5}})
    assert p["order"][0] == "Web" and p["order"][1] == "X.com"
    assert set(p["order"]) == set(jc.PRIORITY_NAMES)
    assert p["weights"]["Web"] == jc.PRIORITY_WEIGHT_MAX          # clamped up from 99
    assert p["weights"]["X.com"] == jc.PRIORITY_WEIGHT_MIN        # clamped up from -3
    assert "junk" not in p["weights"]                              # unknown dropped


def test_resolve_env_emits_scraper_priority_json(isolated):
    """A user-set override projects into SCRAPER_PRIORITY_JSON — the string the
    supervisor consumes at spawn to decide agent order + weights."""
    jc, _ = isolated
    job = jc.create_job("P", subject="s")
    job = jc.set_override(job, "scrapers.priority", {
        "order": ["Web", "Gallery-DL", "X.com"],
        "weights": {"Web": 9, "Gallery-DL": 4},
    })
    env = jc.resolve_env(job)
    payload = json.loads(env["SCRAPER_PRIORITY_JSON"])
    assert payload["order"][0] == "Web"
    assert payload["order"][1] == "Gallery-DL"
    assert payload["order"][2] == "X.com"
    # Every PRIORITY_NAME is covered (including untouched ones defaulted).
    assert set(payload["order"]) == set(jc.PRIORITY_NAMES)
    assert payload["weights"]["Web"] == 9
    assert payload["weights"]["Gallery-DL"] == 4
    assert payload["weights"]["Civitai-Com"] == jc.PRIORITY_WEIGHT_DEFAULT


# ── projection ───────────────────────────────────────────────────────────────

def test_project_categories_uses_effective(isolated):
    jc, _ = isolated
    import categories
    job = jc.create_job("Tax", subject="S")
    job = jc.set_override(job, "categories", [{"name": "Keepers", "hint": "k"}, {"name": "Maybe", "hint": ""}])
    job = jc.set_override(job, "category_rules", "strict")
    jc.project_categories(job)
    assert categories.get_categories() == ("Keepers", "Maybe")


def test_activate_sets_active_and_projects(isolated):
    jc, _ = isolated
    import categories
    job = jc.create_job("Act", subject="S")
    job = jc.set_override(job, "categories", [{"name": "OnlyCat", "hint": ""}])
    jc.save_job(job)
    jc.activate("act")
    assert jc.get_active_slug() == "act"
    assert categories.get_categories() == ("OnlyCat",)


# ── queue / active pointer / advance (schema-independent) ─────────────────────

def test_active_pointer_set_and_get(isolated):
    jc, _ = isolated
    jc.create_job("Job A")
    assert jc.get_active_slug() is None
    jc.set_active("job_a")
    assert jc.get_active_slug() == "job_a"


def test_enqueue_dequeue_and_order(isolated):
    jc, _ = isolated
    for n in ("One", "Two", "Three"):
        jc.create_job(n)
    jc.enqueue("one"); jc.enqueue("two"); jc.enqueue("three")
    assert jc.get_index()["queue"] == ["one", "two", "three"]
    jc.dequeue("two")
    assert jc.get_index()["queue"] == ["one", "three"]
    jc.set_queue(["three", "one"])
    assert jc.get_index()["queue"] == ["three", "one"]


def test_advance_promotes_head(isolated):
    """v2: advance() ADDS the queue head to the active set (multi-active) —
    the previous active job stays running. ``get_active_slug()`` still
    returns the HEAD (first slug), so 'first' remains at the head after we
    add 'second' as a second active job."""
    jc, _ = isolated
    for n in ("First", "Second"):
        jc.create_job(n)
    jc.set_active("first")
    jc.enqueue("second")
    assert jc.advance() == "second"
    assert jc.get_active_slug() == "first"                 # head unchanged
    assert set(jc.get_active_slugs()) == {"first", "second"}
    assert jc.advance() is None


def test_advance_skips_orphaned_slug(isolated):
    jc, _ = isolated
    for n in ("First", "Second", "Third"):
        jc.create_job(n)
    jc.set_active("first")
    jc.enqueue("second"); jc.enqueue("third")
    (jc.jobs_dir() / "second.json").unlink()
    # Advance skips the orphaned 'second' and adds 'third' to the active set.
    assert jc.advance() == "third"
    assert set(jc.get_active_slugs()) == {"first", "third"}


def test_delete_refuses_active_and_removes_from_queue(isolated):
    jc, _ = isolated
    jc.create_job("Keep"); jc.create_job("Q")
    jc.set_active("keep"); jc.enqueue("q")
    with pytest.raises(ValueError):
        jc.delete_job("keep")
    jc.delete_job("q")
    assert jc.get_index()["queue"] == []
    assert jc.get_job("q") is None


# ── slug safety ──────────────────────────────────────────────────────────────

def test_get_job_rejects_malformed_slug(isolated):
    jc, _ = isolated
    assert jc.get_job("../../etc/passwd") is None


def test_delete_job_rejects_traversal_slug(isolated):
    jc, _ = isolated
    with pytest.raises(ValueError):
        jc.delete_job("../../../boot")


# ── clone ────────────────────────────────────────────────────────────────────

def test_clone_via_base_on_copies_subject_preset_overrides(isolated):
    jc, _ = isolated
    src = jc.create_job("Source", subject="Src Subject", preset="default")
    src = jc.set_override(src, "scoring.ovr_min", 77)
    jc.save_job(src)
    clone = jc.create_job("Source Copy", base_on="source")
    assert clone.subject == "Src Subject"
    assert clone.preset == "default"
    assert jc.effective_config(clone)["scoring"]["ovr_min"] == 77


# ── migration (v1 env → v2) ──────────────────────────────────────────────────

def test_migrate_env_creates_default_preset_and_job(isolated, monkeypatch):
    jc, _ = isolated
    monkeypatch.setenv("PIPELINE_SLUG", "myslug")
    monkeypatch.setenv("PIPELINE_TOPIC", "My Topic")
    monkeypatch.setenv("TOPIC_KEYWORDS_EXTRA", "k1,k2")
    monkeypatch.setenv("SCRAPER_DISABLED", "X.com")
    monkeypatch.setenv("VISION_OVR_MIN_SCORE", "42")
    job = jc.migrate_env_to_default_job()
    assert job is not None
    assert job.slug == "myslug"
    assert job.subject == "My Topic"
    assert job.preset == "default"
    eff = jc.effective_config(job)
    assert eff["topic"]["keywords_extra"] == ["k1", "k2"]
    assert eff["scrapers"]["enabled"]["X.com"] is False
    assert eff["scoring"]["ovr_min"] == 42
    assert "default" in jc.list_presets()["presets"]
    assert jc.get_active_slug() == "myslug"


def test_migrate_env_folds_legacy_local_source(isolated, monkeypatch):
    jc, _ = isolated
    monkeypatch.setenv("PIPELINE_SLUG", "s")
    monkeypatch.setenv("LOCAL_IMPORT_ENABLED", "true")
    monkeypatch.setenv("LOCAL_IMPORT_DIR", "/data/local")
    monkeypatch.setenv("LOCAL_IMPORT_NAME", "mylocal")
    job = jc.migrate_env_to_default_job()
    li = jc.effective_config(job)["scrapers"]["local_imports"]
    names = {f["name"] for f in li}
    assert "mylocal" in names
    assert any(f["dir"] == "/data/local" for f in li)


def test_migrate_idempotent(isolated):
    jc, _ = isolated
    assert jc.migrate_env_to_default_job() is not None
    assert jc.migrate_env_to_default_job() is None


def test_discover_data_slugs_finds_existing(isolated, tmp_path):
    jc, _ = isolated
    (tmp_path / "queue" / "old_a" / "civitai").mkdir(parents=True)
    (tmp_path / "sorted" / "old_b" / "Keepers").mkdir(parents=True)
    assert set(jc.discover_data_slugs()) == {"old_a", "old_b"}


# ── robustness (review findings) ─────────────────────────────────────────────

def test_effective_config_does_not_mutate_preset(isolated):
    jc, _ = isolated
    job = jc.create_job("J", subject="S")
    eff1 = jc.effective_config(job)
    eff1["scrapers"]["x_accounts"].append("mutant")
    eff1["scoring"]["ovr_min"] = 999
    eff2 = jc.effective_config(job)
    assert eff2["scrapers"]["x_accounts"] == []          # preset untouched
    assert eff2["scoring"]["ovr_min"] == 40              # default preset value, not the mutated 999


def test_job_with_unknown_preset_falls_back_to_default(isolated):
    jc, _ = isolated
    job = jc.create_job("J", subject="S").with_updates(preset="ghost")
    eff = jc.effective_config(job)                        # must not raise
    assert eff["scoring"]["ovr_min"] == 40               # falls back to the default preset


def test_resolve_env_tolerates_malformed_override(isolated):
    jc, _ = isolated
    job = jc.create_job("J", subject="S").with_updates(
        overrides={"scoring": "not_a_dict", "scrapers": "nope"})
    env = jc.resolve_env(job)                             # must not raise
    assert env["VISION_OVR_MIN_SCORE"] == "0"
    assert env["SCRAPER_DISABLED"] == ""


def test_sparse_enabled_override_only_disables_named(isolated):
    jc, _ = isolated
    job = jc.create_job("J", subject="S")
    job = jc.set_override(job, "scrapers.enabled", {"X.com": False})
    assert jc.resolve_env(job)["SCRAPER_DISABLED"] == "X.com"


def test_override_categories_replaces_wholesale(isolated):
    jc, _ = isolated
    job = jc.create_job("J", subject="S")
    job = jc.set_override(job, "categories", [{"name": "Solo", "hint": ""}])
    assert [c["name"] for c in jc.effective_config(job)["categories"]] == ["Solo"]


def test_v1_file_round_trips_through_save(isolated):
    jc, _ = isolated
    jc.jobs_dir().mkdir(parents=True, exist_ok=True)
    v1 = {"slug": "leg", "name": "Leg", "topic": {"topic": "T", "keywords_extra": ["a"]},
          "scrapers": {"enabled": {"X.com": False}}, "categories": [{"name": "K", "hint": ""}],
          "scoring": {"ovr_min": 42}}
    (jc.jobs_dir() / "leg.json").write_text(json.dumps(v1), encoding="utf-8")
    jc.save_job(jc.get_job("leg"))                        # upgrade-on-read, then persist
    on_disk = json.loads((jc.jobs_dir() / "leg.json").read_text(encoding="utf-8"))
    assert "overrides" in on_disk and "subject" in on_disk
    assert jc.effective_config(jc.get_job("leg"))["scoring"]["ovr_min"] == 42


def test_from_dict_v1_with_stray_subject_key_not_misdetected(isolated):
    jc, _ = isolated
    v1 = {"slug": "x", "name": "X", "subject": "oops",
          "topic": {"topic": "Real", "keywords_extra": ["k"]},
          "scrapers": {"enabled": {"Web": True}}}
    job = jc.Job.from_dict(v1)
    assert jc.effective_config(job)["topic"]["keywords_extra"] == ["k"]   # v1 cfg kept


def test_corrupt_presets_file_returns_default(isolated):
    jc, _ = isolated
    jc.jobs_dir().mkdir(parents=True, exist_ok=True)
    (jc.jobs_dir() / "_presets.json").write_text("{ not json", encoding="utf-8")
    assert "default" in jc.list_presets()["presets"]


def test_delete_unknown_preset_raises(isolated):
    jc, _ = isolated
    with pytest.raises(ValueError):
        jc.delete_preset("never_saved")


# ── multi-active + per-job priority (v2) ─────────────────────────────────────

def test_get_active_slugs_returns_head_and_full_list(isolated):
    jc, _ = isolated
    for n in ("A", "B"):
        jc.create_job(n)
    assert jc.get_active_slugs() == []
    assert jc.get_active_slug() is None
    jc.set_active("a")
    assert jc.get_active_slugs() == ["a"]
    assert jc.get_active_slug() == "a"


def test_activate_additive_by_default(isolated):
    jc, _ = isolated
    jc.create_job("A"); jc.create_job("B")
    jc.activate("a")
    jc.activate("b")                     # additive
    assert jc.get_active_slugs() == ["a", "b"]
    assert jc.get_active_slug() == "a"   # head unchanged
    # Re-activating an already-active slug is idempotent — no duplicate.
    jc.activate("a")
    assert jc.get_active_slugs() == ["a", "b"]


def test_activate_exclusive_resets_active_set(isolated):
    jc, _ = isolated
    jc.create_job("A"); jc.create_job("B")
    jc.activate("a")
    jc.activate("b", exclusive=True)     # resets to [b]
    assert jc.get_active_slugs() == ["b"]


def test_deactivate_removes_from_active_set(isolated):
    jc, _ = isolated
    jc.create_job("A"); jc.create_job("B")
    jc.activate("a"); jc.activate("b")
    jc.deactivate("a")
    assert jc.get_active_slugs() == ["b"]
    jc.deactivate("a")                   # idempotent no-op
    assert jc.get_active_slugs() == ["b"]


def test_set_active_slugs_replaces_wholesale(isolated):
    jc, _ = isolated
    jc.create_job("A"); jc.create_job("B"); jc.create_job("C")
    jc.set_active_slugs(["a", "b", "c"])
    assert jc.get_active_slugs() == ["a", "b", "c"]
    jc.set_active_slugs(["c"])
    assert jc.get_active_slugs() == ["c"]


def test_set_active_slugs_rejects_unknown(isolated):
    jc, _ = isolated
    jc.create_job("A")
    with pytest.raises(ValueError):
        jc.set_active_slugs(["a", "ghost"])
    # Active set must be untouched on rejection.
    assert jc.get_active_slugs() == []


def test_priority_default_and_clamp(isolated):
    jc, _ = isolated
    jc.create_job("A")
    assert jc.get_job_priority("a") == jc.PRIORITY_WEIGHT_DEFAULT
    assert jc.set_job_priority("a", 20) == jc.PRIORITY_WEIGHT_MAX
    assert jc.set_job_priority("a", -3) == jc.PRIORITY_WEIGHT_MIN
    assert jc.set_job_priority("a", 7) == 7
    assert jc.get_job_priority("a") == 7


def test_priority_rejects_unknown_slug(isolated):
    jc, _ = isolated
    with pytest.raises(ValueError):
        jc.set_job_priority("ghost", 5)


def test_index_migration_from_legacy_single_string(isolated, tmp_path):
    """A legacy _index.json with string ``active`` must upgrade on read."""
    jc, _ = isolated
    jc.create_job("A")
    jc.jobs_dir().mkdir(parents=True, exist_ok=True)
    (jc.jobs_dir() / "_index.json").write_text(
        json.dumps({"active": "a", "queue": ["b"]}), encoding="utf-8"
    )
    idx = jc.get_index()
    assert idx["active"] == ["a"]                # coerced to list
    assert idx["queue"] == ["b"]
    assert idx["priority"] == {}


def test_index_migration_priority_absent_defaults(isolated):
    jc, _ = isolated
    jc.create_job("A")
    # No priority on disk → get_job_priority returns default.
    assert jc.get_job_priority("a") == jc.PRIORITY_WEIGHT_DEFAULT


def test_delete_refuses_any_active_slug(isolated):
    """A job in the active SET (not just at the head) can't be deleted."""
    jc, _ = isolated
    jc.create_job("A"); jc.create_job("B")
    jc.activate("a"); jc.activate("b")
    with pytest.raises(ValueError):
        jc.delete_job("b")                       # b is in the active set
    jc.deactivate("b")
    jc.delete_job("b")                           # now allowed


def test_delete_removes_priority_entry(isolated):
    jc, _ = isolated
    jc.create_job("A")
    jc.set_job_priority("a", 8)
    jc.delete_job("a")
    # Fresh get for a non-existent slug returns the DEFAULT weight (5).
    assert jc.get_job_priority("a") == jc.PRIORITY_WEIGHT_DEFAULT


def test_advance_adds_to_active_set(isolated):
    """v2: advance is ADDITIVE — the queue head joins the active set rather
    than replacing it."""
    jc, _ = isolated
    for n in ("A", "B", "C"):
        jc.create_job(n)
    jc.activate("a")
    jc.enqueue("b"); jc.enqueue("c")
    assert jc.advance() == "b"
    assert set(jc.get_active_slugs()) == {"a", "b"}
    assert jc.advance() == "c"
    assert set(jc.get_active_slugs()) == {"a", "b", "c"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
