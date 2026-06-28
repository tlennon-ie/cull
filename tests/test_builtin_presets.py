"""Tests for the shipped starter preset library (builtin_presets.py).

The Jobs v2 model seeds a global preset library a new job can inherit before
customising. The shipped library must be general-purpose (NO influencer/woman
bias in the default), cover the common dataset themes, and project cleanly into
the runtime env vars via job_config.

Runnable:
    pytest tests/test_builtin_presets.py
"""
from __future__ import annotations

import copy
import importlib
import json
import sys
from pathlib import Path

import pytest

PIPELINE_CODE = Path(__file__).resolve().parent.parent / "pipeline_code"
if str(PIPELINE_CODE) not in sys.path:
    sys.path.insert(0, str(PIPELINE_CODE))

import builtin_presets as bp  # noqa: E402


# Themed presets the user signed off on (general "default" + 7 image themes +
# the video-dataset family added in the roadmap wave-1 push).
EXPECTED_PRESETS = {
    "default",            # general dataset-prep triage — the new shipped default
    "aerial_drone",
    "underwater_marine",
    "wildlife_macro",
    "product_ecommerce",
    "anime_illustration",
    "photoreal_portrait",  # retained influencer/portrait taxonomy (NOT default)
    "quality_only",        # retained quality-only triage
    # video-dataset presets (LTX-Video / Wan / Hunyuan / CogVideoX audiences)
    "video_default",
    "video_cinematic",
    "video_anime",
    "video_product",
    "video_nature",
}

# Real photo themes curate image-first (often no generation prompt); AI-gen
# themes expect a prompt.
IMAGE_FIRST = {"aerial_drone", "underwater_marine", "wildlife_macro",
               "product_ecommerce", "quality_only"}
PROMPT_FIRST = {"default", "anime_illustration", "photoreal_portrait"}

INHERITABLE_BLOCKS = ("topic_filters", "categories", "category_rules",
                      "scoring", "captioning")


# ── library shape ────────────────────────────────────────────────────────────

def test_library_default_pointer_is_default_key():
    lib = bp.builtin_library()
    assert lib["default"] == "default"
    assert "default" in lib["presets"]


def test_library_has_all_expected_presets():
    names = set(bp.builtin_library()["presets"])
    assert names == EXPECTED_PRESETS


def test_builtin_library_returns_a_fresh_copy_each_call():
    a = bp.builtin_library()
    a["presets"]["default"]["scoring"]["ovr_min"] = 999
    b = bp.builtin_library()
    assert b["presets"]["default"]["scoring"]["ovr_min"] != 999


def test_every_preset_has_the_inheritable_blocks():
    for name, cfg in bp.builtin_library()["presets"].items():
        for block in INHERITABLE_BLOCKS:
            assert block in cfg, f"{name} missing {block}"
        assert isinstance(cfg["categories"], list) and cfg["categories"]
        for cat in cfg["categories"]:
            assert set(cat) >= {"name", "hint"}
            assert cat["name"] and isinstance(cat["name"], str)
        assert isinstance(cfg["category_rules"], str) and cfg["category_rules"].strip()
        sc = cfg["scoring"]
        assert {"ovr_min", "rel_min", "notes"} <= set(sc)
        cap = cfg["captioning"]
        assert {"enabled", "style", "overwrite"} <= set(cap)
        assert cap["enabled"] is False          # never auto-spend caption tokens


# ── the complaint: default must not be influencer-biased ─────────────────────

def test_default_preset_is_general_not_influencer():
    cats = [c["name"] for c in bp.builtin_library()["presets"]["default"]["categories"]]
    assert cats == ["Keep", "Borderline", "OffTopic"]
    assert "InstagramInfluencer" not in cats
    assert "NSFW" not in cats


def test_non_portrait_presets_drop_the_woman_gate():
    for name, cfg in bp.builtin_library()["presets"].items():
        if name == "photoreal_portrait":
            continue
        assert "woman_present" not in cfg["category_rules"], name
        assert "InstagramInfluencer" not in json.dumps(cfg), name


def test_portrait_preset_is_retained_with_its_taxonomy():
    cats = [c["name"] for c in
            bp.builtin_library()["presets"]["photoreal_portrait"]["categories"]]
    assert "InstagramInfluencer" in cats
    assert "NSFW" in cats


def test_no_preset_mentions_removed_zforfree_source():
    assert "ZForFree" not in json.dumps(bp.builtin_library())


# ── topic-filter / caption taste defaults ────────────────────────────────────

def test_require_prompt_matches_theme_kind():
    presets = bp.builtin_library()["presets"]
    for name in IMAGE_FIRST:
        assert presets[name]["topic_filters"]["require_prompt"] is False, name
    for name in PROMPT_FIRST:
        assert presets[name]["topic_filters"]["require_prompt"] is True, name


def test_caption_styles_are_valid_and_themed():
    valid = {"sd_prompt", "booru_tags", "natural_language"}
    presets = bp.builtin_library()["presets"]
    for name, cfg in presets.items():
        assert cfg["captioning"]["style"] in valid, name
    # anime is naturally tagged with booru tags
    assert presets["anime_illustration"]["captioning"]["style"] == "booru_tags"


# ── integration with job_config (projection must work) ───────────────────────

@pytest.fixture()
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_BASE_DIR", str(tmp_path))
    import categories
    monkeypatch.setattr(categories, "ACTIVE_PATH", tmp_path / "cull_categories.json")
    monkeypatch.setattr(categories, "_cache", None, raising=False)
    monkeypatch.setattr(categories, "_cache_mtime", 0.0, raising=False)
    import job_config
    importlib.reload(job_config)
    return job_config, tmp_path


def test_job_config_seeds_the_full_builtin_library(isolated):
    jc, _ = isolated
    lib = jc.list_presets()
    assert set(lib["presets"]) >= EXPECTED_PRESETS
    assert lib["default"] == "default"


def test_get_preset_fills_full_scraper_shape(isolated):
    jc, _ = isolated
    cfg = jc.get_preset("aerial_drone")
    # sparse builtin preset gets merged over the default shape
    assert "enabled" in cfg["scrapers"]
    assert "local_imports" in cfg["scrapers"]
    assert cfg["categories"][0]["name"] == "Keep"


def test_resolve_env_projects_require_prompt_per_preset(isolated):
    jc, _ = isolated
    aerial = jc.create_job("Aerial Set", preset="aerial_drone")
    general = jc.create_job("General Set", preset="default")
    assert jc.resolve_env(aerial)["REQUIRE_PROMPT"] == "false"
    assert jc.resolve_env(general)["REQUIRE_PROMPT"] == "true"


def test_list_presets_merges_builtins_into_existing_library(isolated):
    jc, tmp_path = isolated
    # Simulate an older install whose presets file predates the themed presets.
    presets_path = tmp_path / "jobs" / "_presets.json"
    presets_path.parent.mkdir(parents=True, exist_ok=True)
    custom = jc.get_preset("default")
    custom["scoring"]["notes"] = "user customised"
    presets_path.write_text(json.dumps({
        "default": "default",
        "presets": {"default": custom, "my_custom": custom},
    }), encoding="utf-8")
    lib = jc.list_presets()
    # builtin themed presets are added…
    assert "aerial_drone" in lib["presets"]
    assert "underwater_marine" in lib["presets"]
    # …without clobbering the user's own presets
    assert lib["presets"]["my_custom"]["scoring"]["notes"] == "user customised"
    assert lib["presets"]["default"]["scoring"]["notes"] == "user customised"


# ── seeded scraper targets / keywords / scoring floors ───────────────────────

def test_presets_seed_both_civitai_domains():
    for name, cfg in bp.builtin_library()["presets"].items():
        assert cfg["scrapers"]["civitai_domains"] == ["civitai.com", "civitai.red"], name


def test_themed_presets_seed_subreddits_and_keywords():
    presets = bp.builtin_library()["presets"]
    for name in ("aerial_drone", "underwater_marine", "wildlife_macro",
                 "product_ecommerce", "anime_illustration"):
        assert presets[name]["scrapers"]["reddit_subreddits"], name
        assert presets[name]["topic_filters"]["keywords_extra"], name


def test_all_presets_have_a_nonzero_ovr_floor():
    # The user asked for Min OVR/REL configured to suitable (non-zero) values.
    for name, cfg in bp.builtin_library()["presets"].items():
        assert cfg["scoring"]["ovr_min"] > 0, name


def test_every_preset_seeds_banned_keywords():
    # No preset relies on the invisible runtime spam-list fallback — each carries
    # its own banned list (spam terms + theme exclusions), so the field is visible.
    for name, cfg in bp.builtin_library()["presets"].items():
        banned = cfg["topic_filters"]["banned_keywords"]
        assert banned, name
        assert "onlyfans" in banned, name          # spam baseline carried forward


def test_generation_hints_seeded_except_lenient_presets():
    presets = bp.builtin_library()["presets"]
    for name in ("aerial_drone", "underwater_marine", "wildlife_macro",
                 "product_ecommerce", "anime_illustration", "photoreal_portrait"):
        assert presets[name]["topic_filters"]["generation_hints"], name
    # default + quality_only stay lenient: a generation-hint gate there would
    # silently reject any prompt missing those markers.
    assert presets["default"]["topic_filters"]["generation_hints"] == []
    assert presets["quality_only"]["topic_filters"]["generation_hints"] == []


def test_resolve_env_projects_seeded_scraper_targets(isolated):
    jc, _ = isolated
    job = jc.create_job("Aerial Set", preset="aerial_drone")
    env = jc.resolve_env(job)
    assert env["CIVITAI_DOMAINS"] == "civitai.com,civitai.red"
    assert "drones" in env["REDDIT_SUBREDDITS"]
    assert "DroneDJ" in env["X_ACCOUNTS"]
    assert int(env["VISION_OVR_MIN_SCORE"]) == 50


# ── managed-defaults refresh: stale builtins pick up new seed data ────────────
# (the reported bug: builtin_presets.py has seeded keywords/subreddits but an
# install whose _presets.json predates the seeding kept showing empty fields)

def _write_legacy_presets(tmp_path, presets: dict) -> Path:
    """Write a pre-baseline _presets.json (no _builtin_baselines key)."""
    p = tmp_path / "jobs" / "_presets.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"default": "default", "presets": presets}),
                 encoding="utf-8")
    return p


def test_list_presets_fills_stale_builtin_seed_fields(isolated):
    jc, tmp_path = isolated
    # An older install: aerial_drone exists but its seed fields predate the
    # seeding (empty keywords / subreddits / accounts, zero scoring floors).
    stale = jc.get_preset("aerial_drone")
    stale["topic_filters"]["keywords_extra"] = []
    stale["scrapers"]["reddit_subreddits"] = []
    stale["scrapers"]["x_accounts"] = []
    stale["scoring"]["ovr_min"] = 0
    stale["scoring"]["rel_min"] = 0
    user_made = jc.get_preset("default")
    user_made["scoring"]["notes"] = "hand tuned"
    _write_legacy_presets(tmp_path, {"aerial_drone": stale, "my_custom": user_made})

    lib = jc.list_presets()
    aerial = lib["presets"]["aerial_drone"]
    assert aerial["topic_filters"]["keywords_extra"], "keywords not refreshed"
    assert aerial["scrapers"]["reddit_subreddits"], "subreddits not refreshed"
    assert aerial["scrapers"]["x_accounts"], "x_accounts not refreshed"
    assert aerial["scoring"]["ovr_min"] > 0, "ovr floor not seeded"
    assert aerial["scoring"]["rel_min"] > 0, "rel floor not seeded"
    # a user-CREATED preset is never touched
    assert lib["presets"]["my_custom"]["scoring"]["notes"] == "hand tuned"


def test_list_presets_preserves_explicit_user_customisation_on_legacy_builtin(isolated):
    jc, tmp_path = isolated
    # On a legacy file we only fill EMPTY seed fields — a non-empty customisation
    # the user typed into a builtin must survive (no clobber).
    edited = jc.get_preset("aerial_drone")
    edited["scoring"]["notes"] = "do not clobber me"
    edited["category_rules"] = "MY CUSTOM RULES"
    _write_legacy_presets(tmp_path, {"aerial_drone": edited})

    aerial = jc.list_presets()["presets"]["aerial_drone"]
    assert aerial["scoring"]["notes"] == "do not clobber me"
    assert aerial["category_rules"] == "MY CUSTOM RULES"


def test_list_presets_preserves_user_edited_builtin_after_baseline(isolated):
    jc, _ = isolated
    jc.list_presets()                          # seed library + record baselines
    edited = jc.get_preset("aerial_drone")
    edited["scoring"]["notes"] = "my notes"
    jc.save_preset("aerial_drone", edited)     # user edits a builtin in the UI
    lib = jc.list_presets()                    # must NOT auto-refresh over it
    assert lib["presets"]["aerial_drone"]["scoring"]["notes"] == "my notes"


def test_list_presets_refreshes_unmodified_builtin_on_upgrade(isolated, monkeypatch):
    jc, _ = isolated
    jc.list_presets()                          # baselines recorded at current ship
    import builtin_presets as bp
    base = bp.builtin_library()

    def newer_ship():
        lib = copy.deepcopy(base)
        lib["presets"]["aerial_drone"]["scoring"]["notes"] = "ship v2 notes"
        return lib

    monkeypatch.setattr(bp, "builtin_library", newer_ship)
    lib = jc.list_presets()
    # unmodified builtin tracks the newer ship
    assert lib["presets"]["aerial_drone"]["scoring"]["notes"] == "ship v2 notes"


def test_reset_preset_to_builtin_restores_shipped(isolated):
    jc, _ = isolated
    jc.list_presets()
    edited = jc.get_preset("aerial_drone")
    edited["scoring"]["notes"] = "changed"
    edited["topic_filters"]["keywords_extra"] = []
    jc.save_preset("aerial_drone", edited)
    restored = jc.reset_preset_to_builtin("aerial_drone")
    assert restored["scoring"]["notes"] != "changed"
    assert restored["topic_filters"]["keywords_extra"]          # back from ship
    # re-baselined: a subsequent reconcile keeps the restored content
    assert jc.list_presets()["presets"]["aerial_drone"]["scoring"]["notes"] != "changed"


def test_reset_preset_rejects_non_builtin(isolated):
    jc, _ = isolated
    jc.list_presets()
    with pytest.raises(ValueError):
        jc.reset_preset_to_builtin("not_a_builtin_name")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
