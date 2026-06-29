"""Tests for the active-learning loop (``active_learning``).

``active_learning`` records user accept/reject/recategorize corrections and
turns them into two explainable signals:

  1. bounded nudges to the per-preset OVR / REL score thresholds, and
  2. a small few-shot exemplar set the classification prompt can inject.

Everything is deterministic and persisted per-slug under
``data/active_learning/<slug>.json`` (``paths.base_dir()``). These tests feed
synthetic corrections and assert the thresholds move in the correct direction
with bounded magnitude, that exemplars surface the most recent corrections,
and that state round-trips through a temp ``PIPELINE_BASE_DIR``.

Runnable:
    pytest tests/test_active_learning.py -q
    python tests/test_active_learning.py
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


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def al(tmp_path, monkeypatch):
    """A freshly-reloaded active_learning module pointed at a temp data dir."""
    monkeypatch.setenv("PIPELINE_BASE_DIR", str(tmp_path))
    import active_learning
    return importlib.reload(active_learning)


def _scores(ovr: int, rel: int) -> dict:
    """Build a scores dict using the real vision-schema keys."""
    return {"OVR_Quality_Score": ovr, "REL_Quality_Score": rel}


# ── persistence round-trip ─────────────────────────────────────────────────────

def test_record_creates_per_slug_file(al, tmp_path):
    al.record_correction("demo", "img1", "Keep", "DISCARD", _scores(40, 50))
    expected = tmp_path / "active_learning" / "demo.json"
    assert expected.exists()
    payload = json.loads(expected.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    assert payload.get("corrections")


def test_persistence_round_trips_across_reload(al, tmp_path, monkeypatch):
    al.record_correction("demo", "img1", "Keep", "DISCARD", _scores(40, 50))
    al.record_correction("demo", "img2", "DISCARD", "Keep", _scores(80, 70))
    # Simulate a fresh process: reload the module, same base dir.
    import active_learning
    reloaded = importlib.reload(active_learning)
    stats = reloaded.stats("demo")
    assert stats.get("DISCARD") == 1
    assert stats.get("Keep") == 1


def test_slugs_are_isolated(al):
    al.record_correction("alpha", "a1", "Keep", "DISCARD", _scores(30, 30))
    al.record_correction("beta", "b1", "DISCARD", "Keep", _scores(90, 90))
    assert al.stats("alpha") == {"DISCARD": 1}
    assert al.stats("beta") == {"Keep": 1}


def test_unknown_slug_is_empty_not_error(al):
    assert al.stats("nope") == {}
    assert al.exemplars("nope", "Keep") == []
    adj = al.suggest_threshold_adjustments("nope")
    assert adj["ovr_min_delta"] == 0
    assert adj["rel_min_delta"] == 0
    assert isinstance(adj["rationale"], str)


# ── stats ───────────────────────────────────────────────────────────────────

def test_stats_counts_per_corrected_category(al):
    al.record_correction("demo", "i1", "Keep", "DISCARD", _scores(40, 40))
    al.record_correction("demo", "i2", "Keep", "DISCARD", _scores(45, 45))
    al.record_correction("demo", "i3", "DISCARD", "Keep", _scores(85, 85))
    stats = al.stats("demo")
    assert stats == {"DISCARD": 2, "Keep": 1}


def test_recording_same_key_twice_updates_not_duplicates(al):
    al.record_correction("demo", "i1", "Keep", "DISCARD", _scores(40, 40))
    al.record_correction("demo", "i1", "DISCARD", "Keep", _scores(80, 80))
    # The same image corrected again should reflect the latest decision only.
    stats = al.stats("demo")
    assert stats == {"Keep": 1}


# ── threshold adjustments: direction ───────────────────────────────────────────

def test_false_keeps_push_thresholds_up(al):
    # Model KEPT these (non-terminal) but the user DROPPED them to DISCARD.
    # Their scores cleared whatever threshold was used, so the bar was too low
    # -> nudge thresholds UP (positive delta).
    for i in range(6):
        al.record_correction("demo", f"fk{i}", "Keep", "DISCARD", _scores(45, 48))
    adj = al.suggest_threshold_adjustments("demo")
    assert adj["ovr_min_delta"] > 0
    assert adj["rel_min_delta"] > 0
    assert "DISCARD" in adj["rationale"] or "drop" in adj["rationale"].lower()


def test_false_drops_pull_thresholds_down(al):
    # Model DISCARDED these but the user KEPT them. The threshold was too high
    # -> nudge thresholds DOWN (negative delta).
    for i in range(6):
        al.record_correction("demo", f"fd{i}", "DISCARD", "Keep", _scores(55, 52))
    adj = al.suggest_threshold_adjustments("demo")
    assert adj["ovr_min_delta"] < 0
    assert adj["rel_min_delta"] < 0


def test_no_corrections_means_no_movement(al):
    adj = al.suggest_threshold_adjustments("demo")
    assert adj["ovr_min_delta"] == 0
    assert adj["rel_min_delta"] == 0


def test_recategorize_between_keep_buckets_does_not_move_thresholds(al):
    # Moving between two non-terminal keep buckets is a taxonomy correction,
    # not a quality-threshold signal -> deltas stay zero.
    for i in range(5):
        al.record_correction("demo", f"rc{i}", "Professional", "Amateur", _scores(60, 60))
    adj = al.suggest_threshold_adjustments("demo")
    assert adj["ovr_min_delta"] == 0
    assert adj["rel_min_delta"] == 0


# ── threshold adjustments: bounded magnitude ───────────────────────────────────

def test_threshold_delta_is_bounded(al):
    # Even an extreme, lopsided pile of false-keeps must not blow past the cap.
    for i in range(500):
        al.record_correction("demo", f"x{i}", "Keep", "DISCARD", _scores(99, 99))
    adj = al.suggest_threshold_adjustments("demo")
    assert 0 < adj["ovr_min_delta"] <= al.MAX_DELTA
    assert 0 < adj["rel_min_delta"] <= al.MAX_DELTA


def test_threshold_delta_bounded_on_the_downside(al):
    for i in range(500):
        al.record_correction("demo", f"y{i}", "DISCARD", "Keep", _scores(1, 1))
    adj = al.suggest_threshold_adjustments("demo")
    assert -al.MAX_DELTA <= adj["ovr_min_delta"] < 0
    assert -al.MAX_DELTA <= adj["rel_min_delta"] < 0


def test_few_corrections_below_min_support_do_not_move(al):
    # A single correction shouldn't swing thresholds; we want enough evidence.
    al.record_correction("demo", "lonely", "Keep", "DISCARD", _scores(45, 45))
    adj = al.suggest_threshold_adjustments("demo")
    assert adj["ovr_min_delta"] == 0
    assert adj["rel_min_delta"] == 0


def test_rationale_is_explainable_string(al):
    for i in range(6):
        al.record_correction("demo", f"r{i}", "Keep", "DISCARD", _scores(40, 40))
    adj = al.suggest_threshold_adjustments("demo")
    assert isinstance(adj["rationale"], str)
    assert adj["rationale"].strip()


# ── exemplars ──────────────────────────────────────────────────────────────────

def test_exemplars_return_recent_corrections_for_category(al):
    al.record_correction("demo", "k1", "DISCARD", "Keep", _scores(80, 80),
                          caption="a woman by a window")
    al.record_correction("demo", "k2", "DISCARD", "Keep", _scores(82, 82),
                          caption="a portrait outdoors")
    ex = al.exemplars("demo", "Keep")
    assert len(ex) == 2
    keys = {e["image_key"] for e in ex}
    assert keys == {"k1", "k2"}
    # Caption is surfaced for prompt injection when present.
    assert any(e.get("caption") for e in ex)


def test_exemplars_respect_k_and_recency(al):
    for i in range(5):
        al.record_correction("demo", f"k{i}", "DISCARD", "Keep", _scores(70 + i, 70))
    ex = al.exemplars("demo", "Keep", k=3)
    assert len(ex) == 3
    # Most recent first: the last-recorded keys should be present.
    keys = [e["image_key"] for e in ex]
    assert "k4" in keys and "k3" in keys and "k2" in keys


def test_exemplars_filter_by_category(al):
    al.record_correction("demo", "keep1", "DISCARD", "Keep", _scores(80, 80))
    al.record_correction("demo", "drop1", "Keep", "DISCARD", _scores(20, 20))
    keep_ex = al.exemplars("demo", "Keep")
    drop_ex = al.exemplars("demo", "DISCARD")
    assert [e["image_key"] for e in keep_ex] == ["keep1"]
    assert [e["image_key"] for e in drop_ex] == ["drop1"]


def test_exemplars_include_note_field(al):
    al.record_correction("demo", "k1", "Professional", "Amateur", _scores(60, 60))
    ex = al.exemplars("demo", "Amateur")
    assert len(ex) == 1
    # A note documenting the predicted->corrected transition aids the prompt.
    assert "note" in ex[0]
    assert "Professional" in ex[0]["note"] and "Amateur" in ex[0]["note"]


def test_exemplars_default_k_is_three(al):
    for i in range(10):
        al.record_correction("demo", f"k{i}", "DISCARD", "Keep", _scores(75, 75))
    assert len(al.exemplars("demo", "Keep")) == 3


# ── input tolerance ────────────────────────────────────────────────────────────

def test_missing_scores_do_not_crash(al):
    al.record_correction("demo", "i1", "Keep", "DISCARD", {})
    # No usable scores: should be recorded for stats but not crash adjustments.
    assert al.stats("demo") == {"DISCARD": 1}
    adj = al.suggest_threshold_adjustments("demo")
    assert isinstance(adj["ovr_min_delta"], int)


def test_accepts_lowercase_score_aliases(al):
    # The index uses ``ovr``/``rel``; accept those as aliases of the schema keys.
    for i in range(6):
        al.record_correction("demo", f"i{i}", "Keep", "DISCARD", {"ovr": 45, "rel": 45})
    adj = al.suggest_threshold_adjustments("demo")
    assert adj["ovr_min_delta"] > 0


def test_corrupt_state_file_recovers(al, tmp_path):
    al.record_correction("demo", "i1", "Keep", "DISCARD", _scores(40, 40))
    state_file = tmp_path / "active_learning" / "demo.json"
    state_file.write_text("{ this is not json", encoding="utf-8")
    # A corrupt file must not raise; it falls back to empty and can be rewritten.
    assert al.stats("demo") == {}
    al.record_correction("demo", "i2", "DISCARD", "Keep", _scores(80, 80))
    assert al.stats("demo") == {"Keep": 1}


def test_blank_image_key_is_rejected(al):
    al.record_correction("demo", "", "Keep", "DISCARD", _scores(40, 40))
    al.record_correction("demo", "   ", "Keep", "DISCARD", _scores(40, 40))
    assert al.stats("demo") == {}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
