"""Tests for the keyword query language in ``topic_filter``.

Covers the parser (parse_query), the dual-shape dispatcher (matches_query),
the env-var round-trip (via load_config), and the wire-up into ``passes``.
Every field (``keywords_extra``, ``banned_keywords``, ``generation_hints``) has
to keep accepting the legacy list-of-strings shape while ALSO accepting the
new query-expression string shape. Bare terms must remain substring-matched
(case-insensitive) so today's setups keep behaving.
"""
from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1] / "pipeline_code"
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import topic_filter as tf  # noqa: E402


# ── parse_query: primitives ──────────────────────────────────────────────────

def test_bare_word_is_substring_match():
    pred = tf.parse_query("product")
    assert pred("a nice product shot") is True
    assert pred("a landscape photo") is False


def test_bare_word_is_case_insensitive_via_lowercased_hay():
    pred = tf.parse_query("Product")  # query terms are lowered
    assert pred("shopify product listing") is True


def test_and_operator_requires_all_terms():
    pred = tf.parse_query("product AND packshot")
    assert pred("product packshot studio") is True
    assert pred("product only") is False
    assert pred("packshot only") is False


def test_or_operator_matches_any_term():
    pred = tf.parse_query("product OR packshot")
    assert pred("just a product") is True
    assert pred("packshot on white") is True
    assert pred("landscape photo") is False


def test_not_operator_negates_term():
    pred = tf.parse_query("product AND NOT nsfw")
    assert pred("clean product shot") is True
    assert pred("product with nsfw content") is False


def test_grouping_with_parens():
    pred = tf.parse_query(
        "(product AND packshot) OR (catalog AND studio)")
    assert pred("product packshot on white") is True
    assert pred("catalog studio spread") is True
    assert pred("product on the street") is False
    assert pred("catalog only") is False


def test_operator_keywords_are_case_insensitive():
    assert tf.parse_query("foo and bar")("foo bar baz") is True
    assert tf.parse_query("foo or bar")("bar") is True
    assert tf.parse_query("foo and not bar")("foo") is True
    assert tf.parse_query("foo and not bar")("foo bar") is False


def test_quoted_phrase_matches_phrase_with_spaces():
    pred = tf.parse_query('"studio lighting" AND packshot')
    assert pred("packshot with studio lighting") is True
    assert pred("packshot with natural lighting") is False


def test_implicit_or_between_bare_terms_backcompat():
    # foo bar baz  ==  foo OR bar OR baz (today's list-of-terms behaviour)
    pred = tf.parse_query("apple banana cherry")
    assert pred("i have an apple") is True
    assert pred("just a banana") is True
    assert pred("nothing here") is False


# ── parse_query: edge cases per spec ─────────────────────────────────────────

def test_empty_expression_returns_false():
    """Empty input yields the safe default: predicate returns False."""
    assert tf.parse_query("")("anything") is False
    assert tf.parse_query("   ")("anything") is False


def test_non_string_input_returns_false():
    assert tf.parse_query(None)("anything") is False  # type: ignore[arg-type]


def test_malformed_unclosed_paren_is_permissive(caplog):
    """A typo must NEVER take a scraper down — permissive predicate + warn."""
    caplog.set_level(logging.WARNING, logger="topic_filter")
    pred = tf.parse_query("(product AND packshot")  # missing close-paren
    assert pred("literally anything") is True
    # a warning was logged
    assert any(
        "invalid keyword query" in r.getMessage() for r in caplog.records
    )


def test_malformed_leading_operator_is_permissive(caplog):
    caplog.set_level(logging.WARNING, logger="topic_filter")
    pred = tf.parse_query("AND product")  # a bare operator can't lead
    assert pred("random text") is True
    assert caplog.records, "expected a warning to be logged"


def test_malformed_dangling_operator_is_permissive(caplog):
    caplog.set_level(logging.WARNING, logger="topic_filter")
    pred = tf.parse_query("product AND")
    assert pred("random text") is True


def test_deeply_nested_grouping_still_parses():
    pred = tf.parse_query("((a AND b) OR (c AND (d OR e)))")
    assert pred("a b") is True
    assert pred("c d") is True
    assert pred("c e") is True
    assert pred("a only") is False
    assert pred("c only") is False


def test_double_negation():
    pred = tf.parse_query("NOT NOT product")
    assert pred("product") is True
    assert pred("landscape") is False


# ── matches_query: dispatcher shape (list vs str) ────────────────────────────

def test_matches_query_list_is_implicit_or_of_substrings():
    """Legacy behaviour: list-of-terms is implicit OR of substring matches."""
    q = ["product", "packshot", "catalog"]
    assert tf.matches_query(q, "shopify product listing") is True
    assert tf.matches_query(q, "catalog page") is True
    assert tf.matches_query(q, "landscape photo") is False


def test_matches_query_empty_list_returns_false():
    assert tf.matches_query([], "anything") is False


def test_matches_query_string_uses_parser():
    assert tf.matches_query("product AND packshot",
                            "product packshot studio") is True
    assert tf.matches_query("product AND packshot",
                            "product only") is False


def test_matches_query_none_returns_false():
    assert tf.matches_query(None, "anything") is False


def test_matches_query_list_terms_are_case_insensitive():
    # Legacy substring match must be case-insensitive so today's dumped
    # config keeps behaving after a case flip in the source text.
    assert tf.matches_query(["Product"], "shopify product listing") is True


# ── env-var parsing: legacy CSV vs query auto-detection ──────────────────────

def test_env_field_empty_yields_empty_list():
    assert tf._parse_env_field("") == []
    assert tf._parse_env_field("   ") == []


def test_env_field_plain_csv_stays_list():
    assert tf._parse_env_field("a, b, c") == ["a", "b", "c"]


def test_env_field_with_parens_flips_to_query_string():
    q = tf._parse_env_field("(a OR b) AND c")
    assert isinstance(q, str)
    assert q == "(a OR b) AND c"


def test_env_field_with_and_or_word_flips_to_query_string():
    assert tf._parse_env_field("product AND packshot") == "product AND packshot"
    assert tf._parse_env_field("product OR packshot") == "product OR packshot"
    assert tf._parse_env_field("NOT nsfw") == "NOT nsfw"


def test_env_field_quoted_flips_to_query_string():
    assert tf._parse_env_field('"studio lighting"') == '"studio lighting"'


# ── passes(): end-to-end wire-up ─────────────────────────────────────────────

def _cfg(*, required=(), banned=(), hints=(), min_len: int = 0):
    """Build a TopicConfig for a single passes() call without env pollution."""
    return tf.TopicConfig(
        topic="test",
        required_keywords=required if isinstance(required, str) else list(required),
        banned_keywords=banned if isinstance(banned, str) else list(banned),
        generation_hints=hints if isinstance(hints, str) else list(hints),
        min_prompt_length=min_len,
    )


def test_passes_legacy_list_still_works(monkeypatch):
    """Backward compat: list-of-terms keeps rejecting on missing keyword."""
    monkeypatch.setenv("REQUIRE_PROMPT", "false")
    cfg = _cfg(required=["product", "packshot"])
    ok, reason = tf.passes("shopify catalog packshot", "", cfg=cfg)
    assert ok is True, reason
    ok, reason = tf.passes("landscape photo", "", cfg=cfg)
    assert ok is False
    assert "required" in reason


def test_passes_query_string_and_terms(monkeypatch):
    """New shape: an AND-of-terms query gates on BOTH being present."""
    monkeypatch.setenv("REQUIRE_PROMPT", "false")
    cfg = _cfg(required="product AND packshot")
    ok, _ = tf.passes("product packshot studio", "", cfg=cfg)
    assert ok is True
    ok, reason = tf.passes("product only", "", cfg=cfg)
    assert ok is False
    assert "required" in reason


def test_passes_query_string_with_not(monkeypatch):
    """A single-line required-plus-forbidden works via NOT."""
    monkeypatch.setenv("REQUIRE_PROMPT", "false")
    cfg = _cfg(required="product AND NOT nsfw")
    ok, _ = tf.passes("clean product shot", "", cfg=cfg)
    assert ok is True
    ok, reason = tf.passes("product with nsfw content", "", cfg=cfg)
    assert ok is False
    assert "required" in reason


def test_passes_banned_query_string_rejects(monkeypatch):
    monkeypatch.setenv("REQUIRE_PROMPT", "false")
    cfg = _cfg(required=(), banned="watermark OR onlyfans")
    ok, _ = tf.passes("clean shot", "", cfg=cfg)
    assert ok is True
    ok, reason = tf.passes("shot with watermark", "", cfg=cfg)
    assert ok is False
    assert "banned" in reason


def test_passes_empty_query_is_no_filter(monkeypatch):
    """An empty query means no filter, matching the legacy empty-list default."""
    monkeypatch.setenv("REQUIRE_PROMPT", "false")
    cfg = _cfg(required="")  # explicit empty string
    ok, reason = tf.passes("literally anything", "", cfg=cfg)
    assert ok is True, reason


# ── load_config(): env-var integration ───────────────────────────────────────

def test_load_config_reads_query_string_from_env(monkeypatch):
    monkeypatch.setenv("TOPIC_KEYWORDS_EXTRA", "(a OR b) AND NOT c")
    monkeypatch.setenv("TOPIC_BANNED_KEYWORDS", "junk")           # legacy CSV
    monkeypatch.setenv("TOPIC_GENERATION_HINTS", "cinematic")     # legacy CSV
    monkeypatch.setenv("REQUIRE_PROMPT", "false")
    cfg = tf.load_config()
    assert isinstance(cfg.required_keywords, str)
    assert cfg.required_keywords == "(a OR b) AND NOT c"
    assert isinstance(cfg.banned_keywords, list)
    assert cfg.banned_keywords == ["junk"]


def test_load_config_legacy_csv_still_becomes_list(monkeypatch):
    monkeypatch.setenv("TOPIC_KEYWORDS_EXTRA", "aerial, drone, overhead")
    cfg = tf.load_config()
    assert cfg.required_keywords == ["aerial", "drone", "overhead"]


# ── module surface + docstring smoke ─────────────────────────────────────────

def test_public_api_exports_new_helpers():
    assert "parse_query" in tf.__all__
    assert "matches_query" in tf.__all__


def test_module_docstring_documents_the_grammar():
    doc = tf.__doc__ or ""
    assert "AND" in doc and "OR" in doc and "NOT" in doc
    assert "or_expr" in doc  # grammar is spelled out
    assert "Examples" in doc


# ── round-trip: preset -> env -> load_config -> passes ───────────────────────

def test_query_string_preset_survives_env_projection(monkeypatch, tmp_path):
    """The primary flow: a preset stores keywords_extra as a string, resolve_env
    passes it through verbatim, and topic_filter.load_config picks it up as
    a query. Uses job_config in isolation so we don't touch the shared repo."""
    monkeypatch.setenv("PIPELINE_BASE_DIR", str(tmp_path))
    import categories
    monkeypatch.setattr(categories, "ACTIVE_PATH", tmp_path / "cull_categories.json")
    monkeypatch.setattr(categories, "_cache", None, raising=False)
    monkeypatch.setattr(categories, "_cache_mtime", 0.0, raising=False)
    import job_config as jc
    importlib.reload(jc)

    # Create a preset with a query-string keywords_extra and a job that uses it.
    base = jc.get_preset("default")
    base["topic_filters"]["keywords_extra"] = "(product OR packshot) AND NOT selfie"
    jc.save_preset("my_query_preset", base)
    job = jc.create_job("Query Job", preset="my_query_preset")

    env = jc.resolve_env(job)
    assert env["TOPIC_KEYWORDS_EXTRA"] == "(product OR packshot) AND NOT selfie"

    # Now feed that env into topic_filter.load_config and exercise passes().
    monkeypatch.setenv("TOPIC_KEYWORDS_EXTRA", env["TOPIC_KEYWORDS_EXTRA"])
    monkeypatch.setenv("REQUIRE_PROMPT", "false")
    monkeypatch.delenv("TOPIC_BANNED_KEYWORDS", raising=False)
    monkeypatch.delenv("TOPIC_GENERATION_HINTS", raising=False)
    cfg = tf.load_config()
    assert isinstance(cfg.required_keywords, str)

    ok, _ = tf.passes("shopify product listing", "", cfg=cfg)
    assert ok is True
    ok, reason = tf.passes("product selfie", "", cfg=cfg)
    assert ok is False
    assert "required" in reason


# ── config_io validation for the new dual shape ──────────────────────────────

def test_config_io_accepts_query_string_for_keyword_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_BASE_DIR", str(tmp_path))
    import categories
    monkeypatch.setattr(categories, "ACTIVE_PATH", tmp_path / "cull_categories.json")
    monkeypatch.setattr(categories, "_cache", None, raising=False)
    monkeypatch.setattr(categories, "_cache_mtime", 0.0, raising=False)
    import job_config  # noqa: F401 - required by config_io
    importlib.reload(job_config)
    import config_io
    importlib.reload(config_io)

    cfg = {
        "topic_filters": {
            "keywords_extra": "(a OR b) AND NOT c",
            "banned_keywords": ["nsfw"],       # legacy list still fine
            "generation_hints": "cinematic AND cfg",
        },
        "categories": [{"name": "Keep", "hint": "on topic"}],
    }
    cleaned = config_io.validate_inheritable_cfg(cfg, partial=False)
    assert cleaned["topic_filters"]["keywords_extra"] == "(a OR b) AND NOT c"
    assert cleaned["topic_filters"]["banned_keywords"] == ["nsfw"]
    assert cleaned["topic_filters"]["generation_hints"] == "cinematic AND cfg"


def test_config_io_rejects_non_string_non_list_keywords(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPELINE_BASE_DIR", str(tmp_path))
    import categories
    monkeypatch.setattr(categories, "ACTIVE_PATH", tmp_path / "cull_categories.json")
    monkeypatch.setattr(categories, "_cache", None, raising=False)
    monkeypatch.setattr(categories, "_cache_mtime", 0.0, raising=False)
    import job_config  # noqa: F401
    importlib.reload(job_config)
    import config_io
    importlib.reload(config_io)

    with pytest.raises(config_io.ValidationError):
        config_io.validate_inheritable_cfg(
            {"topic_filters": {"keywords_extra": {"bad": "shape"}}},
            partial=True,
        )
    with pytest.raises(config_io.ValidationError):
        config_io.validate_inheritable_cfg(
            {"topic_filters": {"keywords_extra": [1, 2, 3]}},
            partial=True,
        )
