"""tests/test_fleet_health.py — pytest suite for fleet_health.py + gbnf_grammar.py.

TDD RED phase: written before the implementation exists.

Run with:
    .venv/Scripts/python.exe -m pytest tests/test_fleet_health.py -q

NO real network. ``fleet_health.probe`` routes every HTTP call through the
private ``_http_get`` helper, which the tests monkeypatch. Telemetry and the
GBNF converter are pure (no IO), so they're exercised directly.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

# Flat-namespace import of pipeline_code, mirroring the other suites in tests/.
PIPELINE_CODE = Path(__file__).resolve().parent.parent / "pipeline_code"
if str(PIPELINE_CODE) not in sys.path:
    sys.path.insert(0, str(PIPELINE_CODE))


def _import(mod_name: str):
    """Import (or re-import) a module from pipeline_code/ with a clean slate."""
    sys.modules.pop(mod_name, None)
    spec = importlib.util.spec_from_file_location(
        mod_name, PIPELINE_CODE / f"{mod_name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def fh():
    return _import("fleet_health")


@pytest.fixture()
def gg():
    return _import("gbnf_grammar")


# Fleet instances in the shape clean_vision_fleet() emits.
def _ep(eid: str, provider: str, base_url: str) -> dict:
    return {"id": eid, "name": eid.upper(), "provider": provider,
            "base_url": base_url, "model": "", "api_key": ""}


def _patch_get(monkeypatch, fh, responses):
    """Replace fleet_health._http_get.

    ``responses`` maps a base_url -> (status, latency_ms) OR an Exception
    instance to raise. The latency is what the fake clock advances by so
    probe() reports a deterministic value.
    """
    calls = []

    def fake_get(url, *, headers=None, timeout=5.0):
        calls.append({"url": url, "headers": headers or {}, "timeout": timeout})
        for base, outcome in responses.items():
            if url.startswith(base):
                if isinstance(outcome, Exception):
                    raise outcome
                return outcome  # (status, latency_ms)
        return (404, 1)

    monkeypatch.setattr(fh, "_http_get", fake_get)
    return calls


# ===========================================================================
# 1. probe() — provider-correct endpoint, latency, ok/err shape
# ===========================================================================

class TestProbe:
    def test_lmstudio_probes_v1_models(self, fh, monkeypatch):
        calls = _patch_get(monkeypatch, fh, {"http://h:1234": (200, 12)})
        result = fh.probe(_ep("a", "lmstudio", "http://h:1234"))
        assert calls[0]["url"].rstrip("/").endswith("/v1/models")
        assert result["ok"] is True

    def test_llamacpp_probes_v1_models(self, fh, monkeypatch):
        calls = _patch_get(monkeypatch, fh, {"http://h:8080": (200, 5)})
        fh.probe(_ep("a", "llamacpp", "http://h:8080"))
        assert "/v1/models" in calls[0]["url"]

    def test_ollama_probes_api_tags(self, fh, monkeypatch):
        calls = _patch_get(monkeypatch, fh, {"http://h:11434": (200, 7)})
        fh.probe(_ep("a", "ollama", "http://h:11434"))
        assert "/api/tags" in calls[0]["url"]

    def test_result_shape_has_ok_latency_error(self, fh, monkeypatch):
        _patch_get(monkeypatch, fh, {"http://h:1234": (200, 9)})
        result = fh.probe(_ep("a", "lmstudio", "http://h:1234"))
        assert set(("ok", "latency_ms", "error")).issubset(result.keys())

    def test_ok_true_has_int_latency(self, fh, monkeypatch):
        _patch_get(monkeypatch, fh, {"http://h:1234": (200, 33)})
        result = fh.probe(_ep("a", "lmstudio", "http://h:1234"))
        assert isinstance(result["latency_ms"], int)
        assert result["latency_ms"] >= 0
        assert result["error"] is None

    def test_non_2xx_is_not_ok(self, fh, monkeypatch):
        _patch_get(monkeypatch, fh, {"http://h:1234": (500, 4)})
        result = fh.probe(_ep("a", "lmstudio", "http://h:1234"))
        assert result["ok"] is False
        assert result["error"]  # non-empty error string

    def test_401_is_not_ok(self, fh, monkeypatch):
        _patch_get(monkeypatch, fh, {"http://h:1234": (401, 4)})
        result = fh.probe(_ep("a", "lmstudio", "http://h:1234"))
        assert result["ok"] is False

    def test_bearer_header_set_when_api_key_present(self, fh, monkeypatch):
        calls = _patch_get(monkeypatch, fh, {"http://h:1234": (200, 4)})
        ep = _ep("a", "lmstudio", "http://h:1234")
        ep["api_key"] = "sk-secret"
        fh.probe(ep)
        assert calls[0]["headers"].get("Authorization") == "Bearer sk-secret"

    def test_no_auth_header_without_api_key(self, fh, monkeypatch):
        calls = _patch_get(monkeypatch, fh, {"http://h:1234": (200, 4)})
        fh.probe(_ep("a", "lmstudio", "http://h:1234"))
        assert "Authorization" not in calls[0]["headers"]

    def test_connection_error_is_not_ok_and_never_raises(self, fh, monkeypatch):
        import requests
        _patch_get(monkeypatch, fh,
                   {"http://h:1234": requests.exceptions.ConnectionError("refused")})
        result = fh.probe(_ep("a", "lmstudio", "http://h:1234"))
        assert result["ok"] is False
        assert "refused" in result["error"] or "conn" in result["error"].lower()

    def test_timeout_is_not_ok(self, fh, monkeypatch):
        import requests
        _patch_get(monkeypatch, fh,
                   {"http://h:1234": requests.exceptions.Timeout("slow")})
        result = fh.probe(_ep("a", "lmstudio", "http://h:1234"))
        assert result["ok"] is False
        assert result["latency_ms"] is None

    def test_unknown_provider_is_not_ok_no_network(self, fh, monkeypatch):
        calls = _patch_get(monkeypatch, fh, {"http://h:1": (200, 1)})
        result = fh.probe(_ep("a", "bogus", "http://h:1"))
        assert result["ok"] is False
        assert calls == []  # bad provider short-circuits before any HTTP

    def test_blank_base_url_is_not_ok_no_network(self, fh, monkeypatch):
        calls = _patch_get(monkeypatch, fh, {"http://h:1": (200, 1)})
        result = fh.probe(_ep("a", "lmstudio", ""))
        assert result["ok"] is False
        assert calls == []

    def test_trailing_slash_base_url_not_double_slashed(self, fh, monkeypatch):
        calls = _patch_get(monkeypatch, fh, {"http://h:1234": (200, 1)})
        fh.probe(_ep("a", "lmstudio", "http://h:1234/"))
        assert "//v1/models" not in calls[0]["url"].split("://", 1)[1]


# ===========================================================================
# 1b. _http_get scheme guard + no-redirect (SSRF hardening)
# ===========================================================================

class TestHttpGetGuard:
    """The real ``_http_get`` (not the monkeypatched stub) must reject a non-
    http(s) scheme and disable redirects so a hostile endpoint can't 302-bounce
    the probe at a cloud-metadata address (169.254.169.254). Mirrors
    ``vision_model_catalog._safe_get``."""

    def test_non_http_scheme_raises_valueerror(self, fh, monkeypatch):
        def boom(*a, **k):  # pragma: no cover - must never run
            raise AssertionError("requests.get must not be called for a bad scheme")

        monkeypatch.setattr(fh.requests, "get", boom)
        with pytest.raises(ValueError):
            fh._http_get("file:///etc/passwd")
        with pytest.raises(ValueError):
            fh._http_get("ftp://host/x")

    def test_passes_allow_redirects_false(self, fh, monkeypatch):
        captured: dict = {}

        class _Resp:
            status_code = 200

        def fake_get(url, **kwargs):
            captured.update(kwargs)
            captured["url"] = url
            return _Resp()

        monkeypatch.setattr(fh.requests, "get", fake_get)
        status, latency = fh._http_get("http://h:1234/v1/models")
        assert status == 200
        assert isinstance(latency, int) and latency >= 0
        assert captured["allow_redirects"] is False

    def test_probe_with_non_http_base_url_is_not_ok_never_raises(self, fh):
        # probe() must surface the _http_get ValueError as a failed probe, not an
        # exception (it has its own catch-all). No monkeypatch: the real guard runs.
        result = fh.probe(_ep("a", "lmstudio", "file:///etc/passwd"))
        assert result["ok"] is False
        assert result["error"]


# ===========================================================================
# 2. pick_healthy() — order by health then latency, drop failures
# ===========================================================================

class TestPickHealthy:
    def test_healthy_before_unhealthy(self, fh):
        fleet = [_ep("slow", "lmstudio", "http://a:1"),
                 _ep("down", "ollama", "http://b:1"),
                 _ep("fast", "lmstudio", "http://c:1")]
        probes = {
            "slow": {"ok": True, "latency_ms": 80, "error": None},
            "down": {"ok": False, "latency_ms": None, "error": "boom"},
            "fast": {"ok": True, "latency_ms": 10, "error": None},
        }
        ordered = fh.pick_healthy(fleet, probes)
        ids = [w["id"] for w in ordered]
        assert ids == ["fast", "slow"]  # healthy, latency-ascending; down dropped

    def test_orders_strictly_by_latency(self, fh):
        fleet = [_ep("c", "lmstudio", "http://c:1"),
                 _ep("a", "lmstudio", "http://a:1"),
                 _ep("b", "lmstudio", "http://b:1")]
        probes = {
            "c": {"ok": True, "latency_ms": 300, "error": None},
            "a": {"ok": True, "latency_ms": 100, "error": None},
            "b": {"ok": True, "latency_ms": 200, "error": None},
        }
        assert [w["id"] for w in fh.pick_healthy(fleet, probes)] == ["a", "b", "c"]

    def test_drops_all_when_none_healthy(self, fh):
        fleet = [_ep("x", "lmstudio", "http://x:1")]
        probes = {"x": {"ok": False, "latency_ms": None, "error": "down"}}
        assert fh.pick_healthy(fleet, probes) == []

    def test_missing_probe_treated_as_unhealthy(self, fh):
        fleet = [_ep("x", "lmstudio", "http://x:1"),
                 _ep("y", "lmstudio", "http://y:1")]
        probes = {"y": {"ok": True, "latency_ms": 5, "error": None}}  # x absent
        assert [w["id"] for w in fh.pick_healthy(fleet, probes)] == ["y"]

    def test_none_latency_on_healthy_sorts_last(self, fh):
        # A healthy probe with no measured latency must not crash the sort and
        # should rank behind a healthy probe that DID report a latency.
        fleet = [_ep("nolat", "lmstudio", "http://a:1"),
                 _ep("haslat", "lmstudio", "http://b:1")]
        probes = {
            "nolat": {"ok": True, "latency_ms": None, "error": None},
            "haslat": {"ok": True, "latency_ms": 50, "error": None},
        }
        assert [w["id"] for w in fh.pick_healthy(fleet, probes)] == ["haslat", "nolat"]

    def test_returns_full_endpoint_dicts(self, fh):
        fleet = [_ep("a", "lmstudio", "http://a:1")]
        probes = {"a": {"ok": True, "latency_ms": 5, "error": None}}
        out = fh.pick_healthy(fleet, probes)
        assert out[0]["base_url"] == "http://a:1"
        assert out[0]["provider"] == "lmstudio"

    def test_empty_fleet_returns_empty(self, fh):
        assert fh.pick_healthy([], {}) == []


# ===========================================================================
# 3. Telemetry — pure aggregation (tokens/sec, p50/p95, err rate, ETA)
# ===========================================================================

class TestTelemetry:
    def _agg(self, fh):
        return fh.Telemetry()

    def test_empty_report_is_safe(self, fh):
        rep = self._agg(fh).report()
        assert rep["count"] == 0
        assert rep["error_rate"] == 0.0
        assert rep["tokens_per_sec"] == 0.0
        # percentiles default to 0 with no samples
        assert rep["latency_p50_ms"] == 0.0
        assert rep["latency_p95_ms"] == 0.0

    def test_error_rate(self, fh):
        agg = self._agg(fh)
        for ok in (True, True, True, False):  # 1 of 4 failed
            agg.add(tokens=10, latency_ms=100, ok=ok)
        assert agg.report()["error_rate"] == pytest.approx(0.25)

    def test_p50_p95_on_known_distribution(self, fh):
        agg = self._agg(fh)
        for v in range(1, 101):  # latencies 1..100
            agg.add(tokens=1, latency_ms=v, ok=True)
        rep = agg.report()
        # Nearest-rank: p50 -> index ceil(0.50*100)=50 -> value 50;
        # p95 -> ceil(0.95*100)=95 -> value 95.
        assert rep["latency_p50_ms"] == pytest.approx(50, abs=1.0)
        assert rep["latency_p95_ms"] == pytest.approx(95, abs=1.0)

    def test_single_sample_percentiles_equal_value(self, fh):
        agg = self._agg(fh)
        agg.add(tokens=5, latency_ms=42, ok=True)
        rep = agg.report()
        assert rep["latency_p50_ms"] == pytest.approx(42)
        assert rep["latency_p95_ms"] == pytest.approx(42)

    def test_tokens_per_sec_uses_summed_tokens_over_summed_latency(self, fh):
        agg = self._agg(fh)
        # 100 tokens total over 0.5 s of wall-equivalent latency -> 200 tok/s
        agg.add(tokens=40, latency_ms=200, ok=True)
        agg.add(tokens=60, latency_ms=300, ok=True)
        assert agg.report()["tokens_per_sec"] == pytest.approx(200.0, rel=1e-6)

    def test_failed_samples_excluded_from_latency_percentiles(self, fh):
        agg = self._agg(fh)
        agg.add(tokens=0, latency_ms=9999, ok=False)  # error shouldn't skew p50
        agg.add(tokens=10, latency_ms=10, ok=True)
        agg.add(tokens=10, latency_ms=20, ok=True)
        rep = agg.report()
        assert rep["latency_p95_ms"] <= 20

    def test_queue_drain_eta_seconds(self, fh):
        agg = self._agg(fh)
        # throughput 2 img/s, 100 remaining -> 50 s
        eta = agg.queue_drain_eta(remaining_count=100, throughput_per_sec=2.0)
        assert eta == pytest.approx(50.0)

    def test_queue_drain_eta_zero_remaining(self, fh):
        agg = self._agg(fh)
        assert agg.queue_drain_eta(remaining_count=0, throughput_per_sec=2.0) == 0.0

    def test_queue_drain_eta_zero_throughput_is_infinite(self, fh):
        agg = self._agg(fh)
        eta = agg.queue_drain_eta(remaining_count=10, throughput_per_sec=0.0)
        assert eta == math.inf or eta is None

    def test_report_includes_throughput_field(self, fh):
        agg = self._agg(fh)
        agg.add(tokens=10, latency_ms=100, ok=True)
        rep = agg.report()
        assert "count" in rep and rep["count"] == 1


# ===========================================================================
# 4. json_schema_to_gbnf — well-formed grammar for the REAL vision schema
# ===========================================================================

class TestGbnfGrammar:
    def _real_schema(self):
        import vision_prompt
        return vision_prompt.build_response_format()["json_schema"]["schema"]

    def test_returns_str(self, gg):
        out = gg.json_schema_to_gbnf({"type": "object", "properties": {},
                                      "required": []})
        assert isinstance(out, str) and out.strip()

    def test_has_root_rule(self, gg):
        out = gg.json_schema_to_gbnf(self._real_schema())
        # llama.cpp GBNF must define a `root` rule.
        assert any(line.lstrip().startswith("root ") or line.lstrip().startswith("root:")
                   or line.lstrip().startswith("root ::=")
                   for line in out.splitlines())
        assert "::=" in out  # uses GBNF production syntax

    def test_contains_required_keys_as_literals(self, gg):
        out = gg.json_schema_to_gbnf(self._real_schema())
        for key in ("description", "primary_subject", "category",
                    "OVR_Quality_Score", "caption"):
            assert f'"{key}"' in out or f'\\"{key}\\"' in out, f"{key} missing"

    def test_enum_alternation_present(self, gg):
        out = gg.json_schema_to_gbnf(self._real_schema())
        # art_medium and category are enums -> alternation with "photograph".
        assert "photograph" in out
        assert "|" in out  # GBNF alternation operator

    def test_category_enum_values_present(self, gg):
        from categories import get_schema_categories
        out = gg.json_schema_to_gbnf(self._real_schema())
        for cat in get_schema_categories():
            assert cat in out, f"category enum value {cat!r} missing from grammar"

    def test_boolean_and_number_primitives_defined(self, gg):
        out = gg.json_schema_to_gbnf(self._real_schema())
        low = out.lower()
        # boolean fields (is_screenshot, nsfw, ...) and integer scores need rules.
        assert "true" in low and "false" in low
        # a digit class for the numeric score fields
        assert "0-9" in out or "[0-9]" in out

    def test_string_rule_defined(self, gg):
        out = gg.json_schema_to_gbnf(self._real_schema())
        # Some string production must exist (escaped-char class is typical).
        assert "string" in out.lower()

    def test_braces_present_for_object(self, gg):
        out = gg.json_schema_to_gbnf(self._real_schema())
        # The object rule must emit literal { and } for the JSON object.
        assert "{" in out and "}" in out

    def test_required_keys_count_matches_schema(self, gg):
        schema = self._real_schema()
        out = gg.json_schema_to_gbnf(schema)
        for key in schema["required"]:
            assert f'"{key}"' in out, f"required key {key!r} not in grammar"

    def test_pure_function_no_mutation_of_input(self, gg):
        import copy
        schema = self._real_schema()
        snapshot = copy.deepcopy(schema)
        gg.json_schema_to_gbnf(schema)
        assert schema == snapshot  # input dict untouched

    def test_empty_object_schema_still_valid(self, gg):
        out = gg.json_schema_to_gbnf({"type": "object", "properties": {},
                                      "required": []})
        assert "root" in out and "{" in out and "}" in out


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
