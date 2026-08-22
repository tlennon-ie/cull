"""tests/test_fleet_wiring.py — last-mile wiring for the vision fleet.

Covers two gated, default-OFF behaviours:

  * Supervisor failover (``run_pipeline``): when ``VISION_FAILOVER_ENABLED`` is
    set, ``_vision_fleet_specs`` health-probes the fleet and skips endpoints whose
    probe failed. When unset it must spawn EVERY enabled instance and never probe.

  * llama.cpp GBNF grammar (``vision_worker_balanced_openai``): when the worker
    talks to llama.cpp (``OPENAI_COMPAT_PROVIDER=llamacpp`` or
    ``VISION_GRAMMAR_ENABLED``) the chat-completions request body carries a
    ``grammar`` field; otherwise the body is unchanged (no ``grammar`` key).

Run with:
    .venv/Scripts/python.exe -m pytest tests/test_fleet_wiring.py -q

NO real network: failover probing is mocked at ``fleet_health.probe_fleet`` and
the HTTP call is mocked at ``requests.post`` on the worker module.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

PIPELINE_CODE = Path(__file__).resolve().parent.parent / "pipeline_code"
if str(PIPELINE_CODE) not in sys.path:
    sys.path.insert(0, str(PIPELINE_CODE))


@pytest.fixture(autouse=True)
def _restore_environ():
    """Snapshot/restore os.environ around every test.

    Importing ``run_pipeline`` runs its module-level ``load_dotenv()`` which can
    mutate the real environment; snapshotting here (autouse, so it runs first)
    keeps this suite from leaking the repo ``.env`` into sibling suites.
    """
    snapshot = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(snapshot)


# ════════════════════════════════════════════════════════════════════════════
# TASK 1 — supervisor failover
# ════════════════════════════════════════════════════════════════════════════

@pytest.fixture()
def rp(monkeypatch):
    """run_pipeline with PIPELINE_CODE_DIR pinned so script-existence checks pass.

    The repo ``.env`` ships ``PIPELINE_CODE_DIR=""`` → ``Path(".")`` at import,
    which would make ``compute_desired_agents``'s add() existence check look in
    cwd and drop every worker. Pin it to the real source dir (mirrors
    test_supervisor_jobs). Registry workers + local folders are silenced so the
    only agents in play are the vision fleet under test.
    """
    import run_pipeline
    monkeypatch.setattr(run_pipeline, "PIPELINE_CODE_DIR", PIPELINE_CODE)
    monkeypatch.setenv("PIPELINE_VISION_WORKERS", "")  # no registry/Groq workers
    monkeypatch.setenv("LOCAL_IMPORTS_JSON", "[]")     # no local feeders
    monkeypatch.setenv("SCRAPER_DISABLED", "")
    return run_pipeline


_TWO_ENDPOINT_FLEET = [
    {"id": "live", "name": "Live", "provider": "lmstudio",
     "base_url": "http://up:1234", "model": "m", "api_key": ""},
    {"id": "dead", "name": "Dead", "provider": "lmstudio",
     "base_url": "http://down:1234", "model": "m", "api_key": ""},
]


def _patch_probe(monkeypatch, healthy_ids):
    """Replace fleet_health.probe_fleet so 'down' endpoints report ok=False.

    Returns the list of fleets actually probed, so a test can assert probing did
    NOT happen on the default path.
    """
    import fleet_health
    probed: list = []

    def fake_probe_fleet(fleet, timeout=5.0):
        probed.append({"fleet": fleet, "timeout": timeout})
        return {
            str(ep.get("id")): {
                "ok": str(ep.get("id")) in healthy_ids,
                "latency_ms": 10,
                "error": None if str(ep.get("id")) in healthy_ids else "down",
            }
            for ep in fleet
        }

    monkeypatch.setattr(fleet_health, "probe_fleet", fake_probe_fleet)
    return probed


def _vision_labels(specs) -> set[str]:
    return {s.label for s in specs}


def test_failover_off_spawns_all_and_never_probes(rp, monkeypatch):
    """Default (flag unset): both instances fan out and probe_fleet is NOT called."""
    monkeypatch.setenv("VISION_WORKERS_JSON", json.dumps(_TWO_ENDPOINT_FLEET))
    monkeypatch.delenv("VISION_FAILOVER_ENABLED", raising=False)
    probed = _patch_probe(monkeypatch, healthy_ids={"live"})

    specs = rp._vision_fleet_specs()

    assert _vision_labels(specs) == {"Vision-Live", "Vision-Dead"}
    assert probed == [], "probe_fleet must not run when failover is disabled"


def test_failover_on_skips_unhealthy_endpoint(rp, monkeypatch):
    """Flag set: the dead endpoint is probed, fails, and is skipped — only the
    healthy worker is fanned out."""
    monkeypatch.setenv("VISION_WORKERS_JSON", json.dumps(_TWO_ENDPOINT_FLEET))
    monkeypatch.setenv("VISION_FAILOVER_ENABLED", "true")
    probed = _patch_probe(monkeypatch, healthy_ids={"live"})

    specs = rp._vision_fleet_specs()

    assert _vision_labels(specs) == {"Vision-Live"}
    assert len(probed) == 1, "probe_fleet must run exactly once when failover is on"


def test_failover_on_all_healthy_spawns_all(rp, monkeypatch):
    """Flag set but every endpoint healthy → nothing is skipped."""
    monkeypatch.setenv("VISION_WORKERS_JSON", json.dumps(_TWO_ENDPOINT_FLEET))
    monkeypatch.setenv("VISION_FAILOVER_ENABLED", "true")
    _patch_probe(monkeypatch, healthy_ids={"live", "dead"})

    specs = rp._vision_fleet_specs()
    assert _vision_labels(specs) == {"Vision-Live", "Vision-Dead"}


def test_failover_on_passes_probe_timeout(rp, monkeypatch):
    """VISION_PROBE_TIMEOUT is forwarded to probe_fleet (default 5 when unset)."""
    monkeypatch.setenv("VISION_WORKERS_JSON", json.dumps(_TWO_ENDPOINT_FLEET))
    monkeypatch.setenv("VISION_FAILOVER_ENABLED", "1")
    monkeypatch.setenv("VISION_PROBE_TIMEOUT", "3")
    probed = _patch_probe(monkeypatch, healthy_ids={"live"})

    rp._vision_fleet_specs()
    assert probed and probed[0]["timeout"] == 3.0


def test_failover_on_probe_error_falls_back_to_full_fleet(rp, monkeypatch):
    """A probing exception must not starve the pipeline — fall back to all."""
    import fleet_health
    monkeypatch.setenv("VISION_WORKERS_JSON", json.dumps(_TWO_ENDPOINT_FLEET))
    monkeypatch.setenv("VISION_FAILOVER_ENABLED", "true")

    def boom(fleet, timeout=5.0):
        raise RuntimeError("probe exploded")

    monkeypatch.setattr(fleet_health, "probe_fleet", boom)
    specs = rp._vision_fleet_specs()
    assert _vision_labels(specs) == {"Vision-Live", "Vision-Dead"}


def test_failover_off_compute_desired_byte_identical(rp, monkeypatch):
    """End-to-end through compute_desired_agents: with the flag unset, the dead
    endpoint still yields a desired Vision agent (no probing/skipping)."""
    monkeypatch.setenv("VISION_WORKERS_JSON", json.dumps(_TWO_ENDPOINT_FLEET))
    monkeypatch.delenv("VISION_FAILOVER_ENABLED", raising=False)
    _patch_probe(monkeypatch, healthy_ids={"live"})

    agents = rp.compute_desired_agents("topic")
    vision = {label for label in agents if label.startswith("Vision-")}
    assert vision == {"Vision-Live", "Vision-Dead"}


# ════════════════════════════════════════════════════════════════════════════
# TASK 2 — llama.cpp GBNF grammar in the request body
# ════════════════════════════════════════════════════════════════════════════

class _FakeResponse:
    """Minimal requests.Response stand-in returning a valid schema-shaped JSON."""
    status_code = 200

    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self) -> dict:
        return self._payload


_VALID_MESSAGE = {
    "choices": [{"message": {"content": json.dumps({"description": "x"})}}]
}


def _make_worker(monkeypatch, **env):
    """Construct BalancedOpenAICompatWorker with the given env, HTTP mocked.

    Returns ``(worker, captured)`` where ``captured`` accumulates the kwargs of
    every ``requests.post`` call so a test can inspect the JSON body. A model is
    pre-set so ``classify_image_bytes`` runs without an autodetect round-trip.
    The vision-JSON parser is stubbed to a trivial pass-through so the fake
    response is accepted regardless of schema specifics.
    """
    import vision_worker_balanced_openai as mod

    monkeypatch.setenv("OPENAI_COMPAT_MODEL", "test-model")
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    captured: list = []

    def fake_post(url, **kwargs):
        captured.append({"url": url, **kwargs})
        return _FakeResponse(_VALID_MESSAGE)

    monkeypatch.setattr(mod.requests, "post", fake_post)
    # Make parsing deterministic and independent of the response schema.
    monkeypatch.setattr(mod, "_safe_parse_vision_json", lambda raw: {"ok": True})

    worker = mod.BalancedOpenAICompatWorker()
    return worker, captured


def _body_of(captured) -> dict:
    assert captured, "requests.post was never called"
    return captured[0]["json"]


def test_grammar_present_for_llamacpp_provider(monkeypatch):
    """OPENAI_COMPAT_PROVIDER=llamacpp → request body carries a non-empty GBNF
    grammar string."""
    worker, captured = _make_worker(monkeypatch, OPENAI_COMPAT_PROVIDER="llamacpp")
    worker.classify_image_bytes("ZmFrZQ==", "instruction")

    body = _body_of(captured)
    assert "grammar" in body
    assert isinstance(body["grammar"], str) and body["grammar"].strip()
    assert "root ::=" in body["grammar"]            # looks like real GBNF
    assert body["response_format"]["type"] == "json_schema"  # kept as fallback


def test_grammar_present_when_flag_enabled(monkeypatch):
    """VISION_GRAMMAR_ENABLED=true also turns the grammar on (provider unset)."""
    worker, captured = _make_worker(monkeypatch, VISION_GRAMMAR_ENABLED="true")
    worker.classify_image_bytes("ZmFrZQ==", "instruction")
    assert "grammar" in _body_of(captured)


def test_grammar_absent_for_lmstudio_default(monkeypatch):
    """Default / lmstudio provider → NO grammar key (body unchanged from before)."""
    worker, captured = _make_worker(monkeypatch, OPENAI_COMPAT_PROVIDER="lmstudio")
    worker.classify_image_bytes("ZmFrZQ==", "instruction")
    assert "grammar" not in _body_of(captured)


def test_grammar_absent_when_gate_off(monkeypatch):
    """No provider hint and no flag → grammar gate off, no grammar key."""
    worker, captured = _make_worker(monkeypatch)  # nothing set
    worker.classify_image_bytes("ZmFrZQ==", "instruction")
    body = _body_of(captured)
    assert "grammar" not in body
    # response_format path is intact.
    assert body["response_format"]["type"] == "json_schema"


def test_grammar_build_failure_falls_back(monkeypatch):
    """If grammar conversion raises, the request still goes out via response_format
    (best-effort) with no grammar key — the worker must not hard-fail."""
    import gbnf_grammar

    worker, captured = _make_worker(monkeypatch, OPENAI_COMPAT_PROVIDER="llamacpp")
    monkeypatch.setattr(
        gbnf_grammar, "json_schema_to_gbnf",
        lambda schema: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    result = worker.classify_image_bytes("ZmFrZQ==", "instruction")
    body = _body_of(captured)
    assert "grammar" not in body                # build failed → omitted
    assert body["response_format"]["type"] == "json_schema"
    assert result == {"ok": True}               # request still succeeded


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
