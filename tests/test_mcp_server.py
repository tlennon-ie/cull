"""Tests for the cull MCP server (``pipeline_code/mcp_server.py``).

The MCP SDK is optional (``pip install cull[mcp]``); every test in this file
uses ``pytest.importorskip("mcp")`` because the tool implementations reference
SDK types (TextContent / ImageContent / CallToolResult) even in the
test-friendly ``dispatch`` path.

The tests DELIBERATELY do NOT spin up a stdio JSON-RPC session — the SDK's
own tests already exercise that. What we own here is:

  * every registered tool has a stable name + schema;
  * every tool goes through the SAME public APIs the dashboard/CLI use, so a
    round trip through job_config lands correctly and errors surface as MCP
    error envelopes instead of unhandled exceptions;
  * secret masking is intact (a stored API key never round-trips);
  * path-injection is barred on the one tool that dereferences a caller path.

Every test runs against a temp ``PIPELINE_BASE_DIR`` (same pattern as
``tests/test_cull_cli.py`` / ``tests/test_job_config.py``) so nothing touches
the real data dir, and no network-touching tool (``cull_export_hf``) is
exercised without a stub.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")  # skip the whole module when the SDK is absent

PIPELINE_CODE = Path(__file__).resolve().parent.parent / "pipeline_code"
if str(PIPELINE_CODE) not in sys.path:
    sys.path.insert(0, str(PIPELINE_CODE))


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture()
def isolated(tmp_path, monkeypatch):
    """Point every path helper at a temp dir and hand back (mcp_server, job_config, tmp).

    Mirrors ``tests/test_cull_cli.py``'s ``isolated`` fixture. Reloads
    ``job_config`` so its import-time path constants resolve into the temp
    dir; reloads ``mcp_server`` so its module-level state is fresh per test.
    """
    monkeypatch.setenv("PIPELINE_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("PIPELINE_QUEUE", str(tmp_path / "queue"))
    monkeypatch.setenv("PIPELINE_SORTED", str(tmp_path / "sorted"))
    monkeypatch.delenv("PIPELINE_SLUG", raising=False)

    import categories
    monkeypatch.setattr(categories, "ACTIVE_PATH", tmp_path / "cull_categories.json")
    monkeypatch.setattr(categories, "_cache", None, raising=False)
    monkeypatch.setattr(categories, "_cache_mtime", 0.0, raising=False)

    import job_config
    importlib.reload(job_config)
    import mcp_server
    importlib.reload(mcp_server)
    return mcp_server, job_config, tmp_path


def _text_payload(result) -> dict:
    """Unwrap a happy-path tool response (``list[TextContent]``) → parsed JSON."""
    assert isinstance(result, list) and result, f"expected non-empty list, got {result!r}"
    text_block = result[0]
    assert getattr(text_block, "type", None) == "text"
    return json.loads(text_block.text)


def _error_payload(result) -> dict:
    """Unwrap an error tool response (``CallToolResult`` with isError=True)."""
    from mcp.types import CallToolResult
    assert isinstance(result, CallToolResult), f"expected CallToolResult, got {type(result).__name__}"
    assert result.isError is True
    body = result.content[0]
    assert getattr(body, "type", None) == "text"
    return json.loads(body.text)


# ── Registry / smoke ────────────────────────────────────────────────────────

def test_module_imports_without_running_server(isolated):
    mcp_server, _jc, _ = isolated
    # ``main()`` must be exposed for the console script entry point to resolve.
    assert callable(mcp_server.main)


def test_registered_tool_names(isolated):
    mcp_server, _jc, _ = isolated
    names = set(mcp_server.list_tool_names())
    expected = {
        "cull_list_jobs", "cull_get_job", "cull_create_job", "cull_delete_job",
        "cull_activate_job", "cull_deactivate_job", "cull_set_job_priority",
        "cull_list_presets", "cull_get_preset", "cull_clone_preset",
        "cull_start_pipeline", "cull_stop_pipeline", "cull_pipeline_status",
        "cull_set_scoring", "cull_get_scoring",
        "cull_add_scraper_url", "cull_toggle_scraper",
        "cull_stats", "cull_sample_gallery", "cull_get_vision_meta",
        "cull_export_kohya", "cull_export_hf",
    }
    missing = expected - names
    assert not missing, f"missing tools: {sorted(missing)}"


def test_every_tool_has_object_schema_with_no_additional_properties(isolated):
    mcp_server, _jc, _ = isolated
    for name, (_desc, _fn, schema) in mcp_server.TOOLS.items():
        assert schema.get("type") == "object", f"{name}: schema.type must be 'object'"
        assert schema.get("additionalProperties") is False, \
            f"{name}: schema must set additionalProperties=false"


def test_build_server_wires_tools_defs(isolated):
    """Building the SDK Server registers every tool as a Tool object."""
    mcp_server, _jc, _ = isolated
    server = mcp_server._build_server()
    # ``server._tool_cache`` is only populated on list_tools invocation; instead
    # verify the decorator was called and the request-handlers table is armed.
    from mcp.types import ListToolsRequest, CallToolRequest
    assert ListToolsRequest in server.request_handlers
    assert CallToolRequest in server.request_handlers


# ── Presets ─────────────────────────────────────────────────────────────────

def test_list_presets_includes_builtin_default(isolated):
    mcp_server, jc, _ = isolated
    payload = _text_payload(mcp_server.dispatch("cull_list_presets", {}))
    names = {row["name"] for row in payload["presets"]}
    assert "default" in names
    assert payload["default"] == jc.default_preset_name()


def test_get_preset_unknown_is_error(isolated):
    mcp_server, _jc, _ = isolated
    err = _error_payload(mcp_server.dispatch("cull_get_preset", {"name": "no_such_preset"}))
    assert "unknown preset" in err["error"]


def test_clone_preset_round_trip(isolated):
    mcp_server, jc, _ = isolated
    result = mcp_server.dispatch("cull_clone_preset",
                                  {"source_name": "default", "new_name": "mine"})
    payload = _text_payload(result)
    assert payload["cloned"] is True
    lib = jc.list_presets()
    assert "mine" in lib["presets"]


# ── Jobs ────────────────────────────────────────────────────────────────────

def test_create_get_list_activate_delete_job_end_to_end(isolated):
    mcp_server, jc, _ = isolated
    # create
    created = _text_payload(mcp_server.dispatch(
        "cull_create_job", {"slug": "cyberpunk", "subject": "cyberpunk portraits"}))
    assert created["slug"] == "cyberpunk"

    # list — has our new job
    listed = _text_payload(mcp_server.dispatch("cull_list_jobs", {}))
    slugs = [row["slug"] for row in listed["jobs"]]
    assert "cyberpunk" in slugs

    # get — echoes subject + preset
    got = _text_payload(mcp_server.dispatch("cull_get_job", {"slug": "cyberpunk"}))
    assert got["subject"] == "cyberpunk portraits"
    assert got["preset"] == jc.default_preset_name()

    # activate — projects env + categories
    activated = _text_payload(mcp_server.dispatch(
        "cull_activate_job", {"slug": "cyberpunk", "exclusive": True}))
    assert activated["active"] == ["cyberpunk"]

    # cannot delete the active job
    err = _error_payload(mcp_server.dispatch("cull_delete_job", {"slug": "cyberpunk"}))
    assert "active" in err["error"].lower()

    # deactivate, then delete
    _text_payload(mcp_server.dispatch("cull_deactivate_job", {"slug": "cyberpunk"}))
    deleted = _text_payload(mcp_server.dispatch("cull_delete_job", {"slug": "cyberpunk"}))
    assert deleted["deleted"] is True


def test_invalid_slug_is_rejected_before_touching_disk(isolated):
    """Slug validator is the path-injection barrier for anything that indexes a
    per-slug file. Bad slugs never reach ``job_config.get_job``."""
    mcp_server, _jc, _ = isolated
    err = _error_payload(mcp_server.dispatch("cull_get_job", {"slug": "../etc/passwd"}))
    assert "invalid slug" in err["error"]


def test_get_job_unknown_slug_is_error(isolated):
    mcp_server, _jc, _ = isolated
    err = _error_payload(mcp_server.dispatch("cull_get_job", {"slug": "nope"}))
    assert "unknown job" in err["error"]


def test_set_job_priority_clamps_and_persists(isolated):
    mcp_server, jc, _ = isolated
    jc.create_job("mine")
    payload = _text_payload(mcp_server.dispatch(
        "cull_set_job_priority", {"slug": "mine", "priority": 42}))
    assert payload["priority"] == 10  # clamped
    assert jc.get_job_priority("mine") == 10


# ── Scoring + scraper mutation ─────────────────────────────────────────────

def test_set_scoring_writes_overrides(isolated):
    mcp_server, jc, _ = isolated
    jc.create_job("scored")
    _text_payload(mcp_server.dispatch(
        "cull_set_scoring",
        {"slug": "scored", "min_ovr": 70, "min_rel": 60, "require_prompt": False},
    ))
    got = _text_payload(mcp_server.dispatch("cull_get_scoring", {"slug": "scored"}))
    assert got["scoring"]["ovr_min"] == 70
    assert got["scoring"]["rel_min"] == 60
    assert got["require_prompt"] is False


def test_set_scoring_no_fields_is_error(isolated):
    mcp_server, jc, _ = isolated
    jc.create_job("scored")
    err = _error_payload(mcp_server.dispatch("cull_set_scoring", {"slug": "scored"}))
    assert "no scoring fields" in err["error"]


def test_add_scraper_url_dedupes_and_persists(isolated):
    mcp_server, jc, _ = isolated
    jc.create_job("gdl")
    first = _text_payload(mcp_server.dispatch("cull_add_scraper_url",
        {"slug": "gdl", "source": "gallery_dl", "url": "https://pixiv.net/user/1"}))
    assert first["count"] == 1
    # adding the same URL is a no-op
    again = _text_payload(mcp_server.dispatch("cull_add_scraper_url",
        {"slug": "gdl", "source": "gallery-dl", "url": "https://pixiv.net/user/1"}))
    assert again["count"] == 1
    # unknown source is an error
    err = _error_payload(mcp_server.dispatch("cull_add_scraper_url",
        {"slug": "gdl", "source": "bogus", "url": "x"}))
    assert "unsupported source" in err["error"]


def test_toggle_scraper_updates_enabled_map(isolated):
    mcp_server, jc, _ = isolated
    jc.create_job("t")
    result = _text_payload(mcp_server.dispatch("cull_toggle_scraper",
        {"slug": "t", "name": "X.com", "enabled": False}))
    assert result["enabled_map"]["X.com"] is False
    # unknown scraper is refused
    err = _error_payload(mcp_server.dispatch("cull_toggle_scraper",
        {"slug": "t", "name": "NotAScraper", "enabled": True}))
    assert "unknown scraper" in err["error"]


# ── Secret masking ─────────────────────────────────────────────────────────

def test_get_job_masks_fleet_api_key(isolated):
    """A vision-worker fleet entry with an api_key round-trips as the mask,
    never as the real credential."""
    mcp_server, jc, _ = isolated
    jc.create_job("secret")
    job = jc.get_job("secret")
    fleet = [{
        "id": "w1", "name": "custom", "provider": "lmstudio",
        "base_url": "http://127.0.0.1:1234", "model": "", "api_key": "sk-supersecret",
        "enabled": True,
    }]
    updated = jc.set_override(job, "vision.workers", fleet)
    jc.save_job(updated)
    payload = _text_payload(mcp_server.dispatch("cull_get_job", {"slug": "secret"}))
    workers = payload["effective_config"]["vision"]["workers"]
    assert workers[0]["api_key"] == mcp_server.SECRET_MASK
    assert "sk-supersecret" not in json.dumps(payload)


# ── Path safety on cull_get_vision_meta ────────────────────────────────────

def test_get_vision_meta_rejects_path_outside_roots(isolated, tmp_path):
    """A path outside queue/sorted is refused — the MCP surface must not
    dereference arbitrary files even when the caller lies about the path."""
    mcp_server, _jc, _ = isolated
    outside = tmp_path.parent / "not_in_queue.jpg.vision.json"
    outside.write_text("{}", encoding="utf-8")
    err = _error_payload(mcp_server.dispatch(
        "cull_get_vision_meta", {"image_path": str(outside).replace(".vision.json", "")},
    ))
    assert "not inside queue or sorted" in err["error"]


# ── Stats + gallery empty paths (index_store empty on a fresh temp dir) ────

def test_stats_on_empty_index(isolated):
    """A fresh install has no indexed images — stats must return empty shells
    instead of crashing."""
    mcp_server, _jc, _ = isolated
    import index_store
    index_store.configure(Path(str(isolated[2] / "index.sqlite3")))
    payload = _text_payload(mcp_server.dispatch("cull_stats", {}))
    assert payload["by_source"] == {"queue": {}, "sorted": {}}
    assert payload["by_category"] == {}


def test_sample_gallery_empty_returns_zero_count(isolated):
    mcp_server, jc, _ = isolated
    import index_store
    index_store.configure(Path(str(isolated[2] / "index.sqlite3")))
    jc.create_job("empty")
    payload = _text_payload(mcp_server.dispatch(
        "cull_sample_gallery", {"slug": "empty", "category": "Keep"}))
    assert payload["count"] == 0
    assert payload["items"] == []


# ── Export tools ───────────────────────────────────────────────────────────

def test_export_kohya_reports_zero_samples_on_empty_slug(isolated, tmp_path):
    """No sorted samples → export_dataset returns sample_count=0; the tool
    surfaces that cleanly instead of throwing."""
    mcp_server, jc, _ = isolated
    jc.create_job("kexp")
    out = tmp_path / "kohya_out"
    payload = _text_payload(mcp_server.dispatch(
        "cull_export_kohya", {"slug": "kexp", "out_dir": str(out)}))
    assert payload["sample_count"] == 0
    assert payload["profile"] == "kohya"
    assert Path(payload["out_dir"]).is_dir()


def test_export_hf_rejects_bad_repo(isolated):
    mcp_server, jc, _ = isolated
    jc.create_job("hf")
    err = _error_payload(mcp_server.dispatch(
        "cull_export_hf", {"slug": "hf", "repo": "no-slash"}))
    assert "namespace/name" in err["error"]


def test_export_hf_surfaces_credential_failure(isolated, monkeypatch):
    """When ``push_to_hf`` raises (no HF token available), the error surfaces
    as an MCP error envelope — it must NOT propagate SystemExit or tear the
    server down."""
    mcp_server, jc, _ = isolated
    jc.create_job("hf2")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)

    # Force a deterministic failure via a stub on push_to_hf so we don't touch
    # the network even if credentials happened to be present.
    import hf_export

    def _boom(*_a, **_k):
        raise ValueError("no token configured")

    monkeypatch.setattr(hf_export, "push_to_hf", _boom)
    err = _error_payload(mcp_server.dispatch(
        "cull_export_hf", {"slug": "hf2", "repo": "user/dataset"}))
    assert "HF push failed" in err["error"]


# ── Pipeline control (dashboard is down in tests) ──────────────────────────

def test_pipeline_status_when_dashboard_absent(isolated):
    """Without a live dashboard, the tool still returns filesystem-derived
    state; ``running`` degrades to False instead of raising."""
    mcp_server, _jc, _ = isolated
    payload = _text_payload(mcp_server.dispatch("cull_pipeline_status", {}))
    assert payload["running"] is False
    assert payload["active_slugs"] == []


def test_start_stop_pipeline_when_dashboard_absent(isolated):
    """Start/stop are reflexive — they call the local dashboard's REST API.
    Without one running, they surface a clear "dashboard unreachable" error."""
    mcp_server, _jc, _ = isolated
    for tool_name in ("cull_start_pipeline", "cull_stop_pipeline"):
        err = _error_payload(mcp_server.dispatch(tool_name, {}))
        assert "dashboard" in err["error"].lower()


# ── Unknown tool ───────────────────────────────────────────────────────────

def test_unknown_tool_yields_error_envelope(isolated):
    mcp_server, _jc, _ = isolated
    err = _error_payload(mcp_server.dispatch("cull_does_not_exist", {}))
    assert "unknown tool" in err["error"]


if __name__ == "__main__":  # python tests/test_mcp_server.py
    raise SystemExit(pytest.main([__file__, "-q"]))
