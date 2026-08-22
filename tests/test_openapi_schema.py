"""Tests for the OpenAPI 3.1 spec generator + its dashboard endpoint.

The generator scans ``app.url_map`` so it always stays in sync with the current
route set — no hand-maintained spec file to drift. These tests pin down:

* ``/api/openapi.json`` returns a valid OpenAPI 3.1 skeleton (``openapi``,
  ``info``, ``paths``, ``components``).
* Every ``/api/*`` route in the dashboard's URL map is represented.
* Path parameters (``<slug>``, ``<key>``) become ``{slug}`` / ``{key}`` OpenAPI
  path parameters with ``required: true``.
* Security schemes (``bearerAuth``, ``apiKeyAuth``) are declared under
  ``components.securitySchemes``.
* The Swagger UI HTML shell is served at ``/api/docs``.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

import pytest
from flask import Flask

PIPELINE_CODE = Path(__file__).resolve().parent.parent / "pipeline_code"
if str(PIPELINE_CODE) not in sys.path:
    sys.path.insert(0, str(PIPELINE_CODE))

try:
    import dotenv as _dotenv
    _dotenv.load_dotenv = lambda *a, **k: False  # type: ignore[assignment]
except Exception:  # pragma: no cover
    pass


# ── Isolated fixture (no dashboard reload cost) ──────────────────────────────

def _small_app():
    """Tiny Flask app that mirrors the shapes the dashboard uses."""
    app = Flask(__name__)

    @app.route("/api/status", methods=["GET"])
    def status():
        """Return status."""
        return {"ok": True}

    @app.route("/api/jobs/<slug>", methods=["GET", "PUT"])
    def job(slug):  # noqa: ARG001
        """Job endpoint with a slug parameter."""
        return {"ok": True}

    @app.route("/api/jobs/<slug>/webhooks", methods=["GET", "POST"])
    def webhooks(slug):  # noqa: ARG001
        return {"ok": True}

    @app.route("/plain")
    def plain():
        return "ignored"

    return app


def _get_spec():
    """Force a reload so we're testing the current source."""
    import openapi_schema
    return importlib.reload(openapi_schema)


def test_spec_has_openapi_3_1_skeleton():
    openapi_schema = _get_spec()
    spec = openapi_schema.build_spec(_small_app())
    assert spec["openapi"] == "3.1.0"
    assert "info" in spec and spec["info"]["title"] == "cull"
    assert "paths" in spec and isinstance(spec["paths"], dict)
    assert spec["servers"] == [{"url": "/"}]


def test_path_params_are_rewritten():
    openapi_schema = _get_spec()
    spec = openapi_schema.build_spec(_small_app())
    assert "/api/jobs/{slug}" in spec["paths"]
    assert "/api/jobs/{slug}/webhooks" in spec["paths"]
    # And ``<slug>`` is NEVER in the output as a raw fragment.
    assert not any("<slug>" in p for p in spec["paths"])


def test_path_params_are_declared_as_parameters():
    openapi_schema = _get_spec()
    spec = openapi_schema.build_spec(_small_app())
    op = spec["paths"]["/api/jobs/{slug}"]["get"]
    params = op.get("parameters", [])
    assert any(p["in"] == "path" and p["name"] == "slug" and p["required"] for p in params)


def test_non_api_routes_are_excluded():
    openapi_schema = _get_spec()
    spec = openapi_schema.build_spec(_small_app())
    assert all(p.startswith("/api/") for p in spec["paths"])


def test_security_schemes_present():
    openapi_schema = _get_spec()
    spec = openapi_schema.build_spec(_small_app())
    schemes = spec["components"]["securitySchemes"]
    assert "bearerAuth" in schemes
    assert schemes["bearerAuth"]["scheme"] == "bearer"
    assert "apiKeyAuth" in schemes
    assert schemes["apiKeyAuth"]["in"] == "header"


def test_docstring_becomes_summary():
    openapi_schema = _get_spec()
    spec = openapi_schema.build_spec(_small_app())
    assert spec["paths"]["/api/status"]["get"]["summary"] == "Return status."


def test_write_methods_get_request_body_slot():
    openapi_schema = _get_spec()
    spec = openapi_schema.build_spec(_small_app())
    post = spec["paths"]["/api/jobs/{slug}/webhooks"]["post"]
    assert "requestBody" in post
    assert post["requestBody"]["content"]["application/json"]["schema"]["type"] == "object"


def test_head_and_options_stripped():
    """Werkzeug adds HEAD/OPTIONS to every rule; they shouldn't appear."""
    openapi_schema = _get_spec()
    spec = openapi_schema.build_spec(_small_app())
    op_keys = set()
    for methods in spec["paths"].values():
        op_keys.update(methods.keys())
    assert "head" not in op_keys
    assert "options" not in op_keys


# ── End-to-end via the real dashboard ────────────────────────────────────────


@pytest.fixture()
def dashboard_client(tmp_path, monkeypatch):
    """Bring up the real dashboard on a temp store — exercises the routes."""
    monkeypatch.setenv("PIPELINE_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("PIPELINE_QUEUE", str(tmp_path / "queue"))
    monkeypatch.setenv("PIPELINE_SORTED", str(tmp_path / "sorted"))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("WORKSPACE_ROOT", str(tmp_path))
    (tmp_path / ".env").write_text("BLUR_NSFW_THUMBS=true\n", encoding="utf-8")
    for key in ("CULL_API_KEYS", "CULL_API_KEY_SCOPES", "CULL_API_KEY_RATE_LIMIT"):
        monkeypatch.delenv(key, raising=False)

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
    return dashboard.app.test_client(), dashboard


def test_dashboard_openapi_endpoint_returns_valid_spec(dashboard_client):
    client, _dash = dashboard_client
    resp = client.get("/api/openapi.json")
    assert resp.status_code == 200
    spec = resp.get_json()
    assert spec["openapi"] == "3.1.0"
    assert spec["info"]["title"] == "cull"
    # A representative API route the dashboard is known to expose.
    assert "/api/status" in spec["paths"]


def test_dashboard_openapi_covers_every_api_route(dashboard_client):
    client, dash = dashboard_client
    spec = client.get("/api/openapi.json").get_json()
    expected: set[str] = set()
    for rule in dash.app.url_map.iter_rules():
        raw = str(rule.rule or "")
        if not raw.startswith("/api/"):
            continue
        # Normalise the same way the generator does.
        import openapi_schema
        expected.add(openapi_schema._rule_to_openapi_path(raw))
    assert expected.issubset(set(spec["paths"].keys()))


def test_dashboard_swagger_docs_served(dashboard_client):
    client, _dash = dashboard_client
    resp = client.get("/api/docs")
    assert resp.status_code == 200
    assert b"swagger-ui" in resp.data
    assert b"/api/openapi.json" in resp.data
    assert resp.mimetype.startswith("text/html")


def test_openapi_spec_serialises_cleanly(dashboard_client):
    """The spec must round-trip through json.dumps (no non-serialisable values)."""
    client, _dash = dashboard_client
    payload = client.get("/api/openapi.json").get_json()
    # If this raises, the spec has a value json can't handle.
    json.dumps(payload)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
