"""Tests for the ``api_auth`` middleware.

Covers the four behaviour promises:

1. **Disabled by default** — with ``CULL_API_KEYS`` unset every existing route
   works exactly as before (this is the load-bearing back-compat contract).
2. **Enforced when tokens are set** — a non-loopback caller without a valid
   token gets 401; wrong token → 401; loopback bypasses entirely.
3. **Scope enforcement** — a read-only token can't write; a curator token
   can't hit admin URLs (settings / credentials / publish); admin covers all.
4. **Rate limiting on writes** — burst past the cap → 429; reads never trip
   the limiter.

Uses the middleware's injectable ``config`` / ``is_loopback`` hooks so we
never touch ``os.environ``: swapping the config on the app in the fixture is
cleaner and faster than reloading the dashboard.
"""
from __future__ import annotations

import importlib
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

for _k in ("CULL_API_KEYS", "CULL_API_KEY_SCOPES", "CULL_API_KEY_RATE_LIMIT"):
    os.environ.pop(_k, None)


TOKEN_A = "a" * 32
TOKEN_ADMIN = "b" * 40
TOKEN_READONLY = "c" * 32


def _fresh_app(monkeypatch, *, is_loopback=lambda: False):
    """Return a minimal Flask app + a couple of test routes.

    Kept independent of ``dashboard_enhanced`` — the middleware is standalone
    and can be exercised against any Flask app. That also keeps this test file
    from paying the ~2s cost of a dashboard reload.
    """
    import api_auth
    importlib.reload(api_auth)
    app = Flask(__name__)

    @app.route("/api/read", methods=["GET"])
    def read_route():
        return {"ok": True}

    @app.route("/api/write", methods=["POST"])
    def write_route():
        return {"ok": True}

    @app.route("/api/settings", methods=["GET"])
    def admin_route():
        return {"ok": True}

    @app.route("/plain")
    def plain():
        return "hello"

    return app, api_auth


def test_middleware_disabled_when_no_tokens(monkeypatch):
    """Empty ``CULL_API_KEYS`` → every request passes through.

    This is THE back-compat contract for the 995+ existing tests.
    """
    app, api_auth = _fresh_app(monkeypatch)
    api_auth.install(app, config=api_auth.AuthConfig(), is_loopback=lambda: False)
    client = app.test_client()

    assert client.get("/api/read").status_code == 200
    assert client.post("/api/write").status_code == 200
    assert client.get("/api/settings").status_code == 200
    assert client.get("/plain").status_code == 200


def test_missing_token_returns_401_when_enabled(monkeypatch):
    app, api_auth = _fresh_app(monkeypatch)
    cfg = api_auth.AuthConfig(tokens=frozenset({TOKEN_A}))
    api_auth.install(app, config=cfg, is_loopback=lambda: False)
    client = app.test_client()

    resp = client.get("/api/read")
    assert resp.status_code == 401
    body = resp.get_json()
    assert body["error"] == "invalid API key"
    assert body["success"] is False


def test_wrong_token_returns_401(monkeypatch):
    app, api_auth = _fresh_app(monkeypatch)
    cfg = api_auth.AuthConfig(tokens=frozenset({TOKEN_A}))
    api_auth.install(app, config=cfg, is_loopback=lambda: False)
    client = app.test_client()

    resp = client.get("/api/read", headers={"Authorization": "Bearer WRONG" * 8})
    assert resp.status_code == 401


def test_valid_bearer_token_accepted(monkeypatch):
    app, api_auth = _fresh_app(monkeypatch)
    cfg = api_auth.AuthConfig(tokens=frozenset({TOKEN_A}))
    api_auth.install(app, config=cfg, is_loopback=lambda: False)
    client = app.test_client()

    resp = client.get("/api/read", headers={"Authorization": f"Bearer {TOKEN_A}"})
    assert resp.status_code == 200


def test_valid_apikey_header_accepted(monkeypatch):
    """The custom ``X-Cull-API-Key`` header works too."""
    app, api_auth = _fresh_app(monkeypatch)
    cfg = api_auth.AuthConfig(tokens=frozenset({TOKEN_A}))
    api_auth.install(app, config=cfg, is_loopback=lambda: False)
    client = app.test_client()

    resp = client.get("/api/read", headers={"X-Cull-API-Key": TOKEN_A})
    assert resp.status_code == 200


def test_loopback_bypasses_auth(monkeypatch):
    """A loopback caller doesn't need a token even when tokens are configured."""
    app, api_auth = _fresh_app(monkeypatch)
    cfg = api_auth.AuthConfig(tokens=frozenset({TOKEN_A}))
    api_auth.install(app, config=cfg, is_loopback=lambda: True)
    client = app.test_client()

    resp = client.get("/api/read")
    assert resp.status_code == 200


def test_readonly_scope_cannot_write(monkeypatch):
    app, api_auth = _fresh_app(monkeypatch)
    cfg = api_auth.AuthConfig(
        tokens=frozenset({TOKEN_READONLY}),
        scopes={TOKEN_READONLY: frozenset({api_auth.SCOPE_READ})},
    )
    api_auth.install(app, config=cfg, is_loopback=lambda: False)
    client = app.test_client()

    read = client.get("/api/read", headers={"X-Cull-API-Key": TOKEN_READONLY})
    write = client.post("/api/write", headers={"X-Cull-API-Key": TOKEN_READONLY})

    assert read.status_code == 200
    assert write.status_code == 403
    assert write.get_json()["error"] == "insufficient scope"


def test_curator_scope_cannot_hit_admin(monkeypatch):
    """Even a GET on an admin URL requires the ``admin`` scope."""
    app, api_auth = _fresh_app(monkeypatch)
    cfg = api_auth.AuthConfig(
        tokens=frozenset({TOKEN_A}),
        scopes={TOKEN_A: frozenset({api_auth.SCOPE_READ, api_auth.SCOPE_CURATOR})},
    )
    api_auth.install(app, config=cfg, is_loopback=lambda: False)
    client = app.test_client()

    resp = client.get("/api/settings", headers={"X-Cull-API-Key": TOKEN_A})
    assert resp.status_code == 403


def test_admin_scope_covers_all(monkeypatch):
    """``admin`` implicitly grants ``curator`` and ``read``."""
    app, api_auth = _fresh_app(monkeypatch)
    cfg = api_auth.AuthConfig(
        tokens=frozenset({TOKEN_ADMIN}),
        scopes={TOKEN_ADMIN: frozenset({api_auth.SCOPE_ADMIN})},
    )
    api_auth.install(app, config=cfg, is_loopback=lambda: False)
    client = app.test_client()

    assert client.get("/api/read", headers={"X-Cull-API-Key": TOKEN_ADMIN}).status_code == 200
    assert client.post("/api/write", headers={"X-Cull-API-Key": TOKEN_ADMIN}).status_code == 200
    assert client.get("/api/settings", headers={"X-Cull-API-Key": TOKEN_ADMIN}).status_code == 200


def test_rate_limit_trips_on_burst(monkeypatch):
    app, api_auth = _fresh_app(monkeypatch)
    cfg = api_auth.AuthConfig(
        tokens=frozenset({TOKEN_A}),
        rate_limit_per_min=3,
    )
    api_auth.install(app, config=cfg, is_loopback=lambda: False)
    client = app.test_client()
    headers = {"X-Cull-API-Key": TOKEN_A}

    for _ in range(3):
        assert client.post("/api/write", headers=headers).status_code == 200
    resp = client.post("/api/write", headers=headers)
    assert resp.status_code == 429
    assert resp.get_json()["error"] == "rate limit exceeded"


def test_reads_dont_consume_rate_limit(monkeypatch):
    """Reads never trip the limiter (a polling dashboard can't lock itself out)."""
    app, api_auth = _fresh_app(monkeypatch)
    cfg = api_auth.AuthConfig(
        tokens=frozenset({TOKEN_A}),
        rate_limit_per_min=2,
    )
    api_auth.install(app, config=cfg, is_loopback=lambda: False)
    client = app.test_client()
    headers = {"X-Cull-API-Key": TOKEN_A}

    for _ in range(10):
        assert client.get("/api/read", headers=headers).status_code == 200


def test_constant_time_match_helper_exists():
    """The helper is an explicit part of the module surface (used in reviews)."""
    import api_auth
    importlib.reload(api_auth)
    assert callable(api_auth.constant_time_match)
    assert api_auth.constant_time_match(TOKEN_A, [TOKEN_A]) == TOKEN_A
    assert api_auth.constant_time_match(TOKEN_A, [TOKEN_ADMIN]) is None
    assert api_auth.constant_time_match("", [TOKEN_A]) is None


def test_short_tokens_are_dropped(monkeypatch):
    """A too-short token in ``CULL_API_KEYS`` is skipped, not passed through."""
    import api_auth
    importlib.reload(api_auth)
    cfg = api_auth.load_config_from_env({"CULL_API_KEYS": "short," + TOKEN_A})
    assert "short" not in cfg.tokens
    assert TOKEN_A in cfg.tokens


def test_scopes_json_ignored_when_malformed(monkeypatch):
    import api_auth
    importlib.reload(api_auth)
    cfg = api_auth.load_config_from_env({
        "CULL_API_KEYS": TOKEN_A,
        "CULL_API_KEY_SCOPES": "not-json",
    })
    # Falls back to the default scope set — auth still enabled.
    assert cfg.enabled
    assert cfg.scopes == {}


def test_rate_limit_defaults_when_env_garbage(monkeypatch):
    import api_auth
    importlib.reload(api_auth)
    cfg = api_auth.load_config_from_env({
        "CULL_API_KEYS": TOKEN_A,
        "CULL_API_KEY_RATE_LIMIT": "abc",
    })
    assert cfg.rate_limit_per_min == api_auth.DEFAULT_RATE_LIMIT


def test_non_api_paths_bypass_middleware(monkeypatch):
    """The middleware only gates /api/* — the HTML shell stays reachable."""
    app, api_auth = _fresh_app(monkeypatch)
    cfg = api_auth.AuthConfig(tokens=frozenset({TOKEN_A}))
    api_auth.install(app, config=cfg, is_loopback=lambda: False)
    client = app.test_client()

    assert client.get("/plain").status_code == 200


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
