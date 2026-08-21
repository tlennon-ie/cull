"""Tests for the theme_config module + the /api/themes* dashboard routes.

Covers:
* slug validation + slugify preserves hyphens (unlike job_config.slugify).
* Round-trip write/read/delete for a user theme.
* Unknown vars keys are dropped.
* Malicious values (``<script>``, ``url(``, ``;`` injection) are rejected.
* /api/themes lists builtin themes on a clean workspace.
* /api/themes/<name> PUT + DELETE happy paths.
* /api/themes/<name>/publish is loopback-only (matches the presets publish
  contract — an unrelated attacker on the LAN cannot force a git push).
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pytest

PIPELINE_CODE = Path(__file__).resolve().parent.parent / "pipeline_code"
if str(PIPELINE_CODE) not in sys.path:
    sys.path.insert(0, str(PIPELINE_CODE))

try:
    import dotenv as _dotenv
    _dotenv.load_dotenv = lambda *a, **k: False  # type: ignore[assignment]
except Exception:  # pragma: no cover
    pass


@pytest.fixture()
def isolated_data(tmp_path, monkeypatch):
    """Point PIPELINE_BASE_DIR at a fresh dir so user themes stay isolated."""
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv("PIPELINE_BASE_DIR", str(data))
    return data


@pytest.fixture()
def theme_repo(tmp_path, monkeypatch, isolated_data):
    """A fake WORKSPACE_ROOT that has themes/builtin/ pre-populated."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "themes" / "builtin").mkdir(parents=True)
    (workspace / "themes" / "community").mkdir(parents=True)
    # Ship one fake builtin so list_themes has something to return.
    (workspace / "themes" / "builtin" / "sample.theme.json").write_text(json.dumps({
        "name": "sample",
        "font_family": "Inter, sans-serif",
        "vars": {"--color-bg": "#111111", "--color-accent": "#e879f9"},
    }))
    monkeypatch.setenv("WORKSPACE_ROOT", str(workspace))
    # Reload theme_config so it picks up the new WORKSPACE_ROOT env var (which
    # is only read at call time by _repo_root, so no reload needed for the
    # module — but we still return the workspace for assertions).
    return workspace


def test_slugify_preserves_hyphens():
    import theme_config
    assert theme_config.slugify("Ai Slop") == "ai_slop"
    assert theme_config.slugify("ai-slop") == "ai-slop"
    assert theme_config.slugify("  Cyber Punk!!! ") == "cyber_punk"


def test_is_valid_name_regex():
    import theme_config
    assert theme_config.is_valid_name("ai-slop")
    assert theme_config.is_valid_name("beige_v2")
    assert not theme_config.is_valid_name("")
    assert not theme_config.is_valid_name("A")  # uppercase rejected
    assert not theme_config.is_valid_name("has spaces")
    assert not theme_config.is_valid_name("path/traversal")
    assert not theme_config.is_valid_name("a" * 41)


def test_write_and_read_round_trip(theme_repo):
    import theme_config
    payload = {
        "font_family": "Inter, sans-serif",
        "vars": {"--color-bg": "#123456", "--color-accent": "#abcdef"},
    }
    path = theme_config.write_theme("my-theme", payload)
    assert path.is_file()
    got = theme_config.read_theme("my-theme")
    assert got is not None
    assert got["source"] == "user"
    assert got["font_family"] == "Inter, sans-serif"
    assert got["vars"]["--color-bg"] == "#123456"


def test_write_drops_unknown_vars(theme_repo):
    import theme_config
    theme_config.write_theme("guarded", {
        "font_family": "",
        "vars": {
            "--color-bg": "#000000",
            "--not-a-token": "#123",       # not in THEME_VAR_KEYS → dropped
            "--color-danger": "#ff0000",
        },
    })
    got = theme_config.read_theme("guarded")
    assert "--not-a-token" not in got["vars"]
    assert got["vars"]["--color-bg"] == "#000000"
    assert got["vars"]["--color-danger"] == "#ff0000"


def test_write_rejects_unsafe_values(theme_repo):
    import theme_config
    # Every one of these should be dropped from the stored vars — none of them
    # is a valid CSS colour value and each carries a CSS-injection risk.
    hostile = {
        "font_family": "",
        "vars": {
            "--color-bg": "red; background:url(//evil.example)",
            "--color-fg": "javascript:alert(1)",
            "--color-accent": "<script>alert(1)</script>",
            "--color-danger": "expression(alert(1))",
            "--color-warn": "#00ff00",   # sane value — should survive
        },
    }
    theme_config.write_theme("hostile", hostile)
    got = theme_config.read_theme("hostile")
    assert got is not None
    # Only the sane one survives.
    assert got["vars"] == {"--color-warn": "#00ff00"}


def test_delete_only_touches_user_layer(theme_repo):
    import theme_config
    theme_config.write_theme("temp", {"font_family": "", "vars": {"--color-bg": "#000000"}})
    assert theme_config.read_theme("temp") is not None
    assert theme_config.delete_theme("temp") is True
    assert theme_config.read_theme("temp") is None
    # Deleting a builtin should not touch the shipped file.
    assert theme_config.delete_theme("sample") is False
    assert theme_config.read_theme("sample") is not None


def test_list_themes_returns_builtin(theme_repo):
    import theme_config
    items = theme_config.list_themes()
    names = [t["name"] for t in items]
    assert "sample" in names


def test_read_theme_user_wins_over_builtin(theme_repo):
    import theme_config
    # User override of the same slug should shadow the builtin.
    theme_config.write_theme("sample", {"font_family": "", "vars": {"--color-bg": "#eeeeee"}})
    got = theme_config.read_theme("sample")
    assert got["source"] == "user"
    assert got["vars"]["--color-bg"] == "#eeeeee"


# ── Dashboard integration ────────────────────────────────────────────────────

@pytest.fixture()
def client(theme_repo, monkeypatch):
    """Flask test client with a fresh WORKSPACE_ROOT + PIPELINE_BASE_DIR."""
    import importlib
    if "dashboard_enhanced" in sys.modules:
        importlib.reload(sys.modules["dashboard_enhanced"])
    import dashboard_enhanced as dashboard
    monkeypatch.setattr(dashboard, "WORKSPACE_ROOT", theme_repo)
    dashboard.app.config.update(TESTING=True)
    return dashboard.app.test_client()


def test_api_themes_list(client, theme_repo):
    resp = client.get("/api/themes")
    assert resp.status_code == 200
    body = resp.get_json()
    assert "themes" in body
    assert isinstance(body["themes"], list)
    names = [t["name"] for t in body["themes"]]
    assert "sample" in names
    assert "core" in body
    assert set(body["core"]) == {"dark", "light", "hc"}
    assert "var_keys" in body
    assert "--color-bg" in body["var_keys"]


def test_api_themes_put_and_delete(client):
    payload = {
        "font_family": "system-ui, sans-serif",
        "vars": {"--color-bg": "#123456"},
    }
    resp = client.put("/api/themes/pytest-theme",
                      data=json.dumps(payload),
                      content_type="application/json")
    assert resp.status_code == 200, resp.get_json()
    assert resp.get_json()["ok"]

    resp = client.get("/api/themes/pytest-theme")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["ok"] and body["theme"]["source"] == "user"

    resp = client.delete("/api/themes/pytest-theme")
    assert resp.status_code == 200
    assert resp.get_json()["ok"]

    resp = client.get("/api/themes/pytest-theme")
    assert resp.status_code == 404


def test_api_themes_put_rejects_invalid_name(client):
    resp = client.put("/api/themes/BAD_NAME",
                      data="{}",
                      content_type="application/json")
    assert resp.status_code == 400
    assert not resp.get_json()["ok"]


def test_api_themes_delete_refuses_builtin(client):
    resp = client.delete("/api/themes/dark")   # a CORE_THEMES entry
    assert resp.status_code == 400
    assert not resp.get_json()["ok"]


def test_publish_is_loopback_only(client, monkeypatch):
    # Force the loopback probe to return False so a hostile caller is refused
    # even when running against 127.0.0.1 in tests.
    import dashboard_enhanced as dashboard
    monkeypatch.setattr(dashboard, "_wave_is_loopback_request", lambda: False)
    resp = client.post("/api/themes/whatever/publish")
    assert resp.status_code == 403


def test_install_is_loopback_only(client, monkeypatch):
    import dashboard_enhanced as dashboard
    monkeypatch.setattr(dashboard, "_wave_is_loopback_request", lambda: False)
    resp = client.post("/api/themes/sample/install")
    assert resp.status_code == 403
